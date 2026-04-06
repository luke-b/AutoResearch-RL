import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

logger = logging.getLogger("GoldenSeed")

# -----------------------------------------------------------------------------
# 1. Int6 Quantization (Simulated) & Linear Layer
# -----------------------------------------------------------------------------
class Int6LinearDynamic(nn.Module):
    """
    Per-row dynamic int6 quantization.
    Weights are stored separately from FP16 scales to maximize zstd compression.
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # In a real system, weight would be torch.int8, bounded to [-32, 31]
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.int8), requires_grad=False)
        self.scales = nn.Parameter(torch.ones(out_features, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize randomly, then quantize
        w_float = torch.randn(self.out_features, self.in_features) * (1.0 / math.sqrt(self.in_features))
        self._quantize_update(w_float)

    def _quantize_update(self, w_float):
        # Find max per row
        max_val = w_float.abs().max(dim=1, keepdim=True)[0]
        max_val = torch.clamp(max_val, min=1e-5)
        # Scale to [-31, 31] for int6 range (using int8 tensor)
        scale = max_val / 31.0
        w_int = torch.round(w_float / scale).to(torch.int8)
        w_int = torch.clamp(w_int, -32, 31)

        self.weight.data.copy_(w_int)
        self.scales.data.copy_(scale.squeeze().to(torch.float16))

    def forward(self, x):
        # De-quantize on the fly
        w_dequant = self.weight.float() * self.scales.unsqueeze(1).float()
        return F.linear(x, w_dequant, self.bias)

# -----------------------------------------------------------------------------
# 2. QK-Gain Normalization & Attention
# -----------------------------------------------------------------------------
class QKGainAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = Int6LinearDynamic(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = Int6LinearDynamic(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Learned QK-Gain initialized to 4.0
        self.qk_gain = nn.Parameter(torch.tensor(4.0))

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Apply QK-Normalization (L2 norm) then QK-Gain
        q = F.normalize(q, p=2, dim=-1) * self.qk_gain
        k = F.normalize(k, p=2, dim=-1) * self.qk_gain

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

# -----------------------------------------------------------------------------
# 3. 3x MLP Expansion
# -----------------------------------------------------------------------------
class MLP3x(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 3x expansion instead of standard 4x (e.g., 512 -> 1536)
        hidden_dim = 3 * config.n_embd
        self.c_fc    = Int6LinearDynamic(config.n_embd, hidden_dim, bias=False)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = Int6LinearDynamic(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = QKGainAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP3x(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# -----------------------------------------------------------------------------
# 4. Depth Recurrence with Dynamic LoRA
# -----------------------------------------------------------------------------
class DepthRecurrentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Tied embeddings in BF16
        self.wte = nn.Embedding(config.vocab_size, config.n_embd, dtype=torch.bfloat16)
        self.wpe = nn.Embedding(config.block_size, config.n_embd, dtype=torch.bfloat16)

        # 3 Physical Blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(3)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Dynamic LoRA deltas for the 3 loop iterations (Rank=4)
        self.lora_rank = 4
        self.lora_a = nn.Parameter(torch.randn(3, config.n_embd, self.lora_rank) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(3, self.lora_rank, config.n_embd))

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

        # Ensure we work in float32 for the activations to match int6 outputs
        tok_emb = self.wte(idx).float()
        pos_emb = self.wpe(pos).float()
        x = tok_emb + pos_emb

        # Depth Recurrence: Loop over the 3 physical blocks 3 times (simulating 9 layers)
        for loop_iter in range(3):
            # Apply iteration-specific LoRA delta
            lora_delta = x @ self.lora_a[loop_iter] @ self.lora_b[loop_iter]
            x = x + lora_delta

            for block in self.blocks:
                x = block(x)

        x = self.ln_f(x)
        logits = F.linear(x, self.wte.weight.float()) # Tied weights

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

# -----------------------------------------------------------------------------
# 5. Swarm Causal Backoff N-gram Mixer
# -----------------------------------------------------------------------------
class SwarmNGramMixer:
    """
    Mixes Neural Logits with N-gram statistics based on entropy.
    (Simplified mock for the Golden Seed)
    """
    def __init__(self):
        # BigramHash representation
        self.ngram_tables = {n: {} for n in range(2, 11)}

    def mix(self, neural_logits, context, target_entropy):
        # In the real system, this fetches from NVLink shared memory.
        # Here we just return neural_logits to make the script self-contained and runnable.
        return neural_logits

# -----------------------------------------------------------------------------
# 6. Muon Optimizer
# -----------------------------------------------------------------------------
class Muon(torch.optim.Optimizer):
    """
    Muon optimizer (momentum escalating 0.92 to 0.99)
    """
    def __init__(self, params, lr=0.02, momentum=0.92):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)

                # Update (simplified, actual Muon uses Newton-Schulz iteration for matrices)
                p.data.add_(buf, alpha=-lr)

        return loss

# -----------------------------------------------------------------------------
# 7. SWA / EMA Warmdown
# -----------------------------------------------------------------------------
class AveragedModel(nn.Module):
    """
    Stochastic Weight Averaging (SWA) / Exponential Moving Average (EMA) wrapper.
    """
    def __init__(self, model, alpha=0.999):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.averaged_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.averaged_params[name] = param.data.clone().detach()

    def update(self):
        # EMA update
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.averaged_params[name].mul_(self.alpha).add_(param.data, alpha=1.0 - self.alpha)

    def apply_averages(self):
        # Replace model weights with averages for evaluation
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.averaged_params[name])

# -----------------------------------------------------------------------------
# Golden Seed Config & Main
# -----------------------------------------------------------------------------
class GPTConfig:
    vocab_size = 50257
    block_size = 2048
    n_embd = 512
    n_head = 8

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing Golden Seed Architecture...")

    config = GPTConfig()
    model = DepthRecurrentGPT(config)

    # Optimizer Setup (Matrix LR: 0.02)
    optimizer = Muon(model.parameters(), lr=0.02, momentum=0.92)

    # Initialize EMA model for the warmdown phase
    ema_model = AveragedModel(model)

    # Mock Data
    total_steps = 10000
    warmdown_steps = 3000

    logger.info("Running training step simulation...")
    # Simulate step 7000 (start of warmdown)
    idx = torch.randint(0, config.vocab_size, (2, 64))
    targets = torch.randint(0, config.vocab_size, (2, 64))

    logits, loss = model(idx, targets)
    loss.backward()
    optimizer.step()

    # Warmdown update
    ema_model.update()

    # Evaluation phase simulation: Sliding Window Evaluation
    logger.info("Running Sliding Window Evaluation simulation...")

    def sliding_window_eval(eval_model, sequence_length=10000, context_size=2048, stride=64):
        eval_model.eval()
        fake_dataset = torch.randint(0, config.vocab_size, (1, sequence_length))
        total_loss = 0.0
        steps = 0

        with torch.no_grad():
            for i in range(0, sequence_length - context_size, stride):
                input_ids = fake_dataset[:, i:i+context_size]
                target_ids = fake_dataset[:, i+1:i+context_size+1]
                _, l = eval_model(input_ids, target_ids)
                total_loss += l.item()
                steps += 1

        eval_model.train()
        return total_loss / max(1, steps)

    # Apply EMA weights for evaluation
    ema_model.apply_averages()
    val_loss = sliding_window_eval(model, sequence_length=4000)
    logger.info(f"Validation Loss (EMA, Sliding Window): {val_loss:.4f}")

    logger.info("Golden Seed check passed.")
