# AutoResearch-RL — Semantic Analysis One-Pager

## Executive Thesis
AutoResearch-RL models **ML research itself** as a closed-loop control problem: a PPO-style agent edits a seed training script, a deterministic orchestration layer enforces hard constraints, and a telemetry-driven evaluator converts outcomes into structured reward for the next policy update. The system’s core semantic idea is **“code as action space”** rather than “hyperparameters as fixed knobs.”

---

## 1) System Intent and Optimization Target
- **Primary objective:** Minimize validation BPB (bits-per-byte), while respecting strict artifact-size, time, and causality constraints.
- **Search object:** Not just scalar params; the agent mutates Python source via structured diff patches.
- **Operational mode:** A perpetual loop with SOTA-tracking, experiment logging, and artifact promotion.

This turns the framework into an autonomous researcher that continuously proposes, tests, rejects, or adopts code mutations.

---

## 2) Semantic Architecture (Control-Flow View)
1. **State ingestion:** Current best code + recent history + runtime telemetry are transformed into an internal state vector.
2. **Action generation:** The policy samples a temperature/control signal and asks an LLM (or mock) to emit JSON patches.
3. **Patch realization:** AST-aware / whitespace-robust diff application rewrites candidate code.
4. **Guardrail gate:** Causality auditor + syntax smoke tests + artifact capacity estimation pre-filter invalid candidates.
5. **Execution stage:** GPU dispatcher runs candidate code in isolated subprocess/docker, streams JSON telemetry.
6. **Adaptive early stopping:** SPRT fits a power-law loss curve and aborts statistically non-competitive runs.
7. **Reward and learning:** Environment computes multi-term reward (delta-BPB, novelty, penalties) and PPO updates policy.
8. **Memory and promotion:** Structured logs persist trajectory; improved candidates become new SOTA artifacts.

**Semantic role split:**
- CPU side = verification, economics, control.
- GPU side = expensive evidence generation.
- RL layer = long-horizon decision policy over code edits.

---

## 3) Core Concepts and Their Meaning
### A. Code-Mutation MDP
The “state” includes the best script and prior outcomes; the “action” is a patch; the “reward” encodes scientific utility under constraints. This captures exploration/exploitation in research iterations.

### B. Constraint-First Intelligence
The orchestrator treats constraints as physical laws (size/time/causality). This sharply reduces wasted compute and prevents policy drift toward invalid shortcuts.

### C. Statistical Compute Triaging (SPRT)
Power-law extrapolation with confidence-aware bounds makes early-stop decisions before full training budget is spent, converting noisy trajectories into expected value of continuation.

### D. Reward Shaping for Search Behavior
The reward surface blends:
- performance gains (BPB improvement),
- novelty incentives (anti-mode-collapse),
- penalties for syntax/capacity/causality failures,
- graduated waste penalties for late aborts.

This creates an implicit “research economy” where useful novelty is rewarded and expensive dead-ends are discouraged.

### E. Seed-Centric Evolution
The seed script defines a high-leverage prior (compressed architecture ideas + tunable config), and the RL loop performs localized evolution around this prior rather than unconstrained generation.

---

## 4) Module-Level Semantic Responsibilities
- **`agent/ppo_agent.py`**: Converts system context into policy-guided mutation proposals; bridges policy output and LLM patch generation; applies patches robustly.
- **`agent/mdp_env.py`**: Encodes objective function semantics; transforms experiment outcomes into scalar learning signal.
- **`orchestrator/orchestrator.py`**: Static validation and artifact-size feasibility modeling; prevents invalid jobs from consuming expensive runtime.
- **`orchestrator/docker_runner.py`**: Isolated execution, telemetry ingestion, timeout/abort enforcement.
- **`gpu_cluster/sprt.py`**: Online statistical stopping policy for compute efficiency.
- **`auditor/causality_auditor.py`**: Static leak detector protecting benchmark integrity.
- **`main.py`**: Integrator of the perpetual research loop, logging, and SOTA artifact lifecycle.

---

## 5) Strengths, Risks, and Strategic Next Steps
### Strengths
- Well-separated control planes (policy vs validation vs execution).
- Practical safeguards against reward hacking via causality checks.
- Compute-aware experimentation via SPRT-based pruning.
- Persistent memory/logging that supports longitudinal optimization.

### Risks / Technical Debt
- Some behavior remains mock/simplified (telemetry source, synthetic patches, approximated capacity assumptions).
- Auditor heuristics may miss subtle leak channels or produce false positives.
- PPO update logic is intentionally lightweight vs production-grade on-policy training stacks.
- Potential brittleness when patching complex structural code edits.

### Recommended high-level roadmap
1. Expand causality auditing to hybrid static+dynamic trace-based checks.
2. Tighten reward calibration with explicit uncertainty handling and richer cost models.
3. Move from mostly single-step updates to batched trajectory PPO with better credit assignment.
4. Add semantic patch validation (AST equivalence/safety constraints) before dispatch.
5. Formalize experiment ontology for better retrieval and anti-regression memory.

---

## 6) One-Sentence System Summary
**AutoResearch-RL is an autonomous, constraint-aware research optimizer where PPO-guided code mutations are filtered by deterministic safety/economics gates and statistically triaged execution feedback to iteratively discover better compression-oriented training programs.**
