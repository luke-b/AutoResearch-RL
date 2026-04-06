from seed.train_gpt import *

config = GPTConfig()
config.mlp_expansion = 4
model = DepthRecurrentGPT(config)
