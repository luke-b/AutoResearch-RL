from orchestrator.orchestrator import Orchestrator

orc = Orchestrator()
size = orc.simulate_compression_and_capacity("def foo():\n    return 'bar'\n" * 1000)
print(f"Calculated size: {size} bytes")
