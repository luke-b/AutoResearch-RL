import logging
from orchestrator.docker_runner import GPUDispatcher
import uuid

logging.basicConfig(level=logging.INFO)

# Let's write a mock SPRT that *will* abort it
def strict_sprt(step, loss):
    if step > 100 and loss > 0.5: # The mock runner's loss is ~1.1 at step 100
        return True
    return False

dispatcher = GPUDispatcher(sprt_callback=strict_sprt)
result = dispatcher.dispatch(str(uuid.uuid4()), "print('hello')", 12_000_000)
print(f"Final Result: {result}")
