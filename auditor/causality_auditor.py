import ast
import logging

logger = logging.getLogger("CausalityAuditor")

class CausalityAuditor(ast.NodeVisitor):
    def __init__(self):
        self.found_violation = False
        self.violation_details = []

    def visit_Subscript(self, node):
        # Look for suspicious slicing like data[i+1:] or data[t+1] which might indicate forward-looking
        if isinstance(node.slice, ast.Slice):
            # Standard generation slices like data[i:i+seq_len] are permitted.
            # We must be careful not to trigger false positives on standard context windows.
            # E.g., looking at `lower` for negative shifts is more important than `upper` for windowing.
            if self._is_forward_looking(node.slice.lower):
                self.found_violation = True
                self.violation_details.append(f"Suspicious forward slice (lower bound) at line {node.lineno}")
        elif isinstance(node.slice, ast.BinOp):
            if self._is_forward_looking(node.slice):
                self.found_violation = True
                self.violation_details.append(f"Suspicious forward indexing at line {node.lineno}")

        self.generic_visit(node)

    def visit_Call(self, node):
        # Look for calls that might load future data, e.g., dataset.get_future(), shift(-1)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ['shift', 'roll']:
                # Heuristic: shifting by a negative amount is often looking into the future
                if node.args and isinstance(node.args[0], ast.UnaryOp) and isinstance(node.args[0].op, ast.USub):
                     self.found_violation = True
                     self.violation_details.append(f"Suspicious shift/roll call at line {node.lineno}")

        self.generic_visit(node)

    def _is_forward_looking(self, node):
        """
        Simple heuristic: looking for things like `i + 1` or `t + something_positive`.
        In reality, this needs to be highly robust to catch sophisticated cheating.
        """
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                # E.g. t + 1
                if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)) and node.right.value > 0:
                    return True
                if isinstance(node.left, ast.Constant) and isinstance(node.left.value, (int, float)) and node.left.value > 0:
                    return True
        return False

def check_causality_leak(source_code: str) -> bool:
    """
    Checks the source code for potential causality leaks (forward-looking).
    Returns True if a leak is detected, False otherwise.
    """
    try:
        tree = ast.parse(source_code)
        auditor = CausalityAuditor()
        auditor.visit(tree)

        if auditor.found_violation:
            for detail in auditor.violation_details:
                logger.warning(f"Causality Violation: {detail}")
            return True

        return False
    except SyntaxError as e:
        logger.error(f"Cannot parse code for causality audit: {e}")
        # If we can't parse it, we consider it a failure, though technically it's a syntax error, not a leak.
        # It's safer to reject it.
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    clean_code = """
def get_batch(data, i):
    return data[i:i+seq_len]  # Wait, i to i+seq_len is normal for generating input.
    # Actually, predicting i+1 from i is standard.
    # The leak is using data[i+2] to predict data[i+1].
"""

    dirty_code = """
def eval_step(data, i):
    # Looking into the future to cheat on the current prediction
    future_token = data[i + 1]
    return evaluate(prediction, future_token)
"""

    print("Testing Clean Code:")
    leak1 = check_causality_leak(clean_code)
    print(f"Leak detected: {leak1}")

    print("\nTesting Dirty Code:")
    leak2 = check_causality_leak(dirty_code)
    print(f"Leak detected: {leak2}")
