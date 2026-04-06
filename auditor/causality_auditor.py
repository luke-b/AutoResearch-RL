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
        Heuristic: Recursively looks for things like `i + 1` or `t + k` where `k > 0`.
        """
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                # E.g. t + 1
                if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)) and node.right.value > 0:
                    return True
                if isinstance(node.left, ast.Constant) and isinstance(node.left.value, (int, float)) and node.left.value > 0:
                    return True

            # Recursively check nested operations (e.g. data[(t + 1) * 2])
            return self._is_forward_looking(node.left) or self._is_forward_looking(node.right)

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
