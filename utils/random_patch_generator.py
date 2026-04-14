"""
Random patch generator for baseline experiments.
"""

import random
import logging

logger = logging.getLogger("RandomPatchGenerator")


class RandomPatchGenerator:
    """Generates random hyperparameter mutations for baseline comparison."""

    # Valid ranges for each hyperparameter
    PARAM_RANGES = {
        "mlp_expansion": [2, 3, 4, 5],
        "lora_rank": [1, 2, 3, 4],
        "depth_loops": [1, 2, 3, 4],
        "block_size": [512, 1024, 2048],
        "n_embd": [128, 256, 384, 512],
        "n_head": [2, 4, 8],
    }

    @staticmethod
    def extract_current_values(code: str) -> dict:
        """Extract current hyperparameter values from code."""
        import ast
        import re

        values = {}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and hasattr(node.targets[0], "id"):
                    var_name = node.targets[0].id
                    if var_name in RandomPatchGenerator.PARAM_RANGES:
                        if isinstance(node.value, ast.Constant):
                            values[var_name] = node.value.value
        except Exception as e:
            logger.warning(f"Failed to extract values: {e}")

        return values

    @staticmethod
    def generate_random_patch(current_code: str) -> dict:
        """Generate a random hyperparameter mutation patch."""
        current_values = RandomPatchGenerator.extract_current_values(current_code)

        # Pick a random parameter that has at least 2 different possible values
        valid_params = [
            p for p in RandomPatchGenerator.PARAM_RANGES.keys()
            if len(RandomPatchGenerator.PARAM_RANGES[p]) > 1
        ]
        
        if not valid_params:
            raise ValueError("No valid parameters to mutate")
        
        param = random.choice(valid_params)

        # Get current and new values
        old_val = current_values.get(param, RandomPatchGenerator.PARAM_RANGES[param][0])
        
        # Ensure new value is different
        available_vals = [v for v in RandomPatchGenerator.PARAM_RANGES[param] if v != old_val]
        if not available_vals:
            # If all values are the same, pick any value
            available_vals = RandomPatchGenerator.PARAM_RANGES[param]
        
        new_val = random.choice(available_vals)

        # Generate search/replace format with proper indentation
        # Search: find the line with this assignment (may have leading spaces)
        search = f"    {param} = {old_val}"
        replace = f"    {param} = {new_val}"

        patch = {
            "search": search,
            "replace": replace,
        }

        logger.debug(f"Generated random patch: {param}: {old_val} → {new_val}")
        return patch

    @staticmethod
    def generate_json_array(current_code: str) -> str:
        """Generate a JSON array with a single random patch (for compatibility)."""
        import json

        patch = RandomPatchGenerator.generate_random_patch(current_code)
        return json.dumps([patch])
