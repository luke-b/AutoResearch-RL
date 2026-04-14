import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.ppo_agent import ASTDiffParser

def test_exact_match():
    code = "def foo():\n    return 1"
    search = "return 1"
    replace = "return 2"
    result = ASTDiffParser.apply_patch(code, search, replace)
    assert result == "def foo():\n    return 2"

def test_whitespace_insensitive_match():
    code = "def foo():\n\n    # A comment\n    return 1\n"
    search = "    # A comment\nreturn 1" # Different indentation and missing blank line
    replace = "    # New comment\n    return 2"
    result = ASTDiffParser.apply_patch(code, search, replace)
    assert "return 2" in result
    assert "New comment" in result

def test_search_not_found():
    code = "def foo():\n    return 1"
    search = "return 2"
    replace = "return 3"
    with pytest.raises(ValueError, match="Search block not found"):
        ASTDiffParser.apply_patch(code, search, replace)

def test_ast_assignment_match():
    code = "class GPTConfig:\n    mlp_expansion = 3\n    depth = 5"
    search = "mlp_expansion = 3"
    replace = "    mlp_expansion = 4"
    result = ASTDiffParser.apply_patch(code, search, replace)
    assert "mlp_expansion = 4" in result
    assert "depth = 5" in result # Unchanged

def test_ast_class_replacement():
    original_code = """
class OldClass:
    def method(self):
        return 1

class TargetClass:
    def method(self):
        return 2

class OtherClass:
    pass
"""
    search = """class TargetClass:
    def method(self):
        return 2"""
    replace = """class TargetClass:
    def method(self):
        return 3
    def new_method(self):
        return 4"""

    result = ASTDiffParser.apply_patch(original_code, search, replace)
    assert "return 3" in result
    assert "new_method" in result
    assert "OldClass" in result
    assert "return 1" in result
    assert "OtherClass" in result


def test_parse_llm_json_search_replace():
    llm_response = '```json\n[{"search": "mlp_expansion = 3", "replace": "mlp_expansion = 4"}]\n```'
    patches = ASTDiffParser.parse_llm_json(llm_response)
    assert patches[0]['search'] == 'mlp_expansion = 3'
    assert patches[0]['replace'] == 'mlp_expansion = 4'


def test_parse_llm_json_json_patch():
    llm_response = '```json\n[{"op": "replace", "path": "mlp_expansion", "value": 4}]\n```'
    patches = ASTDiffParser.parse_llm_json(llm_response)
    assert patches[0]['op'] == 'replace'
    assert patches[0]['path'] == 'mlp_expansion'
    assert patches[0]['value'] == 4


def test_parse_llm_json_operation_field():
    """Test that 'operation' field is normalized to 'op'."""
    llm_response = '```json\n[{"operation": "replace", "path": "mlp_expansion", "value": 4, "comment": "test"}]\n```'
    patches = ASTDiffParser.parse_llm_json(llm_response)
    assert patches[0]['op'] == 'replace'
    assert patches[0]['path'] == 'mlp_expansion'
    assert patches[0]['value'] == 4
    # original 'operation' field is preserved in the patch dict
    assert patches[0]['operation'] == 'replace'


def test_apply_json_patch_replace_assignment():
    original_code = "mlp_expansion = 3\nother = 5\n"
    patch = {"op": "replace", "path": "mlp_expansion", "value": 4}
    result = ASTDiffParser._apply_json_patch(original_code, patch)
    assert "mlp_expansion = 4" in result
    assert "other = 5" in result
