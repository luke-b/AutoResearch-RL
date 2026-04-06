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
