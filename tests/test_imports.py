import pytest

def test_imports():
    try:
        from kinetics import functions
        assert True  # If no ImportError is raised, the test passes
    except ImportError as e:
        assert False, f"ImportError occurred: {e}"