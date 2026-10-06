"""The solver-free guard must still catch a foreign package after subtracting the environment's baseline."""
import importlib.util

import pytest

from _solver_free import foreign_modules


def test_a_bare_array_stack_is_clean():
    assert foreign_modules("import numpy, jax.numpy") == []


@pytest.mark.skipif(importlib.util.find_spec("cvxpy") is None, reason="needs some optimization package installed")
def test_an_optimization_package_is_caught():
    assert "cvxpy" in foreign_modules("import numpy, cvxpy")
