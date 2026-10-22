"""Smoke tests: package imports and a tiny toy example (no data required)."""

import sweettv


def test_version():
    assert sweettv.__version__ == "0.1.0"


def test_toy_sum():
    assert sum([1, 2, 3]) == 6
