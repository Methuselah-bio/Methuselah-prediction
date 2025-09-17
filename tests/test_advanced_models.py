"""Tests for the advanced_models module.

These tests ensure that the function ``build_advanced_models`` returns
a dictionary of model specifications.  The test does not require
optional dependencies like transformers or torch to be installed;
missing dependencies should simply result in fewer models being
included.
"""

import importlib


def test_build_advanced_models_returns_dict() -> None:
    """build_advanced_models should always return a dict."""
    advanced_models = importlib.import_module('src.advanced_models')
    models = advanced_models.build_advanced_models({})
    assert isinstance(models, dict)