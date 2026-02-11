"""Basic package tests for deeplogbot."""

import importlib


def test_import_package():
    """Verify the deeplogbot package can be imported."""
    mod = importlib.import_module("deeplogbot")
    assert hasattr(mod, "run_bot_annotator")


def test_import_config():
    """Verify config module loads without errors."""
    from deeplogbot.config import load_config
    config = load_config()
    assert "classification" in config
    assert "deep_reconciliation" in config


def test_import_rules_classifier():
    """Verify rule-based classifier can be imported."""
    from deeplogbot.models.classification.rules import classify_locations
    assert callable(classify_locations)


def test_cli_entry_point():
    """Verify CLI entry point is importable."""
    from deeplogbot.main import main
    assert callable(main)
