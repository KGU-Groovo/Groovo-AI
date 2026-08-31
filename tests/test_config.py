import importlib

import pytest

import app.config as config_module


def test_production_with_default_jwt_secret_fails_fast(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", "change-me")
    try:
        with pytest.raises(RuntimeError):
            importlib.reload(config_module)
    finally:
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        monkeypatch.delenv("JWT_SECRET", raising=False)
        importlib.reload(config_module)


def test_production_with_real_jwt_secret_starts(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", "a-real-shared-secret")
    try:
        importlib.reload(config_module)
        assert config_module.settings.jwt_secret == "a-real-shared-secret"
    finally:
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        monkeypatch.delenv("JWT_SECRET", raising=False)
        importlib.reload(config_module)
