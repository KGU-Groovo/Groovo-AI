import importlib

import pytest

import app.config as config_module


def test_default_reference_keypoints_register_all_frontend_song_ids():
    assert set(config_module.settings.reference_keypoints) == {
        "hollywood-action",
        "rude",
        "its-me",
        "wda",
        "bad",
    }


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
