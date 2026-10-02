import pytest

from call_companion.config import ConfigError, Settings, load_settings
from call_companion.models import ContextItem, sensitivity_rank


def test_unknown_sensitivity_fails_closed():
    item = ContextItem(id="1", source="x", sensitivity="top-secret-ish")
    assert item.sensitivity == "secret"
    assert sensitivity_rank("???") == sensitivity_rank("secret")


def test_missing_sensitivity_defaults_to_work():
    assert ContextItem(id="1", source="x", sensitivity=None).sensitivity == "work"


def test_kernel_may_add_fields_without_breaking_the_client():
    item = ContextItem.model_validate({"id": "1", "source": "x", "brand_new_field": 42})
    assert item.id == "1"


def test_defaults_without_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CALLC_CONFIG", str(tmp_path / "missing.toml"))
    settings = load_settings()
    assert settings.llm.model == "claude-opus-5-5"
    assert settings.live.assist == "on_hesitation"
    assert settings.ui.host == "127.0.0.1"


def test_load_toml_and_aliases(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        '[user]\nname = "Даня"\naliases = ["Даня", "Dan"]\n[live]\nassist = "always"\n'
        '[kernel.http]\nbase_url = "http://100.64.0.1:8765"\n',
        encoding="utf-8",
    )
    settings = load_settings(cfg)
    assert settings.user_aliases == ["Даня", "Dan"]
    assert settings.live.assist == "always"
    assert settings.kernel.http.base_url.endswith(":8765")


def test_explicit_missing_path_is_an_error(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "nope.toml")


def test_invalid_value_is_reported(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[live]\nassist = "sometimes"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(cfg)


def test_kernel_token_comes_from_env(monkeypatch):
    monkeypatch.setenv("KERNEL_COMPANION_TOKEN", "s3cret")
    assert Settings().kernel_token() == "s3cret"
    monkeypatch.delenv("KERNEL_COMPANION_TOKEN")
    assert Settings().kernel_token() is None


def test_the_example_config_is_valid_and_matches_the_defaults():
    import tomllib
    from pathlib import Path

    raw = tomllib.loads((Path(__file__).parents[1] / "examples" / "config.example.toml").read_text(encoding="utf-8"))
    example, defaults = Settings.model_validate(raw), Settings()
    for section in ("llm", "live", "asr", "capture", "ui", "privacy", "policy"):
        assert getattr(example, section) == getattr(defaults, section), section
    assert example.user.name == "Даня" and example.kernel.http.base_url


def test_kernel_side_modules_import_without_the_rest_of_the_package():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import call_companion.kernel_api, call_companion.kernel_server, call_companion.redact\n"
        "heavy = {'numpy', 'anthropic', 'rich', 'psutil', 'tomllib', 'httpx'} & set(sys.modules)\n"
        "assert not heavy, heavy\n"
    )
    src = str(__import__("pathlib").Path(__file__).parents[1] / "src")
    done = subprocess.run([sys.executable, "-c", code], env={"PYTHONPATH": src, "PATH": ""}, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
