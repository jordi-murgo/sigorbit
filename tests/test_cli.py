import pytest

from sigorbit import cli


def test_cli_explains_missing_api_extra(monkeypatch):
    monkeypatch.setattr(cli, "find_spec", lambda name: None if name == "fastapi" else object())
    with pytest.raises(SystemExit, match=r"sigorbit\[api\].*fastapi"):
        cli.run()
