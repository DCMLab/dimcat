import pytest
from dimcat.base import DimcatConfig, load_settings
from marshmallow import ValidationError


def test_settings():
    settings: DimcatConfig = load_settings()
    with pytest.raises(ValidationError):
        settings["non-field"] = "option"
    for setting in settings:
        if setting in ("dtype", "default_basepath"):
            continue
        with pytest.raises(ValidationError):
            settings[setting] = "invalid option"
