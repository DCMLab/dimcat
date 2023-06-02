import pytest
from dimcat import DimcatConfig
from dimcat.base import load_settings
from marshmallow import ValidationError


def test_settings():
    settings: DimcatConfig = load_settings()
    with pytest.raises(ValidationError):
        settings["non-field"] = "option"
    for setting in settings:
        if setting == "dtype":
            continue
        with pytest.raises(ValidationError):
            settings[setting] = "invalid option"
