import logging
import os.path
from dataclasses import fields

import pytest
from dimcat.base import (
    DimcatConfig,
    DimcatSettings,
    load_settings,
    make_settings_from_config_file,
)
from marshmallow import ValidationError

logger = logging.getLogger(__name__)


@pytest.fixture()
def settings_ini_path():
    here = os.path.dirname(__file__)
    one_up = os.path.abspath(os.path.join(here, ".."))
    path = os.path.join(one_up, "src", "dimcat", "settings.ini")
    assert os.path.isfile(path)
    return path


def test_settings():
    # if the following line passes it means that the current settings.ini has been successfully
    # validated with DimcatSettings.Schema
    settings: DimcatConfig = load_settings(raise_exception=True)
    with pytest.raises(ValidationError):
        settings["non-field"] = "option"


def test_settings_complete(settings_ini_path):
    settings_ini: DimcatConfig = make_settings_from_config_file(
        settings_ini_path, fallback_to_default=False
    )
    settings_dataclass = DimcatSettings
    settings_schema = DimcatSettings.schema
    seti_keys = set(settings_ini)
    setd_keys = set(f.name for f in fields(settings_dataclass))
    sets_keys = set(f for f in settings_schema.declared_fields)
    seti_keys.remove("dtype")
    sets_keys.remove("dtype")
    assert seti_keys == setd_keys
    assert setd_keys == sets_keys
