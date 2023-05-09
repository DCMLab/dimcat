from pprint import pprint

import pytest
from dimcat.config import DimCatConfig
from marshmallow import fields


class DummyObject:
    def __init__(self):
        self.A = None
        self.string = "SomeString"


class DummyDimCatConfig(DimCatConfig):
    configured_object = DummyObject
    string = fields.String()


class DummyDimCatConfig2(DimCatConfig):
    pass


@pytest.fixture
def dummy_object():
    return DummyObject()


@pytest.fixture()
def dummy_config():
    return DummyDimCatConfig()


def test_object_gets_serialized(dummy_object, dummy_config):
    result = dummy_config.dumps(dummy_object)
    assert result["string"] == "SomeString"
    # Assert that unspecified attribute in the config are NOT serialized :
    assert "A" not in result.keys()


def test_class_builder_is_serialized_and_deserialized(dummy_object, dummy_config):
    serialized = dummy_config.dumps(dummy_object)
    assert "_configured_type" in serialized

    ds = DummyDimCatConfig.from_json(serialized)
    # Check that the serialized deserialized it with the good type
    assert ds["_configured_type"] == dummy_object.__class__
