from __future__ import annotations

import json
from itertools import product
from pprint import pprint

import pytest
from dimcat.base import DimcatObject
from dimcat.config import DimcatConfig, DimcatSchema
from dimcat.utils import get_schema
from marshmallow import ValidationError, fields
from marshmallow.class_registry import _registry as MM_REGISTRY


def obj_from_dict(obj_data):
    config = DimcatConfig(obj_data)
    return config.create()


def obj_from_json(json_data):
    obj_data = json.loads(json_data)
    return obj_from_dict(obj_data)


class BaseObject(DimcatObject):
    @classmethod
    def from_dict(cls, config, **kwargs):
        config = dict(config, **kwargs)
        config["dtype"] = cls.name
        schema = cls.Schema()
        return schema.load(config)

    @classmethod
    def from_json(cls, config):
        schema = cls.Schema()
        return schema.loads(config)

    class Schema(DimcatSchema):
        strong = fields.String(required=True)

    def __init__(self, strong: str):
        self.schema = get_schema(
            self.name
        )  # each object gets the same, cached instance
        self.strong = strong

    def to_dict(self) -> dict:
        return DimcatConfig.from_object(self)

    def to_json(self) -> dict:
        return self.schema.dumps(self)


class SubClass(BaseObject):
    def __init__(self, strong: str, weak: bool):
        super().__init__(strong=strong)
        self.weak = weak

    class Schema(BaseObject.Schema):
        weak = fields.Boolean(required=True)


class SubSubClass(SubClass):
    pass


def test_config():
    conf = DimcatConfig.from_object(BaseObject(strong="FORT"))
    print(conf)
    print(conf.validate())
    try:
        conf["invalid_key"] = 1
        raise RuntimeError("Should have raise ValueError")
    except ValueError:
        pass
    conf["strong"] = "test"
    new_base = conf.create()
    base_options = new_base.to_dict()
    sc = SubClass.from_dict(base_options, weak=True)
    sc_options = sc.to_dict()
    sc_options["dtype"] = BaseObject.name
    try:
        DimcatConfig(sc_options)
        raise RuntimeError("Should have raise ValueVError")
    except ValueError:
        pass


def test_subclass():
    b_s = BaseObject.Schema()
    sc_s = SubClass.Schema()
    ssc_s = SubSubClass.Schema()
    print(BaseObject.Schema.__qualname__, b_s.name)
    print(SubClass.Schema.__qualname__, sc_s.name)
    print(SubSubClass.Schema.__qualname__, ssc_s.name)
    b_before = BaseObject(strong="Schtrong")
    sc_before = SubClass(strong="strung", weak=True)
    ssc_before = SubSubClass(strong="Strunk", weak=False)
    for sch, obj in product((b_s, sc_s, ssc_s), (b_before, sc_before, ssc_before)):
        print(f"{sch.name} dumps {obj.name}:")
        try:
            dump = sch.dump(obj)
            print(dump)
        except ValidationError as e:
            print(e)
            continue
        new_obj = obj_from_dict(dump)
        assert obj.__dict__ == new_obj.__dict__
        json = obj.to_json()
        print(json)
        new_obj = obj_from_json(json)
        assert obj.__dict__ == new_obj.__dict__


def test_base():
    base = BaseObject(strong="sTrOnG")
    schema = base.Schema()
    config1 = schema.dump(base)
    config2 = base.to_dict()
    assert config1 == config2
    new_base = BaseObject.from_dict(config1)
    print(new_base.__dict__)


@pytest.fixture
def dummy_object():
    return BaseObject(strong="Dummy")


@pytest.fixture()
def dummy_config():
    return BaseObject.Schema()


def test_mm_registry():
    print(MM_REGISTRY)


def test_config_comparison():
    full1 = BaseObject.Schema()
    full2 = BaseObject.Schema()
    assert full1 != full2


def test_config_validation(dummy_object, dummy_config):
    serialized = dummy_config.dumps(dummy_object)
    print(dummy_object.__dict__)
    pprint(dummy_config.__dict__)
    report = dummy_config.validate(serialized)
    print(report)
