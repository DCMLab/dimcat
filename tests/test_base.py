from __future__ import annotations

import json
import os
import tempfile
from itertools import product
from pprint import pprint
from typing import List, Tuple, Type

import pandas as pd
import pytest
from dimcat.base import (
    DimcatConfig,
    DimcatObject,
    deserialize_dict,
    deserialize_json_file,
    deserialize_json_str,
)
from dimcat.data.base import Data
from dimcat.data.catalogs.base import DimcatCatalog
from dimcat.data.packages.dc import DimcatPackage
from dimcat.data.resources.base import Resource
from dimcat.data.resources.dc import DimcatResource
from dimcat.data.resources.features import Notes
from dimcat.steps.analyzers.base import Analyzer
from dimcat.steps.analyzers.counters import Counter
from dimcat.steps.base import FeatureProcessingStep
from dimcat.steps.loaders.base import Loader
from marshmallow import ValidationError, fields
from marshmallow.class_registry import _registry as MM_REGISTRY

from .conftest import CORPUS_PATH


class TestBaseObjects:
    @pytest.fixture(params=[DimcatObject, Data, FeatureProcessingStep, DimcatConfig])
    def dimcat_class(self, request):
        return request.param

    @pytest.fixture()
    def dimcat_object(self, dimcat_class):
        """Initializes each of the objects to be tested."""
        if dimcat_class == DimcatConfig:
            feature_config = DimcatConfig(dtype="Feature")
            return DimcatConfig(dtype="Analyzer", features=[feature_config])
        return dimcat_class()

    @pytest.fixture()
    def as_dict(self, dimcat_object: DimcatObject) -> dict:
        return dimcat_object.to_dict()

    @pytest.fixture()
    def as_config(self, dimcat_object: DimcatObject) -> DimcatConfig:
        return dimcat_object.to_config()

    def test_init(self, dimcat_object, dimcat_class):
        assert isinstance(dimcat_object, DimcatObject)
        assert isinstance(dimcat_object, dimcat_class)

    def test_serialization(self, as_dict, as_config):
        print(as_dict)
        print(as_config)
        assert as_dict == as_config

    def test_deserialization_via_constructor(self, as_dict, as_config, dimcat_class):
        from_dict = dimcat_class.from_dict(as_dict)
        assert isinstance(from_dict, dimcat_class)
        from_config = dimcat_class.from_config(as_config)
        assert isinstance(from_config, dimcat_class)
        assert from_dict.__dict__ == from_config.__dict__

    def test_deserialization_via_config(self, as_dict, as_config, dimcat_class):
        from_dict = DimcatConfig(as_dict).create()
        assert isinstance(from_dict, dimcat_class)
        from_config = as_config.create()
        assert isinstance(from_config, dimcat_class)
        assert from_dict.__dict__ == from_config.__dict__

    def test_deserialization_from_scratch(self, dimcat_class):
        options = dict(dtype=dimcat_class.dtype)
        if dimcat_class == DimcatConfig:
            options["options"] = dict(dtype="DimcatObject")
        config = DimcatConfig(options)
        obj = config.create()
        assert isinstance(obj, dimcat_class)


DUMMY_CONFIG_OPTIONS = dict(dtype="Feature")


def dummy_config() -> DimcatConfig:
    """Returns a dummy config for use in the test cases."""
    return DimcatConfig(**DUMMY_CONFIG_OPTIONS)


DIMCAT_OBJECT_TEST_CASES: List[Tuple[Type[DimcatObject], dict]] = [
    (DimcatObject, {}),
    (Data, {}),
    (FeatureProcessingStep, {}),
    (DimcatConfig, dummy_config()),
    (Resource, {}),
    (
        DimcatResource,
        {},
    ),
    (Notes, {}),
    (Analyzer, dict(features=dummy_config())),
    (Counter, dict(features=dummy_config())),
    (DimcatPackage, dict(package_name="test_package")),
    (DimcatCatalog, {}),
    (Loader, dict(basepath=CORPUS_PATH)),
]


def unpack_dimcat_object_params(params: tuple) -> Tuple[Type[DimcatObject], dict]:
    """Takes a tuple from DIMCAT_OBJECT_TEST_CASES and returns its components."""
    constructor, options = params
    return constructor, options


def arg_val2str(val) -> str:
    """Converts a value to a string for use in test ids."""
    if isinstance(val, str) and os.path.isfile(val):
        return os.path.basename(val)
    if isinstance(val, dict):
        return f"{{{kwargs2str(val)}}}"
    if isinstance(val, pd.DataFrame):
        return "[DataFrame]"
    return f"{val!r}"


def kwargs2str(options):
    """Converts a dictionary of keyword arguments to a string for use in test ids."""
    arg_str = ", ".join(f"{k}={arg_val2str(v)}" for k, v in options.items())
    return arg_str


def make_test_id(params: tuple) -> str:
    """Makes a test id from the parameters of DIMCAT_OBJECT_TEST_CASES."""
    constructor, options = unpack_dimcat_object_params(params)
    arg_str = kwargs2str(options)
    return f"{constructor.name}({arg_str})"


@pytest.fixture(
    scope="class",
    params=DIMCAT_OBJECT_TEST_CASES,
    ids=make_test_id,
)
def dimcat_object(request, tmp_path_factory):
    """Initializes each of the objects to be tested and injects them into the test class."""
    Constructor, options = unpack_dimcat_object_params(request.param)
    request.cls.object_constructor = Constructor
    dimcat_object = Constructor(**options)
    if isinstance(dimcat_object, DimcatResource) and not dimcat_object.is_frozen:
        tmp_path = str(tmp_path_factory.mktemp("dimcat_resource"))
        dimcat_object.basepath = tmp_path
        options["basepath"] = tmp_path
    request.cls.dimcat_object = dimcat_object
    request.cls.options = options


@pytest.mark.usefixtures("dimcat_object")
class TestSerialization:
    object_constructor: Type[DimcatObject]
    """The class of the tested object."""
    options: dict
    """The arguments for the tested object."""
    dimcat_object: DimcatObject
    """The initialized object."""

    def test_equality_with_other(self):
        new_object = self.object_constructor(**self.options)
        assert new_object == self.dimcat_object

    def test_equality_with_config(self):
        config = self.dimcat_object.to_config()
        assert config == self.dimcat_object

    def test_dict_config_equal(self):
        config = self.dimcat_object.to_config()
        as_dict = self.dimcat_object.to_dict()
        assert config == as_dict

    def test_creation_from_config(self):
        config = self.dimcat_object.to_config()
        new_object = config.create()
        a = self.dimcat_object.to_dict()
        b = new_object.to_dict()
        pprint(a, sort_dicts=False)
        pprint(b, sort_dicts=False)
        assert new_object == self.dimcat_object

    def test_creation_from_manual_config(self):
        options = dict(self.options)
        config = DimcatConfig(dtype=self.dimcat_object.name, options=options)
        new_object = config.create()
        assert new_object == self.dimcat_object

    def test_creation_from_dict(self):
        new_object = deserialize_dict(self.dimcat_object.to_dict())
        assert new_object == self.dimcat_object

    def test_serialization_to_json(self):
        json_str = self.dimcat_object.to_json()
        assert isinstance(json_str, str)
        assert json_str == json.dumps(self.dimcat_object.to_dict())
        new_object = deserialize_json_str(json_str)
        assert new_object == self.dimcat_object

    def test_serialization_to_json_file(self):
        with tempfile.NamedTemporaryFile(
            mode="r+", suffix=".json", encoding="utf-8"
        ) as tmp_file:
            self.dimcat_object.to_json_file(tmp_file.name)
            new_object = deserialize_json_file(tmp_file.name)
        assert new_object == self.dimcat_object


# BELOW IS PLAYGROUND CODE WAITING TO BE FACTORED IN


class BaseObject(DimcatObject):
    class Schema(DimcatObject.Schema):
        strong = fields.String(required=True)

    def __init__(self, strong: str):
        self.strong = strong


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
        raise RuntimeError("Should have raise ValidationError")
    except ValidationError:
        pass
    conf["strong"] = "test"
    new_base = conf.create()
    base_options = new_base.to_dict()
    sc = SubClass.from_dict(base_options, weak=True)
    sc_options = sc.to_dict()
    sc_options["dtype"] = BaseObject.name
    try:
        DimcatConfig(sc_options)
        raise RuntimeError("Should have raise ValidationError")
    except ValidationError:
        pass


def test_subclass():
    base_schema = BaseObject.Schema()
    subclass_schema = SubClass.Schema()
    subsubclass_schema = SubSubClass.Schema()
    print(BaseObject.Schema.__qualname__, base_schema.name)
    print(SubClass.Schema.__qualname__, subclass_schema.name)
    print(SubSubClass.Schema.__qualname__, subsubclass_schema.name)
    base_object_before = BaseObject(strong="Schtrong")
    subobject_before = SubClass(strong="strung", weak=True)
    subsubobject_before = SubSubClass(strong="Strunk", weak=False)
    for serialization_schema, serialized_object in product(
        (base_schema, subclass_schema, subsubclass_schema),
        (base_object_before, subobject_before, subsubobject_before),
    ):
        print(f"{serialization_schema.name} dumps {serialized_object.name}:")
        try:
            dump = serialization_schema.dump(serialized_object)
            print(dump)
        except ValidationError as e:
            print(e)
            continue
        try:
            new_obj = deserialize_dict(dump)
        except ValidationError as e:
            print(e)
            continue
        assert serialized_object.__dict__ == new_obj.__dict__
        json = serialized_object.to_json()
        print(json)
        new_obj = deserialize_json_str(json)
        assert serialized_object.__dict__ == new_obj.__dict__


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
