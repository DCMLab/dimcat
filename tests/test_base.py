from __future__ import annotations

import json
import os
import tempfile
from itertools import product
from pprint import pprint
from typing import Iterable, List, Tuple, Type

import pandas as pd
import pytest
from dimcat.analyzers import Counter
from dimcat.analyzers.base import Analyzer
from dimcat.base import (
    Data,
    DimcatConfig,
    DimcatObject,
    DimcatSchema,
    PipelineStep,
    deserialize_dict,
    deserialize_json_file,
    deserialize_json_str,
)
from dimcat.dataset.base import DimcatCatalog
from dimcat.resources.base import DimcatResource
from dimcat.resources.features import Notes
from marshmallow import ValidationError, fields, pre_load, validate
from marshmallow.class_registry import _registry as MM_REGISTRY

from tests.conftest import single_resource_path


class DummyAnalyzer(PipelineStep):
    """This class simulates the aspect of embedding a list of configs."""

    class Schema(PipelineStep.Schema):
        features = fields.List(
            fields.Nested(DimcatConfig.Schema), validate=validate.Length(min=1)
        )

        @pre_load()
        def features_as_list(self, obj, **kwargs):
            features = self.get_attribute(obj, "features", None)
            if features is not None and not isinstance(features, list):
                try:
                    obj.features = [obj.features]
                except AttributeError:
                    obj["features"] = [obj["features"]]
            return obj

    def __init__(
        self,
        features: DimcatConfig | Iterable[DimcatConfig],
    ):
        self._features: List[DimcatConfig] = []
        self.features = features

    @property
    def features(self) -> List[DimcatConfig]:
        return self._features

    @features.setter
    def features(self, features):
        if isinstance(features, DimcatConfig):
            features = [features]
        configs = []
        for config in features:
            if isinstance(config, DimcatConfig):
                configs.append(config)
            else:
                raise ValueError(f"Not a DimcatConfig: {config}")
        if len(configs) == 0:
            raise ValidationError(
                f"{self.name} requires at least one feature to analyze."
            )
        self._features = configs


def test_dummy_analyzer():
    config = DimcatConfig(dtype="DimcatObject")
    report = DummyAnalyzer.schema.validate({"features": [config]})
    assert len(report) == 0


class TestBaseObjects:
    @pytest.fixture(
        params=[DimcatObject, Data, PipelineStep, DimcatConfig, DummyAnalyzer]
    )
    def dimcat_class(self, request):
        return request.param

    @pytest.fixture()
    def dimcat_object(self, dimcat_class):
        """Initializes each of the objects to be tested."""
        if dimcat_class == DimcatConfig:
            feature_config = DimcatConfig(dtype="DimcatObject")
            return DimcatConfig(dtype="DummyAnalyzer", features=[feature_config])
        if dimcat_class == DummyAnalyzer:
            feature_config = DimcatConfig(dtype="DimcatObject")
            return DummyAnalyzer(features=feature_config)
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
        elif dimcat_class == DummyAnalyzer:
            options["features"] = [DimcatConfig(dtype="DimcatObject")]
        config = DimcatConfig(options)
        obj = config.create()
        assert isinstance(obj, dimcat_class)


DUMMY_CONFIG_OPTIONS = dict(dtype="Notes")


def dummy_config() -> DimcatConfig:
    """Returns a dummy config for use in the test cases."""
    return DimcatConfig(**DUMMY_CONFIG_OPTIONS)


DIMCAT_OBJECT_TEST_CASES: List[Tuple[Type[DimcatObject], dict]] = [
    (DimcatObject, {}),
    (Data, {}),
    (PipelineStep, {}),
    (DimcatConfig, dummy_config()),
    (DimcatResource, dict(resource=single_resource_path())),
    (DummyAnalyzer, dict(features=dummy_config())),
    (Notes, dict(resource=single_resource_path())),
    (Analyzer, dict(features=dummy_config())),
    (Counter, dict(features=dummy_config())),
    (DimcatCatalog, {}),
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
    request.cls.dtype = Constructor
    dimcat_object = Constructor(**options)
    if isinstance(dimcat_object, DimcatResource) and not dimcat_object.is_frozen:
        tmp_path = tmp_path_factory.mktemp("dimcat_resource")
        dimcat_object.basepath = tmp_path
        options["basepath"] = tmp_path
    request.cls.dimcat_object = dimcat_object
    request.cls.options = options


@pytest.mark.usefixtures("dimcat_object")
class TestSerialization:
    dtype: Type[DimcatObject]
    """The class of the tested object."""
    options: dict
    """The arguments for the tested object."""
    dimcat_object: DimcatObject
    """The initialized object."""

    def test_equality_with_other(self):
        new_object = self.dtype(**self.options)
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
        a = new_object.to_dict()
        b = self.dimcat_object.to_dict()
        print(a, type(a))
        print(b, type(b))
        assert new_object == self.dimcat_object

    def test_creation_from_manual_config(self):
        options = dict(self.options)
        if "basepath" in options:
            tmp_path = options.pop("basepath")
            os.chdir(tmp_path)
        config = DimcatConfig(dtype=self.dtype.name, options=options)
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
    class Schema(DimcatSchema):
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
        new_obj = deserialize_dict(dump)
        assert obj.__dict__ == new_obj.__dict__
        json = obj.to_json()
        print(json)
        new_obj = deserialize_json_str(json)
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
