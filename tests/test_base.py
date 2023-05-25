from __future__ import annotations

import os
from itertools import product
from pprint import pprint
from typing import Iterable, List, Tuple, Type

import frictionless as fl
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
    deserialize_json_str,
)
from dimcat.data.base import DimcatResource, ResourceStatus
from dimcat.features import Notes
from git import Repo
from marshmallow import ValidationError, fields, pre_load, validate
from marshmallow.class_registry import _registry as MM_REGISTRY

# Directory holding your clones of DCMLab/unittest_metacorpus
CORPUS_DIR = "~"


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


def corpus_path():
    """Compose the paths for the test corpora."""
    print("Path was requested")
    repo_name, test_commit = ("unittest_metacorpus", "e6fec84")
    path = os.path.join(CORPUS_DIR, repo_name)
    path = os.path.expanduser(path)
    assert os.path.isdir(path)
    repo = Repo(path)
    commit = repo.commit("HEAD")
    sha = commit.hexsha[: len(test_commit)]
    assert (
        sha == test_commit
    ), f"Your {path} is @ {sha}. Please do\n\tgit checkout {test_commit}."
    return path


CORPUS_PATH = corpus_path()

RESOURCE_PATHS = {
    file: os.path.join(CORPUS_PATH, file)
    for file in os.listdir(CORPUS_PATH)
    if file.endswith(".resource.yaml")
}


@pytest.fixture(params=RESOURCE_PATHS.values(), ids=RESOURCE_PATHS)
def resource_path(request):
    return request.param


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


def single_resource_path() -> str:
    """Returns the path to a single resource."""
    return RESOURCE_PATHS["unittest_notes.resource.yaml"]


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
        return kwargs2str(val)
    return str(val)


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
def dimcat_object(request):
    """Initializes each of the objects to be tested and injects them into the test class."""
    Constructor, options = unpack_dimcat_object_params(request.param)
    request.cls.dtype = Constructor
    request.cls.options = options
    request.cls.dimcat_object = Constructor(**options)


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


class TestResource:
    @pytest.fixture()
    def resource_from_descriptor(self, resource_path):
        resource = DimcatResource(resource=resource_path)
        return resource

    @pytest.fixture()
    def resource_schema(self, resource_path):
        res = fl.Resource(resource_path)
        return res.schema

    @pytest.fixture()
    def as_dict(self, resource_from_descriptor):
        return resource_from_descriptor.to_dict()

    @pytest.fixture()
    def as_config(self, resource_from_descriptor):
        return resource_from_descriptor.to_config()

    @pytest.fixture()
    def resource_dataframe(self, resource_from_descriptor):
        return resource_from_descriptor.get_dataframe()

    @pytest.fixture()
    def resource_from_dataframe(self, resource_dataframe, resource_schema):
        return DimcatResource(df=resource_dataframe, column_schema=resource_schema)

    def test_resource_from_disk(self, resource_from_descriptor):
        assert resource_from_descriptor.status == ResourceStatus.FROZEN
        print(resource_from_descriptor.__dict__)
        with pytest.raises(RuntimeError):
            resource_from_descriptor.basepath = "~"

    def test_resource_from_df(self, resource_from_dataframe):
        print(resource_from_dataframe)
        assert resource_from_dataframe.status in (
            ResourceStatus.VALIDATED,
            ResourceStatus.DATAFRAME,
        )
        as_config = resource_from_dataframe.to_config()
        print(as_config)
        os.remove(resource_from_dataframe.normpath)

    def test_serialization(self, as_dict, as_config):
        print(as_dict)
        print(as_config)
        assert as_dict == as_config

    def test_deserialization_via_config(self, as_dict, as_config):
        dc_config = DimcatConfig(as_dict)
        print(dc_config.dtype)
        print(dc_config.schema)
        from_dict = dc_config.create()
        assert isinstance(from_dict, DimcatResource)
        from_config = as_config.create()
        assert isinstance(from_config, DimcatResource)
        assert from_dict.__dict__.keys() == from_config.__dict__.keys()


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
