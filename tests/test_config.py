from __future__ import annotations

import json
from itertools import product
from pprint import pprint
from typing import Optional, Union
from zipfile import ZipFile

import frictionless as fl
import ms3
import pandas as pd
import pytest
from dimcat.base import DimcatObject, WrappedDataframe
from dimcat.config import DimcatConfig, DimcatSchema
from dimcat.utils import get_schema
from marshmallow import ValidationError, fields
from marshmallow.class_registry import _registry as MM_REGISTRY
from typing_extensions import Self


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

    def __init__(self, **kwargs):
        self.schema = get_schema(
            self.name
        )  # each object gets the same, cached instance

    def to_dict(self) -> dict:
        return DimcatConfig.from_object(self)

    def to_json(self) -> dict:
        return self.schema.dumps(self)


class DimcatResource(BaseObject):
    class Schema(DimcatSchema):
        resource = fields.Method(
            serialize="get_descriptor", deserialize="load_descriptor"
        )

        def get_descriptor(self, dc_resource: DimcatResource):
            if dc_resource.path is None:
                raise ValidationError(
                    f"Cannot serialize this {dc_resource.name} because it doesn't have a file path."
                )
            descriptor = dc_resource.resource.to_descriptor()
            return descriptor

        def load_descriptor(self, descriptor):
            return fl.Resource.from_descriptor(descriptor)

    def __init__(self, resource: Optional[fl.Resource] = None) -> None:
        super().__init__()
        self.resource: fl.Resource = fl.Resource()
        if resource is not None:
            if isinstance(resource, fl.Resource):
                self.resource = resource
            else:
                obj_name = self.__name__
                r_type = type(resource)
                msg = f"{obj_name} takes a frictionless.Resource, not {r_type}."
                if issubclass(r_type, str):
                    msg += f" Try {obj_name}.from_descriptor()"
                raise ValueError(msg)

    @property
    def path(self):
        return self.resource.path

    @path.setter
    def path(self, new_path):
        self.resource.path = new_path

    @classmethod
    def from_dataframe(cls, df, **kwargs):
        resource = fl.describe(df, **kwargs)
        # fl.validate(resource)
        return cls(resource=resource)

    @classmethod
    def from_descriptor(
        cls, descriptor: Union[fl.interfaces.IDescriptor, str], **options
    ) -> Self:
        resource = fl.Resource.from_descriptor(descriptor, **options)
        return cls(resource=resource)

    def __str__(self):
        return str(self.resource)

    def __repr__(self):
        return repr(self.resource)

    def get_pandas(
        self, wrapped=True
    ) -> Union[WrappedDataframe[pd.DataFrame], pd.DataFrame]:
        r = self.resource
        if r.path is None:
            raise ValidationError(
                "The resource does not refer to a file path and cannot be restored."
            )
        s = r.schema
        if r.normpath.endswith(".zip") or r.compression == "zip":
            zip_file_handler = ZipFile(r.normpath)
            df = ms3.load_tsv(zip_file_handler.open(r.innerpath))
        else:
            raise NotImplementedError()
        if len(s.primary_key) > 0:
            df = df.set_index(s.primary_key)
        if wrapped:
            return WrappedDataframe.from_df(df)
        return df


def test_dc_resource():
    df = ms3.load_tsv("~/dcml_corpora/tchaikovsky_seasons/harmonies/op37a10.tsv")
    resource = DimcatResource.from_dataframe(df=df)
    serialized = resource.to_dict()
    print(serialized)
    deserialized = obj_from_dict(serialized)
    restored_df = deserialized.get_pandas()
    print(restored_df)


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
