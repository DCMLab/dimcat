import os

import frictionless as fl
import pytest
from dimcat.base import deserialize_dict
from dimcat.resources.base import DimcatResource, ResourceStatus


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
        assert resource_from_descriptor.status == ResourceStatus.ON_DISK_NOT_LOADED
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
        from_dict = deserialize_dict(as_dict)
        assert isinstance(from_dict, DimcatResource)
        from_config = as_config.create()
        assert isinstance(from_config, DimcatResource)
        assert from_dict.__dict__.keys() == from_config.__dict__.keys()
