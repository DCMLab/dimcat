import pytest
from dimcat.data.resources import Notes, ResourceStatus


def test_feature_creation(resource_object):
    if resource_object.status < ResourceStatus.DATAFRAME:
        pytest.skip("Empty resource.")
    _ = Notes.from_resource(resource_object)
