import os

import pytest
from dimcat.base import get_setting
from dimcat.data.resources.base import (
    DimcatIndex,
    DimcatResource,
    PieceIndex,
    Resource,
    ResourceStatus,
)
from dimcat.utils import make_valid_frictionless_name

from .conftest import CORPUS_PATH, get_mixed_score_paths

# region Resource


class TestBaseResource:
    @pytest.fixture()
    def resource_obj(self):
        return Resource()

    @pytest.fixture()
    def expected_basepath(self):
        return None

    @pytest.fixture()
    def expected_filepath(self):
        return ""

    @pytest.fixture()
    def expected_normpath(self):
        return None

    @pytest.fixture()
    def expected_resource_name(self):
        return get_setting("default_resource_name")

    def test_basepath_after_init(self, resource_obj, expected_basepath):
        assert resource_obj.basepath == expected_basepath

    def test_filepath_after_init(self, resource_obj, expected_filepath):
        assert resource_obj.filepath == expected_filepath

    def test_normpath_after_init(self, resource_obj, expected_normpath):
        assert resource_obj.normpath == expected_normpath

    def test_resource_name_after_init(self, resource_obj, expected_resource_name):
        assert resource_obj.resource_name == expected_resource_name


class TestResourceFromAbsolute(TestBaseResource):
    @pytest.fixture(
        params=get_mixed_score_paths(),
        ids=os.path.basename,
    )
    def score_path(self, request):
        return request.param

    @pytest.fixture()
    def resource_obj(self, score_path):
        return Resource(score_path)

    @pytest.fixture()
    def expected_basepath(self, score_path):
        return os.path.split(score_path)[0]

    @pytest.fixture()
    def expected_filepath(self, score_path):
        return os.path.split(score_path)[1]

    @pytest.fixture()
    def expected_normpath(self, score_path):
        return score_path

    @pytest.fixture()
    def expected_resource_name(self, score_path):
        file_name, _ = os.path.splitext(os.path.basename(score_path))
        return make_valid_frictionless_name(file_name)


class TestResourceReconciledAbsolute(TestResourceFromAbsolute):
    @pytest.fixture()
    def basepath(self, score_path):
        path, _ = os.path.split(score_path)
        basepath, _ = os.path.split(path)
        return basepath

    @pytest.fixture()
    def resource_obj(self, score_path, basepath):
        return Resource(score_path, basepath=basepath)

    @pytest.fixture()
    def expected_basepath(self, basepath):
        return basepath

    @pytest.fixture()
    def expected_filepath(self, score_path, expected_basepath):
        return os.path.relpath(score_path, expected_basepath)


# endregion Resource

# region DimcatResource


class TestEmptyResource:
    expected_resource_status: ResourceStatus = ResourceStatus.EMPTY
    """The expected status of the resource after initialization."""
    should_be_frozen: bool = False
    """Whether the resource should be frozen, i.e., immutable after initialization."""
    should_be_serialized: bool = False
    """Whether the resource should figure as serialized after initialization."""
    should_be_loaded: bool = False
    """Whether or not we expect the resource to have a dataframe loaded into memory."""
    should_have_descriptor: bool = False
    """Whether or not we expect the resource to have a descriptor file on disk."""

    @pytest.fixture()
    def expected_basepath(self):
        """The expected basepath of the resource after initialization."""
        return None

    @pytest.fixture()
    def dc_resource(self, empty_resource):
        """For each subclass of TestVanillaResource, this fixture should be overridden and yield the
        tested DimcatResource object."""
        return empty_resource

    def test_basepath_after_init(self, dc_resource, expected_basepath):
        assert dc_resource.basepath == expected_basepath

    def test_status_after_init(self, dc_resource):
        assert dc_resource.status == self.expected_resource_status

    def test_frozen(self, dc_resource):
        assert dc_resource.is_frozen == self.should_be_frozen

    def test_valid(self, dc_resource):
        report = dc_resource.validate()
        assert report is None or report.valid
        assert dc_resource.is_valid

    def test_serialized(self, dc_resource):
        assert dc_resource.is_serialized == self.should_be_serialized

    def test_loaded(self, dc_resource):
        assert dc_resource.is_loaded == self.should_be_loaded

    def test_descriptor_path(self, dc_resource):
        descriptor_path = dc_resource.descriptor_filepath
        if self.should_have_descriptor:
            assert descriptor_path is not None
        else:
            assert descriptor_path is None

    def test_copy(self, dc_resource):
        copy = dc_resource.copy()
        assert copy == dc_resource
        assert copy is not dc_resource
        assert copy.status == dc_resource.status


class TestDiskResource(TestEmptyResource):
    expected_resource_status = ResourceStatus.STANDALONE_NOT_LOADED
    should_be_frozen: bool = True
    should_be_serialized = True
    should_have_descriptor = True

    @pytest.fixture()
    def expected_basepath(self):
        return CORPUS_PATH

    @pytest.fixture()
    def dc_resource(self, resource_from_descriptor):
        return resource_from_descriptor


class TestResourceFromFrozen(TestDiskResource):
    @pytest.fixture()
    def dc_resource(self, resource_from_frozen_resource):
        return resource_from_frozen_resource


class TestMemoryResource(TestEmptyResource):
    """MemoryResources are those instantiated from a dataframe. They have in common that, in this
    test suite, their basepath is a temporary path where they can be serialized."""

    should_be_loaded = False  # because this one is empty

    @pytest.fixture()
    def expected_basepath(self, tmp_serialization_path):
        return tmp_serialization_path

    @pytest.fixture()
    def dc_resource(self, empty_resource_with_paths):
        return empty_resource_with_paths

    @pytest.mark.skip(
        reason="column_schema currently expresses types for reading from disk, not for loaded daata."
    )
    def test_valid(self, dc_resource):
        pass


class TestSchemaResource(TestEmptyResource):
    expected_resource_status = ResourceStatus.SCHEMA

    @pytest.fixture()
    def dc_resource(self, schema_resource) -> DimcatResource:
        return schema_resource


class TestFromDataFrame(TestMemoryResource):
    expected_resource_status = ResourceStatus.DATAFRAME
    should_be_loaded = True

    @pytest.fixture()
    def dc_resource(self, resource_from_dataframe) -> DimcatResource:
        return resource_from_dataframe


class TestResourceFromMemoryResource(TestFromDataFrame):
    @pytest.fixture()
    def dc_resource(self, resource_from_memory_resource):
        return resource_from_memory_resource


class TestAssembledResource(TestMemoryResource):
    expected_resource_status = ResourceStatus.DATAFRAME
    should_be_loaded = True

    @pytest.fixture()
    def dc_resource(self, assembled_resource):
        return assembled_resource


class TestSerializedResource(TestMemoryResource):
    expected_resource_status = ResourceStatus.STANDALONE_LOADED
    should_be_frozen: bool = True
    should_be_serialized = True
    should_be_loaded = True
    should_have_descriptor = True

    @pytest.fixture()
    def dc_resource(self, serialized_resource):
        return serialized_resource


class TestFromFrictionless(TestDiskResource):
    expected_resource_status = ResourceStatus.STANDALONE_NOT_LOADED

    @pytest.fixture()
    def dc_resource(self, resource_from_fl_resource):
        return resource_from_fl_resource


class TestFromDict(TestDiskResource):
    expected_resource_status = ResourceStatus.STANDALONE_NOT_LOADED

    @pytest.fixture()
    def dc_resource(self, resource_from_dict):
        return resource_from_dict


class TestFromConfig(TestDiskResource):
    expected_resource_status = ResourceStatus.STANDALONE_NOT_LOADED

    @pytest.fixture()
    def dc_resource(self, resource_from_config):
        return resource_from_config


@pytest.fixture(scope="session")
def package_descriptor_filepath(score_path) -> str:
    """Returns the path to the descriptor file."""
    return os.path.relpath(score_path, CORPUS_PATH)


class TestFromFlPackage(TestDiskResource):
    expected_resource_status = ResourceStatus.PACKAGED_NOT_LOADED
    should_have_descriptor = True

    @pytest.fixture()
    def dc_resource(self, zipped_resource_from_fl_package):
        return zipped_resource_from_fl_package


class TestFromDcPackage(TestDiskResource):
    expected_resource_status = ResourceStatus.PACKAGED_NOT_LOADED
    should_have_descriptor = True

    @pytest.fixture()
    def dc_resource(self, zipped_resource_from_dc_package):
        return zipped_resource_from_dc_package


# endregion DimcatResource

#
# @pytest.fixture(
#     scope="class",
# )
# def dimcat_resource(request,
#                     resource_path,
#                     fl_resource,
#                     resource_from_descriptor,
#                     resource_from_fl_resource,
#                     resource_from_dict,
#                     resource_from_config,
#                     dataframe_from_tsv,
#                     resource_from_dataframe,
#                     tmp_serialization_path):
#     """Initializes each of the resource objects to be tested and injects them into the test class."""
#     request.cls.tmp_path = tmp_serialization_path
#     request.cls.descriptor_path = resource_path
#     request.cls.fl_resource = fl_resource
#     request.cls.resource_from_descriptor = resource_from_descriptor
#     request.cls.resource_from_fl_resource = resource_from_fl_resource
#     request.cls.dataframe_from_tsv = dataframe_from_tsv
#     request.cls.resource_from_dataframe = resource_from_dataframe
#     request.cls.resource_from_dict = resource_from_dict
#     request.cls.resource_from_config = resource_from_config
#
#
#
#
# @pytest.mark.usefixtures("dimcat_resource")
# class TestResourceOld:
#     """Each method uses the same objects, and therefore should restore their original status after
#     changing it."""
#
#     tmp_path: str
#     """Path to the temporary directory where not-frozen test resources are stored during serialization."""
#     descriptor_path: str
#     """Path to the JSON resource descriptor on disk."""
#     fl_resource: fl.Resource
#     """Frictionless resource object created from the descriptor on disk."""
#     resource_from_descriptor: DimcatResource
#     """DimcatResource object created from the descriptor on disk."""
#     resource_from_fl_resource: DimcatResource
#     """Dimcat resource object created from the frictionless.Resource object."""
#     dataframe_from_tsv: pd.DataFrame
#     """Pandas dataframe created from the TSV file described by the resource."""
#     resource_from_dataframe: DimcatResource
#     """Dimcat resource object created from the dataframe."""
#     resource_from_dict: DimcatResource
#     """Dimcat resource object created from the descriptor source."""
#     resource_from_config: DimcatResource
#     """Dimcat resource object created from a serialized ."""
#
#     def test_equivalence(self):
#         assert self.resource_from_descriptor == self.resource_from_fl_resource
#         assert self.resource_from_descriptor == self.resource_from_dataframe
#
#     def test_basepath(self):
#         assert self.resource_from_descriptor.basepath == os.path.dirname(self.descriptor_path)
#         assert self.resource_from_fl_resource.basepath == os.path.dirname(self.descriptor_path)
#         assert self.resource_from_dataframe.basepath == self.tmp_path
#         with pytest.raises(RuntimeError):
#             self.resource_from_descriptor.basepath = "~"
#         with pytest.raises(RuntimeError):
#             self.resource_from_fl_resource.basepath = "~"
#         self.resource_from_dataframe.basepath = "~"
#         assert self.resource_from_dataframe.basepath == os.path.expanduser("~")
#         self.resource_from_dataframe.basepath = self.tmp_path
#
#     def test_filepath(self):
#         assert self.resource_from_descriptor.filepath == self.resource_from_descriptor.resource_name + ".tsv"
#         assert self.resource_from_fl_resource.filepath == self.resource_from_fl_resource.resource_name + ".tsv"
#         assert self.resource_from_dataframe.filepath == self.fl_resource.name + ".tsv"
#         with pytest.raises(RuntimeError):
#             self.resource_from_descriptor.filepath = "test.tsv"
#         with pytest.raises(RuntimeError):
#             self.resource_from_fl_resource.filepath = "test.tsv"
#         self.resource_from_dataframe.filepath = "subfolder/test.tsv"
#         assert self.resource_from_dataframe.normpath == os.path.join(self.tmp_path, "subfolder", "test.tsv")
#
#     def test_store_dataframe(self):
#         """"""
#         ms3.assert_dfs_equal(self.resource_from_descriptor.df, self.dataframe_from_tsv)
#         ms3.assert_dfs_equal(self.resource_from_fl_resource.df, self.dataframe_from_tsv)
#         config = self.resource_from_dataframe.to_config()
#         fresh_copy = config.create()
#         ms3.assert_dfs_equal(fresh_copy.df, self.dataframe_from_tsv)
#
#
#
#     def test_resource_from_df(self):
#         print(self.resource_from_dataframe)
#         assert self.resource_from_dataframe.status in (
#             ResourceStatus.VALIDATED,
#             ResourceStatus.DATAFRAME,
#         )
#         as_config = self.resource_from_dataframe.to_config()
#         print(as_config)
#
#     def test_serialization(self, as_dict, as_config):
#         print(as_dict)
#         print(as_config)
#         assert as_dict == as_config
#
#     def test_deserialization_via_config(self, as_dict, as_config):
#         from_dict = deserialize_dict(as_dict)
#         assert isinstance(from_dict, DimcatResource)
#         from_config = as_config.create()
#         assert isinstance(from_config, DimcatResource)
#         assert from_dict.__dict__.keys() == from_config.__dict__.keys()
#


def test_index_from_resource(resource_object):
    was_loaded_before = resource_object.is_loaded
    idx = DimcatIndex.from_resource(resource_object)
    print(idx)
    if resource_object.is_empty:
        assert len(idx) == 0
    else:
        assert len(idx) > 0
    assert resource_object.is_loaded == was_loaded_before
    piece_idx1 = resource_object.get_piece_index()
    piece_idx2 = PieceIndex.from_index(idx)
    assert len(piece_idx1) == len(piece_idx2)
    assert piece_idx1 == piece_idx2
    assert piece_idx1[:3] in piece_idx2
    if resource_object.is_empty or piece_idx1.nlevels == idx.nlevels:
        assert piece_idx1 in idx
    else:
        assert piece_idx1 not in idx
    serialized = idx.to_json()
    deserialized = DimcatIndex.from_json(serialized)
    assert idx == deserialized
