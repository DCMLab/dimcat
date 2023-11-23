import logging
import os

import pytest
from dimcat.base import deserialize_dict, deserialize_json_file, get_setting
from dimcat.data.resources.base import PathResource, Resource, ResourceStatus
from dimcat.data.resources.dc import DimcatIndex, DimcatResource, PieceIndex
from dimcat.data.utils import make_rel_path
from dimcat.dc_exceptions import BasePathNotDefinedError, FilePathNotDefinedError
from dimcat.utils import make_valid_frictionless_name

from .conftest import CORPUS_PATH, get_mixed_score_paths

logger = logging.getLogger(__name__)
# region Resource


class TestBaseResource:
    @pytest.fixture(params=[Resource, DimcatResource])
    def resource_constructor(self, request):
        return request.param

    @pytest.fixture(params=[None, -1, -2], ids=["no_bp", "bp-1", "bp-2"])
    def init_basepath(self, request, corpus_path):
        """Different basepath arguments for initilizing objects."""
        if request.param is None:
            return None
        basepath, _ = os.path.split(corpus_path)
        init_basepath = os.sep.join(basepath.split(os.sep)[: request.param])
        return init_basepath

    @pytest.fixture()
    def resource_obj(self, resource_constructor, init_basepath):
        return resource_constructor(basepath=init_basepath)

    @pytest.fixture()
    def expected_basepath(self, init_basepath):
        return init_basepath

    @pytest.fixture()
    def expected_filepath(self):
        return ""

    @pytest.fixture()
    def expected_normpath(self):
        return None

    @pytest.fixture()
    def expected_resource_name(self):
        return get_setting("default_resource_name")

    @pytest.fixture()
    def expected_status(self):
        return ResourceStatus.EMPTY

    def test_basepath_after_init(self, resource_obj, expected_basepath):
        assert resource_obj.basepath == expected_basepath

    def test_filepath_after_init(self, resource_obj, expected_filepath):
        assert resource_obj.filepath == expected_filepath

    def test_normpath_after_init(self, resource_obj, init_basepath):
        if init_basepath is None:
            with pytest.raises(BasePathNotDefinedError):
                resource_obj.normpath
        else:
            with pytest.raises(FilePathNotDefinedError):
                resource_obj.normpath

    def test_resource_name_after_init(self, resource_obj, expected_resource_name):
        assert resource_obj.resource_name == expected_resource_name

    def test_status_after_init(self, resource_obj, expected_status):
        assert resource_obj.status == expected_status

    def test_copying(self, resource_obj):
        copy = resource_obj.copy()
        assert copy == resource_obj
        assert copy is not resource_obj

    def test_serialization_via_json(self, resource_obj, tmp_serialization_path):
        json_path = os.path.join(tmp_serialization_path, f"{resource_obj.name}.json")
        logger.warning(f"json_path: {json_path}")
        resource_obj.to_json_file(json_path)
        copy = deserialize_json_file(json_path)
        assert copy == resource_obj

    def test_descriptor(self, resource_obj):
        # json_path = resource_obj.store_descriptor()
        # print(json_path)
        descriptor = resource_obj.make_descriptor()
        copy1 = deserialize_dict(descriptor)
        # copy2 = deserialize_json_file(json_path)
        assert resource_obj == copy1
        # assert copy1 == copy2

    def test_validating(self, resource_obj):
        resource_obj.validate(raise_exception=True)


class TestResourceFromScorePath(TestBaseResource):
    @pytest.fixture(
        params=get_mixed_score_paths(),
        ids=os.path.basename,
    )
    def score_path(self, request):
        return request.param

    @pytest.fixture(params=[Resource, PathResource])
    def resource_constructor(self, request):
        return request.param

    @pytest.fixture(params=[None, -2], ids=["no_bp", "bp-2"])
    def init_basepath(self, request, score_path):
        """Different basepath arguments for initilizing objects.
        In the case of absolute path resources this works only if basepath is None or -2;
        in the latter case, the unittest_corpus path, two levels up, is "site-packages" so it
        works as a common ancestor for the music21 score paths"""
        if request.param is None:
            return None
        basepath, _ = os.path.split(score_path)
        init_basepath = os.sep.join(basepath.split(os.sep)[: request.param])
        return init_basepath

    @pytest.fixture()
    def resource_obj(self, resource_constructor, score_path, init_basepath):
        # if request.node.callspec.params["init_basepath"] != -1:
        return resource_constructor.from_resource_path(
            score_path, basepath=init_basepath
        )

    @pytest.fixture()
    def expected_basepath(self, init_basepath, score_path):
        """If init_basepath is None, we expect the directory where the score is located."""
        if init_basepath is None:
            basepath, _ = os.path.split(score_path)
            return basepath
        return init_basepath

    @pytest.fixture()
    def expected_filepath(self, score_path, expected_basepath):
        return make_rel_path(score_path, expected_basepath)

    @pytest.fixture()
    def expected_normpath(self, score_path):
        return score_path

    @pytest.fixture()
    def expected_resource_name(self, expected_filepath):
        return make_valid_frictionless_name(expected_filepath)

    @pytest.fixture()
    def expected_status(self):
        return ResourceStatus.PATH_ONLY

    def test_normpath_after_init(self, resource_obj, expected_normpath):
        assert resource_obj.normpath == expected_normpath


class TestResourceFromDescriptorPath(TestBaseResource):
    @pytest.fixture(params=[None, -1, -2], ids=["no_bp", "bp-1", "bp-2"])
    def init_basepath(self, request, resource_descriptor_path):
        """Different basepath arguments for initializing objects."""
        if request.param is None:
            return None
        basepath, _ = os.path.split(resource_descriptor_path)
        init_basepath = os.sep.join(basepath.split(os.sep)[: request.param])
        return init_basepath

    @pytest.fixture()
    def resource_obj(self, resource_descriptor_path, init_basepath):
        return Resource.from_descriptor_path(
            descriptor_path=resource_descriptor_path,
        )

    @pytest.fixture()
    def expected_basepath(self, resource_descriptor_path):
        """If init_basepath is None, we expect the directory where the score is located."""
        basepath, _ = os.path.split(resource_descriptor_path)
        return basepath

    @pytest.fixture()
    def expected_filepath(self, expected_basepath, expected_normpath):
        return make_rel_path(expected_normpath, expected_basepath)

    @pytest.fixture()
    def expected_resource_name(self, fl_resource):
        return fl_resource.name

    @pytest.fixture()
    def expected_status(self):
        return ResourceStatus.STANDALONE_NOT_LOADED

    @pytest.fixture()
    def expected_normpath(self, fl_resource):
        return fl_resource.normpath

    def test_normpath_after_init(self, resource_obj, expected_normpath):
        assert resource_obj.normpath == expected_normpath


class TestResourceFromDescriptor(TestResourceFromDescriptorPath):
    @pytest.fixture()
    def init_basepath(self, request, resource_descriptor_path):
        """When initializing from descriptor, the basepath always needs to be accurate so that the descriptor's
        'filepath' property resolves to the existing file."""
        basepath, _ = os.path.split(resource_descriptor_path)
        return basepath

    @pytest.fixture()
    def resource_obj(self, fl_resource, init_basepath):
        descriptor = fl_resource.to_dict()
        return PathResource.from_descriptor(
            descriptor=descriptor,
            basepath=init_basepath,
        )

    @pytest.fixture()
    def expected_basepath(self, init_basepath):
        return init_basepath

    @pytest.fixture()
    def expected_filepath(self, fl_resource):
        return fl_resource.path

    @pytest.fixture()
    def expected_normpath(self, init_basepath, expected_filepath):
        if init_basepath is None:
            return
        return os.path.join(init_basepath, expected_filepath)

    @pytest.fixture()
    def expected_status(self):
        return ResourceStatus.PATH_ONLY

    def test_normpath_after_init(self, resource_obj, init_basepath, expected_normpath):
        if init_basepath is None:
            with pytest.raises(BasePathNotDefinedError):
                resource_obj.normpath
        else:
            assert resource_obj.normpath == expected_normpath


# endregion Resource

# region DimcatResource


class TestEmptyDimcatResource:
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

    def test_frozen_after_init(self, dc_resource):
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
        descriptor_path = dc_resource.descriptor_filename
        has_descriptor = descriptor_path is not None
        if self.should_have_descriptor:
            assert has_descriptor
        else:
            assert not has_descriptor

    def test_copy(self, dc_resource):
        copy = dc_resource.copy()
        assert copy == dc_resource
        assert copy is not dc_resource
        assert copy.status == dc_resource.status


class TestDiskDimcatResource(TestEmptyDimcatResource):
    expected_resource_status = ResourceStatus.STANDALONE_NOT_LOADED
    should_be_frozen: bool = True
    should_be_serialized = True
    should_be_loaded: bool = False
    should_have_descriptor = True

    @pytest.fixture()
    def expected_basepath(self):
        return CORPUS_PATH

    @pytest.fixture()
    def dc_resource(self, resource_from_descriptor):
        return resource_from_descriptor


class TestResourceFromFrozen(TestDiskDimcatResource):
    @pytest.fixture()
    def dc_resource(self, resource_from_frozen_resource):
        return resource_from_frozen_resource


class TestMemoryDimcatResource(TestEmptyDimcatResource):
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


class TestSchemaDimcatResource(TestEmptyDimcatResource):
    expected_resource_status = ResourceStatus.SCHEMA_ONLY

    @pytest.fixture()
    def dc_resource(self, schema_resource) -> DimcatResource:
        return schema_resource


class TestFromDataFrame(TestMemoryDimcatResource):
    expected_resource_status = ResourceStatus.DATAFRAME
    should_be_loaded = True

    @pytest.fixture()
    def dc_resource(self, resource_from_dataframe) -> DimcatResource:
        return resource_from_dataframe


class TestResourceFromMemoryResource(TestFromDataFrame):
    @pytest.fixture()
    def dc_resource(self, resource_from_memory_resource):
        return resource_from_memory_resource


class TestAssembledResource(TestMemoryDimcatResource):
    expected_resource_status = ResourceStatus.DATAFRAME
    should_be_loaded = True

    @pytest.fixture()
    def dc_resource(self, assembled_resource):
        return assembled_resource


class TestSerializedResource(TestMemoryDimcatResource):
    expected_resource_status = ResourceStatus.STANDALONE_LOADED
    should_be_frozen: bool = True
    should_be_serialized = True
    should_be_loaded = True
    should_have_descriptor = True

    @pytest.fixture()
    def dc_resource(self, serialized_resource):
        return serialized_resource


class TestFromFrictionless(TestDiskDimcatResource):
    expected_resource_status = ResourceStatus.STANDALONE_NOT_LOADED

    @pytest.fixture()
    def dc_resource(self, resource_from_fl_resource):
        return resource_from_fl_resource


class TestFromDict(TestDiskDimcatResource):
    expected_resource_status = ResourceStatus.STANDALONE_NOT_LOADED

    @pytest.fixture()
    def dc_resource(self, resource_from_dict):
        return resource_from_dict


class TestFromConfig(TestDiskDimcatResource):
    expected_resource_status = ResourceStatus.STANDALONE_NOT_LOADED

    @pytest.fixture()
    def dc_resource(self, resource_from_config):
        return resource_from_config


@pytest.fixture(scope="session")
def package_descriptor_filename(package_descriptor_path) -> str:
    """Returns the path to the descriptor file."""
    return make_rel_path(package_descriptor_path, CORPUS_PATH)


class TestFromFlPackage(TestDiskDimcatResource):
    expected_resource_status = ResourceStatus.PACKAGED_NOT_LOADED
    should_have_descriptor = True

    @pytest.fixture()
    def dc_resource(self, zipped_resource_from_fl_package):
        return zipped_resource_from_fl_package


class TestFromDcPackage(TestDiskDimcatResource):
    expected_resource_status = ResourceStatus.PACKAGED_NOT_LOADED
    should_have_descriptor = True

    @pytest.fixture()
    def dc_resource(self, zipped_resource_copied_from_dc_package):
        return zipped_resource_copied_from_dc_package


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
