import os.path

import frictionless as fl
import ms3
import pytest
from dimcat.dataset import DimcatPackage
from dimcat.dataset.base import Dataset, PackageStatus
from dimcat.dataset.processed import (
    AnalyzedDataset,
    GroupedAnalyzedDataset,
    GroupedDataset,
)
from dimcat.resources.base import DimcatResource, ResourceStatus, get_default_basepath

from tests.conftest import CORPUS_PATH

# region helper fixtures


@pytest.fixture(scope="session")
def fl_resource(resource_path):
    """Returns a frictionless resource object."""
    return fl.Resource(resource_path)


@pytest.fixture(scope="session")
def dataframe_from_tsv(fl_resource):
    """Returns a dataframe read directly from the normpath of the fl_resource."""
    return ms3.load_tsv(fl_resource.normpath)


@pytest.fixture()
def tmp_serialization_path(request, tmp_path_factory):
    """Returns the path to the directory where serialized resources are stored."""
    return str(tmp_path_factory.mktemp(request.cls.__name__))


# endregion helper fixtures

# region test DimcatResource objects


@pytest.fixture()
def empty_resource():
    return DimcatResource(resource_name="empty_resource")


class TestVanillaResource:
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
        return get_default_basepath()

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


@pytest.fixture()
def resource_from_descriptor(resource_path):
    """Returns a DimcatResource object created from the descriptor on disk."""
    return DimcatResource.from_descriptor(descriptor_path=resource_path)


class TestDiskResource(TestVanillaResource):
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


@pytest.fixture()
def resource_from_frozen_resource(resource_from_descriptor):
    """Returns a DimcatResource object created from a frozen resource."""
    return DimcatResource.from_resource(resource_from_descriptor)


class TestResourceFromFrozen(TestDiskResource):
    @pytest.fixture()
    def dc_resource(self, resource_from_frozen_resource):
        return resource_from_frozen_resource


@pytest.fixture()
def empty_resource_with_paths(tmp_serialization_path):
    return DimcatResource(
        basepath=tmp_serialization_path, filepath="empty_resource.tsv"
    )


class TestMemoryResource(TestVanillaResource):
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


@pytest.fixture()
def schema_resource(fl_resource):
    """Returns a DimcatResource with a pre-set frictionless.Schema object."""
    return DimcatResource(column_schema=fl_resource.schema)


class TestSchemaResource(TestVanillaResource):
    expected_resource_status = ResourceStatus.SCHEMA

    @pytest.fixture()
    def dc_resource(self, schema_resource) -> DimcatResource:
        return schema_resource


@pytest.fixture()
def resource_from_dataframe(
    dataframe_from_tsv,
    fl_resource,
    tmp_serialization_path,
    resource_descriptor_filepath,
) -> DimcatResource:
    """Returns a DimcatResource object created from the dataframe."""
    return DimcatResource.from_dataframe(
        df=dataframe_from_tsv,
        resource_name=fl_resource.name,
        basepath=tmp_serialization_path,
        column_schema=fl_resource.schema,
    )


class TestFromDataFrame(TestMemoryResource):
    expected_resource_status = ResourceStatus.DATAFRAME
    should_be_loaded = True

    @pytest.fixture()
    def dc_resource(self, resource_from_dataframe) -> DimcatResource:
        return resource_from_dataframe


@pytest.fixture()
def resource_from_memory_resource(resource_from_dataframe):
    """Returns a DimcatResource object created from a frozen resource."""
    return DimcatResource.from_resource(resource_from_dataframe)


class TestResourceFromMemoryResource(TestFromDataFrame):
    @pytest.fixture()
    def dc_resource(self, resource_from_memory_resource):
        return resource_from_memory_resource


@pytest.fixture()
def assembled_resource(
    dataframe_from_tsv, fl_resource, tmp_serialization_path
) -> DimcatResource:
    resource = DimcatResource(
        basepath=tmp_serialization_path,
        resource_name=fl_resource.name,
    )
    resource.df = dataframe_from_tsv
    return resource


class TestAssembledResource(TestMemoryResource):
    expected_resource_status = ResourceStatus.DATAFRAME
    should_be_loaded = True

    @pytest.fixture()
    def dc_resource(self, assembled_resource):
        return assembled_resource


@pytest.fixture()
def serialized_resource(resource_from_dataframe) -> DimcatResource:
    resource_from_dataframe.store_dataframe()
    return resource_from_dataframe


class TestSerializedResource(TestMemoryResource):
    expected_resource_status = ResourceStatus.STANDALONE_LOADED
    should_be_frozen: bool = True
    should_be_serialized = True
    should_be_loaded = True
    should_have_descriptor = True

    @pytest.fixture()
    def dc_resource(self, serialized_resource):
        return serialized_resource


@pytest.fixture()
def resource_from_fl_resource(
    fl_resource, resource_descriptor_filepath
) -> DimcatResource:
    """Returns a Dimcat resource object created from the frictionless.Resource object."""
    return DimcatResource(
        resource=fl_resource, descriptor_filepath=resource_descriptor_filepath
    )


class TestFromFrictionless(TestDiskResource):
    expected_resource_status = ResourceStatus.STANDALONE_NOT_LOADED

    @pytest.fixture()
    def dc_resource(self, resource_from_fl_resource):
        return resource_from_fl_resource


@pytest.fixture()
def resource_from_dict(resource_from_descriptor):
    """Returns a DimcatResource object created from the descriptor source."""
    as_dict = resource_from_descriptor.to_dict()
    return DimcatResource.from_dict(as_dict)


class TestFromDict(TestDiskResource):
    expected_resource_status = ResourceStatus.STANDALONE_NOT_LOADED

    @pytest.fixture()
    def dc_resource(self, resource_from_dict):
        return resource_from_dict


@pytest.fixture()
def resource_from_config(resource_from_descriptor):
    """Returns a DimcatResource object created from the descriptor on disk."""
    config = resource_from_descriptor.to_config()
    return DimcatResource.from_config(config)


class TestFromConfig(TestDiskResource):
    expected_resource_status = ResourceStatus.STANDALONE_NOT_LOADED

    @pytest.fixture()
    def dc_resource(self, resource_from_config):
        return resource_from_config


@pytest.fixture(scope="session")
def resource_descriptor_filepath(resource_path) -> str:
    """Returns the path to the descriptor file."""
    return os.path.relpath(resource_path, CORPUS_PATH)


@pytest.fixture(scope="session")
def package_descriptor_filepath(package_path) -> str:
    """Returns the path to the descriptor file."""
    return os.path.relpath(package_path, CORPUS_PATH)


@pytest.fixture()
def zipped_resource_from_fl_package(
    fl_package,
    package_descriptor_filepath,
) -> DimcatResource:
    """Returns a DimcatResource object created from the dataframe."""
    fl_resource = fl_package.get_resource("notes")
    return DimcatResource(
        resource=fl_resource, descriptor_filepath=package_descriptor_filepath
    )


class TestFromFlPackage(TestDiskResource):
    expected_resource_status = ResourceStatus.PACKAGED_NOT_LOADED
    should_have_descriptor = True

    @pytest.fixture()
    def dc_resource(self, zipped_resource_from_fl_package):
        return zipped_resource_from_fl_package


@pytest.fixture()
def zipped_resource_from_dc_package(
    package_from_fl_package, package_descriptor_filepath
) -> DimcatResource:
    dc_resource = package_from_fl_package.get_resource("notes")
    return DimcatResource.from_resource(
        dc_resource, descriptor_filepath=package_descriptor_filepath
    )


class TestFromDcPackage(TestDiskResource):
    expected_resource_status = ResourceStatus.PACKAGED_NOT_LOADED
    should_have_descriptor = True

    @pytest.fixture()
    def dc_resource(self, zipped_resource_from_dc_package):
        return zipped_resource_from_dc_package


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

# endregion test DimcatResource objects

# region test DimcatPackage objects


@pytest.fixture()
def empty_package():
    return DimcatPackage(package_name="empty_package")


@pytest.fixture()
def package_from_fl_package(fl_package) -> DimcatPackage:
    """Returns a DimcatPackage object."""
    return DimcatPackage(fl_package)


class TestDimcatPackage:
    expected_package_status: ResourceStatus = PackageStatus.EMPTY
    """The expected status of the package after initialization."""

    @pytest.fixture()
    def expected_basepath(self):
        """The expected basepath of the resource after initialization."""
        return get_default_basepath()

    @pytest.fixture()
    def package_obj(self, empty_package):
        """For each subclass of TestDimcatPackage, this fixture should be overridden and yield the
        tested DimcatPackage object."""
        return empty_package

    def test_basepath_after_init(self, package_obj, expected_basepath):
        assert package_obj.basepath == expected_basepath

    def test_status_after_init(self, package_obj):
        assert package_obj.status == self.expected_package_status

    def test_serialization(self, package_obj):
        as_config = package_obj.to_config()
        new_object = as_config.create()
        assert new_object == package_obj

    def test_copy(self, package_obj):
        copy = package_obj.copy()
        assert copy == package_obj
        assert copy is not package_obj


@pytest.fixture()
def package_from_descriptor(package_path):
    return DimcatPackage(package_path)


class TestPackageFromDescriptor(TestDimcatPackage):
    expected_package_status = PackageStatus.FULLY_SERIALIZED

    @pytest.fixture()
    def expected_basepath(self):
        """The expected basepath of the resource after initialization."""
        return CORPUS_PATH

    @pytest.fixture()
    def package_obj(self, package_from_descriptor):
        return package_from_descriptor


@pytest.fixture()
def fl_package(package_path) -> fl.Package:
    """Returns a frictionless package object."""
    return fl.Package(package_path)


class TestPackageFromFL(TestPackageFromDescriptor):
    @pytest.fixture()
    def package_obj(self, package_from_fl_package):
        return package_from_fl_package


# endregion test DimcatPackage objects


def test_dataset(package_path):
    dataset = Dataset()
    dataset.load_package(package_path)
    new_dataset = Dataset.from_dataset(dataset)
    assert new_dataset == dataset
    assert new_dataset is not dataset


def test_processed_dataset(package_path):
    dataset = Dataset()
    dataset.load_package(package_path)
    a_dataset = AnalyzedDataset.from_dataset(dataset)
    assert not hasattr(dataset, "get_result")
    assert hasattr(a_dataset, "get_result")
    assert isinstance(a_dataset, AnalyzedDataset)
    assert a_dataset.inputs == dataset.inputs
    assert a_dataset.outputs.has_package("results")

    ag_dataset = GroupedDataset.from_dataset(a_dataset)
    assert isinstance(ag_dataset, GroupedAnalyzedDataset)
    assert isinstance(ag_dataset, GroupedDataset)
    assert isinstance(ag_dataset, AnalyzedDataset)
    assert ag_dataset.inputs == dataset.inputs
