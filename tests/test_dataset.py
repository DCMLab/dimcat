import frictionless as fl
import ms3
import pytest
from dimcat.dataset import DimcatPackage
from dimcat.resources.base import DimcatResource, ResourceStatus, get_default_basepath

from tests.conftest import CORPUS_PATH


@pytest.fixture(scope="session")
def fl_resource(resource_path):
    """Returns a frictionless resource object."""
    return fl.Resource(resource_path)


@pytest.fixture(scope="session")
def fl_package(package_path) -> fl.Package:
    """Returns a frictionless package object."""
    return fl.Package(package_path)


@pytest.fixture(scope="session")
def dc_package(fl_package) -> DimcatPackage:
    """Returns a DimcatPackage object."""
    return DimcatPackage(fl_package)


@pytest.fixture(scope="session")
def dataframe_from_tsv(fl_resource):
    """Returns a dataframe read directly from the normpath of the fl_resource."""
    return ms3.load_tsv(fl_resource.normpath)


@pytest.fixture(scope="session")
def tmp_serialization_path(tmp_path_factory, fl_resource):
    """Returns the path to the directory where serialized resources are stored."""
    return str(tmp_path_factory.mktemp(fl_resource.name))


@pytest.fixture(scope="session")
def empty_resource():
    return DimcatResource(resource_name="empty_resource")


class TestVanillaResource:
    expected_resource_status: ResourceStatus = ResourceStatus.EMPTY
    """The expected status of the resource after initialization."""
    should_be_frozen: bool = False
    """Whether the resource should be frozen, i.e., immutable after initialization."""

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


@pytest.fixture(scope="session")
def resource_from_descriptor(resource_path):
    """Returns a DimcatResource object created from the descriptor on disk."""
    return DimcatResource.from_descriptor(descriptor_path=resource_path)


class TestDiskResource(TestVanillaResource):
    expected_resource_status = ResourceStatus.ON_DISK_NOT_LOADED
    should_be_frozen: bool = True

    @pytest.fixture()
    def expected_basepath(self):
        return CORPUS_PATH

    @pytest.fixture()
    def dc_resource(self, resource_from_descriptor):
        return resource_from_descriptor


@pytest.fixture(scope="session")
def empty_resource_with_paths(tmp_serialization_path):
    return DimcatResource(
        basepath=tmp_serialization_path, filepath="empty_resource.tsv"
    )


class TestMemoryResource(TestVanillaResource):
    """MemoryResources are those instantiated from a dataframe. They have in common that, in this
    test suite, their basepath is a temporary path where they can be serialized."""

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


@pytest.fixture(scope="session")
def schema_resource(fl_resource):
    """Returns a DimcatResource with a pre-set frictionless.Schema object."""
    return DimcatResource(column_schema=fl_resource.schema)


class TestSchemaResource(TestVanillaResource):
    expected_resource_status = ResourceStatus.SCHEMA

    @pytest.fixture()
    def dc_resource(self, schema_resource) -> DimcatResource:
        return schema_resource


@pytest.fixture(scope="session")
def resource_from_dataframe(
    dataframe_from_tsv, fl_resource, tmp_serialization_path
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

    @pytest.fixture()
    def dc_resource(self, resource_from_dataframe) -> DimcatResource:
        return resource_from_dataframe


@pytest.fixture(scope="session")
def assembled_resource(
    dataframe_from_tsv, fl_resource, tmp_serialization_path
) -> DimcatResource:
    resource = DimcatResource(
        resource_name=fl_resource.name, column_schema=fl_resource.schema
    )
    resource.df = dataframe_from_tsv
    resource.basepath = tmp_serialization_path
    return resource


class TestAssembledResource(TestMemoryResource):
    expected_resource_status = ResourceStatus.DATAFRAME

    @pytest.fixture()
    def dc_resource(self, assembled_resource):
        return assembled_resource


@pytest.fixture(scope="session")
def serialized_resource(resource_from_dataframe) -> DimcatResource:
    resource_from_dataframe.store_dataframe()
    return resource_from_dataframe


class TestSerializedResource(TestMemoryResource):
    expected_resource_status = ResourceStatus.ON_DISK_AND_LOADED
    should_be_frozen: bool = True

    @pytest.fixture()
    def dc_resource(self, serialized_resource):
        return serialized_resource


@pytest.fixture(scope="session")
def resource_from_fl_resource(fl_resource) -> DimcatResource:
    """Returns a Dimcat resource object created from the frictionless.Resource object."""
    return DimcatResource(resource=fl_resource)


class TestFromFrictionless(TestDiskResource):
    expected_resource_status = ResourceStatus.ON_DISK_NOT_LOADED

    @pytest.fixture()
    def dc_resource(self, resource_from_fl_resource):
        return resource_from_fl_resource


@pytest.fixture(scope="session")
def resource_from_dict(resource_from_descriptor):
    """Returns a DimcatResource object created from the descriptor source."""
    as_dict = resource_from_descriptor.to_dict()
    return DimcatResource.from_dict(as_dict)


class TestFromDict(TestDiskResource):
    expected_resource_status = ResourceStatus.ON_DISK_NOT_LOADED

    @pytest.fixture()
    def dc_resource(self, resource_from_dict):
        return resource_from_dict


@pytest.fixture(scope="session")
def resource_from_config(resource_from_descriptor):
    """Returns a DimcatResource object created from the descriptor on disk."""
    config = resource_from_descriptor.to_config()
    return DimcatResource.from_config(config)


class TestFromConfig(TestDiskResource):
    expected_resource_status = ResourceStatus.ON_DISK_NOT_LOADED

    @pytest.fixture()
    def dc_resource(self, resource_from_config):
        return resource_from_config


@pytest.fixture(scope="session")
def zipped_resource_from_fl_package(fl_package) -> DimcatResource:
    """Returns a DimcatResource object created from the dataframe."""
    fl_resource = fl_package.get_resource("notes")
    return DimcatResource(resource=fl_resource)


class TestFromFlPackage(TestDiskResource):
    expected_resource_status = ResourceStatus.ON_DISK_NOT_LOADED

    @pytest.fixture()
    def dc_resource(self, zipped_resource_from_fl_package):
        return zipped_resource_from_fl_package


@pytest.fixture(scope="session")
def zipped_resource_from_dc_package(dc_package) -> DimcatResource:
    dc_resource = dc_package.get_resource("notes")
    return DimcatResource.from_resource(dc_resource)


class TestFromDcPackage(TestDiskResource):
    expected_resource_status = ResourceStatus.ON_DISK_NOT_LOADED

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
#     def test_is_serialized(self):
#         assert self.resource_from_descriptor.is_serialized
#         assert self.resource_from_fl_resource.is_serialized
#         assert not self.resource_from_dataframe.is_serialized
#
#     def test_is_loaded(self):
#         assert not self.resource_from_descriptor.is_loaded
#         assert not self.resource_from_fl_resource.is_loaded
#         assert self.resource_from_dataframe.is_loaded
#
#     def test_descriptor_path(self):
#         assert self.resource_from_descriptor.descriptor_path == self.descriptor_path
#         assert self.resource_from_fl_resource.descriptor_path is None
#         assert self.resource_from_dataframe.descriptor_path is None
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
