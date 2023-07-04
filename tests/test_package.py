import pytest
from dimcat.data import DimcatPackage, PackageStatus, ResourceStatus

from tests.conftest import CORPUS_PATH


@pytest.fixture()
def empty_package():
    return DimcatPackage(package_name="empty_package")


class TestDimcatPackage:
    expected_package_status: ResourceStatus = PackageStatus.EMPTY
    """The expected status of the package after initialization."""

    @pytest.fixture()
    def expected_basepath(self):
        """The expected basepath of the resource after initialization."""
        return None

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

    def test_piece_index(self, package_obj):
        idx1 = package_obj.get_piece_index()
        table = package_obj.get_boolean_resource_table()
        idx2 = table.index
        assert idx1 == idx2


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


class TestPackageFromFL(TestPackageFromDescriptor):
    @pytest.fixture()
    def package_obj(self, package_from_fl_package):
        return package_from_fl_package
