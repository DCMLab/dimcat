import pytest
from dimcat.dataset import DimcatPackage
from dimcat.dataset.base import Dataset, PackageStatus
from dimcat.dataset.processed import (
    AnalyzedDataset,
    GroupedAnalyzedDataset,
    GroupedDataset,
)
from dimcat.resources.base import ResourceStatus
from dimcat.utils import get_default_basepath

from tests.conftest import CORPUS_PATH

# region test DimcatPackage objects


@pytest.fixture()
def empty_package():
    return DimcatPackage(package_name="empty_package")


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


class TestPackageFromFL(TestPackageFromDescriptor):
    @pytest.fixture()
    def package_obj(self, package_from_fl_package):
        return package_from_fl_package


# endregion test DimcatPackage objects


def test_copying_dataset(dataset_from_single_package):
    new_dataset = Dataset.from_dataset(dataset_from_single_package)
    assert new_dataset == dataset_from_single_package
    assert new_dataset is not dataset_from_single_package
    as_config = dataset_from_single_package.to_config()
    new_dataset = as_config.create()
    assert new_dataset == dataset_from_single_package


def test_processed_dataset(dataset_from_single_package):
    a_dataset = AnalyzedDataset.from_dataset(dataset_from_single_package)
    assert not hasattr(dataset_from_single_package, "get_result")
    assert hasattr(a_dataset, "get_result")
    assert isinstance(a_dataset, AnalyzedDataset)
    assert a_dataset.inputs == dataset_from_single_package.inputs
    assert a_dataset.outputs.has_package("results")

    ag_dataset = GroupedDataset.from_dataset(a_dataset)
    assert isinstance(ag_dataset, GroupedAnalyzedDataset)
    assert isinstance(ag_dataset, GroupedDataset)
    assert isinstance(ag_dataset, AnalyzedDataset)
    assert ag_dataset.inputs == dataset_from_single_package.inputs
