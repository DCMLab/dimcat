import pytest
from dimcat.dataset import DimcatPackage
from dimcat.dataset.base import Dataset, PackageStatus
from dimcat.dataset.processed import (
    AnalyzedDataset,
    GroupedAnalyzedDataset,
    GroupedDataset,
)
from dimcat.resources.base import DimcatIndex, PieceIndex, ResourceStatus
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


@pytest.fixture()
def dataset_from_single_package(package_path):
    dataset = Dataset()
    dataset.load_package(package_path)
    return dataset


def test_dataset(dataset_from_single_package):
    new_dataset = Dataset.from_dataset(dataset_from_single_package)
    assert new_dataset == dataset_from_single_package
    assert new_dataset is not dataset_from_single_package


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


@pytest.fixture(
    params=[
        "resource_from_descriptor",
        "resource_from_fl_resource",
        "resource_from_dataframe",
    ]
)
def resource_object(
    request,
    resource_from_descriptor,
    resource_from_fl_resource,
    resource_from_dataframe,
):
    yield request.getfixturevalue(request.param)


def test_index_from_resource(resource_object):
    was_loaded_before = resource_object.is_loaded
    idx = DimcatIndex.from_resource(resource_object)
    print(idx)
    assert len(idx) > 0
    assert resource_object.is_loaded == was_loaded_before
    piece_idx1 = PieceIndex.from_resource(resource_object)
    piece_idx2 = PieceIndex.from_index(idx)
    assert len(piece_idx1) == len(piece_idx2)
    assert piece_idx1 == piece_idx2
    assert piece_idx1[:3] in piece_idx2
    if piece_idx1.nlevels == idx.nlevels:
        assert piece_idx1 in idx
    else:
        assert piece_idx1 not in idx
