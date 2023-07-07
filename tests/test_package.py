import logging

import pytest
from dimcat.data.package.base import PackageStatus
from dimcat.data.package.dimcat import DimcatPackage
from dimcat.data.resources.base import ResourceStatus

from tests.conftest import CORPUS_PATH

logging.basicConfig(level=logging.DEBUG)


class TestDimcatPackage:
    expected_package_status: ResourceStatus = PackageStatus.EMPTY
    """The expected status of the package after initialization."""

    @pytest.fixture()
    def expected_basepath(self):
        """The expected basepath of the resource after initialization."""
        return None

    @pytest.fixture()
    def package_obj(self):
        """For each subclass of TestDimcatPackage, this fixture should be overridden and yield the
        tested DimcatPackage object."""
        return DimcatPackage(package_name="empty_package")

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
        if package_obj.status == PackageStatus.PATHS_ONLY:
            return
        idx1 = package_obj.get_piece_index()
        table = package_obj.get_boolean_resource_table()
        idx2 = table.index
        assert idx1 == idx2


class TestPackageFromFilePaths(TestDimcatPackage):
    expected_package_status: ResourceStatus = PackageStatus.PATHS_ONLY

    @pytest.fixture()
    def package_obj(self, list_of_mixed_score_paths):
        """For each subclass of TestDimcatPackage, this fixture should be overridden and yield the
        tested DimcatPackage object."""
        package = DimcatPackage.from_filepaths(
            filepaths=list_of_mixed_score_paths, package_name="mixed_files"
        )
        return package


class TestPackageFromFileDirectory(TestDimcatPackage):
    expected_package_status: ResourceStatus = PackageStatus.PATHS_ONLY

    @pytest.fixture()
    def package_obj(self, corpus_path):
        """For each subclass of TestDimcatPackage, this fixture should be overridden and yield the
        tested DimcatPackage object."""
        package = DimcatPackage.from_directory(
            corpus_path, package_name="unittest_corpus"
        )
        return package

    @pytest.fixture()
    def expected_basepath(self, corpus_path):
        """The expected basepath of the resource after initialization."""
        return corpus_path


# region PackageFromDescriptor


class TestPackageFromDescriptor(TestDimcatPackage):
    expected_package_status = PackageStatus.FULLY_SERIALIZED

    @pytest.fixture()
    def expected_basepath(self):
        """The expected basepath of the resource after initialization."""
        return CORPUS_PATH

    @pytest.fixture()
    def package_obj(self, score_path):
        return DimcatPackage.from_descriptor_path(score_path)


class TestPackageFromFL(TestPackageFromDescriptor):
    @pytest.fixture()
    def package_obj(self, package_from_fl_package):
        return package_from_fl_package


# endregion PackageFromDescriptor
