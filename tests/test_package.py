import logging
import os.path
from typing import Type

import pytest
from dimcat.base import change_setting
from dimcat.data.packages.base import Package, PackageMode, PackageStatus, PathPackage
from dimcat.data.packages.dc import DimcatPackage
from dimcat.data.resources.base import PathResource
from dimcat.dc_exceptions import ResourceIsFrozenError

from tests.conftest import CORPUS_PATH

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture()
def single_path_resource(list_of_m21_score_paths):
    selected_path = list_of_m21_score_paths[-1]
    return PathResource.from_resource_path(selected_path)


class TestPackage:
    @pytest.fixture(params=[Package, PathPackage, DimcatPackage])
    def package_constructor(self, request):
        return request.param

    @pytest.fixture()
    def expected_basepath(self, tmp_package_path):
        """The expected basepath of the resource after initialization."""
        return tmp_package_path

    @pytest.fixture()
    def expected_package_status(self, package_constructor):
        return PackageStatus.EMPTY

    @pytest.fixture(
        params=[
            mode.value
            for mode in PackageMode
            if mode.value in ["RAISE", "ALLOW_MISALIGNMENT"]
        ]
    )
    def package_mode(self, request):
        """This is not tested with RECONCILE modes because it would copy files to the basepath, which is the
        unittest_metacorpus by default (the path from where the descriptor is loaded).
        """
        return request.param

    @pytest.fixture()
    def package_obj(self, package_constructor, tmp_package_path):
        """For each subclass of TestDimcatPackage, this fixture should be overridden and yield the
        tested DimcatPackage object."""
        return package_constructor(
            package_name="empty_package", basepath=tmp_package_path
        )

    def test_basepath_after_init(self, package_obj, expected_basepath):
        assert package_obj.basepath == expected_basepath

    def test_status_after_init(self, package_obj, expected_package_status):
        assert package_obj.status == expected_package_status

    def test_serialization(self, package_obj):
        as_config = package_obj.to_config()
        new_object = as_config.create()
        assert new_object == package_obj

    def test_copy(self, package_obj):
        copy = package_obj.copy()
        assert copy == package_obj
        assert copy is not package_obj

    def test_piece_index(self, package_obj):
        if not isinstance(package_obj, DimcatPackage):
            return
        idx1 = package_obj.get_piece_index()
        table = package_obj.get_boolean_resource_table()
        idx2 = table.index
        assert idx1 == idx2

    def test_adding_resource(self, package_obj, single_path_resource, package_mode):
        if isinstance(package_obj, DimcatPackage):
            with pytest.raises(TypeError):
                package_obj._add_resource(single_path_resource, mode=package_mode)
        elif package_mode == PackageMode.RAISE:
            with pytest.raises(ResourceIsFrozenError):
                package_obj._add_resource(single_path_resource, mode=package_mode)
        else:
            n_before = package_obj.n_resources
            package_obj._add_resource(single_path_resource, mode=package_mode)
            n_after = package_obj.n_resources
            assert n_after == n_before + 1


class TestPackageFromFilePaths(TestPackage):
    @pytest.fixture()
    def package_constructor(self, request):
        return PathPackage

    @pytest.fixture()
    def expected_package_status(self, package_constructor):
        return PackageStatus.PATHS_ONLY

    @pytest.fixture()
    def package_obj(
        self, package_constructor, list_of_mixed_score_paths, tmp_package_path
    ):
        """For each subclass of TestDimcatPackage, this fixture should be overridden and yield the
        tested DimcatPackage object."""
        package = package_constructor.from_filepaths(
            basepath=tmp_package_path,
            filepaths=list_of_mixed_score_paths,
            package_name="mixed_files",
        )
        return package


class TestPackageFromFileDirectory(TestPackage):
    @pytest.fixture(params=[Package, PathPackage])
    def package_constructor(self, request):
        return request.param

    @pytest.fixture()
    def expected_package_status(self, package_constructor):
        if package_constructor == Package:
            return PackageStatus.MISALIGNED
        else:
            return PackageStatus.PATHS_ONLY

    @pytest.fixture()
    def package_obj(
        self, package_constructor: Type[Package], corpus_path, tmp_package_path
    ):
        """For each subclass of TestDimcatPackage, this fixture should be overridden and yield the
        tested DimcatPackage object."""
        change_setting("default_basepath", tmp_package_path)

        def resource_names(path):
            return os.sep.join(path.split(os.sep)[-2:])

        package = package_constructor.from_directory(
            corpus_path,
            package_name="unittest_corpus",
            exclude_re="(?:yml|py)$",  # needed as long as the three corpora contain the .github/workflow
            resource_names=resource_names,
        )
        return package

    @pytest.fixture()
    def expected_basepath(self, corpus_path):
        """The expected basepath of the resource after initialization."""
        return corpus_path


# region PackageFromDescriptor


class TestPackageFromDescriptor(TestPackage):
    @pytest.fixture()
    def expected_basepath(self):
        """The expected basepath of the resource after initialization."""
        return CORPUS_PATH

    @pytest.fixture(params=[Package, DimcatPackage])
    def package_constructor(self, request):
        return request.param

    @pytest.fixture()
    def expected_package_status(self, package_constructor):
        return PackageStatus.FULLY_SERIALIZED

    @pytest.fixture()
    def package_obj(
        self,
        package_constructor,
        package_descriptor_path,
    ):
        return package_constructor.from_descriptor_path(package_descriptor_path)


class TestPackageFromFL(TestPackageFromDescriptor):
    @pytest.fixture()
    def package_obj(self, package_from_fl_package):
        return package_from_fl_package


# endregion PackageFromDescriptor
