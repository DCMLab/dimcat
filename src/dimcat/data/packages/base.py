from __future__ import annotations

import os
import re
from collections import defaultdict
from enum import IntEnum, auto
from inspect import isclass
from pathlib import Path
from pprint import pformat
from typing import (
    Callable,
    ClassVar,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeAlias,
    Union,
)

import frictionless as fl
import marshmallow as mm
from dimcat.base import DimcatConfig, FriendlyEnum, get_class
from dimcat.data.base import Data
from dimcat.data.resources.base import (
    F,
    FeatureName,
    PathResource,
    R,
    Resource,
    reconcile_base_and_file,
)
from dimcat.data.resources.dc import DimcatResource, Feature, FeatureSpecs, PieceIndex
from dimcat.data.resources.facets import Facet, MuseScoreFacet
from dimcat.data.resources.features import Metadata
from dimcat.data.resources.utils import feature_specs2config
from dimcat.data.utils import (
    check_descriptor_filename_argument,
    make_rel_path,
    store_as_json_or_yaml,
)
from dimcat.dc_exceptions import (
    BaseFilePathMismatchError,
    BasePathNotDefinedError,
    EmptyPackageError,
    FilePathNotDefinedError,
    NoMatchingResourceFoundError,
    PackageDescriptorHasWrongTypeError,
    PackageInconsistentlySerializedError,
    PackageNotFullySerializedError,
    PackagePathsNotAlignedError,
    ResourceIsFrozenError,
    ResourceIsMisalignedError,
    ResourceIsPackagedError,
    ResourceNamesNonUniqueError,
    ResourceNotFoundError,
)
from dimcat.utils import (
    check_file_path,
    make_valid_frictionless_name,
    make_valid_frictionless_name_from_filepath,
    resolve_path,
    scan_directory,
    treat_basepath_argument,
)
from typing_extensions import Self


class PackageMode(FriendlyEnum):
    """The behaviour of a Package when adding a resource with incompatible paths."""

    RAISE = "RAISE"
    """Raises an error when adding a resource with an incompatible path."""
    RECONCILE_SAFELY = "RECONCILE_SAFELY"
    """Copies newly added resources to the package's basepath if necessary but without overwriting existing files."""
    RECONCILE_EVERYTHING = "RECONCILE_EVERYTHING"
    """Copies newly added resources to the package's basepath if necessary, overwriting existing files."""
    ALLOW_MISALIGNMENT = "ALLOW_MISALIGNMENT"
    """Reconcile the resource and add a physical copy to the package ZIP."""


class PackageStatus(IntEnum):
    """Expresses the status of a :clas:`Package` with respect to the paths of the included resources being aligned
    with the package's basepath and serialized to the package's ZIP file or not. The enum members have increasing
    integer values starting with EMPTY == 0.

    +----------------------+------------+---------------------+---------------+----------------+
    | PackageStatus        | is_aligned | package_exists      | R.is_packaged | Resource types |
    |                      |            | & descriptor_exists |               |                |
    +======================+============+=====================+===============+================+
    | EMPTY                | True       | ?                   | True          | any            |
    +----------------------+------------+---------------------+---------------+----------------+
    | PATHS_ONLY           | ?          | ?                   | ?             | PathResource   |
    +----------------------+------------+---------------------+---------------+----------------+
    | MISALIGNED           | False      | ?                   | False         | any            |
    +----------------------+------------+---------------------+---------------+----------------+
    | ALIGNED              | True       | False               | True          | any            |
    +----------------------+------------+---------------------+---------------+----------------+
    | PARTIALLY_SERIALIZED | True       | True                | True          | any            |
    +----------------------+------------+---------------------+---------------+----------------+
    | FULLY_SERIALIZED     | True       | True                | True          | any            |
    +----------------------+------------+---------------------+---------------+----------------+
    """

    EMPTY = 0
    PATHS_ONLY = auto()
    MISALIGNED = auto()
    ALIGNED = auto()
    PARTIALLY_SERIALIZED = auto()
    FULLY_SERIALIZED = auto()


class PackageSchema(Data.Schema):
    resources = mm.fields.List(
        mm.fields.Nested(DimcatResource.PickleSchema),
        metadata=dict(description="The resources contained in the package."),
    )
    package_name = mm.fields.Str(
        required=True,
        metadata=dict(description="The name of the package."),
        data_key="name",
    )
    descriptor_filename = mm.fields.String(allow_none=True, metadata={"expose": False})
    auto_validate = mm.fields.Boolean(metadata={"expose": False})
    # ToDo: accept the rest as additional metadata dict as the "custom" field

    @mm.pre_load
    def catch_package_name_argument(self, data, **kwargs):
        if "package_name" in data:
            data["name"] = data.pop("package_name")
        return data


class Package(Data):
    """Wrapper for a :obj:`frictionless.Package`. The purpose of a Package is to create, load, and
    store a collection of :obj:`Resource` objects. The default way of storing a
    :obj:`DimcatResource` package is a ``[name.]datapackage.json`` descriptor and a .zip file
    containing one .tsv file per DimcatResource contained in the package.

    Attributes
    ----------

    * ``package`` (:obj:`frictionless.Package`) - The frictionless Package object that is wrapped
      by this class.
    * ``package_name`` (:obj:`str`) - The name of the package that can be used to access it.
    * ``basepath`` (:obj:`str`) - The basepath where the package and its .json descriptor are stored.
    """

    _accepted_resource_types: ClassVar[Tuple[Type[Resource], ...]] = (Resource,)
    """:meth:`add_resource` if a given resource is not an instance of one of these. The first one
    is used as default constructor in :meth:`create_and_add_resource`.
    """

    _auto_serialize: ClassVar[bool] = False
    """If True, the package is serialized to disk after each resource is added."""

    _detects_extensions: ClassVar[Iterable[str]] = None
    """Determines which files are detected by :meth:`from_directory` if ``extensions`` is not specified.
    If None, all files are detected.
    """
    _default_mode: ClassVar[PackageMode] = PackageMode.ALLOW_MISALIGNMENT
    """How the class deals with newly added resources. See :class:`PackageMode` for details."""

    _store_zipped: ClassVar[bool] = True
    """Whether, upon serialization, the resources are to be stored in a single ZIP file or as individual files."""

    @classmethod
    def _make_new_resource(
        cls,
        filepath: str,
        resource_name: Optional[str] = None,
        corpus_name: Optional[str] = None,
        basepath: Optional[str] = None,
    ) -> PathResource:
        """Create a new Resource from a filepath.

        Args:
            filepath: The filepath of the new resource.
            resource_name: The name of the new resource. If None, the filename is used.
            corpus_name: The name of the new resource. If None, the default is used.

        Returns:
            The new Resource.
        """
        Constructor = cls._accepted_resource_types[0]
        new_resource = Constructor.from_filepath(
            filepath=filepath,
            resource_name=resource_name,
            basepath=basepath,
        )
        if corpus_name:
            new_resource.corpus_name = make_valid_frictionless_name(corpus_name)
        return new_resource

    @classmethod
    def from_descriptor(
        cls,
        descriptor: dict | fl.Package,
        descriptor_filename: Optional[str] = None,
        auto_validate: Optional[bool] = None,
        basepath: Optional[str] = None,
    ) -> Self:
        """Create a new Package from a frictionless descriptor dictionary.

        Args:
            descriptor: Dictionary corresponding to a frictionless descriptor.
            basepath: The basepath for all resources in the package.
            auto_validate: Whether to automatically validate the package.

        Returns:
            The new Package.
        """
        if isinstance(descriptor, fl.Package):
            fl_package = descriptor
            if basepath is None:
                basepath = fl_package.basepath
        elif isinstance(descriptor, str):
            raise ValueError(
                f"{cls.name}.from_descriptor() expects a descriptor, not a string. Did you mean "
                f"{cls.name}.from_descriptor_path()?"
            )
        else:
            fl_package = fl.Package.from_descriptor(descriptor)
        if auto_validate is None:
            value_in_descriptor = fl_package.custom.get("auto_validate")
            if value_in_descriptor is None:
                auto_validate = False
            else:
                auto_validate = value_in_descriptor
        package_name = fl_package.name
        if dtype := fl_package.custom.get("dtype"):
            # the descriptor.custom dict contains serialization data for a DiMCAT object so we will deserialize
            # it with the appropriate dtype class constructor
            Constructor = get_class(dtype)
            if not issubclass(Constructor, cls):
                raise PackageDescriptorHasWrongTypeError(
                    cls.name, Constructor, fl_package.name
                )
            descriptor = fl_package.to_dict()
            descriptor = dict(
                descriptor,
                descriptor_filename=descriptor_filename,
                auto_validate=auto_validate,
                basepath=basepath,
            )
            return Constructor.schema.load(descriptor)
        if (creator := fl_package.custom.get("creator")) and creator["name"] == "ms3":
            Constructor = get_class("MuseScorePackage")
            ResourceConstructor = MuseScoreFacet
        else:
            Constructor = cls
            ResourceConstructor = Resource
        resources = [
            ResourceConstructor.from_descriptor(
                descriptor=resource,
                basepath=basepath,
                descriptor_filename=descriptor_filename,
                auto_validate=auto_validate,
            )
            for resource in fl_package.resources
        ]
        return Constructor(
            package_name=package_name,
            resources=resources,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            metadata=fl_package.custom,
        )

    @classmethod
    def from_descriptor_path(
        cls,
        descriptor_path: str,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
    ) -> Self:
        """Create a new Package from a descriptor path.

        Args:
            descriptor_path: The path to the descriptor file.
            basepath: The basepath for all resources in the package.
            auto_validate: Whether to automatically validate the package.

        Returns:
            The new Package.
        """
        if basepath is None:
            basepath, descriptor_filename = os.path.split(descriptor_path)
        else:
            basepath, descriptor_filename = reconcile_base_and_file(
                basepath, descriptor_path
            )
        fl_package = fl.Package.from_descriptor(descriptor_path)
        return cls.from_descriptor(
            fl_package,
            descriptor_filename=descriptor_filename,
            auto_validate=auto_validate,
            basepath=basepath,
        )

    @classmethod
    def from_filepaths(
        cls,
        filepaths: Iterable[str],
        package_name: str,
        resource_names: Optional[Iterable[str] | Callable[[str], Optional[str]]] = None,
        corpus_names: Optional[
            Iterable[str] | Callable[[str], Optional[str]] | str
        ] = None,
        auto_validate: bool = False,
        basepath: Optional[str] = None,
    ) -> Self:
        """Create a new Package from an iterable of filepaths.

        Args:
            filepaths: The filepaths that are to be turned into :class:`Resource` objects and packaged.
            package_name: The name of the new package. If None, the name of the original package is used.
            resource_names:
                Names of (or name factory for) the created resources serving as piece identifiers.
                By default, the filename is used. To override this behaviour you can pass an iterable
                of names corresponding to paths, or a callable that takes a path and returns a name.
                When the callable returns None, the default is used (i.e., the filename).
                Whatever the name turns out to be, it will always be turned into a valid
                frictionless name via :func:`make_valid_frictionless_name`.
            corpus_names:
                Names of (or name factory for) the corpus that each resource (=piece) belongs to
                and that is used in the ('corpus', 'piece') ID.
                By default, the name of the package is used. To override this behaviour you can pass
                an iterable of names corresponding to paths, or a callable that takes a path and
                returns a name. When the callable returns None, the default is used  (i.e., the
                package_name).
                Whatever the name turns out to be, it will always be turned into a valid
                frictionless name via :func:`make_valid_frictionless_name`.
            auto_validate: Set True to validate the new package after copying it.
            basepath: The basepath where the new package will be stored. If None, the basepath of the original package
        """
        if isinstance(filepaths, (str, Path)):
            raise TypeError(f"Expecting an iterable of paths, got {filepaths!r}")
        resource_creation_kwargs = [dict(filepath=fp) for fp in filepaths]
        if not resource_names:
            resource_names = make_valid_frictionless_name_from_filepath
        if callable(resource_names):
            resource_names = [resource_names(fp) for fp in filepaths]
        resource_creation_kwargs = (
            []
        )  # dicts with kwargs to be passed to :meth:`_make_new_resource`
        name2paths = defaultdict(list)  # gather {name -> [paths]} for error reporting
        for filepath, resource_name in zip(filepaths, resource_names):
            if resource_name is None:
                name = make_valid_frictionless_name_from_filepath(filepath)
            else:
                name = resource_name
            name2paths[name].append(filepath)
            resource_creation_kwargs.append(dict(filepath=filepath, resource_name=name))
        show_paths = {
            name: paths for name, paths in name2paths.items() if len(paths) > 1
        }
        if len(show_paths) > 1:
            raise ResourceNamesNonUniqueError(show_paths)
        if corpus_names:
            if callable(corpus_names):
                corpus_names = [corpus_names(fp) for fp in filepaths]
            elif isinstance(corpus_names, str):
                corpus_names = [corpus_names] * len(resource_creation_kwargs)
            corpus_names = [
                corpus_name if corpus_name else package_name
                for corpus_name in corpus_names
            ]
            resource_creation_kwargs = [
                dict(kwargs, corpus_name=name)
                for kwargs, name in zip(resource_creation_kwargs, corpus_names)
            ]
        if basepath:
            resource_creation_kwargs = [
                dict(kwargs, basepath=basepath) for kwargs in resource_creation_kwargs
            ]
        new_resources = [
            cls._make_new_resource(**kwargs) for kwargs in resource_creation_kwargs
        ]
        return cls.from_resources(
            new_resources,
            package_name=package_name,
            auto_validate=auto_validate,
            basepath=basepath,
        )

    @classmethod
    def from_directory(
        cls,
        directory: str,
        package_name: Optional[str] = None,
        extensions: Optional[Iterable[str]] = None,
        file_re: Optional[str] = None,
        exclude_re: Optional[str] = None,
        resource_names: Optional[Callable[[str], Optional[str]]] = None,
        corpus_names: Optional[Callable[[str], Optional[str]]] = None,
        auto_validate: bool = False,
    ) -> Self:
        """Create a new Package from an iterable of filepaths.

        Args:
            directory: The directory that is to be scanned for files with particular extensions.
            package_name:
                The name of the new package. If None, the base of the directory is used.
            extensions:
                The extensions of the files to be discovered under ``directory`` and which are to be turned into
                :class:`Resource` objects via :meth:`from_filepaths`.
            resource_names:
                Name factory for the resources created from the paths. Names also serve as piece
                identifiers.
                By default, the filename is used. To override this behaviour you can pass a callable
                that takes a filepath and returns a name. When the callable returns None, the
                default is used (i.e., the filename).
                Whatever the name turns out to be, it will always be turned into a valid
                frictionless name via :func:`make_valid_frictionless_name`.
            file_re:
                Pass a regular expression in order to select only files that (partially) match it.
            corpus_names:
                Names of (or name factory for) the corpus that each resource (=piece) belongs to
                and that is used in the ('corpus', 'piece') ID.
                By default, the name of the package is used. To override this behaviour you can pass
                a callable that takes a path and returns a name. When the callable returns None,
                the default is used  (i.e., the package_name).
                Whatever the name turns out to be, it will always be turned into a valid
                frictionless name via :func:`make_valid_frictionless_name`.
            auto_validate: Set True to validate the new package after copying it.
        """
        directory = resolve_path(directory)
        if extensions is None and cls._detects_extensions:
            extensions = cls._detects_extensions
        elif isinstance(extensions, str):
            extensions = (extensions,)
        paths = list(
            scan_directory(
                directory,
                extensions=extensions,
                file_re=file_re,
                exclude_re=exclude_re,
            )
        )
        cls.logger.info(f"Found {len(paths)} files in {directory}.")
        if not package_name:
            package_name = os.path.basename(directory)
        return cls.from_filepaths(
            paths,
            package_name=package_name,
            resource_names=resource_names,
            corpus_names=corpus_names,
            auto_validate=auto_validate,
            basepath=directory,
        )

    @classmethod
    def from_package(
        cls,
        package: Package,
        package_name: Optional[str] = None,
        descriptor_filename: Optional[str] = None,
        auto_validate: Optional[bool] = None,
        basepath: Optional[str] = None,
    ) -> Self:
        """Create a new Package from an existing Package by copying all resources.

        Args:
            package: The Package to copy.
            package_name: The name of the new package. If None, the name of the original package is used.
            descriptor_filename:
                Pass a JSON or YAML filename or relative filepath to override the default (``<package_name>.json``).
                Following frictionless specs it should end on ".datapackage.[json|yaml]".
            auto_validate: Set a value to override the value set in ``package``.
            basepath: The basepath where the new package will be stored. If None, the basepath of the original package
        """
        if not isinstance(package, Package):
            if isinstance(package, fl.Package):
                cls.logger.debug(
                    f"Received a frictionless.Package, passing it on to {cls.name}.from_descriptor()."
                )
                return cls.from_descriptor(package)
            raise TypeError(f"Expected a Package, got {type(package)!r}")
        fl_package = package._package.to_copy()
        if package_name is None:
            package_name = package.package_name
        if basepath is None:
            basepath = package.basepath
        if descriptor_filename is None:
            descriptor_filename = package.descriptor_filename
        if auto_validate is not None:
            if package.auto_validate is not None:
                auto_validate = package.auto_validate
            else:
                auto_validate = False
        new_package = cls(
            package_name=package_name,
            descriptor_filename=descriptor_filename,
            auto_validate=auto_validate,
            basepath=basepath,
        )
        new_package._package = fl_package
        for resource in package._resources:
            new_package._resources.append(resource.copy())
        new_package._status = package._status
        return new_package

    @classmethod
    def from_resources(
        cls,
        resources: Iterable[Resource],
        package_name: str,
        descriptor_filename: Optional[str] = None,
        auto_validate: bool = False,
        basepath: Optional[str] = None,
    ) -> Self:
        """Create a new Package from an iterable of :class:`Resource`.

        Args:
            resources: The Resources to package.
            package_name: The name of the new package.
            descriptor_filename:
                Pass a JSON or YAML filename or relative filepath to override the default (``<package_name>.json``).
                Following frictionless specs it should end on ".datapackage.[json|yaml]".
            auto_validate: Set True to validate the new package after copying it.
            basepath: The basepath where the new package will be stored. If None, the basepath of the original package
        """
        new_package = cls(
            package_name=package_name,
            descriptor_filename=descriptor_filename,
            auto_validate=auto_validate,
            basepath=basepath,
        )
        if isinstance(resources, Resource):
            resources = (resources,)
        for resource in resources:
            new_package.add_resource(resource)
        return new_package

    class PickleSchema(PackageSchema):
        pass

    class Schema(PackageSchema, Data.Schema):
        pass

    def __init__(
        self,
        package_name: str,
        resources: Iterable[Resource] = None,
        basepath: Optional[str] = None,
        descriptor_filename: Optional[str] = None,
        auto_validate: bool = False,
        metadata: Optional[dict] = None,
    ) -> None:
        """

        Args:
            metadata:
            package_name:
                Name of the package that can be used to retrieve it.
            resources:
                An iterable of :class:`Resource` objects to add to the package.
            descriptor_filename:
                Pass a JSON or YAML filename or relative filepath to override the default (``<package_name>.json``).
                Following frictionless specs it should end on ".datapackage.[json|yaml]".
            basepath:
                The absolute path on the local file system where the package descriptor and all contained resources
                are stored. The filepaths of all included :class:`DimcatResource` objects need to be relative to the
                basepath and DiMCAT does its best to ensure this.
            auto_validate:
                By default, the package is validated everytime a resource is added. Pass False to disable this.
            metadata:
                Custom metadata to be maintained in the package descriptor.
        """
        if not package_name:
            raise ValueError("package_name cannot be empty")
        self._package = fl.Package(resources=[])
        if metadata:
            self._package.custom.update(metadata)
        self._status = PackageStatus.EMPTY
        self._resources: List[Resource] = []
        self._descriptor_filename: Optional[str] = None
        self.auto_validate = True if auto_validate else False  # catches None => False
        super().__init__(basepath=basepath)
        self.package_name = package_name
        if descriptor_filename is not None:
            self.descriptor_filename = descriptor_filename

        if resources is not None:
            self.extend(resources)

        if auto_validate:
            self.validate(raise_exception=True)

    def __getitem__(self, item: str | int) -> R:
        if isinstance(item, int):
            return self._resources[item]
        try:
            return self.get_resource_by_name(item)
        except Exception as e:
            raise KeyError(str(e)) from e

    def __iter__(self) -> Iterator[R]:
        yield from self._resources

    def __len__(self):
        return len(self._resources)

    @property
    def available_features(self) -> Set[FeatureName]:
        """The set of all available features defined as the union of :attr:`contained_features` and
        :attr:`extractable_features`.
        """
        return self.contained_features.union(self.extractable_features)

    @property
    def basepath(self) -> str:
        return self._basepath

    @basepath.setter
    def basepath(self, basepath: str) -> None:
        basepath_arg = resolve_path(basepath)
        if self._basepath is None:
            self._basepath = treat_basepath_argument(basepath_arg, self.logger)
            self._package.basepath = basepath_arg
            return
        if self.status > PackageStatus.MISALIGNED:
            if basepath_arg == self.basepath:
                return
            state = "partially" if PackageStatus.PARTIALLY_SERIALIZED else "fully"
            raise NotImplementedError(
                f"Cannot change the basepath of a package that has already been {state} serialized. Attempted to "
                f"change from {self.basepath!r} to {basepath_arg!r}."
            )
        assert os.path.isdir(
            basepath_arg
        ), f"Basepath {basepath_arg!r} is not an existing directory."
        self._basepath = basepath_arg
        self._package.basepath = basepath_arg
        for resource in self._resources:
            resource.basepath = basepath_arg  # this is meant to fail RN

    @property
    def contained_features(self) -> Set[FeatureName]:
        """The dtypes of all feature resources included in the package."""
        return {feature.dtype for feature in self.iter_features()}

    @property
    def descriptor_exists(self) -> bool:
        descriptor_path = self.get_descriptor_path()
        if not descriptor_path:
            return False
        return os.path.isfile(descriptor_path)

    @property
    def descriptor_filename(self) -> str:
        """The path to the descriptor file on disk, relative to the basepath."""
        return self._descriptor_filename

    @descriptor_filename.setter
    def descriptor_filename(self, descriptor_filename: str) -> None:
        """The path to the descriptor file on disk, relative to the basepath."""
        check_descriptor_filename_argument(descriptor_filename)
        self._descriptor_filename = descriptor_filename

    @property
    def descriptor_is_complete(self) -> bool:
        """Returns True when the package has a descriptor on disk that contains all resources."""
        if not self.descriptor_exists:
            return False
        descriptor_path = self.get_descriptor_path()
        resource_names_in_descriptor = fl.Package.from_descriptor(
            descriptor_path
        ).resource_names
        for resource in self._resources:
            if resource.name not in resource_names_in_descriptor:
                return False
        return True

    @property
    def extractable_features(self) -> Set[FeatureName]:
        """The dtypes of all features that can be extracted from the facet resources included in the package."""
        f_name_tuples = [facet.extractable_features for facet in self.iter_facets()]
        result = set(sum(f_name_tuples, tuple()))
        result.add(FeatureName.Metadata)
        return result

    @property
    def filepath(self) -> str:
        """The filename of the package's ZIP file on disk, corresponding to ``<package_name>.zip``"""
        return f"{self.package_name}.zip"

    @property
    def is_aligned(self) -> bool:
        """Returns True when the basepaths, filepaths, and descriptor_filenames of all resources are aligned with the
        package."""
        if self.is_empty:
            return True
        if not self.basepath:
            first_resource = self._resources[0]
            basepath = first_resource.basepath
            self.logger.debug(
                f"Checking alignment based on the basepath of the first resource, "
                f"{first_resource.resource_name!r}."
            )
        else:
            basepath = self.basepath
        descriptor_filename = self.get_descriptor_filename()
        for resource in self._resources:
            if resource.basepath != basepath:
                return False
            if resource.descriptor_filename != descriptor_filename:
                return False
        return True

    @property
    def is_empty(self) -> bool:
        """Returns True when the package contains no resources."""
        return len(self._resources) == 0

    @property
    def is_fully_serialized(self) -> bool:
        """Returns True when the package has been fully serialized."""
        if not self.is_aligned:
            return False
        return all(resource.is_serialized for resource in self._resources)

    @property
    def is_partially_serialized(self) -> bool:
        """Returns True when both the resource and descriptor exist on disk but raises if only
        on of them exists."""
        if not self.is_aligned:
            return False
        n_exist = self.descriptor_exists + self.package_exists
        if n_exist == 2:
            return True
        if n_exist == 0:
            return False
        if self.descriptor_exists:
            existing = self.get_descriptor_path()
            missing = dict(
                basepath=self.basepath,
                filepath=self.filepath,
            )
        else:
            existing, missing = self.normpath, self.get_descriptor_path()
        raise PackageInconsistentlySerializedError(self.package_name, existing, missing)

    @property
    def is_paths_only(self) -> bool:
        """Returns True when the package has a basepath but no resources."""
        for resource in self._resources:
            if isinstance(resource, DimcatResource):
                return False
            if isinstance(resource, PathResource):
                continue
            if isinstance(resource, Resource):
                if resource.resource.schema.to_dict() != {}:
                    return False
                continue
            raise TypeError(f"Unknown resource type: {type(resource)}")
        return True

    @property
    def n_resources(self) -> int:
        return len(self._resources)

    @property
    def normpath(self) -> str:
        """Absolute path to the serialized or future tabular file. Raises if basepath is not set."""
        if not self.basepath:
            raise BasePathNotDefinedError
        if not self.filepath:
            raise FilePathNotDefinedError
        return os.path.join(self.basepath, self.filepath)

    # @property
    # def package(self) -> fl.Package:
    #     return self._package
    #
    # @package.setter
    # def package(self, package: str | fl.Package) -> None:
    #     if isinstance(package, Package):
    #         raise TypeError(
    #             f"To create a {self.name} from a {package.name}, use {self.name}.from_package()."
    #         )
    #     fl_package = self._handle_package_argument(package)
    #
    #     self._package = fl_package
    #     dimcat_resource_or_not = []
    #     for fl_resource in self._package.resources:
    #         fl_resource: fl.Resource
    #         dc_resource = self._handle_resource_argument(fl_resource)
    #         is_dimcat_resource = isinstance(dc_resource, DimcatResource)
    #         dimcat_resource_or_not.append(is_dimcat_resource)
    #         self._resources.append(dc_resource)
    #     if len(dimcat_resource_or_not) > 0:
    #         if all(dimcat_resource_or_not):
    #             self._status = PackageStatus.FULLY_SERIALIZED
    #         elif any(dimcat_resource_or_not):
    #             self._status = PackageStatus.PARTIALLY_SERIALIZED
    #         else:
    #             self._status = PackageStatus.PATHS_ONLY

    @property
    def package_exists(self) -> bool:
        """Returns True if the package's normpath exists on disk."""
        try:
            return os.path.isfile(self.normpath)
        except (BasePathNotDefinedError, FilePathNotDefinedError):
            return False

    @property
    def package_name(self) -> str:
        return self._package.name

    @package_name.setter
    def package_name(self, package_name: str) -> None:
        valid_name = make_valid_frictionless_name(package_name)
        if valid_name != package_name:
            self.logger.info(f"Changed {package_name!r} name to {valid_name!r}.")
        self._package.name = valid_name

    @property
    def resources(self) -> List[Resource]:
        """Returns a list of the resources in the package.
        Mutating the list will not affect the package but mutating one of the resources would.
        """
        return [r for r in self._resources]

    @property
    def resource_names(self) -> List[str]:
        return self._package.resource_names

    @property
    def status(self) -> PackageStatus:
        return self._status

    @property
    def zip_file_exists(self) -> bool:
        return os.path.isfile(self.get_zip_path())

    def _verify_creationist_arguments(
        self,
        **kwargs,
    ):
        """Spoiler alert: They are spurious."""
        pass

    def create_and_add_resource(
        self,
        resource: Optional[Resource | fl.Resource | str] = None,
        resource_name: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
    ) -> None:
        """Adds a resource to the package. Parameters are passed to :class:`DimcatResource`."""
        Constructor = self._accepted_resource_types[0]
        if isinstance(resource, self._accepted_resource_types):
            new_resource = resource.__class__.from_resource(
                resource=resource,
                resource_name=resource_name,
                basepath=basepath,
                auto_validate=auto_validate,
            )
            self.add_resource(new_resource)
            return
        if isinstance(resource, str):
            new_resource = Constructor.from_descriptor_path(
                descriptor_path=resource,
                basepath=basepath,
                auto_validate=auto_validate,
            )
            self.add_resource(new_resource)
            return
        if resource is None or isinstance(resource, fl.Resource):
            new_resource = Constructor(
                resource=resource,
                basepath=basepath,
                auto_validate=auto_validate,
            )
            if resource_name is not None:
                new_resource.resource_name = resource_name
            self.add_resource(new_resource)
            return
        raise TypeError(
            f"resource is expected to be a resource or a path to a descriptor, not {type(resource)!r}"
        )

    def add_resource(self, resource: Resource, update_descriptor: bool = False):
        """Adds a resource to the package."""
        resource = self._handle_resource_argument(resource)
        added_resource = self._add_resource(resource)
        if update_descriptor and self.package_exists and added_resource.is_serialized:
            self.store_descriptor(
                overwrite=True,
                allow_partial=True,
            )
            self._update_status()

    def _add_resource(
        self,
        resource: Resource,
        mode: Optional[PackageMode] = None,
    ) -> R:
        """Tries to add resource to the package. Behaviour depends on the ``mode``.

        Args:
            resource:
            mode:

        Returns:

        """
        if not isinstance(resource, self._accepted_resource_types):
            if len(self._accepted_resource_types) > 1:
                expected = self._accepted_resource_types
            else:
                expected = self._accepted_resource_types[0]
            raise TypeError(
                f"{self.name}s accept only {expected}, got {type(resource)!r}"
            )
        if mode is None:
            mode = self._default_mode
        # if len(self._resources) == 0 and self.package_exists:
        #     os.remove(self.normpath)
        resource = self._amend_resource_type(resource)
        resource = self._reconcile_resource(
            resource,
            mode=mode,
        )
        resource._update_status()
        self._resources.append(resource)
        self._package.add_resource(resource.resource)
        self._update_status()
        return resource

    def _amend_resource_type(self, resource) -> R:
        """Change the type of the given resource and perform transformations, if needed, before
        adding it to the package.

        Raises:
            TypeError: If the given resource is not specified in :attr:`accepted_resource_types`.
            ValueError: If the given resource has a name that already exists in the package.
        """
        if (
            isinstance(resource, Resource)
            and resource.resource_name in self.resource_names
        ):
            raise ValueError(
                f"Resource with name {resource.resource_name!r} already exists."
            )
        if isinstance(resource, self._accepted_resource_types):
            return resource
        Constructor = self._accepted_resource_types[0]
        return Constructor.from_resource(resource)

    def check_if_homogeneous(
        self,
        resource_types: Optional[Type[Resource], Tuple[Type[Resource], ...]] = None,
        status_exactly=None,
        status_at_least=None,
        status_at_most=None,
    ) -> bool:
        """Returns True if all resources in the package conform to the specified criteria.

        Args:
            resource_types: If not specified, all resources need to be of the same type.
            status_exactly: If specified, all resources need to have exactly this status.
            status_at_least: If specified, all resources need to have at least this status.
            status_at_most: If specified, all resources need to have at most this status.

        Returns:

        """
        if self.is_empty:
            return True
        if resource_types is None:
            resource_types = (self._resources[0].__class__,)
        else:
            if isclass(resource_types):
                resource_types = (resource_types,)
            resource_types = tuple(
                get_class(typ) if isinstance(typ, str) else typ
                for typ in resource_types
            )
        if not all(isinstance(resource, resource_types) for resource in self.resources):
            return False
        if status_exactly is not None and not all(
            resource.status == status_exactly for resource in self.resources
        ):
            return False
        if status_at_least is not None and not all(
            resource.status >= status_at_least for resource in self.resources
        ):
            return False
        if status_at_most is not None and not all(
            resource.status <= status_at_most for resource in self.resources
        ):
            return False

    def copy(self) -> Self:
        """Returns a copy of the package."""
        return self.from_package(self)

    def extend(self, resources: Iterable[Resource]) -> None:
        """Adds multiple resources to the package."""
        status_before = self.status
        resources = tuple(resources)
        if len(resources) == 0:
            self.logger.debug("Nothing to add.")
            return
        for n_added, resource in enumerate(resources, 1):
            self._add_resource(
                resource,
            )
        self.logger.info(
            f"Package {self.package_name!r} was extended with {n_added} resources to a total "
            f"of {self.n_resources}."
        )
        status_after = self.status
        if status_before != status_after:
            self.logger.debug(
                f"Status changed from {status_before!r} to {status_after!r}"
            )

    def extract_feature(self, feature: FeatureSpecs) -> F:
        feature_config = feature_specs2config(feature)
        feature_name = FeatureName(feature_config.options_dtype)
        if feature_name == FeatureName.Metadata:
            return self.get_metadata()
        if feature_name not in self.extractable_features:
            raise NoMatchingResourceFoundError(feature_config)
        candidate_facets = [
            facet
            for facet in self.iter_facets()
            if feature_name in facet.extractable_features
        ]
        if len(candidate_facets) > 1:
            raise NotImplementedError(
                f"More than one facet allow for extracting {feature_name!r}."
            )
        selected_facet = candidate_facets[0]
        return selected_facet._extract_feature(feature_config)

    def get_descriptor_path(
        self,
        set_default_if_missing=False,
    ) -> Optional[str]:
        """Returns the path to the descriptor file. If basepath or descriptor_filename are not set, they are set
        permanently to their defaults. If ``create_if_missing`` is set to True, the descriptor file is created if it
        does not exist yet."""
        descriptor_path = os.path.join(
            self.get_basepath(set_default_if_missing=set_default_if_missing),
            self.get_descriptor_filename(set_default_if_missing=set_default_if_missing),
        )
        return descriptor_path

    def get_descriptor_filename(
        self,
        set_default_if_missing: bool = False,
    ) -> str:
        """Like :attr:`descriptor_filename` but returning a default value if None.
        If ``set_default_if_missing`` is set to True and no basepath has been set (e.g. during initialization),
        the :attr:`basepath` is permanently set to the  default basepath.
        """
        if self.descriptor_filename:
            return self.descriptor_filename
        if self.package_name:
            descriptor_filename = f"{self.package_name}.datapackage.json"
        else:
            descriptor_filename = "datapackage.json"
        if set_default_if_missing:
            self._descriptor_filename = descriptor_filename
        return descriptor_filename

    def get_feature(self, feature: FeatureSpecs) -> F:
        """Checks if the package includes a feature matching the specs, and extracts it otherwise, if possible.

        Raises:
            NoMatchingResourceFoundError:
                If none of the previously extracted features matches the specs and none of the input resources
                allows for extracting a matching feature.
        """
        feature_config = feature_specs2config(feature)
        try:
            return self.get_resource_by_config(feature_config)
        except NoMatchingResourceFoundError:
            pass
        return self.extract_feature(feature_config)

    def get_piece_index(self) -> PieceIndex:
        """Returns the piece index corresponding to all resources' IDs, sorted."""
        IDs = set()
        for resource in self:
            IDs.add(resource.ID)
        return PieceIndex.from_tuples(sorted(IDs))

    def get_metadata(self) -> Metadata:
        """Returns the metadata of the package."""
        if self.n_resources == 0:
            raise EmptyPackageError(self.package_name)
        resources = self.get_resources_by_type("Metadata")
        if len(resources) == 0:
            raise NoMatchingResourceFoundError("Metadata")
        if len(resources) > 1:
            raise NotImplementedError(
                f"More than one metadata resource found: {resources!r}"
            )
        metadata = resources[0]
        metadata.load()
        return metadata

    def get_resource(self, resource: DimcatConfig | Type[Resource] | str):
        """High-level method that calls one of the other get_resource_* methods depending on the
        type of the argument. A string is interpreted as resource name, not as type."""
        if self.n_resources == 0:
            raise EmptyPackageError(self.package_name)
        if isinstance(resource, DimcatConfig):
            return self.get_resource_by_config(resource)
        if isinstance(resource, type):
            resources = self.get_resources_by_type(resource)
        elif isinstance(resource, str):
            try:
                return self.get_resource_by_name(resource)
            except ResourceNotFoundError:
                resources = self.get_resources_by_regex(resource)
        if len(resources) > 1:
            raise NotImplementedError(
                f"More than one {resource.__name__} resource found for {resource!r}:\n"
                f"{', '.join(r.resource_name for r in resources)}"
            )
        elif len(resources) == 0:
            raise NoMatchingResourceFoundError(resource.name, self.package_name)
        return resources[0]

    def get_resource_by_config(self, config: DimcatConfig) -> R:
        """Returns the first resource that matches the given config.

        Raises:
            EmptyPackageError: If the package is empty.
            NoMatchingResourceFoundError: If no resource matches the config.
        """
        if self.n_resources == 0:
            raise EmptyPackageError(self.package_name)
        for resource in self.resources:
            resource_config = resource.to_config()
            if resource_config.matches(config):
                self.logger.debug(
                    f"Requested config {config!r} matched with {resource_config!r}."
                )
                return resource
        raise NoMatchingResourceFoundError(config)

    def get_resource_by_name(self, name: Optional[str] = None) -> R:
        """Returns the Resource with the given name. If no name is given, returns the last resource.

        Raises:
            EmptyPackageError: If the package is empty.
            ResourceNotFoundError: If the resource with the given name is not found.
        """
        if self.n_resources == 0:
            raise EmptyPackageError(self.package_name)
        if name is None:
            return self._resources[-1]
        for resource in self._resources:
            if resource.resource_name == name:
                return resource
        raise ResourceNotFoundError(name, self.package_name)

    def get_resources_by_regex(self, regex: str) -> List[Resource]:
        """Returns the Resource objects whose names contain the given regex."""
        return [
            resource
            for resource in self._resources
            if re.search(regex, resource.resource_name)
        ]

    def get_resources_by_type(
        self,
        resource_type: Type[Resource] | str,
        include_subclasses: bool = False,
    ) -> List[Resource]:
        """Returns the Resource objects of the given type."""
        if isinstance(resource_type, str):
            resource_type = get_class(resource_type)
        if not issubclass(resource_type, Resource):
            raise TypeError(
                f"Expected a subclass of 'Resource', got {resource_type!r}."
            )
        if include_subclasses:
            return [
                resource
                for resource in self._resources
                if isinstance(resource, resource_type)
            ]
        else:
            return [
                resource
                for resource in self._resources
                if resource.__class__ == resource_type
            ]

    def _get_status(self) -> PackageStatus:
        """Returns the status of the package."""
        if self.is_empty:
            return PackageStatus.EMPTY
        if self.is_paths_only:
            return PackageStatus.PATHS_ONLY
        if not self.is_aligned:
            return PackageStatus.MISALIGNED
        if not self.is_partially_serialized:
            return PackageStatus.ALIGNED
        if self.is_fully_serialized:
            return PackageStatus.FULLY_SERIALIZED
        return PackageStatus.PARTIALLY_SERIALIZED

    def get_zip_filepath(self) -> str:
        """Returns the path of the ZIP file that the resources of this package are serialized to."""
        descriptor_filename = self.get_descriptor_filename()
        if descriptor_filename == "datapackage.json":
            zip_filename = f"{self.package_name}.zip"
        elif descriptor_filename.endswith(
            ".datapackage.json"
        ) or descriptor_filename.endswith(".datapackage.yaml"):
            zip_filename = f"{descriptor_filename[:-17]}.zip"
        return zip_filename

    def get_zip_path(self) -> str:
        """Returns the path of the ZIP file that the resources of this package are serialized to."""
        zip_filename = self.get_zip_filepath()
        return os.path.join(self.get_basepath(), zip_filename)

    def _handle_resource_argument(
        self,
        resource: Resource | fl.Resource,
    ) -> Resource:
        """Turn the argument into some :class:`Resource` object.

        Raises:
            TypeError: If the argument is neither a :class:`Resourcce` nor a frictionless.Resource.
        """
        if isinstance(resource, Resource):
            return resource
        if isinstance(resource, fl.Resource):
            return Resource.from_descriptor(resource=resource.to_dict())
        raise TypeError(
            f"Expected a frictionless.Resource or a DimcatResource, but got {type(resource)!r}."
        )

    def iter_facets(self) -> Iterator[Facet]:
        """Iterates over all facets in the package."""
        for resource in self:
            if isinstance(resource, Facet):
                yield resource

    def iter_features(self) -> Iterator[Feature]:
        """Iterates over all features in the package."""
        for resource in self:
            if isinstance(resource, Feature):
                yield resource

    def make_descriptor(self) -> dict:
        return self.pickle_schema.dump(self)

    def _reconcile_resource(
        self,
        resource: R,
        mode: Optional[PackageMode] = None,
    ) -> R:
        if mode is None:
            mode = self._default_mode
        if mode == PackageMode.ALLOW_MISALIGNMENT:
            return resource

        # try reconciling the paths

        package_descriptor_filename = self.get_descriptor_filename(
            set_default_if_missing=True
        )
        if resource.descriptor_filename is None:
            resource.descriptor_filename = package_descriptor_filename
            resource_descriptor_filename_ok = True
        else:
            resource_descriptor_filename_ok = (
                resource.descriptor_filename == package_descriptor_filename
            )
        if self.basepath is None:
            self.logger.debug(
                "Package basepath is None, resource is being added without reconciling."
            )
            return resource
        package_basepath = self.get_basepath()
        if resource.basepath is None:
            resource.basepath = package_basepath
            basepath_ok = True
        else:
            basepath_ok = resource.basepath == package_basepath
        if basepath_ok and resource_descriptor_filename_ok:
            return resource
        package_filepath = self.filepath if self._store_zipped else None
        if not basepath_ok:
            try:
                resource.basepath = package_basepath
            except (
                ResourceIsFrozenError,
                ResourceIsPackagedError,
                BaseFilePathMismatchError,
            ):
                # resource is currently pointing to a resource file and/or descriptor on disk
                try:
                    # if the resource basepath is a subpath of the package basepath, we can
                    # simply create a copy with adapted paths without having to copy the resource
                    # raises if not allowed
                    adapted_filepath = make_rel_path(
                        resource.normpath, package_basepath
                    )
                    new_fl_resource = resource.resource.to_copy()
                    new_fl_resource.basepath = package_basepath
                    new_fl_resource.path = adapted_filepath
                    new_resource = resource.__class__(
                        resource=new_fl_resource,
                        descriptor_filename=package_descriptor_filename,
                    )
                    new_resource._corpus_name = resource._corpus_name
                    return new_resource
                except BaseFilePathMismatchError:
                    pass
                if mode == PackageMode.RAISE:
                    raise ResourceIsMisalignedError(
                        resource.basepath, package_basepath, self.name
                    )
                if mode == PackageMode.RECONCILE_SAFELY:
                    try:
                        resource = resource.copy_to_new_location(
                            package_basepath,
                            filepath=package_filepath,
                            descriptor_filename=package_descriptor_filename,
                        )
                    except FileExistsError:
                        resource = resource.from_resource(
                            resource=resource,
                            descriptor_filename=package_descriptor_filename,
                            basepath=package_basepath,
                        )
                        if package_filepath is not None:
                            resource.filepath = package_filepath
                        self.logger.info(
                            f"{mode!r}: Using the existing resource at {resource.normpath!r}."
                        )
                elif mode == PackageMode.RECONCILE_EVERYTHING:
                    resource = resource.copy_to_new_location(
                        package_basepath,
                        overwrite=True,
                        filepath=package_filepath,
                        descriptor_filename=package_descriptor_filename,
                    )
                else:
                    raise NotImplementedError(f"Unexpected PackageMode {mode!r}.")
        elif not resource_descriptor_filename_ok:
            if mode == PackageMode.RAISE:
                raise ResourceIsMisalignedError(
                    resource.descriptor_filename, package_descriptor_filename, self.name
                )
            elif mode in (
                PackageMode.RECONCILE_SAFELY,
                PackageMode.RECONCILE_EVERYTHING,
            ):
                resource._set_descriptor_filename(package_descriptor_filename)
        return resource

    def replace_resource(
        self,
        resource: Resource,
        name_of_replaced_resource: Optional[str] = None,
    ) -> None:
        """Replaces the package with the same name as the given package with the given package."""
        if not isinstance(resource, Resource):
            msg = f"{self.name}.replace_resource() takes a Resource, not {type(resource)!r}."
            raise TypeError(msg)
        search_name = (
            name_of_replaced_resource
            if name_of_replaced_resource
            else resource.resource_name
        )
        for i, r in enumerate(self._resources):
            if r.resource_name == search_name:
                self._resources[i] = resource
                self.logger.info(
                    f"Replaced resource {search_name!r} with "
                    f"resource {resource.resource_name!r}."
                )
                return
        raise ResourceNotFoundError(search_name, self.package_name)

    def _set_descriptor_filename(self, descriptor_filename):
        if self.descriptor_exists:
            if (
                descriptor_filename == self._descriptor_filename
                or descriptor_filename == self.get_descriptor_path()
            ):
                self.logger.info(
                    f"Descriptor filepath for {self.name!r} was already set to {descriptor_filename!r}."
                )
            else:
                raise RuntimeError(
                    f"Cannot set descriptor_filename for {self.name!r} to {descriptor_filename} because it already "
                    f"set to the existing one at {self.get_descriptor_path()!r}."
                )
        if os.path.isabs(descriptor_filename):
            filepath = check_file_path(
                descriptor_filename,
                extensions=("package.json", "package.yaml"),
                must_exist=False,
            )
            if self.basepath is None:
                basepath, rel_path = os.path.split(filepath)
                self.basepath = basepath
                self.logger.info(
                    f"The absolute descriptor_path {filepath!r} was used to set the basepath to "
                    f"{basepath!r} and descriptor_filename to {rel_path}."
                )
            else:
                rel_path = make_rel_path(filepath, self.basepath)
                self.logger.info(
                    f"The absolute descriptor_path {filepath!r} was turned into the relative path "
                    f"{rel_path!r} using the basepath {self.basepath!r}."
                )
            self._descriptor_filename = rel_path
        else:
            self.logger.info(f"Setting descriptor_filename to {descriptor_filename!r}.")
            self._descriptor_filename = descriptor_filename

    def store_descriptor(
        self,
        descriptor_path: Optional[str] = None,
        overwrite=True,
        allow_partial=False,
    ) -> str:
        """Stores the descriptor to disk based on the package's configuration and returns its path."""
        if (
            self._default_mode is not PackageMode.ALLOW_MISALIGNMENT
            and not self.is_aligned
        ):
            show_misaligned = dict(
                target_basepath=self.get_basepath(),
                target_descriptor_filename=self.get_descriptor_filename(),
            )
            for r in self.resources:
                misaligned = {
                    attr: val
                    for attr, val in zip(
                        ("basepath", "descriptor_filename"),
                        (r.basepath, r.descriptor_filename),
                    )
                    if val != show_misaligned["target_" + attr]
                }
                if misaligned:
                    show_misaligned[r.resource_name] = misaligned
            raise PackagePathsNotAlignedError(
                f"Cannot store descriptor for this {self.name} because its resources are not aligned:\n"
                f"{pformat(show_misaligned, sort_dicts=False)}"
            )
        if self.status is not PackageStatus.PATHS_ONLY and (
            not self.is_fully_serialized and not allow_partial
        ):
            raise PackageNotFullySerializedError(
                f"Cannot store descriptor for this {self.name} because not all resources have been serialized. "
                f"If you want to allow this, set allow_partial=True."
            )
        if descriptor_path is None:
            descriptor_path = self.get_descriptor_path(set_default_if_missing=False)
            new_descriptor_filename = None
        else:
            new_descriptor_filename = make_rel_path(descriptor_path, self.basepath)
            new_descriptor_filename = check_descriptor_filename_argument(
                new_descriptor_filename
            )
        if not overwrite and os.path.isfile(descriptor_path):
            self.logger.info(
                f"Descriptor exists already and will not be overwritten: {descriptor_path}"
            )
            return descriptor_path
        descriptor_dict = self.make_descriptor()
        store_as_json_or_yaml(descriptor_dict, descriptor_path)
        if new_descriptor_filename is not None:
            self.descriptor_filename = new_descriptor_filename
            self.logger.debug(
                f"Updated descriptor_filename to {new_descriptor_filename!r}."
            )
        if self.auto_validate:
            _ = self.validate(raise_exception=True)
        return descriptor_path

    def summary_dict(self, verbose: bool = False) -> str:
        """Returns a summary of the package."""
        summary = self._package.to_descriptor()
        summary["basepath"] = self.basepath
        if verbose:
            return summary
        summary["resources"] = [f"{r.resource_name!r} ({r.dtype})" for r in self]
        return summary

    def _update_status(self):
        self._status = self._get_status()

    def validate(self, raise_exception: bool = False) -> fl.Report:
        if self.n_resources != len(self._package.resource_names):
            name = (
                "<unnamed Package>"
                if self.package_name is None
                else f"package {self.package_name}"
            )
            raise ValueError(
                f"Number of Resources in {name} ({self.n_resources}) does not match number of resources in "
                f"the wrapped frictionless.Package ({len(self._package.resource_names)})."
            )
        report = self._package.validate()
        if not report.valid and raise_exception:
            errors = [err.message for task in report.tasks for err in task.errors]
            raise fl.FrictionlessException("\n".join(errors))
        return report


class PathPackage(Package):
    """Behaves like :class:`Package` but with the important difference that it never interprets filepaths as
    frictionless resource descriptors (which Package loads as the appropriate :class:`Resource` type).
    """

    _accepted_resource_types = (PathResource,)
    _default_mode = PackageMode.ALLOW_MISALIGNMENT
    _detects_extensions = None  # any


PackageSpecs: TypeAlias = Union[Package, fl.Package, str]
