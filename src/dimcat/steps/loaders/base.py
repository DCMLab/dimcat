"""
A loader reads an existing datapackage or creates one by parsing data from a source.
"""
import dataclasses
import logging
import os
from collections import Counter
from pathlib import Path
from typing import (
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
)

import marshmallow as mm
import pandas as pd
from dimcat.base import FriendlyEnum, get_setting
from dimcat.data.base import Data
from dimcat.data.catalogs.base import DimcatCatalog
from dimcat.data.datasets.base import Dataset
from dimcat.data.packages.base import Package, PathPackage
from dimcat.data.packages.dc import DimcatPackage
from dimcat.data.packages.score import ScorePathPackage
from dimcat.data.resources.base import PathResource, Resource
from dimcat.data.utils import is_default_package_descriptor_path, make_rel_path
from dimcat.dc_exceptions import (
    DuplicateIDError,
    DuplicateResourceIDsError,
    ExcludedFileExtensionError,
    NoPathsSpecifiedError,
)
from dimcat.steps.base import PipelineStep
from dimcat.steps.loaders.utils import store_datapackage
from dimcat.utils import make_valid_frictionless_name, resolve_path, scan_directory
from tqdm.auto import tqdm
from typing_extensions import Self

logger = logging.getLogger(__name__)


class FacetName(FriendlyEnum):
    """The names of the facets that can be extracted from scores."""

    events = "events"
    control = "control"
    structure = "structure"
    annotations = "annotations"
    metadata = "metadata"


@dataclasses.dataclass
class LoadedFacets:
    events: Dict[tuple, pd.DataFrame] = dataclasses.field(default_factory=dict)
    control: Dict[tuple, pd.DataFrame] = dataclasses.field(default_factory=dict)
    structure: Dict[tuple, pd.DataFrame] = dataclasses.field(default_factory=dict)
    annotations: Dict[tuple, pd.DataFrame] = dataclasses.field(default_factory=dict)
    metadata: Dict[tuple, pd.Series] = dataclasses.field(default_factory=dict)

    def get_concatenated_facets(self) -> Dict[str, pd.DataFrame]:
        facet2df = {}
        for field in dataclasses.fields(self):
            facet = field.name
            id2dataframe = getattr(self, facet)
            if len(id2dataframe) == 0:
                continue
            if facet == "metadata":
                obj = pd.concat(id2dataframe).unstack()
                obj.index.rename(["corpus", "piece"], inplace=True)
            else:
                obj = pd.concat(id2dataframe, names=["corpus", "piece", "i"])
            facet2df[facet] = obj
        return facet2df


class Loader(PipelineStep):
    """Base class for all loaders."""

    _accepted_file_extensions: ClassVar[Optional[Tuple[str, ...]]] = None
    """File extensions that this loader accepts. If None, all files are accepted."""

    _conditionally_accepted_file_extensions: ClassVar[Optional[Tuple[str, ...]]] = None
    """File extensions that this loader accepts conditional on whether a particular piece of software is installed."""

    _path_package_type: ClassVar[Type[Package]] = PathPackage
    """The type of package that this loader uses to collect PathResource objects."""

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
        basepath: Optional[str] = None,
        **kwargs,
    ) -> Self:
        """Create a loader from a :obj:`ScorePackage` created on the fly from an iterable of filepaths.

        Args:
            directory: The directory that is to be scanned for files with particular extensions.
            package_name:
                The name of the new package. If None, the base of the directory is used.
            extensions:
                The extensions of the files to be discovered under ``directory`` and which are to be
                turned into :class:`Resource` objects. Defaults to this loader's
                :attr:`_accepted_file_extensions`.
            file_re:
                Pass a regular expression in order to select only files that (partially) match it.
            resource_names:
                Name factory for the resources created from the paths. Names also serve as piece
                identifiers.
                By default, the filename is used. To override this behaviour you can pass a callable
                that takes a filepath and returns a name. When the callable returns None, the
                default is used (i.e., the filename).
                Whatever the name turns out to be, it will always be turned into a valid
                frictionless name via :func:`make_valid_frictionless_name`.
            corpus_names:
                Names of (or name factory for) the corpus that each resource (=piece) belongs to
                and that is used in the ('corpus', 'piece') ID.
                By default, the name of the package is used. To override this behaviour you can pass
                a callable that takes a path and returns a name. When the callable returns None,
                the default is used  (i.e., the package_name).
                Whatever the name turns out to be, it will always be turned into a valid
                frictionless name via :func:`make_valid_frictionless_name`.
            auto_validate: Set True to validate the new package after copying it.
            basepath:
                The basepath where the new package will be stored. If None, the basepath of the
                original package
        """
        if extensions is None:
            extensions = cls._accepted_file_extensions
        new_package = cls._path_package_type.from_directory(
            directory=directory,
            package_name=package_name,
            extensions=extensions,
            file_re=file_re,
            exclude_re=exclude_re,
            resource_names=resource_names,
            corpus_names=corpus_names,
            auto_validate=auto_validate,
        )
        return cls.from_package(package=new_package, basepath=basepath, **kwargs)

    @classmethod
    def from_filepaths(
        cls,
        filepaths: Iterable[str],
        basepath: Optional[str] = None,
    ) -> Self:
        """Create a loader from a DimcatPackage created on the fly from an iterable of filepaths.

        Args:
            filepaths: The filepaths that are to be turned into :class:`Resource` objects and packaged.
            basepath: The basepath where the new package will be stored. If None, the basepath of the original package
        """
        if isinstance(filepaths, (str, Path)):
            filepaths = [filepaths]
        valid, invalid = [], []
        for filepath in filepaths:
            if is_default_package_descriptor_path(filepath):
                valid.append(filepath)
            else:
                invalid.append(filepath)
        if invalid:
            plural = "s" if len(invalid) > 1 else ""
            cls.logger.warning(
                f"Ignoring the following path{plural} because they do not seem to correspond to package "
                f"descriptors: {invalid}"
            )
        if not valid:
            raise NoPathsSpecifiedError
        packages = DimcatCatalog(packages=valid)
        return cls(basepath=basepath, packages=packages)

    @classmethod
    def from_package(
        cls,
        package: DimcatPackage,
        basepath: Optional[str] = None,
    ) -> Self:
        """Create a loader from a DimcatPackage."""
        loader = cls(basepath=basepath)
        loader.add_package(package)
        return loader

    class Schema(PipelineStep.Schema):
        basepath = mm.fields.Str(
            allow_none=True,
            metadata=dict(
                description="The directory where the generated package(s) will be stored."
            ),
        )
        packages = mm.fields.Nested(DimcatCatalog.Schema)

    def __init__(
        self,
        basepath: Optional[str] = None,
        packages: Optional[DimcatCatalog] = None,
    ):
        self._basepath = None
        self._sources: List[str] = []
        self.packages = DimcatCatalog()  # no basepath because it will not be pickled
        if basepath is not None:
            self.basepath = basepath
        if packages is not None:
            self.packages.extend(packages)

    @property
    def basepath(self) -> str:
        return self._basepath

    @basepath.setter
    def basepath(self, basepath: str):
        self._basepath = Data.treat_new_basepath(basepath, other_logger=self.logger)

    @property
    def sources(self) -> List[str]:
        return list(self._sources)

    def add_package(self, package: DimcatPackage) -> None:
        """Add a package to the loader that contains resources to be processed."""
        self.packages.add_package(package)

    def check_resource(self, resource: Resource) -> None:
        """Checks whether the resource at the given path exists."""
        filepath = resource.normpath
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Resource {filepath} does not exist.")

    def create_dataset(self) -> Dataset:
        return Dataset.from_loader(self)

    def fit_to_dataset(self, dataset: Dataset) -> None:
        """Fit this PipelineStep to a :class:`Dataset`."""
        if self.basepath is None:
            if dataset.inputs.basepath is not None:
                self.basepath = dataset.inputs.basepath
                self.logger.info(f"Using basepath {self.basepath} from inputs catalog.")
            elif dataset.outputs.basepath is not None:
                self.basepath = dataset.outputs.basepath
                self.logger.info(
                    f"Using basepath {self.basepath} from outputs catalog."
                )
            else:
                self.basepath = get_setting("default_basepath")
                self.logger.info(f"Using default basepath {self.basepath}.")

    def get_basepath(self) -> str:
        """Get the basepath of the resource. If not specified, the default basepath is returned."""
        if not self.basepath:
            default_basepath = resolve_path(get_setting("default_basepath"))
            self.logger.info(f"Falling back to default basepath {default_basepath!r}.")
            return default_basepath
        return self.basepath

    def iter_resources(self) -> Iterator[Resource]:
        """Iterate over the resources in the package(s)."""
        yield from self.packages.iter_resources()

    def iter_resource_paths(self) -> Iterator[str]:
        """Iterate over the paths of the resources in the package(s)."""
        for resource in self.iter_resources():
            yield resource.normpath

    def iter_package_descriptors(self) -> Iterator[str]:
        """Create datapackage(s) for the input catalog of a Dataset and iterate over their descriptor paths."""
        for package in self.packages:
            if package.descriptor_exists:
                yield package.get_descriptor_path()

    def _process_resource(self, resource: str) -> None:
        """Parse the resource and extract the facets."""
        raise NotImplementedError

    def process_resource(self, resource: Resource) -> None:
        self.check_resource(resource)
        self._process_resource(resource)

    def _process_dataset(self, dataset: Dataset) -> Dataset:
        """Apply this PipelineStep to a :class:`Dataset` and return a copy containing the output(s)."""
        new_dataset = self._make_new_dataset(dataset)
        self.fit_to_dataset(new_dataset)
        for descriptor_path in self.iter_package_descriptors():
            new_dataset.load_package(descriptor_path)
        return new_dataset


class PackageLoader(Loader):
    """Simple loader that discovers and loads frictionless datapackages through their descriptors."""

    _accepted_file_extensions = tuple(get_setting("package_descriptor_endings"))
    default_loader_name = "package_loader"
    _path_package_type = Package

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
        basepath: Optional[str] = None,
        **kwargs,
    ) -> Self:
        """Create a loader from a :obj:`ScorePackage` created on the fly from an iterable of filepaths.

        Args:
            directory: The directory that is to be scanned for files with particular extensions.
            package_name:
                The name of the new package. If None, the base of the directory is used.
            extensions:
                The extensions of the files to be discovered under ``directory`` and which are to be
                turned into :class:`Resource` objects. Defaults to this loader's
                :attr:`_accepted_file_extensions`.
            file_re:
                Pass a regular expression in order to select only files that (partially) match it.
            resource_names:
                Name factory for the resources created from the paths. Names also serve as piece
                identifiers.
                By default, the filename is used. To override this behaviour you can pass a callable
                that takes a filepath and returns a name. When the callable returns None, the
                default is used (i.e., the filename).
                Whatever the name turns out to be, it will always be turned into a valid
                frictionless name via :func:`make_valid_frictionless_name`.
            corpus_names:
                Names of (or name factory for) the corpus that each resource (=piece) belongs to
                and that is used in the ('corpus', 'piece') ID.
                By default, the name of the package is used. To override this behaviour you can pass
                a callable that takes a path and returns a name. When the callable returns None,
                the default is used  (i.e., the package_name).
                Whatever the name turns out to be, it will always be turned into a valid
                frictionless name via :func:`make_valid_frictionless_name`.
            auto_validate: Set True to validate the new package after copying it.
            basepath:
                The basepath where the new package will be stored. If None, the basepath of the
                original package
        """
        if extensions is None:
            extensions = cls._accepted_file_extensions
        paths = list(
            scan_directory(
                directory,
                extensions=extensions,
                file_re=file_re,
                exclude_re=exclude_re,
            )
        )
        return cls.from_filepaths(paths, basepath=basepath, **kwargs)


class ScoreLoader(Loader):
    """Base class for all loaders that parse scores and create a datapackage containing the extracted facets."""

    _default_loader_name: ClassVar[str] = "score_loader"
    """The default name may be used as file name when storing the resulting package."""
    _path_package_type: ClassVar[Type[Package]] = ScorePathPackage
    """The type of package that the loader creates and returns."""

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
        basepath: Optional[str] = None,
        loader_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> Self:
        """Create a loader from a :obj:`ScorePackage` created on the fly from an iterable of filepaths.

        Args:
            directory: The directory that is to be scanned for files with particular extensions.
            package_name:
                The name of the new package. If None, the base of the directory is used.
            extensions:
                The extensions of the files to be discovered under ``directory`` and which are to be
                turned into :class:`Resource` objects. Defaults to this loader's
                :attr:`_accepted_file_extensions`.
            file_re:
                Pass a regular expression in order to select only files that (partially) match it.
            resource_names:
                Name factory for the resources created from the paths. Names also serve as piece
                identifiers.
                By default, the filename is used. To override this behaviour you can pass a callable
                that takes a filepath and returns a name. When the callable returns None, the
                default is used (i.e., the filename).
                Whatever the name turns out to be, it will always be turned into a valid
                frictionless name via :func:`make_valid_frictionless_name`.
            corpus_names:
                Names of (or name factory for) the corpus that each resource (=piece) belongs to
                and that is used in the ('corpus', 'piece') ID.
                By default, the name of the package is used. To override this behaviour you can pass
                a callable that takes a path and returns a name. When the callable returns None,
                the default is used  (i.e., the package_name).
                Whatever the name turns out to be, it will always be turned into a valid
                frictionless name via :func:`make_valid_frictionless_name`.
            auto_validate: Set True to validate the new package after copying it.
            basepath:
                The basepath where the new package will be stored. If None, the basepath of the
                original package
        """
        return super().from_directory(
            directory=directory,
            package_name=package_name,
            extensions=extensions,
            file_re=file_re,
            exclude_re=exclude_re,
            resource_names=resource_names,
            corpus_names=corpus_names,
            auto_validate=auto_validate,
            basepath=basepath,
            loader_name=loader_name,
            overwrite=overwrite,
        )

    # noinspection PyMethodOverriding
    @classmethod
    def from_filepaths(
        cls,
        filepaths: Iterable[str],
        package_name: str,
        resource_names: Optional[Iterable[str] | Callable[[str], str]] = None,
        corpus_names: Optional[Iterable[str] | Callable[[str], Optional[str]]] = None,
        auto_validate: bool = False,
        basepath: Optional[str] = None,
        loader_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> Self:
        """Create a loader from a :obj:`ScorePackage` created on the fly from an iterable of filepaths.

        Args:
            filepaths: The filepaths that are to be turned into :class:`Resource` objects and packaged.
            package_name: The name of the new package.
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
            filepaths = [filepaths]
        new_package = ScorePathPackage.from_filepaths(
            filepaths=filepaths,
            package_name=package_name,
            resource_names=resource_names,
            corpus_names=corpus_names,
            auto_validate=auto_validate,
            basepath=basepath,
        )
        return cls.from_package(
            package=new_package,
            basepath=basepath,
            loader_name=loader_name,
            overwrite=overwrite,
        )

    @classmethod
    def from_package(
        cls,
        package: ScorePathPackage,
        basepath: Optional[str] = None,
        loader_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> Self:
        """Create a loader from a DimcatPackage."""
        loader = cls(
            basepath=basepath,
            loader_name=loader_name,
            overwrite=overwrite,
        )
        loader.add_package(package)
        return loader

    @classmethod
    def from_resources(
        cls,
        resources: Iterable[PathResource] | PathResource,
        package_name: str,
        auto_validate: bool = False,
        basepath: Optional[str] = None,
        loader_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> Self:
        """Create a loader from a :obj:`ScorePackage` created on the fly from an iterable of PathResources.

        Args:
            resources: The :class:`PathResource` objects that will be turned into a package.
            package_name: The name of the new package.
            auto_validate: Set True to validate the new package after copying it.
            basepath: The basepath where the new package will be stored. If None, the basepath of the original package
        """
        if isinstance(resources, Resource):
            resources = [resources]
        new_package = ScorePathPackage.from_resources(
            resources=resources,
            package_name=package_name,
            auto_validate=auto_validate,
            basepath=basepath,
        )
        return cls.from_package(
            package=new_package,
            basepath=basepath,
            loader_name=loader_name,
            overwrite=overwrite,
        )

    class Schema(Loader.Schema):
        loader_name = mm.fields.Str(
            allow_none=True,
            metadata=dict(
                description="The name of the loader. Used to name the generated package."
            ),
        )

    def __init__(
        self,
        basepath: Optional[str] = None,
        loader_name: Optional[str] = None,
        overwrite: bool = False,
    ):
        """

        Args:
            basepath: Directory in which to store the loaded data as a datapackage.
            loader_name: Name of the datapackage containing the loaded data.
            overwrite:
                By default, the loader will not parse anything if the target package ``loader_name``
                already exists in ``basepath``. Set this to True to re-parse and overwrite.
        """
        self._loader_name = None
        self.overwrite = overwrite
        self.loaded_facets = LoadedFacets()
        self._descriptor_path = None
        """Will be set when the loader has created and stored a datapackage."""
        self._processed_ids = set()
        super().__init__(basepath=basepath)
        if loader_name is not None:
            self.loader_name = loader_name

    @property
    def descriptor_exists(self) -> bool:
        descriptor_path = self.get_descriptor_path()
        if not descriptor_path:
            return False
        return os.path.isfile(descriptor_path)

    @property
    def descriptor_path(self) -> Optional[str]:
        return self._descriptor_path

    @property
    def loader_name(self) -> Optional[str]:
        return self._loader_name

    @loader_name.setter
    def loader_name(self, loader_name: Optional[str]):
        valid_name = make_valid_frictionless_name(loader_name)
        if valid_name != loader_name:
            self.logger.info(f"Changed loader_name to {valid_name}.")
        self._loader_name = valid_name

    @property
    def processed_ids(self) -> Set[tuple]:
        return set(self._processed_ids)

    @property
    def zip_file_exists(self) -> bool:
        return os.path.isfile(self.get_zip_path())

    def add_piece_facet_dataframe(
        self, facet_name: FacetName, ID: tuple, df: pd.DataFrame | pd.Series
    ) -> None:
        facet_name = FacetName(facet_name)
        id2dataframe = getattr(self.loaded_facets, facet_name)
        if ID in id2dataframe:
            raise DuplicateIDError(f"Duplicate ID {ID} for facet {facet_name!r}.")
        if df is None or len(df) == 0:
            self.logger.debug(f"Facet {facet_name!r} not available for ID {ID}.")
            return
        id2dataframe[ID] = df
        self._processed_ids.add(ID)

    def check_resource(self, resource: PathResource) -> None:
        super().check_resource(resource)
        admissible_extensions = []
        if self._accepted_file_extensions is not None:
            admissible_extensions.extend(self._accepted_file_extensions)
        if self._conditionally_accepted_file_extensions is not None:
            admissible_extensions.extend(self._conditionally_accepted_file_extensions)
        if not admissible_extensions:
            return
        filepath = resource.normpath
        _, fext = os.path.splitext(filepath)
        if fext not in admissible_extensions:
            raise ExcludedFileExtensionError(fext, admissible_extensions)

    def get_loader_name(self) -> str:
        """Returns :attr:`loader_name` if set, otherwise :attr:`default_loader_name`."""
        if self.loader_name:
            return self.loader_name
        return self._default_loader_name

    def get_descriptor_filename(self) -> str:
        """Returns the filename of the datapackage descriptor."""
        if self.descriptor_path:
            return make_rel_path(self.descriptor_path, self.get_basepath())
        data_filepath = self.get_zip_filepath()[:-4]
        return f"{data_filepath}.datapackage.json"

    def get_descriptor_path(self) -> str:
        """Returns the path of the datapackage descriptor."""
        if self.descriptor_path:
            return self.descriptor_path
        descriptor_filename = self.get_descriptor_filename()
        return os.path.join(self.get_basepath(), descriptor_filename)

    def get_zip_filepath(self) -> str:
        """Returns the filename of the ZIP file that the resources of this package are serialized to."""
        loader_name = self.get_loader_name()
        return f"{loader_name}.zip"

    def get_zip_path(self) -> str:
        """Returns the path of the ZIP file that the resources of this package are serialized to."""
        zip_filename = self.get_zip_filepath()
        return os.path.join(self.get_basepath(), zip_filename)

    def iter_package_descriptors(self) -> Iterator[str]:
        """Create datapackage(s) and iterate over their descriptor paths."""
        try:
            descriptor_path = self.make_and_store_datapackage()
        except FileExistsError:
            descriptor_path = self.get_descriptor_path()
            self.logger.info(f"Using existing datapackage at {descriptor_path}.")
        if os.path.isfile(descriptor_path):
            yield from [descriptor_path]
        else:
            yield from []

    def make_and_store_datapackage(
        self,
        overwrite: Optional[bool] = None,
    ) -> str:
        """

        Args:
            overwrite:
                Set to a boolean to set :attr:`overwrite` to a new value.

        Returns:

        Raises:
            FileExistsError: If the zip file <basepath>/<package_name>.zip already exists.
        """
        if overwrite is not None:
            self.overwrite = overwrite
        if not overwrite and self.zip_file_exists:
            raise FileExistsError(f"File {self.get_zip_path()} already exists.")
        self.parse_and_extract()
        self._store_datapackage()
        return self.get_descriptor_path()

    def parse_and_extract(self) -> None:
        """Iterates over score resources and stores the extracted information in :attr:`loaded_facets`."""
        resources = list(self.iter_resources())
        IDs = [resource.ID for resource in resources]
        if len(IDs) != len(set(IDs)):
            ID_counts = {ID: count for ID, count in Counter(IDs).items() if count > 1}
            raise DuplicateResourceIDsError(ID_counts)
        for resource in (
            pbar := tqdm(
                resources,
                total=len(resources),
                desc="Parsing scores...",
            )
        ):
            pbar.set_description(f"Parsing {resource.filepath}")
            try:
                self.process_resource(resource)
            except Exception as e:
                self.logger.error(f"Error while processing {resource.normpath}: {e}")
                raise
                continue

    def _process_resource(self, resource: Resource) -> None:
        """Parse the resource and extract the facets."""
        ID = resource.ID
        filepath = resource.filepath
        _, fext = os.path.splitext(filepath)
        metadata = pd.Series(dict(rel_path=filepath))
        self.add_piece_facet_dataframe(FacetName.metadata, ID, metadata)

    def _store_datapackage(self) -> None:
        facet2df = self.loaded_facets.get_concatenated_facets()
        if len(facet2df) == 0:
            self.logger.info("No data to store.")
            return
        self._descriptor_path = store_datapackage(
            facet_df_pairs=facet2df.items(),
            name=self.get_loader_name(),
            directory=self.get_basepath(),
            overwrite=True,
        )
        assert (
            self.descriptor_path is not None
        ), "No descriptor_path was set after making the datapackage."
        self.logger.info(f"Stored datapackage at {self.descriptor_path}.")
