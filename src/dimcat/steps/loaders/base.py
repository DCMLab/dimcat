"""
A loader reads an existing datapackage or creates one by parsing data from a source.
"""
import dataclasses
import logging
import os
from pathlib import Path
from typing import (
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Set,
    Tuple,
    TypeAlias,
)

import marshmallow as mm
import pandas as pd
from dimcat import Dataset
from dimcat.base import get_setting
from dimcat.exceptions import (
    DuplicateIDError,
    ExcludedFileExtensionError,
    NoPathsSpecifiedError,
)
from dimcat.steps.base import PipelineStep
from dimcat.steps.loaders.utils import PathFactory, store_datapackage
from dimcat.utils import is_uri, make_valid_frictionless_name, resolve_path

logger = logging.getLogger(__name__)

FacetName: TypeAlias = Literal[
    "events", "control", "structure", "annotations", "metadata"
]


@dataclasses.dataclass
class LoadedFacets:
    events: Dict[tuple, pd.DataFrame] = dataclasses.field(default_factory=dict)
    control: Dict[tuple, pd.DataFrame] = dataclasses.field(default_factory=dict)
    structure: Dict[tuple, pd.DataFrame] = dataclasses.field(default_factory=dict)
    annotations: Dict[tuple, pd.DataFrame] = dataclasses.field(default_factory=dict)
    metadata: Dict[tuple, pd.Series] = dataclasses.field(default_factory=dict)


class Loader(PipelineStep):
    """Base class for all loaders."""

    accepted_file_extensions: ClassVar[Optional[Tuple[str, ...]]] = None
    """File extensions that this loader accepts. If None, all files are accepted."""

    conditionally_accepted_file_extensions: ClassVar[Optional[Tuple[str, ...]]] = None
    """File extensions that this loader accepts conditional on whether a particular piece of software is installed."""

    class Schema(PipelineStep.Schema):
        package_name = mm.fields.Str(required=True)
        basepath = mm.fields.Str()
        source = mm.fields.Str(allow_none=True)

    def __init__(
        self,
        package_name: str,
        basepath: Optional[str] = None,
        source: Optional[str] = None,
    ):
        self._package_name = package_name
        self._basepath = None
        self.basepath = basepath
        self._source = None
        self.source = source
        self._paths: Iterable[str] = []

    @property
    def basepath(self) -> str:
        return self._basepath

    @basepath.setter
    def basepath(self, basepath: str):
        self._basepath = resolve_path(basepath)

    @property
    def package_name(self) -> Optional[str]:
        return self._package_name

    @package_name.setter
    def package_name(self, package_name: Optional[str]):
        valid_name = make_valid_frictionless_name(package_name)
        if valid_name != package_name:
            self.logger.info(f"Changed package name to {valid_name}.")
        self._package_name = valid_name

    @property
    def paths(self) -> Iterable[str]:
        if self._paths:
            return self._paths
        if self._source is None:
            raise NoPathsSpecifiedError
        if isinstance(self._source, str):
            self._paths = PathFactory(self._source)
        else:
            self._paths = self._source
        return self._paths

    @property
    def source(self) -> str:
        return self._source

    @source.setter
    def source(self, source: str):
        if is_uri(source):
            raise NotImplementedError("Loading from remote URLs is not yet supported.")
        new_source = resolve_path(source)
        if not new_source:
            raise ValueError(f"Could not resolve {source}.")
        self._source = new_source

    def check_resource(self, resource: str | Path) -> None:
        """Checks whether the resource at the given path exists."""
        if not os.path.isfile(resource):
            raise FileNotFoundError(f"Resource {resource} does not exist.")

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
                self.basepath = get_setting("default_output_dir")
                self.logger.info(f"Using default basepath {self.basepath}.")

    def get_descriptor_path(self) -> str:
        """Returns the path to the datapackage descriptor."""
        zip_path = self.get_zip_path()
        descriptor_path = zip_path[:-3] + "datapackage.json"
        return descriptor_path

    def get_package_name(self) -> str:
        """Returns :attr:`package_name` if set, otherwise a valid frictionless name generated from the
        :attr:`source`."""
        return self.package_name

    def get_zip_path(self) -> str:
        package_name = self.get_package_name()
        return os.path.join(self.basepath, package_name + ".zip")

    def iter_package_descriptors(self) -> Iterator[str]:
        """Create datapackage(s) and iterate over their descriptor paths."""
        raise NotImplementedError

    def _process_resource(self, resource: str) -> None:
        """Parse the resource and extract the facets."""
        raise NotImplementedError

    def process_resource(self, resource: str | Path) -> None:
        resource = resolve_path(resource)
        self.check_resource(resource)
        return self._process_resource(resource)

    def _process_dataset(self, dataset: Dataset) -> Dataset:
        """Apply this PipelineStep to a :class:`Dataset` and return a copy containing the output(s)."""
        new_dataset = self._make_new_dataset(dataset)
        self.fit_to_dataset(new_dataset)
        for descriptor_path in self.iter_package_descriptors():
            new_dataset.load_package(descriptor_path)
        return new_dataset


class ScoreLoader(Loader):
    """Base class for all loaders that parse scores and create a datapackage containing the extracted facets."""

    class Schema(Loader.Schema):
        pass

    def __init__(
        self,
        package_name: str,
        basepath: Optional[str] = None,
        source: Optional[str] = None,
        overwrite: bool = False,
    ):
        super().__init__(
            package_name=package_name,
            basepath=basepath,
            source=source,
        )
        self.overwrite = overwrite
        self.loaded_facets = LoadedFacets()
        self._processed_ids = set()

    @property
    def processed_ids(self) -> Set[tuple]:
        return set(self._processed_ids)

    def add_piece_facet(
        self, facet_name: FacetName, ID: tuple, df: pd.DataFrame
    ) -> None:
        id2dataframe = getattr(self.loaded_facets, facet_name)
        if ID in id2dataframe:
            raise DuplicateIDError(f"Duplicate ID {ID} for facet {facet_name}.")
        if df is None or len(df) == 0:
            self.logger.debug(ID, facet_name)
            return
        id2dataframe[ID] = df
        self._processed_ids.add(ID)

    def check_resource(self, resource: str | Path) -> None:
        super().check_resource(resource)
        admissible_extensions = []
        if self.accepted_file_extensions is not None:
            admissible_extensions.extend(self.accepted_file_extensions)
        if self.conditionally_accepted_file_extensions is not None:
            admissible_extensions.extend(self.conditionally_accepted_file_extensions)
        if not admissible_extensions:
            return
        _, fext = os.path.splitext(resource)
        if fext not in admissible_extensions:
            raise ExcludedFileExtensionError(fext, admissible_extensions)

    def create_datapackage(
        self,
        overwrite: Optional[bool] = None,
    ):
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
        zip_path = self.get_zip_path()
        if not overwrite:
            if os.path.isfile(zip_path):
                raise FileExistsError(f"File {zip_path} already exists.")

    def _get_concatenated_facets(self) -> Dict[str, pd.DataFrame]:
        facet2df = {}
        for field in dataclasses.fields(self.loaded_facets):
            facet = field.name
            id2dataframe = getattr(self.loaded_facets, facet)
            if len(id2dataframe) == 0:
                continue
            if facet == "metadata":
                obj = pd.concat(id2dataframe).unstack()
                obj.index.rename(["corpus", "piece"], inplace=True)
            else:
                obj = pd.concat(id2dataframe, names=["corpus", "piece", "i"])
            facet2df[facet] = obj
        return facet2df

    def iter_package_descriptors(self) -> Iterator[str]:
        """Create datapackage(s) and iterate over their descriptor paths."""
        try:
            descriptor_path = self.create_datapackage()
        except FileExistsError:
            descriptor_path = self.get_descriptor_path()
            self.logger.info(f"Using existing datapackage at {descriptor_path}.")
        yield from [descriptor_path]

    def store_datapackage(self) -> str:
        package_name = self.get_package_name()
        facet2df = self._get_concatenated_facets()
        self.descriptor_path = store_datapackage(
            facet2df.items(), package_name, self.basepath
        )
        self.logger.info(f"Stored datapackage at {self.descriptor_path}.")
