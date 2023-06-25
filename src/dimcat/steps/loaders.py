"""
A loader reads an existing datapackage or creates one by parsing data from a source.
"""
import dataclasses
import json
import logging
import os
import re
from pathlib import Path
from typing import Collection, Dict, Iterable, Literal, Optional, Set, Tuple, TypeAlias

import marshmallow as mm
import ms3
import pandas as pd
from dimcat.exceptions import DuplicateIDError
from dimcat.steps.base import PipelineStep
from dimcat.utils import check_name, is_uri, make_valid_frictionless_name, resolve_path
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

FacetName: TypeAlias = Literal[
    "events", "control", "structure", "annotations", "metadata"
]


def store_facets_as_zip(
    facet_df_pairs: Iterable[Tuple[str, pd.DataFrame]],
    zip_path: str,
    overwrite: bool = True,
):
    """Stores the dataframes as <name>.tsv within the given ZIP file."""
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    if os.path.isfile(zip_path):
        if overwrite:
            os.remove(zip_path)
        else:
            raise FileExistsError(
                f"File {zip_path} already exists and overwrite is set to False."
            )
    for facet, df in facet_df_pairs:
        df = ms3.no_collections_no_booleans(df)
        df.to_csv(
            zip_path,
            sep="\t",
            mode="a",
            compression=dict(method="zip", archive_name=f"{facet}.tsv"),
        )


def make_datapackage_descriptor(
    facet_df_pairs: Iterable[Tuple[str, pd.DataFrame]], package_name: str
) -> dict:
    package_descriptor = {"name": package_name, "resources": []}
    for facet, df in facet_df_pairs:
        schema = ms3.get_schema_or_url(facet, df.columns, df.index.names)
        resource_descriptor = ms3.assemble_resource_descriptor(
            name=f"{package_name}.{facet}",
            path=f"{package_name}.zip",
            schema=schema,
            innerpath=f"{facet}.tsv",
        )
        package_descriptor["resources"].append(resource_descriptor)
    return package_descriptor


def store_datapackage(
    facet_df_pairs: Iterable[Tuple[str, pd.DataFrame]],
    name: str,
    directory: str,
    overwrite: bool = True,
) -> str:
    zip_path = os.path.join(directory, f"{name}.zip")
    store_facets_as_zip(facet_df_pairs, zip_path, overwrite=overwrite)
    package_descriptor = make_datapackage_descriptor(facet_df_pairs, name)
    descriptor_path = os.path.join(directory, f"{name}.datapackage.json")
    with open(descriptor_path, "w", encoding="utf-8") as f:
        json.dump(package_descriptor, f)
    return descriptor_path


@dataclasses.dataclass
class LoadedFacets:
    events: Dict[tuple, pd.DataFrame] = dataclasses.field(default_factory=dict)
    control: Dict[tuple, pd.DataFrame] = dataclasses.field(default_factory=dict)
    structure: Dict[tuple, pd.DataFrame] = dataclasses.field(default_factory=dict)
    annotations: Dict[tuple, pd.DataFrame] = dataclasses.field(default_factory=dict)
    metadata: Dict[tuple, pd.Series] = dataclasses.field(default_factory=dict)


class Loader(PipelineStep):
    """Base class for all loaders."""

    class Schema(PipelineStep.Schema):
        source = mm.fields.Str(required=True)
        basepath = mm.fields.Str()
        package_name = mm.fields.Str(allow_none=True)

    def __init__(
        self,
        source: str,
        basepath: str,
        package_name: Optional[str] = None,
        autoload: bool = True,
    ):
        self._source = None
        self.source = source
        self._basepath = None
        self.basepath = basepath
        self._package_name = package_name
        self.autoload = autoload
        self.loaded_facets = LoadedFacets()
        self.descriptor_path = None
        self._processed_ids = set()

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
        self._package_name = check_name(package_name)

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
        if not os.path.isfile(resource):
            raise FileNotFoundError(f"Resource {resource} does not exist.")

    def create_datapackage(
        self,
        overwrite: bool = False,
        reload: bool = False,
    ):
        """

        Args:
            overwrite:
                If False (default), raise FileExistsError if zip file already exists.
                If True, overwrite existing zip file.
            reload:
                If False (default), skip resources whose ID's have already been processed.
                If True, reload all resources.

        Returns:

        Raises:
            FileExistsError: If the zip file <basepath>/<package_name>.zip already exists.

        """
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

    def get_package_name(self) -> str:
        if self.package_name is not None:
            return self.package_name
        new_name = os.path.basename(self.source)
        frictionless_name = make_valid_frictionless_name(new_name)
        return frictionless_name

    def get_zip_path(self) -> str:
        package_name = self.get_package_name()
        return os.path.join(self.basepath, package_name + ".zip")

    def _process_resource(self, resource: str) -> None:
        """Parse the resource and extract the facets."""
        raise NotImplementedError

    def process_resource(self, resource: str | Path) -> None:
        resource = resolve_path(resource)
        self.check_resource(resource)
        return self._process_resource(resource)

    def store_datapackage(self) -> str:
        package_name = self.get_package_name()
        facet2df = self._get_concatenated_facets()
        self.descriptor_path = store_datapackage(
            facet2df.items(), package_name, self.basepath
        )
        self.logger.info(f"Stored datapackage at {self.descriptor_path}.")


class MuseScoreLoader(Loader):
    """Wrapper around the ms3 MuseScore parsing library."""

    class Schema(Loader.Schema):
        pass

    def __init__(
        self,
        source: str,
        basepath: str,
        package_name: Optional[str] = None,
        autoload: bool = True,
        as_corpus: bool = False,
        only_metadata_fnames: bool = True,
        include_convertible: bool = False,
        include_tsv: bool = True,
        exclude_review: bool = True,
        file_re: Optional[str | re.Pattern] = None,
        folder_re: Optional[str | re.Pattern] = None,
        exclude_re: Optional[str | re.Pattern] = None,
        paths: Optional[Collection[str]] = None,
        labels_cfg={},
        ms=None,
        **logger_cfg,
    ):
        super().__init__(
            source=source,
            basepath=basepath,
            package_name=package_name,
            autoload=autoload,
        )
        self.parser: ms3.Parse | ms3.Corpus = None
        ms3_arguments = dict(
            directory=self.source,
            only_metadata_fnames=only_metadata_fnames,
            include_convertible=include_convertible,
            include_tsv=include_tsv,
            exclude_review=exclude_review,
            file_re=file_re,
            folder_re=folder_re,
            exclude_re=exclude_re,
            paths=paths,
            labels_cfg=labels_cfg,
            ms=ms,
            **logger_cfg,
        )
        if as_corpus:
            self.parser = ms3.Corpus(**ms3_arguments)
        else:
            self.parser = ms3.Parse(**ms3_arguments)
        if self.autoload:
            _ = self.create_datapackage()

    def create_datapackage(
        self,
        overwrite: bool = False,
        reload: bool = False,
        view_name: Optional[str] = None,
        parsed: bool = True,
        unparsed: bool = True,
        choose: Literal["auto", "ask"] = "auto",
    ) -> str:
        """

        Args:
            overwrite:
                If False (default), raise FileExistsError if zip file already exists.
                If True, overwrite existing zip file.
            reload:
                If False (default), skip resources whose ID's have already been processed.
                If True, reload all resources.
            view_name:
            parsed:
            unparsed:
            choose:

        Returns:

        Raises:
            FileExistsError: If the zip file <basepath>/<package_name>.zip already exists.

        """
        super().create_datapackage(overwrite=overwrite, reload=reload)
        if choose not in ("auto", "ask"):
            raise ValueError(
                f"Invalid value for choose: {choose}. Pass 'auto' (default) or 'ask'."
            )
        self._parse_and_extract(
            reload=reload,
            choose=choose,
            parsed=parsed,
            unparsed=unparsed,
            view_name=view_name,
        )
        self.store_datapackage()
        return self.descriptor_path

    def _parse_and_extract(
        self,
        reload: bool = False,
        choose: Literal["auto", "ask"] = "auto",
        parsed: bool = True,
        unparsed: bool = True,
        view_name: Optional[str] = None,
    ):
        score_files = self.parser.get_files(
            facets="scores",
            view_name=view_name,
            parsed=parsed,
            unparsed=unparsed,
            choose=choose,
            flat=True,
            include_empty=False,
        )  # Dict[str | Tuple[str, str], List[ms3.File]]; the lists are guaranteed to have length 1
        if isinstance(self.parser, ms3.Parse):
            logger_names = {
                (corpus, fname): self.parser[corpus].logger_names[fname]
                for corpus, fname in score_files.keys()
            }
            score_files = {ID: files[0] for ID, files in score_files.items()}
        else:  # ms3.Corpus
            if self.package_name is None:
                corpus_name = make_valid_frictionless_name(self.parser.name)
                self.package_name = corpus_name
            else:
                corpus_name = self.package_name
            logger_names = {
                (corpus_name, fname): self.parser.logger_names[fname]
                for fname in score_files.keys()
            }
            score_files = {
                (corpus_name, fname): files[0] for fname, files in score_files.items()
            }

        for ID, file in (
            pbar := tqdm(
                score_files.items(),
                total=len(score_files),
                desc="Parsing scores...",
            )
        ):
            if not reload and ID in self.processed_ids:
                self.logger.debug(f"Skipping piece {ID} (already processed)")
                continue
            path = file.full_path
            pbar.set_description(f"Parsing {file.file}")
            logger_name = logger_names[ID]
            self._process_resource(path, ID, logger_name)

    def _process_resource(self, path, ID, logger_name):
        score = ms3.Score(
            path,
            read_only=True,
            labels_cfg=self.parser.labels_cfg,
            ms=self.parser.ms,
            name=logger_name,
        )
        for facet_name, obj in zip(
            ("events", "control", "structure", "annotations", "metadata"),
            (
                score.mscx.notes_and_rests(),
                score.mscx.chords(),
                score.mscx.measures(),
                score.mscx.labels(),
                ms3.metadata2series(score.mscx.metadata),
            ),
        ):
            self.add_piece_facet(facet_name, ID, obj)
