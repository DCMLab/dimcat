import logging
import os
import re
from pathlib import Path
from typing import Collection, Literal, Optional

import ms3
from dimcat.data.resources.base import PathResource
from dimcat.dc_exceptions import NoMuseScoreExecutableSpecifiedError
from dimcat.utils import make_valid_frictionless_name, resolve_path

from .base import Loader, ScoreLoader

logger = logging.getLogger(__name__)


class MuseScoreLoader(ScoreLoader):
    """Wrapper around the ms3 MuseScore parsing library."""

    _accepted_file_extensions = (".mscx", ".mscz")
    _conditionally_accepted_file_extensions = (
        ".cap",
        ".capx",
        ".midi",
        ".mid",
        ".musicxml",
        ".mxl",
        ".xml",
    )
    """Convertible file formats accepted if a MuseScore executable is specified (parameter ``ms``)."""
    _default_loader_name = "musescore"

    class Schema(Loader.Schema):
        pass

    @classmethod
    def from_ms3(
        cls,
        directory: str,
        package_name: str = None,
        as_corpus: bool = False,
        only_metadata_pieces: bool = True,
        include_convertible: bool = False,
        include_tsv: bool = True,
        exclude_review: bool = True,
        file_re: Optional[str | re.Pattern] = None,
        folder_re: Optional[str | re.Pattern] = None,
        exclude_re: Optional[str | re.Pattern] = None,
        paths: Optional[Collection[str]] = None,
        choose: Literal["auto", "all", "ask"] = "auto",
        labels_cfg={},
        ms=None,
        logger_cfg: Optional[dict] = None,
        basepath: Optional[str] = None,
        loader_name: Optional[str] = None,
        overwrite: bool = False,
        auto_validate: bool = True,
    ):
        directory = resolve_path(directory)
        if not os.path.isdir(directory):
            raise ValueError(f"Invalid directory: {directory}")
        parser: ms3.Parse | ms3.Corpus = None  # for type hinting
        if logger_cfg is None:
            logger_cfg = {}
        ms3_arguments = dict(
            directory=directory,
            only_metadata_pieces=only_metadata_pieces,
            include_convertible=include_convertible,
            include_tsv=include_tsv,
            exclude_review=exclude_review,
            file_re=file_re,
            folder_re=folder_re,
            exclude_re=exclude_re,
            labels_cfg=labels_cfg,
            ms=ms,
            **logger_cfg,
        )
        if as_corpus:
            ms3_arguments["paths"] = paths
            parser = ms3.Corpus(**ms3_arguments)
        else:
            if paths is not None:
                raise NotImplementedError(
                    "Argument 'paths' currently is only supported for as_corpus=True."
                )
            parser = ms3.Parse(**ms3_arguments)
        score_files = parser.get_files(
            facets="scores",
            choose=choose,
            flat=True,
            include_empty=False,
        )  # Dict[str | Tuple[str, str], List[ms3.File]]; the lists are guaranteed to have length 1
        if package_name is None:
            folder = os.path.basename(directory)
            package_name = make_valid_frictionless_name(folder)
        else:
            package_name = make_valid_frictionless_name(package_name)
            cls.logger.info(f"Assigned the name '{package_name}' to the package.")
        filepaths, corpus_names, piece_names = [], [], []
        if isinstance(parser, ms3.Parse):
            for ID, files in score_files.items():
                corpus_name, piece_name = ID
                first_file = files[0]
                filepaths.append(first_file.full_path)
                corpus_names.append(corpus_name)
                piece_names.append(piece_name)
        else:  # ms3.Corpus
            corpus_name = package_name
            for fname, files in score_files.items():
                first_file = files[0]
                filepaths.append(first_file.full_path)
                corpus_names.append(corpus_name)
                piece_names.append(fname)
        return cls.from_filepaths(
            filepaths=filepaths,
            package_name=package_name,
            resource_names=piece_names,
            corpus_names=corpus_names,
            auto_validate=auto_validate,
            basepath=basepath,
            loader_name=loader_name,
            overwrite=overwrite,
        )

    def __init__(
        self,
        basepath: Optional[str] = None,
        loader_name: Optional[str] = None,
        overwrite: bool = False,
        ms: Optional[str] = None,
    ):
        """

        Args:
            basepath: Directory in which to store the loaded data as a datapackage.
            loader_name: Name of the datapackage containing the loaded data.
            overwrite:
                By default, the loader will not parse anything if the target package ``loader_name``
                already exists in ``basepath``. Set this to True to re-parse and overwrite.
            ms3:
                Path to a MuseScore executable to allow for loading all score formats that MuseScore can open.
                If None, only .mscx and .mscz files can be loaded.
        """
        super().__init__(
            basepath=basepath,
            loader_name=loader_name,
            overwrite=overwrite,
        )
        self.ms = ms

    def check_resource(self, resource: str | Path) -> None:
        super().check_resource(resource)
        filepath = resource.normpath
        _, fext = os.path.splitext(filepath)
        if fext in self._conditionally_accepted_file_extensions and self.ms is None:
            raise NoMuseScoreExecutableSpecifiedError

    # def make_and_store_datapackage(
    #     self,
    #     overwrite: Optional[bool] = None,
    #     view_name: Optional[str] = None,
    #     parsed: bool = True,
    #     unparsed: bool = True,
    #     choose: Literal["auto", "ask"] = "auto",
    # ) -> str:
    #     """
    #
    #     Args:
    #         overwrite:
    #             If False (default), raise FileExistsError if zip file already exists.
    #             If True, overwrite existing zip file.
    #         view_name:
    #         parsed:
    #         unparsed:
    #         choose:
    #
    #     Returns:
    #
    #     Raises:
    #         FileExistsError: If the zip file <basepath>/<package_name>.zip already exists.
    #
    #     """
    #     super().make_and_store_datapackage(overwrite=overwrite)
    #     if choose not in ("auto", "ask"):
    #         raise ValueError(
    #             f"Invalid value for choose: {choose}. Pass 'auto' (default) or 'ask'."
    #         )
    #     self._parse_and_extract(
    #         choose=choose,
    #         parsed=parsed,
    #         unparsed=unparsed,
    #         view_name=view_name,
    #     )
    #     self._store_datapackage()
    #     return self.descriptor_path

    def _process_resource(self, resource: PathResource) -> None:
        ID = resource.ID
        filepath = resource.normpath
        score = ms3.Score(
            filepath,
            read_only=True,
            ms=self.ms,
            name=self.logger.name,
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
            self.add_piece_facet_dataframe(facet_name, ID, obj)
