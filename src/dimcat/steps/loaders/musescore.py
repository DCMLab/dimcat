import os
import re
from pathlib import Path
from typing import ClassVar, Collection, Literal, Optional, Tuple

import ms3
from dimcat.exceptions import NoMuseScoreExecutableSpecifiedError
from dimcat.utils import make_valid_frictionless_name
from tqdm.asyncio import tqdm

from .base import Loader, ScoreLoader


class MuseScoreLoader(ScoreLoader):
    """Wrapper around the ms3 MuseScore parsing library."""

    accepted_file_extensions: ClassVar[Tuple[str, ...]] = (".mscx", ".mscz")
    conditionally_accepted_file_extensions: ClassVar[Tuple[str, ...]] = (".mscz",)
    """Convertible file formats accepted if a MuseScore executable is specified (parameter ``ms``)."""

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
            labels_cfg=labels_cfg,
            ms=ms,
            **logger_cfg,
        )
        if as_corpus:
            ms3_arguments[
                "paths"
            ] = paths  # ToDo: Treat 'paths' argument separately (only for ms3.Parse?)
            self.parser = ms3.Corpus(**ms3_arguments)
        else:
            if paths is not None:
                raise NotImplementedError(
                    "Argument 'paths' currently is only supported for as_corpus=True."
                )
            self.parser = ms3.Parse(**ms3_arguments)
        if self.autoload:
            _ = self.create_datapackage()

    def check_resource(self, resource: str | Path) -> None:
        super().check_resource(resource)
        _, fext = os.path.splitext(resource)
        if fext in self.conditionally_accepted_file_extensions:
            if self.parser.ms is None:
                raise NoMuseScoreExecutableSpecifiedError

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
        super().create_datapackage(overwrite=overwrite)
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

    def _process_resource(self, resource: str, ID: tuple, logger_name: str) -> None:
        score = ms3.Score(
            resource,
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
