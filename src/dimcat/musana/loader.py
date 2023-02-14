from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, List, Literal

import ms3
import pandas as pd
from dimcat.dtypes import Harmonies, Measures, Notes, TabularData, TypedSequence
from dimcat.musana.harmony_types import Key, TonalHarmony
from dimcat.musana.util import determine_era_based_on_year

# _____________________________ AspectInfo ______________________________________


@dataclass(frozen=True)
class KeyInfo(TabularData):
    @cached_property
    def global_key(self) -> Key:
        key_str = self.df["globalkey"][0]
        if not isinstance(key_str, str):
            key = "NA"
        else:
            key = Key.parse(key_str=key_str)
        return key

    @cached_property
    def local_key(self) -> TypedSequence:
        local_key_series = self.df["localkey"]
        sequential_data = TypedSequence.from_series(series=local_key_series)
        return sequential_data


# _____________________________ LevelInfo ______________________________________


@dataclass
class PieceMetaData:
    corpus_path: str
    corpus_name: TypedSequence
    piece_name: TypedSequence
    composed_start: TypedSequence
    composed_end: TypedSequence
    composer: TypedSequence
    annotated_key: TypedSequence
    label_count: int | None
    piece_length: int

    @cached_property
    def era(self) -> str:
        era = determine_era_based_on_year(year=self.composed_end.to_series()[0])
        return era


@dataclass
class CorpusMetaData:
    corpus_name: TypedSequence
    composer: TypedSequence
    composed_start: TypedSequence
    composed_end: TypedSequence
    annotated_key: TypedSequence
    piecename_list: List[str]  # don't count pieces with label_count=0
    pieceinfo_list: List[PieceInfo]  # don't count pieces with label_count=0


@dataclass
class MetaCorporaMetaData:
    corpora_names: TypedSequence
    composer: TypedSequence
    composed_start: TypedSequence
    composed_end: TypedSequence
    annotated_key: TypedSequence
    corpusname_list: List[str]
    corpusinfo_list: List[CorpusInfo]


@dataclass
class PieceInfo:
    # containing the data for a single piece
    meta_info: PieceMetaData
    harmony_info: Harmonies
    measure_info: Measures
    note_info: Notes
    key_info: KeyInfo

    @classmethod
    def from_directory(cls, parent_corpus_path: str, piece_name: str) -> PieceInfo:
        corpus_name: str = os.path.dirname(parent_corpus_path)
        metadata_tsv_df: pd.DataFrame = ms3.load_tsv(
            os.path.join(parent_corpus_path, "metadata.tsv")
        )
        metadata_tsv_df = ms3.enforce_fname_index_for_metadata(metadata_tsv_df)
        try:
            piece_metadata = metadata_tsv_df.loc[piece_name]
        except KeyError:
            raise KeyError(
                f"Piece name {piece_name} not contained in metadata.tsv, only{metadata_tsv_df.index.to_list()}"
            )
        metadata_dict = piece_metadata.to_dict()

        try:
            tsv_name = piece_name + ".tsv"
            harmonies_df: pd.DataFrame = ms3.load_tsv(
                os.path.join(parent_corpus_path, "harmonies", tsv_name)
            )
            measure_df: pd.DataFrame = ms3.load_tsv(
                os.path.join(parent_corpus_path, "measures", tsv_name)
            )
            note_df: pd.DataFrame = ms3.load_tsv(
                os.path.join(parent_corpus_path, "notes", tsv_name)
            )
        except Exception:
            raise Warning(
                "piece does not have all the required .tsv files in this corpus"
            )

        harmony_info = Harmonies.from_df(df=harmonies_df)
        measure_info = Measures.from_df(df=measure_df)
        note_info = Notes.from_df(df=note_df)

        key_df: pd.DataFrame = harmony_info.df[["globalkey", "localkey"]]
        key_info = KeyInfo.from_df(df=key_df)

        piece_length = harmonies_df.shape[0]

        piece_name_SeqData: TypedSequence = TypedSequence([piece_name] * piece_length)
        corpus_name_SeqData: TypedSequence = TypedSequence([corpus_name] * piece_length)

        annotated_key: str = metadata_dict["annotated_key"]
        annotated_key_SeqData: TypedSequence = TypedSequence(
            [annotated_key] * piece_length
        )

        composed_start: int = metadata_dict["composed_start"]
        composed_start_SeqData: TypedSequence = TypedSequence(
            [composed_start] * piece_length
        )

        composed_end: int = metadata_dict["composed_end"]
        composed_end_SeqData: TypedSequence = TypedSequence(
            [composed_end] * piece_length
        )

        composer: TypedSequence = TypedSequence(
            [corpus_name.split("_")[0]] * piece_length
        )
        label_count = metadata_dict["label_count"]

        meta_info = PieceMetaData(
            corpus_path=parent_corpus_path,
            corpus_name=corpus_name_SeqData,
            piece_name=piece_name_SeqData,
            composed_start=composed_start_SeqData,
            composed_end=composed_end_SeqData,
            composer=composer,
            annotated_key=annotated_key_SeqData,
            label_count=label_count,
            piece_length=piece_length,
        )

        instance = cls(
            meta_info=meta_info,
            harmony_info=harmony_info,
            measure_info=measure_info,
            note_info=note_info,
            key_info=key_info,
        )
        return instance

    @cached_property
    def annotated(self) -> bool:
        label_count = self.meta_info.label_count
        if label_count > 0:
            return True
        else:
            return False

    @cached_property
    def get_tonal_harmony_sequential(self) -> TypedSequence:
        """Essentially get the "chord" column from the dataframe and transform each chord to a TonalHarmony object."""
        dropped_nan_df = self.harmony_info.df.dropna(
            how="any", subset=["chord", "globalkey", "localkey"]
        )
        dropped_nan_df = dropped_nan_df.reset_index(drop=True)

        def create_tonal_harmony(row):
            return TonalHarmony.parse(
                globalkey_str=row["globalkey"],
                localkey_str=row["localkey"],
                chord_str=row["chord"],
            )

        # Create a list of TonalHarmony objects using the create_tonal_harmony() function
        tonal_harmony_list = dropped_nan_df.apply(create_tonal_harmony, axis=1)

        # Create a Sequential object from the list of TonalHarmony objects
        tonal_harmony_sequential = TypedSequence(tonal_harmony_list)

        return tonal_harmony_sequential


@dataclass
class BaseCorpusInfo(ABC):
    @abstractmethod
    def filter_pieces_by_condition(
        self,
        condition: Callable[
            [
                PieceInfo,
            ],
            bool,
        ],
    ) -> List[PieceInfo]:
        pass


@dataclass
class CorpusInfo(BaseCorpusInfo):
    # containing data for a single corpus
    meta_info: CorpusMetaData
    harmony_info: Harmonies
    measure_info: Measures
    note_info: Notes
    key_info: KeyInfo

    def filter_pieces_by_condition(
        self,
        condition: Callable[
            [
                PieceInfo,
            ],
            bool,
        ],
    ) -> List[PieceInfo]:
        filtered_pieces = [
            pieceinfo
            for pieceinfo in self.meta_info.pieceinfo_list
            if condition(pieceinfo)
        ]
        return filtered_pieces

    @classmethod
    def from_directory(cls, corpus_path: str) -> CorpusInfo:
        """Assemble all required args for CorpusInfo class"""
        # corpus_name: str = corpus_path.split(os.sep)[-2]
        metadata_tsv_df: pd.DataFrame = ms3.load_tsv(
            os.path.join(corpus_path, "metadata.tsv")
        )
        metadata_tsv_df = ms3.enforce_fname_index_for_metadata(metadata_tsv_df)

        # don't count pieces with label_count=0, and annotated_key is empty
        metadata_tsv_annotated = metadata_tsv_df.loc[
            metadata_tsv_df["label_count"] != 0
        ]
        piecename_list = metadata_tsv_annotated.index.to_list()
        pieceinfo_list = [
            PieceInfo.from_directory(parent_corpus_path=corpus_path, piece_name=item)
            for item in piecename_list
        ]

        try:
            harmonies_df: pd.DataFrame = pd.concat(
                [item.harmony_info.df for item in pieceinfo_list]
            )
            measure_df: pd.DataFrame = pd.concat(
                [item.measure_info.df for item in pieceinfo_list]
            )
            note_df: pd.DataFrame = pd.concat(
                [item.note_info.df for item in pieceinfo_list]
            )
        except Exception:
            raise Warning(
                "piece does not have all the required .tsv files in this corpus"
            )

        harmony_info = Harmonies.from_df(df=harmonies_df)
        measure_info = Measures.from_df(df=measure_df)
        note_info = Notes.from_df(df=note_df)

        key_df: pd.DataFrame = harmony_info.df[["globalkey", "localkey"]]
        key_info = KeyInfo.from_df(df=key_df)

        concat_composed_start_series = sum(
            (item.meta_info.composed_start.values for item in pieceinfo_list), []
        )
        composed_start_SeqData = TypedSequence(concat_composed_start_series)

        concat_composed_end_series = sum(
            (item.meta_info.composed_end.values for item in pieceinfo_list), []
        )
        composed_end_SeqData = TypedSequence(concat_composed_end_series)

        concat_corpus_name = sum(
            (item.meta_info.corpus_name.values for item in pieceinfo_list), []
        )
        corpusname_SeqData = TypedSequence(concat_corpus_name)

        concat_composer = sum(
            (item.meta_info.composer.values for item in pieceinfo_list), []
        )
        composer_SeqData = TypedSequence(concat_composer)

        concat_annotated_key = sum(
            (item.meta_info.annotated_key.values for item in pieceinfo_list), []
        )
        annotated_key_SeqData = TypedSequence(concat_annotated_key)

        meta_info = CorpusMetaData(
            corpus_name=corpusname_SeqData,
            composer=composer_SeqData,
            composed_start=composed_start_SeqData,
            composed_end=composed_end_SeqData,
            piecename_list=piecename_list,
            pieceinfo_list=pieceinfo_list,
            annotated_key=annotated_key_SeqData,
        )

        instance = cls(
            meta_info=meta_info,
            harmony_info=harmony_info,
            measure_info=measure_info,
            note_info=note_info,
            key_info=key_info,
        )
        return instance


@dataclass
class MetaCorporaInfo(BaseCorpusInfo):
    # containing data for a collection corpora
    meta_info: MetaCorporaMetaData
    harmony_info: Harmonies
    measure_info: Measures
    note_info: Notes
    key_info: KeyInfo

    def filter_pieces_by_condition(
        self,
        condition: Callable[
            [
                PieceInfo,
            ],
            bool,
        ],
    ) -> List[PieceInfo]:
        filtered_pieces = sum(
            [
                corpusinfo.filter_pieces_by_condition(condition=condition)
                for corpusinfo in self.meta_info.corpusinfo_list
            ],
            [],
        )
        return filtered_pieces

    def get_annotated_pieces(self) -> List[PieceInfo]:
        def is_annotated(pieceinfo):
            return pieceinfo.annotated

        annotated_pieces = self.filter_pieces_by_condition(condition=is_annotated)
        return annotated_pieces

    @classmethod
    def from_directory(cls, metacorpora_path: str) -> MetaCorporaInfo:
        metacorpora_path = ms3.resolve_dir(metacorpora_path)
        corpusname_list = [
            folder
            for folder in ms3.first_level_subdirs(metacorpora_path)
            if not folder.startswith(".")
        ]
        corpusinfo_list = []
        for item in corpusname_list:
            corpus_path = os.path.join(metacorpora_path, item)
            try:
                corpusinfo_list.append(
                    CorpusInfo.from_directory(corpus_path=corpus_path)
                )
            except FileNotFoundError:
                continue

        try:
            harmonies_df: pd.DataFrame = pd.concat(
                [item.harmony_info.df for item in corpusinfo_list]
            )
            measure_df: pd.DataFrame = pd.concat(
                [item.measure_info.df for item in corpusinfo_list]
            )
            note_df: pd.DataFrame = pd.concat(
                [item.note_info.df for item in corpusinfo_list]
            )
        except Exception:
            raise Warning(
                "Corpus does not have all the required .tsv files in this corpus"
            )

        harmony_info = Harmonies.from_df(df=harmonies_df)
        measure_info = Measures.from_df(df=measure_df)
        note_info = Notes.from_df(df=note_df)

        key_df: pd.DataFrame = harmony_info.df[["globalkey", "localkey"]]
        key_info = KeyInfo.from_df(df=key_df)

        concat_composed_start_series = sum(
            (item.meta_info.composed_start.values for item in corpusinfo_list), []
        )
        composed_start_SeqData = TypedSequence(concat_composed_start_series)

        concat_composed_end_series = sum(
            (item.meta_info.composed_end.values for item in corpusinfo_list), []
        )
        composed_end_SeqData = TypedSequence(concat_composed_end_series)

        concat_corpus_name = sum(
            (item.meta_info.corpus_name.values for item in corpusinfo_list), []
        )
        corporaname_SeqData = TypedSequence(concat_corpus_name)

        concat_composer = sum(
            (item.meta_info.composer.values for item in corpusinfo_list), []
        )
        composer_SeqData = TypedSequence(concat_composer)

        concat_annotated_key = sum(
            (item.meta_info.annotated_key.values for item in corpusinfo_list), []
        )
        annotated_key_SeqData = TypedSequence(concat_annotated_key)

        meta_info = MetaCorporaMetaData(
            corpora_names=corporaname_SeqData,
            composer=composer_SeqData,
            composed_start=composed_start_SeqData,
            composed_end=composed_end_SeqData,
            annotated_key=annotated_key_SeqData,
            corpusname_list=corpusname_list,
            corpusinfo_list=corpusinfo_list,
        )

        instance = cls(
            meta_info=meta_info,
            harmony_info=harmony_info,
            measure_info=measure_info,
            note_info=note_info,
            key_info=key_info,
        )
        return instance


if __name__ == "__main__":
    metacorpora = MetaCorporaInfo.from_directory(
        metacorpora_path="~/romantic_piano_corpus/"
    )

    def piece_operation(pieceinfo: PieceInfo):
        return pieceinfo.harmony_info.get_chord_bigrams()

    # define piece grouping
    eras = ["Renaissance", "Baroque", "Classical", "Romantic"]

    def era_condition(era: Literal["Renaissance", "Baroque", "Classical", "Romantic"]):
        return lambda pieceinfo: pieceinfo.meta_info.era == era

    # automation
    transition_dict = {
        era: [
            piece_operation(piece)
            for piece in metacorpora.filter_pieces_by_condition(era_condition(era))
        ]
        for era in eras
    }

    print(transition_dict)

    piece = PieceInfo.from_directory(
        parent_corpus_path="~/corelli//", piece_name="op01n01a"
    )

    result = piece.harmony_info.df[["globalkey", "localkey", "chord"]].to_numpy()
    print(result)
