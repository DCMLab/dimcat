from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, List

import ms3
import numpy as np
import pandas as pd
from dimcat.dtypes import TypedSequence
from dimcat.musana.harmony_types import Key, TonalHarmony
from dimcat.musana.util import determine_era_based_on_year
from scipy.stats import entropy


@dataclass(frozen=True)
class TabularData(ABC):
    _df: pd.DataFrame

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        instance = cls(_df=df)
        return instance

    def get_aspect(self, key: str) -> SequentialData:
        series: pd.Series = self._df[key]
        sequential_data = SequentialData.from_series(series)
        return sequential_data


@dataclass(frozen=True)
class SequentialData(ABC):
    _series: pd.Series

    @classmethod
    def from_series(cls, series: pd.Series):
        instance = cls(_series=series)
        return instance

    def unique_labels(self) -> SequentialData:
        unique_labels = self._series.unique()
        series = pd.Series(unique_labels)
        sequential_data = SequentialData.from_series(series)
        return sequential_data

    def get_changes(self) -> SequentialData:
        """Transform label seq [A, A, A, B, C, C, A, C, C, C] --->  [A, B, C, A, C]"""
        prev = object()
        occurrence_list = [prev := v for v in self._series if prev != v]  # noqa: F841
        series = pd.Series(occurrence_list)
        sequential_data = SequentialData.from_series(series)
        return sequential_data

    def len(self):
        return self._series.shape[0]

    def count(self):
        """Count the occurrences of objects in the sequence"""
        value_count = pd.value_counts(self._series)
        return value_count

    def mean(self) -> float:
        return self._series.mean()

    def probability(self) -> pd.Series:
        count = self.count()
        prob = count / count.sum()
        return prob

    def entropy(self) -> float:
        """
        The Shannon entropy (information entropy), the expected/average surprisal based on its probability distribution.
        """
        # mean_entropy = self.event_entropy().mean()
        p = self.probability()
        distr_entropy = entropy(p, base=2)
        return distr_entropy

    def surprisal(self) -> pd.Series:
        """The self entropy, information content, surprisal"""
        probs = self.probability()
        self_entropy = -np.log(probs)
        series = pd.Series(data=self_entropy, name="surprisal")
        return series


# _____________________________ AspectInfo ______________________________________


@dataclass(frozen=True)
class HarmonyInfo(TabularData):
    def modulation_bigrams_list(self) -> List[str]:
        """Returns a list of str representing the modulation bigram. e.g., "f#_IV/V_bIII/V" """
        globalkey = self._df["globalkey"][0]
        localkey_list = self.get_aspect(key="localkey").get_changes()
        mod_bigrams = localkey_list.n_grams(n=2)
        mod_bigrams = ["_".join([item[0], item[1]]) for item in mod_bigrams]
        bigrams = [globalkey + "_" + item for item in mod_bigrams]
        return bigrams


@dataclass(frozen=True)
class MeasureInfo(TabularData):
    pass


@dataclass(frozen=True)
class NoteInfo(TabularData):
    @cached_property
    def tpc(self) -> SequentialData:
        series = self._df["tpc"]
        sequential = SequentialData.from_series(series=series)
        return sequential


@dataclass(frozen=True)
class KeyInfo(TabularData):
    @cached_property
    def global_key(self) -> Key:
        key_str = self._df["globalkey"][0]
        if not isinstance(key_str, str):
            key = "NA"
        else:
            key = Key.parse(key_str=key_str)
        return key

    @cached_property
    def local_key(self) -> SequentialData:
        local_key_series = self._df["localkey"]
        sequential_data = SequentialData.from_series(series=local_key_series)
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
        era = determine_era_based_on_year(year=self.composed_end._series[0])
        return era


@dataclass
class CorpusMetaData:
    corpus_name: SequentialData
    composer: SequentialData
    composed_start: SequentialData
    composed_end: SequentialData
    annotated_key: SequentialData
    piecename_list: List[str]  # don't count pieces with label_count=0
    pieceinfo_list: List[PieceInfo]  # don't count pieces with label_count=0


@dataclass
class MetaCorporaMetaData:
    corpora_names: SequentialData
    composer: SequentialData
    composed_start: SequentialData
    composed_end: SequentialData
    annotated_key: SequentialData
    corpusname_list: List[str]
    corpusinfo_list: List[CorpusInfo]


@dataclass
class PieceInfo:
    # containing the data for a single piece
    meta_info: PieceMetaData
    harmony_info: HarmonyInfo
    measure_info: MeasureInfo
    note_info: NoteInfo
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

        harmony_info = HarmonyInfo.from_df(df=harmonies_df)
        measure_info = MeasureInfo.from_df(df=measure_df)
        note_info = NoteInfo.from_df(df=note_df)

        key_df: pd.DataFrame = harmony_info._df[["globalkey", "localkey"]]
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
        dropped_nan_df = self.harmony_info._df.dropna(
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
    harmony_info: HarmonyInfo
    measure_info: MeasureInfo
    note_info: NoteInfo
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
        metadata_tsv_df: pd.DataFrame = pd.read_csv(
            corpus_path + "metadata.tsv", sep="\t"
        )

        # don't count pieces with label_count=0, and annotated_key is empty
        piecename_list = metadata_tsv_df.loc[metadata_tsv_df["label_count"] != 0][
            "fnames"
        ]
        pieceinfo_list = [
            PieceInfo.from_directory(parent_corpus_path=corpus_path, piece_name=item)
            for item in piecename_list
        ]

        try:
            harmonies_df: pd.DataFrame = pd.concat(
                [item.harmony_info._df for item in pieceinfo_list]
            )
            measure_df: pd.DataFrame = pd.concat(
                [item.measure_info._df for item in pieceinfo_list]
            )
            note_df: pd.DataFrame = pd.concat(
                [item.note_info._df for item in pieceinfo_list]
            )
        except Exception:
            raise Warning(
                "piece does not have all the required .tsv files in this corpus"
            )

        harmony_info = HarmonyInfo.from_df(df=harmonies_df)
        measure_info = MeasureInfo.from_df(df=measure_df)
        note_info = NoteInfo.from_df(df=note_df)

        key_df: pd.DataFrame = harmony_info._df[["globalkey", "localkey"]]
        key_info = KeyInfo.from_df(df=key_df)

        concat_composed_start_series = pd.concat(
            [item.meta_info.composed_start._series for item in pieceinfo_list]
        )
        composed_start_SeqData = SequentialData.from_series(
            series=concat_composed_start_series
        )

        concat_composed_end_series = pd.concat(
            [item.meta_info.composed_end._series for item in pieceinfo_list]
        )
        composed_end_SeqData = SequentialData.from_series(
            series=concat_composed_end_series
        )

        corpusname_SeqData = SequentialData.from_series(
            series=pd.concat(
                [item.meta_info.corpus_name._series for item in pieceinfo_list]
            )
        )
        composer_SeqData = SequentialData.from_series(
            series=pd.concat(
                [item.meta_info.composer._series for item in pieceinfo_list]
            )
        )

        annotated_key_SeqData = SequentialData.from_series(
            series=pd.concat(
                [item.meta_info.annotated_key._series for item in pieceinfo_list]
            )
        )

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
    harmony_info: HarmonyInfo
    measure_info: MeasureInfo
    note_info: NoteInfo
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
        corpusname_list = sorted(
            [
                f
                for f in os.listdir(metacorpora_path)
                if not f.startswith(".")
                if not f.startswith("__")
            ]
        )

        corpusinfo_list = [
            CorpusInfo.from_directory(corpus_path=metacorpora_path + item + "/")
            for item in corpusname_list
        ]

        try:
            harmonies_df: pd.DataFrame = pd.concat(
                [item.harmony_info._df for item in corpusinfo_list]
            )
            measure_df: pd.DataFrame = pd.concat(
                [item.measure_info._df for item in corpusinfo_list]
            )
            note_df: pd.DataFrame = pd.concat(
                [item.note_info._df for item in corpusinfo_list]
            )
        except Exception:
            raise Warning(
                "Corpus does not have all the required .tsv files in this corpus"
            )

        harmony_info = HarmonyInfo.from_df(df=harmonies_df)
        measure_info = MeasureInfo.from_df(df=measure_df)
        note_info = NoteInfo.from_df(df=note_df)

        key_df: pd.DataFrame = harmony_info._df[["globalkey", "localkey"]]
        key_info = KeyInfo.from_df(df=key_df)

        concat_composed_start_series = pd.concat(
            [item.meta_info.composed_start._series for item in corpusinfo_list]
        )
        composed_start_SeqData = SequentialData.from_series(
            series=concat_composed_start_series
        )

        concat_composed_end_series = pd.concat(
            [item.meta_info.composed_end._series for item in corpusinfo_list]
        )
        composed_end_SeqData = SequentialData.from_series(
            series=concat_composed_end_series
        )

        corporaname_SeqData = SequentialData.from_series(
            series=pd.concat(
                [item.meta_info.corpus_name._series for item in corpusinfo_list]
            )
        )
        composer_SeqData = SequentialData.from_series(
            series=pd.concat(
                [item.meta_info.composer._series for item in corpusinfo_list]
            )
        )

        annotated_key_SeqData = SequentialData.from_series(
            series=pd.concat(
                [item.meta_info.annotated_key._series for item in corpusinfo_list]
            )
        )

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
    # metacorpora = MetaCorporaInfo.from_directory(metacorpora_path='../romantic_piano_corpus/')
    # piece_operation = lambda pieceinfo: pieceinfo.harmony_info.chord_ngrams(2)
    #
    # # define piece grouping
    # eras = ['Renaissance', 'Baroque', 'Classical', 'Romantic']
    # era_condition = lambda era: (lambda pieceinfo: pieceinfo.meta_info.era() == era)
    #
    # # automation
    # transition_dict = {era: [piece_operation(piece) for piece in
    #                          metacorpora.filter_pieces_by_condition(era_condition(era))] for era in eras}
    #
    # print(transition_dict)

    piece = PieceInfo.from_directory(
        parent_corpus_path="~/corelli//", piece_name="op01n01a"
    )

    result = piece.harmony_info._df[["globalkey", "localkey", "chord"]].to_numpy()
