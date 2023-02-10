import copy
from collections import defaultdict
from functools import lru_cache
from typing import Optional, Union, Dict, List, Any, Iterator, Tuple

import ms3
import pandas as pd
from ms3._typing import ScoreFacet

from dimcat.data.base import _Dataset, SlicedData, PROCESSED_DATA_FIELDS, logger, GroupedData, AnalyzedData
from dimcat.utils.functions import typestrings2types
from dimcat.utils import clean_index_levels
from dimcat._typing import ID, PieceID, SliceID, GroupID


class Dataset(_Dataset):
    """An object that represents one or several corpora issued by the DCML corpus initiative.
    Essentially a wrapper for a ms3.Parse object."""

    def __init__(self, data: Optional[Union["Dataset", ms3.Parse]] = None, **kwargs):
        """

        Parameters
        ----------
        data : Data
            Convert a given Data object into a Corpus if possible.
        kwargs
            All keyword arguments are passed to load().
        """
        super().__init__()
        self.pieces: Dict[ID, ms3.Piece] = {}
        """{(corpus, fname) -> :obj:`ms3.Piece`}
        References to the individual :obj:`ms3.Piece` objects contained in this Dataset. They give access to all data
        and metadata available for one piece.
        """
        self.data = data
        if len(kwargs) > 0:
            self.load(**kwargs)

    @property
    def data(self) -> ms3.Parse:
        """Get the data field in its raw form."""
        return self._data

    @data.setter
    def data(self, data_object: Optional[Union["Dataset", ms3.Parse]]) -> None:
        """Check if the assigned object is suitable for conversion."""
        if data_object is None:
            self._data = ms3.Parse()
            return
        is_dataset_object = isinstance(data_object, Dataset)
        is_parse_object = isinstance(data_object, ms3.Parse)
        if not (is_dataset_object or is_parse_object):
            raise TypeError(
                f"{data_object.__class__} could not be converted to a DCML dataset."
            )
        if is_parse_object:
            self._data = data_object
            return
        # else: got a Dataset object and need to copy its fields
        self._data = data_object._data
        self.pieces = dict(data_object.pieces)
        # ^^^ doesn't copy the ms3.Parse and ms3.Piece objects, only the references ^^^
        # vvv all other fields are deepcopied vvv
        if isinstance(data_object, SlicedData) and not isinstance(self, SlicedData):
            self.indices = sorted(set(ID[:2] for ID in data_object.indices))
        elif self.__class__.__name__ == "Dataset":
            self.indices = []
            self.get_indices()
        else:
            self.indices = list(data_object.indices)
        self.index_levels = copy.deepcopy(data_object.index_levels)
        self.pipeline_steps = list(data_object.pipeline_steps)
        self.group2pandas = data_object.group2pandas
        if self.__class__.__name__ == "Dataset":
            return
        processed_types = typestrings2types(PROCESSED_DATA_FIELDS.keys())
        typestring2dtype = dict(zip(PROCESSED_DATA_FIELDS.keys(), processed_types))
        for typestring, optional_fields in PROCESSED_DATA_FIELDS.items():
            dtype = typestring2dtype[typestring]
            if not (isinstance(data_object, dtype) and isinstance(self, dtype)):
                continue
            for field in optional_fields:
                setattr(self, field, copy.deepcopy(getattr(data_object, field)))

    def copy(self):
        return self.__class__(data=self)

    def get_facet(self, what: ScoreFacet, unfold: bool = False) -> pd.DataFrame:
        """Uses _.iter_facet() to collect and concatenate all DataFrames for a particular facet.

        Parameters
        ----------
        what : {'form_labels', 'events', 'expanded', 'notes_and_rests', 'notes', 'labels',
                'cadences', 'chords', 'measures', 'rests'}
            What facet to retrieve.
        unfold : :obj:`bool`, optional
            Pass True if you need repeats to be unfolded.

        Returns
        -------
        :obj:`pandas.DataFrame`
        """
        dfs = {
            idx: df
            for idx, df in self.iter_facet(
                what=what,
                unfold=unfold,
            )
        }
        if len(dfs) == 1:
            return list(dfs.values())[0]
        concatenated_groups = pd.concat(
            dfs.values(), keys=dfs.keys(), names=self.index_levels["indices"]
        )
        return clean_index_levels(concatenated_groups)

    def convert_group2pandas(self, result_dict) -> Union[pd.Series, pd.DataFrame]:
        """Converts the {ID -> processing_result} dict using the method specified in _.group2pandas."""
        converters = {
            "group_of_values2series": self.group_of_values2series,
            "group_of_series2series": self.group_of_series2series,
            "group2dataframe": self.group2dataframe,
            "group2dataframe_unstacked": self.group2dataframe_unstacked,
        }
        converter = converters[self.group2pandas]
        pandas_obj = converter(result_dict)
        return clean_index_levels(pandas_obj)

    def load(
        self,
        directory: Optional[Union[str, List[str]]] = None,
        parse_tsv: bool = True,
        parse_scores: bool = False,
        ms: Optional[str] = None,
    ):
        """
        Load and parse all of the desired raw data and metadata.

        Parameters
        ----------
        directory : str
            The path to all the data to load.
        parse_tsv : :obj:`bool`, optional
            By default, all detected TSV files are parsed and their type is inferred.
            Pass False to prevent parsing TSV files.
        parse_scores : :obj:`bool`, optional
            By default, detected scores files are not parsed to save time.
            Call super().__init__(parse_scores=True) if an analyzer needs
            to access parsed MuseScore XML.
        ms : :obj:`str`, optional
            If you pass the path to your local MuseScore 3 installation, ms3 will attempt to parse
            musicXML, MuseScore 2, and other formats by temporarily converting them. If you're
            using the standard path, you may try 'auto', or 'win' for Windows, 'mac' for MacOS,
            or 'mscore' for Linux. In case you do not pass the 'file_re' and the MuseScore
            executable is detected, all convertible files are automatically selected, otherwise
            only those that can be parsed without conversion.
        """
        if ms is not None:
            self.data.ms = ms
        if directory is not None:
            if isinstance(directory, str):
                directory = [directory]
            for d in directory:
                self.data.add_dir(
                    directory=d,
                )
        if parse_tsv:
            self.data.parse_tsv()
        if parse_scores:
            self.data.parse_scores()
        if self.data.n_parsed_tsvs == 0 and self.data.n_parsed_scores == 0:
            logger.info("No files have been parsed for analysis.")
        else:
            self.get_indices()

    def get_indices(self):
        """Fills self.pieces with metadata and IDs for all loaded data. This resets previously
        applied groupings."""
        self.pieces = {}
        # self.group_labels = {}
        for corpus_name, ms3_corpus in self.data.iter_corpora():
            for fname, piece in ms3_corpus.iter_pieces():
                ID = (corpus_name, fname)
                self.pieces[ID] = piece
        self.set_indices(list(self.pieces.keys()))

    def set_indices(
        self, new_indices: Union[List[PieceID], Dict[Any, List[PieceID]]]
    ) -> None:
        """Replace :attr:`indices` with a new list of IDs.

        Args:
            new_indices:
                The new list IDs or a dictionary of several lists of IDs. The latter is useful for re-grouping
                freshly sliced IDs of a :class:`GroupedDataset`.
        """
        if isinstance(new_indices, dict):
            new_indices = sum(new_indices.values(), [])
        self.indices = new_indices

    def iter_facet(
        self, what: ScoreFacet, unfold: bool = False
    ) -> Iterator[Tuple[ID, pd.DataFrame]]:
        """Iterate through facet DataFrames.

        Args:
            what: Which type of facet to retrieve.
            unfold: Pass True if you need repeats to be unfolded.

        Yields:
            Index tuple.
            Facet DataFrame.
        """
        for index in self.indices:
            df = self.get_item(index, what=what, unfold=unfold)
            if df is None or len(df.index) == 0:
                logger.info(f"{index} has no {what}.")
                continue
            yield index, df

    @lru_cache()
    def get_item(
        self, index: PieceID, what: ScoreFacet, unfold: bool = False
    ) -> Optional[pd.DataFrame]:
        """Retrieve a DataFrame pertaining to the facet ``what`` of the piece ``index``.

        Args:
            index: (corpus, fname) or (corpus, fname, interval)
            what: What facet to retrieve.
            unfold: Pass True if you need repeats to be unfolded.

        Returns:
            DataFrame representing an entire score facet, or a chunk (slice) of it.
        """
        corpus, fname = index
        file, df = self.data[corpus][fname].get_facet(
            what, unfold=unfold, interval_index=True
        )
        logger.debug(f"Retrieved {what} from {file}.")
        if df is not None and not isinstance(df.index, pd.IntervalIndex):
            logger.info(f"'{what}' of {index} does not come with an IntervalIndex")
            df = None
        if df is None:
            return
        assert (
            df.index.nlevels == 1
        ), f"Retrieved DataFrame has {df.index.nlevels}, not 1"
        return df


class SlicedDataset(SlicedData, Dataset):
    @lru_cache()
    def get_item(
        self, index: SliceID, what: ScoreFacet, unfold: bool = False
    ) -> Optional[pd.DataFrame]:
        """Retrieve a DataFrame pertaining to the facet ``what`` of the piece ``index``. If
        the facet has been sliced before, the sliced DataFrame is returned.

        Args:
            index: (corpus, fname) or (corpus, fname, interval)
            what: What facet to retrieve.
            unfold: Pass True if you need repeats to be unfolded.

        Returns:
            DataFrame representing an entire score facet, or a chunk (slice) of it.
        """
        match index:
            case (_, _, _):
                return self.get_slice(index, what)
            case (corpus, piece):
                return super().get_item(index=(corpus, piece), what=what, unfold=unfold)

    def iter_facet(
        self, what: ScoreFacet, unfold: bool = False
    ) -> Iterator[Tuple[SliceID, pd.DataFrame]]:
        """Iterate through facet DataFrames.

        Args:
            what: Which type of facet to retrieve.
            unfold: Pass True if you need repeats to be unfolded.

        Yields:
            Index tuple.
            Facet DataFrame.
        """
        if not self.slice_facet_if_necessary(what, unfold):
            logger.info(f"No sliced {what} available.")
            raise StopIteration
        yield from super().iter_facet(what=what, unfold=unfold)

    def slice_facet_if_necessary(self, what, unfold):
        """

        Parameters
        ----------
        what : :obj:`str`
            Facet for which to create slices if necessary
        unfold : :obj:`bool`
            Whether repeats should be unfolded.

        Returns
        -------
        :obj:`bool`
            True if slices are available or not needed, False otherwise.
        """
        if not hasattr(self, "slice_info"):
            # no slicer applied
            return True
        if len(self.slice_info) == 0:
            # applying slicer did not yield any slices
            return True
        if what in self.sliced:
            # already sliced
            return True
        self.sliced[what] = {}
        facet_ids = defaultdict(list)
        for corpus, fname, interval in self.slice_info.keys():
            facet_ids[(corpus, fname)].append(interval)
        for id, intervals in facet_ids.items():
            facet_df = self.get_item(id, what, unfold)
            if facet_df is None or len(facet_df.index) == 0:
                continue
            sliced = ms3.overlapping_chunk_per_interval(facet_df, intervals)
            self.sliced[what].update(
                {id + (iv,): chunk for iv, chunk in sliced.items()}
            )
        if len(self.sliced[what]) == 0:
            del self.sliced[what]
            return False
        return True


class GroupedDataset(GroupedData, Dataset):
    def iter_grouped_facet(
        self,
        what: ScoreFacet,
    ) -> Iterator[Tuple[GroupID, pd.DataFrame]]:
        """Iterate through one concatenated facet DataFrame per group.

        Args:
            what: Which type of facet to retrieve.

        Yields:
            Group index.
            Facet DataFrame.
        """
        for group, index_group in self.iter_grouped_indices():
            result = {}
            missing_id = []
            for index in index_group:
                df = self.get_item(index, what=what)
                if df is None or len(df.index) == 0:
                    missing_id.append(index)
                    continue
                result[index] = df
            n_results = len(result)
            if len(missing_id) > 0:
                if n_results == 0:
                    pass
                    # logger.info(f"No '{what}' available for {group}.")
                else:
                    logger.info(
                        f"Group {group} is missing '{what}' for the following indices:\n{missing_id}"
                    )
            if n_results == 0:
                continue
            if n_results == 1:
                # workaround necessary because of nasty "cannot handle overlapping indices;
                # use IntervalIndex.get_indexer_non_unique" error
                result["empty"] = pd.DataFrame()
            result = pd.concat(
                result.values(),
                keys=result.keys(),
                names=self.index_levels["indices"] + ["interval"],
            )
            yield group, result

    def set_indices(
        self, new_indices: Union[List[PieceID], Dict[ID, List[PieceID]]]
    ) -> None:
        """Replace :attr:`indices` with a new list of IDs and update the :attr:`grouped_indices` accordingly.

        Args:
            new_indices:
                The new list of IDs or an {old_id -> [new_id]} dictionary to replace the IDs with a list of new IDs.
        """
        id2group = defaultdict(lambda: ())
        if len(self.indices) > 0:
            id2group.update(
                {
                    ID: group
                    for group, group_ids in self.iter_grouped_indices()
                    for ID in group_ids
                }
            )
        new_grouped_indices = defaultdict(list)
        if isinstance(new_indices, dict):
            for old_id, new_ids in new_indices.items():
                old_group = id2group[old_id]
                new_grouped_indices[old_group].extend(new_ids)
        else:
            for new_id in new_indices:
                old_group = id2group[new_id]
                new_grouped_indices[old_group].append(new_id)
        self.grouped_indices = {
            k: new_grouped_indices[k] for k in sorted(new_grouped_indices.keys())
        }
        new_indices = sum(new_grouped_indices.values(), [])
        self.indices = sorted(new_indices)

    def set_grouped_indices(self, grouped_indices: Dict[GroupID, List[ID]]):
        self.grouped_indices = grouped_indices


class AnalyzedDataset(AnalyzedData, Dataset):
    pass


class GroupedSlicedDataset(GroupedDataset, SlicedDataset):
    assert_types = ["SlicedDataset", "GroupedDataset"]
    excluded_types = ["AnalyzedData"]
    type_mapping = {
        (
            "SlicedDataset",
            "GroupedDataset",
            "GroupedSlicedDataset",
        ): "GroupedSlicedDataset",
    }

    def get_slice_info(self, ignore_groups=False) -> pd.DataFrame:
        """Concatenates slice_info Series and returns them as a DataFrame."""
        group_dfs = {}
        for group, index_group in self.iter_grouped_indices():
            group_info = {ix: self.slice_info[ix] for ix in index_group}
            group_dfs[group] = pd.concat(
                group_info.values(), keys=group_info.keys(), axis=1
            ).T
        concatenated_info = pd.concat(group_dfs.values(), keys=group_dfs.keys())
        concatenated_info.index = self._rename_multiindex_levels(
            concatenated_info.index,
            self.index_levels["groups"] + self.index_levels["indices"],
        )
        return clean_index_levels(concatenated_info)

    # def iter_facet(self, what, unfold=False, concatenate=False, ignore_groups=False):
    #     """Iterate through groups of potentially sliced facet DataFrames.
    #
    #     Parameters
    #     ----------
    #     what : {'form_labels', 'events', 'expanded', 'notes_and_rests', 'notes', 'labels',
    #             'cadences', 'chords', 'measures', 'rests'}
    #         What facet to retrieve.
    #     unfold : :obj:`bool`, optional
    #         Pass True if you need repeats to be unfolded.
    #     concatenate : :obj:`bool`, optional
    #         By default, the returned dict contains one DataFrame per ID in the group.
    #         Pass True to instead concatenate the DataFrames. Then, the dict will contain only
    #         one entry where the key is a tuple containing all IDs and the value is a DataFrame,
    #         the components of which can be distinguished using its MultiIndex.
    #     ignore_groups : :obj:`bool`, False
    #         If set to True, the iteration loop is flattened and yields (index, facet_df) pairs directly. Clashes
    #         with the setting concatenate=True which concatenates facets per group.
    #
    #     Yields
    #     ------
    #     :obj:`tuple`
    #         Group identifier
    #     :obj:`dict` or :obj:`pandas.DataFrame`
    #         Default: {ID -> DataFrame}.
    #         If concatenate=True: DataFrame with MultiIndex identifying ID, and (eventual) interval.
    #     """
    #     if not self.slice_facet_if_necessary(what, unfold):
    #         logger.info(f"No sliced {what} available.")
    #         raise StopIteration
    #     if sum((concatenate, ignore_groups)) > 1:
    #         raise ValueError(
    #             "Arguments 'concatenate' and 'ignore_groups' are in conflict, choose one "
    #             "or use the method get_facet()."
    #         )
    #     for group, index_group in self.iter_grouped_indices():
    #         result = {}
    #         missing_id = []
    #         for index in index_group:
    #             df = self.get_item(index, what=what, unfold=unfold)
    #             if df is None:
    #                 continue
    #             elif ignore_groups:
    #                 yield index, df
    #             if len(df.index) == 0:
    #                 missing_id.append(index)
    #             result[index] = df
    #         if ignore_groups:
    #             continue
    #         n_results = len(result)
    #         if len(missing_id) > 0:
    #             if n_results == 0:
    #                 pass
    #                 # logger.info(f"No '{what}' available for {group}.")
    #             else:
    #                 logger.info(
    #                     f"Group {group} is missing '{what}' for the following indices:\n"
    #                     f"{missing_id}"
    #                 )
    #         if n_results == 0:
    #             continue
    #         if concatenate:
    #             if n_results == 1:
    #                 # workaround necessary because of nasty "cannot handle overlapping indices;
    #                 # use IntervalIndex.get_indexer_non_unique" error
    #                 result["empty"] = pd.DataFrame()
    #             result = pd.concat(
    #                 result.values(),
    #                 keys=result.keys(),
    #                 names=self.index_levels["indices"] + ["interval"],
    #             )
    #             result = {tuple(index_group): result}
    #
    #         yield group, result


class AnalyzedGroupedDataset(AnalyzedDataset, GroupedDataset):
    assert_types = ["GroupedDataset", "AnalyzedDataset"]
    type_mapping = {
        (
            "AnalyzedGroupedDataset",
            "AnalyzedDataset",
            "GroupedDataset",
        ): "AnalyzedGroupedDataset",
    }


class AnalyzedSlicedDataset(AnalyzedDataset, SlicedDataset):
    assert_types = ["SlicedDataset", "AnalyzedDataset"]
    excluded_types = []
    type_mapping = {
        (
            "AnalyzedSlicedDataset",
            "AnalyzedDataset",
            "SlicedDataset",
        ): "AnalyzedSlicedDataset",
    }
    pass


class AnalyzedGroupedSlicedDataset(AnalyzedSlicedDataset, GroupedSlicedDataset):
    assert_types = [
        "GroupedSlicedDataset",
        "AnalyzedGroupedDataset",
        "AnalyzedSlicedDataset",
    ]
    type_mapping = {
        (
            "GroupedSlicedDataset",
            "AnalyzedGroupedDataset",
            "AnalyzedSlicedDataset",
            "AnalyzedGroupedSlicedDataset",
        ): "AnalyzedGroupedSlicedDataset",
    }
    pass
