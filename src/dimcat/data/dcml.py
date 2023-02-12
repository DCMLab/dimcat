from collections import defaultdict
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Tuple, Union

import ms3
import pandas as pd
from dimcat.data.base import AnalyzedData, Dataset, GroupedData, SlicedData, logger
from dimcat.dtypes import ID, GroupID, PieceID, SliceID
from dimcat.utils import clean_index_levels
from ms3._typing import ScoreFacet


class SlicedDataset(SlicedData, Dataset):
    @lru_cache()
    def get_item(self, ID: SliceID, what: ScoreFacet) -> Optional[pd.DataFrame]:
        """Retrieve a DataFrame pertaining to the facet ``what`` of the piece ``index``. If
        the facet has been sliced before, the sliced DataFrame is returned.

        Args:
            ID: (corpus, fname) or (corpus, fname, interval)
            what: What facet to retrieve.

        Returns:
            DataFrame representing an entire score facet, or a chunk (slice) of it.
        """
        match ID:
            case (_, _, _):
                return self.get_slice(ID, what)
            case (corpus, piece):
                return super().get_item(ID=PieceID(corpus, piece), what=what)

    def iter_facet(self, what: ScoreFacet) -> Iterator[Tuple[SliceID, pd.DataFrame]]:
        """Iterate through facet DataFrames.

        Args:
            what: Which type of facet to retrieve.

        Yields:
            Index tuple.
            Facet DataFrame.
        """
        if not self.slice_facet_if_necessary(what):
            logger.info(f"No sliced {what} available.")
            raise StopIteration
        yield from super().iter_facet(what=what)

    def slice_facet_if_necessary(self, what):
        """

        Parameters
        ----------
        what : :obj:`str`
            Facet for which to create slices if necessary

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
            facet_df = self.get_item(id, what)
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
