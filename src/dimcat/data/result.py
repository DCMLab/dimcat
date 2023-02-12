import copy
from functools import reduce
from typing import Iterable, Optional, Tuple, Union

import pandas as pd
from dimcat.base import Data
from dimcat.data.base import AnalyzedData, Dataset, GroupedData, logger
from dimcat.dtypes import ID, GroupID
from ms3 import pretty_dict


class Result(Data):
    """Represents the result of an Analyzer processing a :class:`_Dataset`"""

    def __init__(
        self,
        analyzer: "Analyzer",  # noqa: F821
        dataset_before: Dataset,
        dataset_after: AnalyzedData,
    ):
        self.analyzer = analyzer
        self.dataset_before = dataset_before
        self.dataset_after = dataset_after
        self.config: dict = {}
        self.result_dict: dict = {}
        self._concatenated_results = None

    def _concat_results(
        self,
        index_result_dict: Optional[dict] = None,
        level_names: Optional[Union[Tuple[str], str]] = None,
    ) -> pd.DataFrame:
        if index_result_dict is None:
            index_result_dict = self.result_dict
            if level_names is None:
                level_names = self.dataset_after.index_levels["indices"]
        elif level_names is None:
            raise ValueError("Names of index level(s) need(s) to be specified.")
        df = pd.DataFrame.from_dict(index_result_dict, orient="index")
        try:
            df.index.rename(level_names, inplace=True)
        except TypeError:
            print(f"level_names = {level_names}; nlevels = {df.index.nlevels}")
            raise
        return df

    def get_results(self):
        return self._concat_results()

    def get_group_results(self):
        group_results = dict(self.iter_group_results())
        level_names = tuple(self.dataset_after.index_levels["groups"])
        if len(level_names) == 0:
            level_names = "group"
        return self._concat_results(group_results, level_names=level_names)

    def _aggregate_results_by_ids(self, indices: Iterable[ID]):
        group_results = [
            self.result_dict[idx] for idx in indices if idx in self.result_dict
        ]
        if len(group_results) == 0:
            return
        aggregated = reduce(self.analyzer.aggregate, group_results)
        return aggregated

    def _get_aggregated_result_for_group(self, idx: GroupID):
        indices = self.dataset_after.grouped_indices[idx]
        return self._aggregate_results_by_ids(indices)

    def items(self):
        yield from self.result_dict.items()

    def iter_results(self):
        yield from self.result_dict.values()

    def iter_group_results(self):
        if isinstance(self.dataset_after, GroupedData):
            for group, indices in self.dataset_after.iter_grouped_indices():
                aggregated = self._aggregate_results_by_ids(indices)
                if aggregated is None:
                    logger.warning(
                        f"{self.analyzer.__class__.__name__} yielded no result for group {group}"
                    )
                    continue
                yield group, aggregated
        else:
            aggregated = self._aggregate_results_by_ids(self.iter_results())
            yield self.dataset_before.__class__.__name__, aggregated

    def __copy__(self):
        new_obj = self.__class__(
            analyzer=self.analyzer,
            dataset_before=self.dataset_before,
            dataset_after=self.dataset_after,
        )
        for k, v in self.__dict__.items():
            if k not in ["analyzer", "dataset_before", "dataset_after"]:
                setattr(new_obj, k, copy.copy(v))
        return new_obj

    def __deepcopy__(self, memodict={}):
        new_obj = self.__class__(
            analyzer=self.analyzer,
            dataset_before=self.dataset_before,
            dataset_after=self.dataset_after,
        )
        for k, v in self.__dict__.items():
            if k not in ["analyzer", "dataset_before", "dataset_after"]:
                setattr(new_obj, k, copy.deepcopy(v, memodict))
        return new_obj

    def __setitem__(self, key, value):
        self.result_dict[key] = value

    def __getitem__(self, item):
        if item in self.result_dict:
            return self.result_dict[item]
        return self._get_aggregated_result_for_group[item]

    def __len__(self):
        return len(self.result_dict)

    def __repr__(self):
        name = f"{self.analyzer.__class__.__name__} of {self.dataset_before.__class__.__name__}"
        name += "\n" + "-" * len(name)
        n_results = f"{len(self)} results"
        if len(self.config) > 0:
            config = pretty_dict(
                self.config, heading_key="config", heading_value="value"
            )
        else:
            config = ""
        return "\n\n".join((name, n_results, config))

    def _repr_html_(self):
        return self._concat_results().to_html()


class NotesResult(Result):
    def _concat_results(
        self,
        index_result_dict: Optional[dict] = None,
        level_names: Optional[Union[Tuple[str], str]] = None,
    ) -> pd.DataFrame:
        df = super()._concat_results(
            index_result_dict=index_result_dict, level_names=level_names
        )
        return df[sorted(df.columns)]


class ChordSymbolResult(Result):
    def _concat_results(
        self,
        index_result_dict: Optional[dict] = None,
        level_names: Optional[Union[Tuple[str], str]] = None,
    ) -> pd.DataFrame:
        if index_result_dict is None:
            index_result_dict = self.result_dict
            if level_names is None:
                level_names = self.dataset_after.index_levels["indices"]
        elif level_names is None:
            raise ValueError("Names of index level(s) need(s) to be specified.")
        df = pd.DataFrame.from_dict(index_result_dict, orient="index", dtype="Int64")
        df.fillna(0, inplace=True)
        try:
            df.index.rename(level_names, inplace=True)
        except TypeError:
            print(f"level_names = {level_names}; nlevels = {df.index.nlevels}")
            raise
        # df.sort_values(df.index.to_list(), axis=1, ascending=False, inplace=True)
        return df

    def iter_group_results(self):
        for idx, aggregated in super().iter_group_results():
            yield idx, aggregated.sort_values(ascending=False).astype("Int64")
