from typing import Hashable, Iterable, Optional

from dimcat.data.resources import DimcatResource
from dimcat.data.resources.base import DR
from dimcat.steps.groupers import CorpusGrouper, PieceGrouper


class _FilterMixin:
    def __init__(
        self,
        keep_values: Optional[Hashable | Iterable[Hashable]] = None,
        drop_values: Optional[Hashable | Iterable[Hashable]] = None,
        drop_level: Optional[bool] = None,
        *args,
        **kwargs
    ):
        """Mixin class that post_processes the result of the extended Grouper by filtering its grouping level.

        Args:
            keep_values:
                One or several values to keep (dropping the rest). If a value is specified both for keeping and
                dropping, it is dropped.
            drop_values: One or several values to drop.
            drop_level:
                Boolean specifies whether to keep the filtered level or to drop it. The default (None) corresponds
                to automatic behaviour, where the level is dropped if only one value remains, otherwise kept.
            *args:
            **kwargs:
        """
        self.keep_values = keep_values
        self.drop_values = drop_values
        self.drop_level = drop_level
        super().__init__(*args, **kwargs)

    def _post_process_result(
        self,
        result: DR,
        original_resource: DimcatResource,
    ) -> DR:
        """Perform the filtering on the grouped resource."""
        result = super()._post_process_result(result, original_resource)
        return result.filter_index_level(
            keep_values=self.keep_values,
            drop_values=self.drop_values,
            level=self.level_name,
            drop_level=self.drop_level,
        )


class CorpusFilter(_FilterMixin, CorpusGrouper):
    pass


class PieceFilter(_FilterMixin, PieceGrouper):
    pass
