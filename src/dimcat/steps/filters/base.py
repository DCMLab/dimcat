from typing import Iterable, Optional

import marshmallow as mm
from dimcat.data.resources import DimcatResource
from dimcat.data.resources.base import DR
from dimcat.data.resources.dc import levelvalue_
from dimcat.steps.groupers import CorpusGrouper, PieceGrouper


class FilterSchema(mm.Schema):
    keep_values = mm.fields.List(mm.fields.Raw, allow_none=True)
    drop_values = mm.fields.List(mm.fields.Raw, allow_none=True)
    drop_level = mm.fields.Boolean(allow_none=True)


class _FilterMixin:
    def __init__(
        self,
        keep_values: levelvalue_ | Iterable[levelvalue_] = None,
        drop_values: levelvalue_ | Iterable[levelvalue_] = None,
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
        if drop_level:
            # if the level is dropped, the Filter will not change the Dataset into a GroupedDataset, which is the
            # default value for the ClassVar '_new_dataset_type' inherited from the Grouper. For the automatic
            # behaviour, the decision cannot be made at this point and chances are that some resources will be grouped
            # and others not, so the type of the Dataset should be turned into/left as a GroupedDataset.
            self._new_dataset_type = None
        super().__init__(*args, **kwargs)

    def _post_process_result(
        self,
        result: DR,
        original_resource: DimcatResource,
    ) -> DR:
        """Perform the filtering on the grouped resource."""
        result = super()._post_process_result(result, original_resource)
        # the call to the super method adds the parent grouper's level to the default_groupby;
        # if the subsequent operation removes the level in question, it will also remove it from the default_groupby
        return result.filter_index_level(
            keep_values=self.keep_values,
            drop_values=self.drop_values,
            level=self.level_name,
            drop_level=self.drop_level,
        )


class CorpusFilter(_FilterMixin, CorpusGrouper):
    class Schema(CorpusGrouper.Schema, FilterSchema):
        pass


class PieceFilter(_FilterMixin, PieceGrouper):
    class Schema(PieceGrouper.Schema, FilterSchema):
        pass
