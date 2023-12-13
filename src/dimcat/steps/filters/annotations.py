from typing import Iterable, Optional

from dimcat.data.resources.base import IX
from dimcat.data.resources.dc import levelvalue_
from dimcat.steps.filters.base import FilterSchema, _FilterMixin
from dimcat.steps.groupers import HasCadenceAnnotationsGrouper, HasHarmonyLabelsGrouper


class HasCadenceAnnotationsFilter(_FilterMixin, HasCadenceAnnotationsGrouper):
    class Schema(HasCadenceAnnotationsGrouper.Schema, FilterSchema):
        pass


class HasHarmonyLabelsFilter(_FilterMixin, HasHarmonyLabelsGrouper):
    class Schema(HasHarmonyLabelsGrouper.Schema, FilterSchema):
        pass

    def __init__(
        self,
        keep_values: levelvalue_ | Iterable[levelvalue_] = (True,),
        drop_values: levelvalue_ | Iterable[levelvalue_] = None,
        drop_level: Optional[bool] = None,
        level_name: str = "has_harmony_labels",
        grouped_units: IX = None,
        **kwargs,
    ):
        super().__init__(
            keep_values=keep_values,
            drop_values=drop_values,
            drop_level=drop_level,
            level_name=level_name,
            grouped_units=grouped_units,
            **kwargs,
        )
