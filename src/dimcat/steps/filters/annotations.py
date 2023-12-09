from dimcat.steps.filters.base import FilterSchema, _FilterMixin
from dimcat.steps.groupers import HasCadenceAnnotationsGrouper, HasHarmonyLabelsGrouper


class HasCadenceAnnotationsFilter(_FilterMixin, HasCadenceAnnotationsGrouper):
    class Schema(HasCadenceAnnotationsGrouper.Schema, FilterSchema):
        pass


class HasHarmonyLabelsFilter(_FilterMixin, HasHarmonyLabelsGrouper):
    class Schema(HasHarmonyLabelsGrouper.Schema, FilterSchema):
        pass
