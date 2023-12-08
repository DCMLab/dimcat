from dimcat.steps.filters.base import _FilterMixin
from dimcat.steps.groupers import HasCadenceAnnotationsGrouper, HasHarmonyLabelsGrouper


class HasCadenceAnnotationsFilter(_FilterMixin, HasCadenceAnnotationsGrouper):
    pass


class HasHarmonyLabelsFilter(_FilterMixin, HasHarmonyLabelsGrouper):
    pass
