import logging

from dimcat.data.resources.base import SomeDataframe, SomeSeries
from dimcat.data.resources.dc import DimcatResource
from dimcat.data.resources.features import Feature, FeatureName
from dimcat.data.resources.results import Durations, PitchClassDurations
from dimcat.steps.analyzers.base import Analyzer

logger = logging.getLogger(__name__)


class Proportions(Analyzer):
    new_resource_type = Durations

    @staticmethod
    def compute(feature: DimcatResource | SomeDataframe, **kwargs) -> int:
        result = feature.groupby(feature.value_column).duration_qb.sum().astype(float)
        result = result.to_frame()
        return result

    def groupby_apply(self, feature: Feature, groupby: SomeSeries = None, **kwargs):
        """Performs the computation on a groupby. The value of ``groupby`` needs to be
        a Series of the same length as ``feature`` or otherwise work as positional argument to feature.groupby().
        """
        if groupby is None:
            groupby = feature.get_grouping_levels(self.smallest_unit)
            self.logger.debug(
                f"Using the {feature.resource_name}'s default groupby {groupby!r}"
            )
        groupby += [feature.value_column]
        result = (
            feature.groupby(groupby, group_keys=False).duration_qb.sum().astype(float)
        )
        result = result.to_frame()
        return result

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}_proportions"


class PitchClassVectors(Proportions):
    allowed_features = (FeatureName.Notes,)
    new_resource_type = PitchClassDurations

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}_pitch_class_vectors"
