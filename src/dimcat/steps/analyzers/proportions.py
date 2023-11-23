import logging

from dimcat.data.resources.base import FeatureName, SomeDataframe, SomeSeries
from dimcat.data.resources.dc import DimcatResource, Feature
from dimcat.data.resources.results import (
    Durations,
    PitchClassDurations,
    Result,
    ScaleDegreeDurations,
)
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
        groupby.append(feature.value_column)
        result = (
            feature.groupby(groupby, group_keys=False).duration_qb.sum().astype(float)
        )
        result = result.to_frame()
        return result

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}.proportions"


class PitchClassVectors(Proportions):
    allowed_features = (FeatureName.Notes,)
    new_resource_type = PitchClassDurations

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}.pitch_class_vectors"


class ScaleDegreeVectors(Proportions):
    allowed_features = (FeatureName.BassNotes,)
    new_resource_type = ScaleDegreeDurations

    def _make_new_resource(self, resource: Feature) -> Result:
        result = super()._make_new_resource(resource)
        result.default_format = resource.format
        return result

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}.scale_degree_vectors"
