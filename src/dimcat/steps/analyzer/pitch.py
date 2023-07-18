import logging

from dimcat.data.resource.base import SomeDataframe, SomeSeries
from dimcat.data.resource.dc import DimcatResource
from dimcat.data.resource.features import Feature, FeatureName
from dimcat.data.resource.results import Durations
from dimcat.steps.analyzer.base import Analyzer

logger = logging.getLogger(__name__)


class PitchClassVectors(Analyzer):
    allowed_features = (FeatureName.Notes,)
    new_resource_type = Durations

    @staticmethod
    def compute(feature: DimcatResource | SomeDataframe, **kwargs) -> int:
        result = feature.groupby("tpc").duration_qb.sum().astype(float)
        result = result.to_frame("duration in ♩")
        return result

    def groupby_apply(self, feature: Feature, groupby: SomeSeries = None, **kwargs):
        """Performs the computation on a groupby. The value of ``groupby`` needs to be
        a Series of the same length as ``feature`` or otherwise work as positional argument to feature.groupby().
        """
        if groupby is None:
            groupby = feature.get_default_groupby()
            self.logger.debug(
                f"Using the {feature.resource_name}'s default groupby {groupby!r}"
            )
        groupby += ["tpc"]
        result = (
            feature.groupby(groupby, group_keys=False).duration_qb.sum().astype(float)
        )
        result = result.to_frame("duration in ♩")
        return result

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}_pitch_class_vectors"