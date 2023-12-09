import logging

import marshmallow as mm
from dimcat.data.resources.base import D, FeatureName, SomeSeries
from dimcat.data.resources.dc import DimcatResource, Feature
from dimcat.data.resources.results import Durations
from dimcat.dc_exceptions import FeatureWithUndefinedValueColumnError
from dimcat.steps.analyzers.base import Analyzer

logger = logging.getLogger(__name__)


class Proportions(Analyzer):
    _default_dimension_column = "duration_qb"
    _new_resource_type = Durations

    @staticmethod
    def compute(feature: Feature, **kwargs) -> D:
        groupby = [feature.value_column]
        if (
            feature.formatted_column is not None
            and feature.formatted_column not in groupby
        ):
            groupby.append(feature.formatted_column)
        result = (
            feature.groupby(groupby)[Proportions._default_dimension_column]
            .sum()
            .astype(float)
        )
        result = result.to_frame()
        return result

    class Schema(Analyzer.Schema):
        dimension_column = mm.fields.Str(
            load_default="duration_qb", allow_none=True, metadata=dict(expose=False)
        )

    def check_resource(self, resource: DimcatResource) -> None:
        """Check if the resource has a value column."""
        super().check_resource(resource)
        if resource.value_column is None:
            raise FeatureWithUndefinedValueColumnError(
                resource.resource_name, resource.name
            )

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
        if (
            feature.formatted_column is not None
            and feature.formatted_column not in groupby
        ):
            groupby.append(feature.formatted_column)
        result = (
            feature.groupby(groupby, group_keys=False)[self.dimension_column]
            .sum()
            .astype(float)
        )
        result = result.to_frame()
        return result

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}.proportions"


class PitchClassVectors(Proportions):
    _allowed_features = (FeatureName.Notes,)

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}.pitch_class_vectors"
