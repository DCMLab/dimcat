import logging

from dimcat.data.resource.base import SomeDataframe, SomeSeries
from dimcat.data.resource.dc import DimcatResource
from dimcat.data.resource.features import Feature
from dimcat.steps.analyzers.base import Analyzer

logger = logging.getLogger(__name__)


class Counter(Analyzer):
    @staticmethod
    def compute(feature: DimcatResource | SomeDataframe, **kwargs) -> int:
        return len(feature.index)

    def groupby_apply(self, feature: Feature, groupby: SomeSeries = None, **kwargs):
        """Performs the computation on a groupby. The value of ``groupby`` needs to be
        a Series of the same length as ``feature`` or otherwise work as positional argument to feature.groupby().
        """
        if groupby is None:
            groupby = feature.get_default_groupby()
            self.logger.debug(
                f"Using the {feature.resource_name}'s default groupby {groupby!r}"
            )
        return (
            feature.groupby(groupby).size().to_frame(f"{feature.resource_name} counts")
        )

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}_counted"
