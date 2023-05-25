from dimcat.analyzers.base import Analyzer
from dimcat.resources.base import DimcatResource, SomeSeries
from dimcat.resources.features import Feature


class Counter(Analyzer):
    @staticmethod
    def compute(feature: DimcatResource, **kwargs) -> int:
        return len(feature.index)

    def groupby_apply(self, feature: Feature, groupby: SomeSeries = None, **kwargs):
        """Static method that performs the computation on a groupby. The value of ``groupby`` needs to be
        a Series of the same length as ``feature`` or otherwise work as positional argument to feature.groupby().
        """
        if groupby is None:
            gpb = feature.groupby(level=[0, 1])
        else:
            gpb = feature.groupby(groupby)
        return gpb.size()

    def post_process(self, result):
        """Whatever needs to be done after analyzing the data before passing it to the dataset."""
        name = f"{self.features[0].name} counts"
        return result.from_df(result.rename(name).to_frame())
