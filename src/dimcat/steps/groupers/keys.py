import pandas as pd
from dimcat.data.resources import DimcatResource, Feature, FeatureName
from dimcat.steps.groupers.base import Grouper


class ModeGrouper(Grouper):
    allowed_features = (FeatureName.HarmonyLabels,)

    def __init__(
        self,
        level_name: str = "mode",
        **kwargs,
    ):
        super().__init__(level_name=level_name, **kwargs)

    def check_resource(self, resource: DimcatResource) -> None:
        super().check_resource(resource)
        if "localkey_mode" not in resource.df.columns:
            raise ValueError(f"Expected 'localkey_mode' column in {resource.name}")

    def apply_grouper(self, resource: Feature) -> pd.DataFrame:
        """Apply the grouper to a Feature."""
        resource_df = resource.df
        grouped_df = pd.concat(
            dict(resource_df.groupby("localkey_mode").__iter__()),
            names=[self.level_name],
        )
        # this is equivalent to but slightly faster than:
        # grouped_df = resource_df.groupby('localkey_mode').apply(lambda df: df) # <-- index level would need renaming
        return grouped_df
