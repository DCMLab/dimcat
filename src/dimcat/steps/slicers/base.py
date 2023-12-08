import logging
from typing import ClassVar, Optional

import marshmallow as mm
import pandas as pd
from dimcat.data.datasets.processed import SlicedDataset
from dimcat.data.resources import DimcatResource, FeatureName
from dimcat.data.resources.base import DR
from dimcat.data.resources.dc import SliceIntervals
from dimcat.dc_exceptions import ResourceAlreadyTransformed
from dimcat.steps.base import ResourceTransformation
from dimcat.utils import check_name

logger = logging.getLogger(__name__)


class Slicer(ResourceTransformation):
    # inherited from PipelineStep:
    _new_dataset_type = SlicedDataset
    _new_resource_type = None  # same as input
    _applicable_to_empty_datasets = True
    # inherited from FeatureProcessingStep:
    _allowed_features = None  # any
    _output_package_name = None  # transform 'features'
    _requires_at_least_one_feature = False
    _required_feature: ClassVar[Optional[FeatureName]] = None
    """The type of Feature that needs to be present in a dataset to fit this slicer."""

    class Schema(ResourceTransformation.Schema):
        level_name = mm.fields.Str()

    def __init__(self, level_name: str = "slice", **kwargs):
        super().__init__(**kwargs)
        self._level_name: str = None
        self.level_name = level_name

    @property
    def level_name(self) -> str:
        return self._level_name

    @level_name.setter
    def level_name(self, level_name: str):
        check_name(level_name)
        self._level_name = level_name

    @property
    def required_feature(self) -> Optional[FeatureName]:
        return self._required_feature

    def check_resource(self, resource: DimcatResource) -> None:
        super().check_resource(resource)
        if self.level_name in resource.get_level_names():
            raise ResourceAlreadyTransformed(resource.resource_name, self.name)

    def get_slice_intervals(self, resource: DimcatResource) -> SliceIntervals:
        return resource.get_slice_intervals(
            level_name=self.level_name
        )  # base slicer slices resource by itself

    def _post_process_result(
        self,
        result: DR,
        original_resource: DimcatResource,
    ) -> DR:
        return result

    def transform_resource(self, resource: DimcatResource) -> pd.DataFrame:
        """Apply the grouper to a Feature."""
        slice_intervals = self.get_slice_intervals(resource)
        return resource.apply_slice_intervals(slice_intervals)
