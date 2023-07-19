import logging

import marshmallow as mm
from dimcat.data.dataset.processed import SlicedDataset
from dimcat.steps.base import FeatureProcessingStep

logger = logging.getLogger(__name__)


class Slicer(FeatureProcessingStep):
    new_dataset_type = SlicedDataset
    new_resource_type = None
    output_package_name = None

    class Schema(FeatureProcessingStep.Schema):
        level_name = mm.fields.Str()
