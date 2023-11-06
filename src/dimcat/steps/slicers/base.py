import logging
from collections import defaultdict

import marshmallow as mm
import pandas as pd
from dimcat import Dataset
from dimcat.data.datasets.processed import SlicedDataset
from dimcat.data.resources import DimcatResource, Feature
from dimcat.data.resources.dc import SliceIntervals
from dimcat.dc_exceptions import ResourceNotProcessableError
from dimcat.steps.base import FeatureProcessingStep
from dimcat.utils import check_name

logger = logging.getLogger(__name__)


class Slicer(FeatureProcessingStep):
    # inherited from PipelineStep:
    new_dataset_type = SlicedDataset
    new_resource_type = None  # same as input
    applicable_to_empty_datasets = True
    # inherited from FeatureProcessingStep:
    allowed_features = None  # any
    output_package_name = None  # transform 'features'
    requires_at_least_one_feature = False

    class Schema(FeatureProcessingStep.Schema):
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

    def apply_slicer(self, resource: DimcatResource) -> pd.DataFrame:
        """Apply the grouper to a Feature."""
        slice_intervals = self.get_slice_intervals(resource)
        return resource.apply_slice_intervals(slice_intervals)

    def get_slice_intervals(self, resource: DimcatResource) -> SliceIntervals:
        return resource.get_slice_intervals(
            level_name=self.level_name
        )  # base slicer slices resource by itself

    def _make_new_resource(self, resource: Feature) -> Feature:
        """Apply the grouper to a Feature."""
        if self.level_name in resource.get_level_names():
            self.logger.debug(
                f"Resource {resource.resource_name!r} already has a level named {self.level_name!r}."
            )
            return resource
        result_constructor = self._get_new_resource_type(resource)
        results = self.apply_slicer(resource)
        result_name = self.resource_name_factory(resource)
        return result_constructor.from_dataframe(
            df=results,
            resource_name=result_name,
        )

    def _process_dataset(self, dataset: Dataset) -> Dataset:
        """Apply this PipelineStep to a :class:`Dataset` and return a copy containing the output(s)."""
        new_dataset = self._make_new_dataset(dataset)
        self.fit_to_dataset(new_dataset)
        new_dataset._pipeline.add_step(self)
        package_name_resource_iterator = self._iter_resources(new_dataset)
        processed_resources = defaultdict(list)
        for package_name, resource in package_name_resource_iterator:
            try:
                new_resource = self.process_resource(resource)
            except ResourceNotProcessableError as e:
                self.logger.warning(
                    f"Resource {resource.resource_name!r} could not be grouped and is not included in "
                    f"the new Dataset due to the following error: {e!r}"
                )
                continue
            processed_resources[package_name].append(new_resource)
        for package_name, resources in processed_resources.items():
            new_package = self._make_new_package(package_name)
            new_package.extend(resources)
            n_processed = len(resources)
            if new_package.n_resources < n_processed:
                if new_package.n_resources == 0:
                    self.logger.warning(
                        f"None of the {n_processed} {package_name} were successfully transformed."
                    )
                else:
                    self.logger.warning(
                        f"Transformation was successful only on {new_package.n_resources} of the "
                        f"{n_processed} features."
                    )
            new_dataset.outputs.replace_package(new_package)
        return new_dataset

    def _post_process_result(
        self,
        result: DimcatResource,
        original_resource: DimcatResource,
    ) -> DimcatResource:
        """Change the default_groupby value of the returned Feature."""
        result.update_default_groupby(self.level_name)
        return result
