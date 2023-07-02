from __future__ import annotations

import logging
from typing import Iterable, Iterator, List, Literal, Optional, overload

from dimcat.base import DimcatConfig, DimcatObject, DimcatObjectField
from dimcat.data.dataset import Dataset
from dimcat.data.resources import DimcatResource, FeatureSpecs
from marshmallow import fields

from .base import FeatureProcessingStep, PipelineStep

logger = logging.getLogger(__name__)


class Pipeline(FeatureProcessingStep):
    @classmethod
    def from_step_configs(cls, configs: Iterable[DimcatConfig]) -> Pipeline:
        steps = []
        for config in configs:
            if not isinstance(config, DimcatConfig):
                config = DimcatConfig(config)
            steps.append(config.create())
        return cls(steps)

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline) -> Pipeline:
        return cls(steps=pipeline.steps)

    class Schema(FeatureProcessingStep.Schema):
        steps = fields.List(DimcatObjectField())

    def __init__(
        self,
        steps: Optional[
            FeatureProcessingStep
            | DimcatConfig
            | Iterable[FeatureProcessingStep | DimcatConfig]
        ] = None,
        features: Optional[FeatureSpecs | Iterable[FeatureSpecs]] = None,
        **kwargs,
    ):
        super().__init__(features=features, **kwargs)
        self._steps: List[FeatureProcessingStep] = []
        if steps is not None:
            self.steps = steps

    def __iter__(self) -> Iterator[FeatureProcessingStep]:
        yield from self._steps

    def __len__(self):
        return len(self._steps)

    @property
    def steps(self) -> List[FeatureProcessingStep]:
        """The pipeline steps as a list. Modifying the list does not affect the pipeline."""
        return list(self._steps)

    @steps.setter
    def steps(
        self,
        steps: FeatureProcessingStep
        | DimcatConfig
        | Iterable[FeatureProcessingStep | DimcatConfig],
    ) -> None:
        if isinstance(steps, (DimcatObject, dict)):
            steps = [steps]
        for step in steps:
            self.add_step(step)

    def add_step(self, step: FeatureProcessingStep | DimcatConfig) -> None:
        if isinstance(step, dict):
            assert "dtype" in step, (
                "PipelineStep dict must be config-like and have a 'dtype' key mapping to the "
                "name of a PipelineStep."
            )
            step = DimcatConfig(step)
        if isinstance(step, DimcatConfig):
            step = step.create()
        if not isinstance(step, PipelineStep):
            raise TypeError(f"Pipeline acceppts only PipelineSteps, not {type(step)}.")
        self._steps.append(step)

    @overload
    def info(self, return_str: Literal[False] = False) -> None:
        ...

    @overload
    def info(self, return_str: Literal[True]) -> str:
        ...

    def info(self, return_str=False) -> Optional[str]:
        """Show the names of the included steps."""
        info_str = f"Pipeline([{', '.join(step.name for step in self._steps)}])"
        if return_str:
            return info_str
        print(info_str)

    def _process_dataset(self, dataset: Dataset) -> Dataset:
        if len(self._steps) == 0:
            self.logger.info("Nothing to do.")
            return dataset
        processed_dataset = dataset
        for step in self._steps:
            previous_dataset = processed_dataset
            processed_dataset = step.process(previous_dataset)
            # ToDo: checks?
        return processed_dataset

    def _process_resource(self, resource: DimcatResource) -> DimcatResource:
        if len(self._steps) == 0:
            self.logger.info("Nothing to do.")
            return resource
        processed_resource = resource
        for step in self._steps:
            previous_resource = processed_resource
            processed_resource = step._make_new_resource(previous_resource)
            # ToDo: Pipeline checks the compatibility of steps and data first, uses step._process_resource()
            # ToDo: Check the processed resource and handle errors
        return processed_resource
