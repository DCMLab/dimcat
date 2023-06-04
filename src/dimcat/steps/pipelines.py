from __future__ import annotations

import logging
from typing import Iterable, Iterator, List, Optional

from dimcat.base import DimcatConfig, DimcatObject, DimcatObjectField
from dimcat.data.dataset import Dataset
from dimcat.data.resources import DimcatResource, FeatureSpecs
from marshmallow import fields

from .base import PipelineStep

logger = logging.getLogger(__name__)


class Pipeline(PipelineStep):
    @classmethod
    def from_pipeline(cls, pipeline: Pipeline) -> Pipeline:
        return cls(steps=pipeline.steps)

    class Schema(PipelineStep.Schema):
        steps = fields.List(DimcatObjectField())

    def __init__(
        self,
        steps: Optional[
            PipelineStep | DimcatConfig | Iterable[PipelineStep | DimcatConfig]
        ] = None,
        features: Optional[FeatureSpecs | Iterable[FeatureSpecs]] = None,
        **kwargs,
    ):
        super().__init__(features=features, **kwargs)
        self._steps: List[PipelineStep] = []
        if steps is not None:
            self.steps = steps

    def __iter__(self) -> Iterator[PipelineStep]:
        yield from self._steps

    def __len__(self):
        return len(self._steps)

    @property
    def steps(self) -> List[PipelineStep]:
        """The pipeline steps as a list. Modifying the list does not affect the pipeline."""
        return list(self._steps)

    @steps.setter
    def steps(
        self, steps: PipelineStep | DimcatConfig | Iterable[PipelineStep | DimcatConfig]
    ) -> None:
        if isinstance(steps, (DimcatObject, dict)):
            steps = [steps]
        for step in steps:
            self.add_step(step)

    def add_step(self, step: PipelineStep | DimcatConfig) -> None:
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

    def _process_dataset(self, dataset: Dataset) -> Dataset:
        if len(self._steps) == 0:
            self.logger.info("Nothing to do.")
            return dataset
        previous_dataset = dataset
        for step in self._steps:
            processed_dataset = step.process(previous_dataset)
            # ToDo: checks?
            previous_dataset = processed_dataset
        return processed_dataset

    def _process_resource(self, resource: DimcatResource) -> DimcatResource:
        if len(self._steps) == 0:
            self.logger.info("Nothing to do.")
            return resource
        previous_resource = resource
        for step in self._steps:
            processed_resource = step.dispatch(previous_resource)
            # ToDo: checks?
            previous_resource = processed_resource
        return processed_resource
