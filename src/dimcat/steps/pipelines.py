from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from dimcat.base import DimcatConfig, DimcatObject, deserialize_dict
from marshmallow import fields

from .base import PipelineStep

logger = logging.getLogger(__name__)


class DimcatObjectField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        if isinstance(value, DimcatConfig):
            return dict(value)
        return value.to_dict()

    def _deserialize(self, value, attr, data, **kwargs):
        return deserialize_dict(value)


class Pipeline(PipelineStep):
    class Schema(PipelineStep.Schema):
        steps = fields.List(DimcatObjectField())

    def __init__(
        self,
        steps: Optional[
            PipelineStep | DimcatConfig | Iterable[PipelineStep | DimcatConfig]
        ] = None,
        **kwargs,
    ):
        self._steps: List[PipelineStep] = []
        if steps is not None:
            self.steps = steps

    @property
    def steps(self) -> List[PipelineStep]:
        return self._steps

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
