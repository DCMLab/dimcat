from __future__ import annotations

import logging
from typing import Iterable, Iterator, List, Literal, Optional, Type, overload

from dimcat import Dataset, DimcatConfig
from dimcat.base import (
    DimcatObject,
    DimcatObjectField,
    get_class,
    make_config_from_specs,
    make_object_from_specs,
)
from dimcat.data.resources.base import DR
from dimcat.data.resources.dc import DimcatResource
from dimcat.dc_exceptions import NoMatchingPipelineStepFoundError
from dimcat.steps.base import PipelineStep, StepSpecs
from marshmallow import fields

module_logger = logging.getLogger(__name__)


class Pipeline(PipelineStep):
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

    class Schema(PipelineStep.Schema):
        steps = fields.List(DimcatObjectField())

    def __init__(
        self,
        steps: Optional[StepSpecs | Iterable[StepSpecs]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        self,
        steps: StepSpecs | Iterable[StepSpecs],
    ) -> None:
        if isinstance(steps, (DimcatObject, dict, type, str)):
            steps = [steps]
        for step in steps:
            self.add_step(step)

    def add_step(self, step: StepSpecs) -> None:
        step = make_object_from_specs(step, PipelineStep)
        self._steps.append(step)

    def get_last_step(
        self,
        step_specs: Optional[StepSpecs] = None,
        allow_subclasses: bool = True,
    ) -> PipelineStep:
        """Returns the last step that matches the given specs.

        Args:
            step_specs:
                Specification that can be converted to a :class:`DimcatConfig` describing a :class:`PipelineStep`.
                If None, the last step is returned.
            allow_subclasses:
                By default, matches the last applied :class:`PipelineStep` of the type described by ``step_specs``
                or one of its subclasses. Set to ``False`` to return the last step that matches exactly.

        Returns:
            PipelineStep object that matches the given specs.

        Raises:
            NoMatchingPipelineStepFoundError: If no matching step is found.
        """
        if step_specs is None:
            if len(self._steps) == 0:
                raise NoMatchingPipelineStepFoundError
            return self._steps[-1]
        steps = self.get_steps(step_specs, allow_subclasses=allow_subclasses)
        if len(steps) == 0:
            raise NoMatchingPipelineStepFoundError(step_specs)
        return steps[-1]

    def get_steps(
        self,
        step_specs: Optional[StepSpecs] = None,
        allow_subclasses: bool = True,
    ) -> List[PipelineStep]:
        """Returns all steps that match the given specs.

        Args:
            step_specs:
                Specification that can be converted to a :class:`DimcatConfig` describing a :class:`PipelineStep`.
                If None, all steps are returned (equivalent to :attr:`steps`).
            allow_subclasses:
                By default, matching subclasses of the :class:`PipelineStep` described by ``step_specs`` are also
                included. Set to ``False`` to only return steps that match exactly.

        Returns:
            PipelineStep objects that matches the given specs.
        """
        if step_specs is None:
            return self.steps
        step_config = make_config_from_specs(step_specs, "PipelineStep")
        covariant = allow_subclasses
        steps = []
        for step in self._steps:
            cfg = step.to_config()
            if cfg.matches(step_config, covariant=covariant):
                steps.append(step)
        return steps

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

    def _process_resource(
        self,
        resource: DimcatResource,
        ignore_exceptions: bool = False,
        skip_step_types: Optional[Iterable[Type[PipelineStep] | str]] = None,
    ) -> DR:
        if skip_step_types:
            if isinstance(skip_step_types, (str, type)):
                skip_step_types = [skip_step_types]
            ignored_types = [
                get_class(dtype) if isinstance(dtype, str) else dtype
                for dtype in skip_step_types
            ]
            ignored_types = tuple(set(ignored_types))
            pipeline_steps = [
                step for step in self._steps if not isinstance(step, ignored_types)
            ]
        else:
            pipeline_steps = self._steps
        if len(pipeline_steps) == 0:
            self.logger.info("Nothing to do.")
            return resource
        processed_resource = resource
        for step in pipeline_steps:
            previous_resource = processed_resource
            try:
                processed_resource = step.process_resource(previous_resource)
            except Exception as e:
                if ignore_exceptions:
                    self.logger.info(
                        f"{step.name!r}._process_resource() failed on {previous_resource.resource_name!r} and "
                        f"ignore_exceptions=True, so the {previous_resource.name} will not have been processed by "
                        f"this PipelineStep. Exception:\n{e!r}"
                    )
                    continue
                raise
            # ToDo: Check the processed resource and handle errors
        return processed_resource
