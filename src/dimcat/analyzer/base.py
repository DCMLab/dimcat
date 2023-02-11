"""Analyzers are PipelineSteps that process data and store the results in Data.processed."""
import logging
from abc import ABC, abstractmethod
from typing import Any, Collection, Iterator, Tuple, Type, TypeVar, Union

import dimcat.data as data_module
from dimcat._typing import ID
from dimcat.base import PipelineStep
from dimcat.data import AnalyzedData, Dataset
from dimcat.utils import typestrings2types

logger = logging.getLogger(__name__)


def _typestring2type(typestring: str) -> Type:
    return getattr(data_module, typestring)


R = TypeVar("R")


class Analyzer(PipelineStep, ABC):
    """Analyzers are PipelineSteps that process data and store the results in Data.processed."""

    result_type = "Result"

    assert_steps: Union[str, Collection[str]] = []
    """Analyzer.process_data() raises ValueError if at least one of the names does not belong to
    a :obj:`PipelineStep` that is among the previous PipelineSteps applied to the :obj:`_Dataset`."""

    assert_previous_step: Union[str, Collection[str]] = []
    """Analyzer.process_data() raises ValueError if last :obj:`PipelineStep` applied to the
    :obj:`_Dataset` does not match any of these types."""

    excluded_steps: Union[str, Collection[str]] = []
    """Analyzer.process_data() raises ValueError if any of the previous :obj:`PipelineStep` applied to the
    :obj:`_Dataset` matches one of these types."""

    def __init__(self):
        """Creates essential fields."""
        self.config = {}
        """:obj:`dict`
        This dictionary stores the parameters to be passed to the compute() method."""
        self.group2pandas = None
        """:obj:`str`
        The name of the function that allows displaying one group's results as a single
        pandas object. See data.Corpus.convert_group2pandas()"""
        self.level_names = {}
        """:obj:`dict`
        Define {"indices": "index_level_name"} if the analysis is applied once per group,
        because the index of the DataFrame holding the processed data won't be showing the
        individual indices anymore.
        """

    @staticmethod
    @abstractmethod
    def aggregate(result_a: R, result_b: R) -> R:
        """Static method that combines two results of :meth:`compute`.

        This needs to be equivalent to calling self.compute on the concatenation of the respective data resulting
        in the two arguments."""
        pass

    @staticmethod
    @abstractmethod
    def compute(self, **kwargs) -> R:
        """Static method that performs the actual computation takes place."""

    @abstractmethod
    def data_iterator(self, data: AnalyzedData) -> Iterator[Tuple[ID, Any]]:
        """How a particular analyzer iterates through a dataset, getting the chunks passed to :meth:`compute`."""
        yield from data

    def process_data(self, dataset: Dataset) -> AnalyzedData:
        """Returns an :obj:`AnalyzedData` copy of the Dataset with the added analysis result."""
        analyzer_name = self.__class__.__name__
        if len(self.assert_steps) > 0:
            assert_steps = typestrings2types(self.assert_steps)
            for step in assert_steps:
                if not any(
                    isinstance(previous_step, step)
                    for previous_step in dataset.pipeline_steps
                ):
                    raise ValueError(
                        f"{analyzer_name} require previous application of a {step.__name__}."
                    )
        if len(self.assert_previous_step) > 0:
            assert_previous_step = typestrings2types(self.assert_previous_step)
            previous_step = dataset.pipeline_steps[0]
            if not isinstance(previous_step, assert_previous_step):
                raise ValueError(
                    f"{analyzer_name} requires the previous pipeline step to be an "
                    f"instance of {self.assert_previous_step}, not {previous_step.__name__}."
                )
        if len(self.excluded_steps) > 0:
            excluded_steps = typestrings2types(self.excluded_steps)
            for step in excluded_steps:
                if any(
                    isinstance(previous_step, step)
                    for previous_step in dataset.pipeline_steps
                ):
                    raise ValueError(
                        f"{analyzer_name} cannot be applied when a {step.__name__} has been applied before."
                    )
        new_dataset = AnalyzedData(dataset)
        result_type = _typestring2type(self.result_type)
        result_object = result_type(
            analyzer=self, dataset_before=dataset, dataset_after=new_dataset
        )
        result_object.config = self.config
        for idx, df in self.data_iterator(new_dataset):
            eligible, message = self.check(df)
            if not eligible:
                logger.info(f"{idx}: {message}")
                continue
            result_object[idx] = self.compute(df, **self.config)
        result_object = self.post_process(result_object)
        new_dataset.track_pipeline(
            self, group2pandas=self.group2pandas, **self.level_names
        )
        new_dataset.set_result(self, result_object)
        return new_dataset

    def post_process(self, processed):
        """Whatever needs to be done after analyzing the data before passing it to the dataset."""
        return processed
