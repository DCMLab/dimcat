"""A Pipeline performs computation iteratively on a List of PipelineSteps."""
from abc import ABC, abstractmethod
from typing import List

from .data import Data


class PipelineStep(ABC):
    """
    A PipelineStep object is able to transform some data in a pre-defined way.

    The initializer will set some parameters of the transformation, and then the
    `process_data` function is used to transform an input Data object, returning a copy.
    """

    def __init__(self):
        self.required_facets = []
        """Specifies a list of facets (such as 'notes' or 'labels') that the passed Data object
        needs to provide."""

    def check(self, _):
        """Test piece of data for certain properties before computing analysis.

        Returns
        -------
        :obj:`bool`
            True if the passed data is eligible.
        :obj:`str`
            Error message in case the passed data is not eligible.
        """
        return True, ""

    @abstractmethod
    def process_data(self, data: Data) -> Data:
        """
        Perform a transformation on an input Data object. This should never alter the
        Data or its properties in place, instead returning a copy or view of the input.

        Parameters
        ----------
        data : :obj:`Data`
            The data to be transformed. This should not be altered in place.

        Returns
        -------
        :obj:`Data`
            A copy or view of the input Data, transformed in some way defined by this
            PipelineStep.
        """


class Pipeline(PipelineStep):
    """
    A Pipeline takes at initialization a List of PipelineSteps, and processes the Data
    by iteratievly feeding the input Data into each PipelineStep in that list.

    Note that a Pipeline is itself a PipelineStep, so can be included recursively in
    another Pipeline.
    """

    def __init__(self, pipeline_steps: List[PipelineStep]):
        """
        Create a new Pipeline for performing an iterative computation on some Data.

        Parameters
        ----------
        pipeline_steps : List[PipelineStep]
            A List of the computations to perform iteratively on the input Data.
        """
        self.pipeline_steps = pipeline_steps

    def process_data(self, data: Data) -> Data:
        """
        Process the given Data. This will feed the Data iteratively to each PipelineStep
        contained in its initialized pipeline_steps list.

        Parameters
        ----------
        data : Data
            The input Data on which some computation will be performed.

        Returns
        -------
        Data
            The result of feeding the input Data into each step in this Pipeline. This
            will be exactly the output from the process_data function of the final
            PipelineStep.
        """
        for pipeline_step in self.pipeline_steps:
            data = pipeline_step.process_data(data)

        return data
