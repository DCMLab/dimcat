"""A Pipeline performs computation iteratively on a List of PipelineSteps."""
from typing import List

from dimcat.base import PipelineStep, Data




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
