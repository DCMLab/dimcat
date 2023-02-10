from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type


class Data(ABC):
    """
    Subclasses are the dtypes that this library uses. Every PipelineStep accepts one or several
    dtypes.

    The initializer can set parameters influencing how the contained data will look and is able
    to create an object from an existing Data object to enable type conversion.
    """

    _registry: Dict[str, Type] = {}
    """Register of all subclasses."""

    def __init_subclass__(cls, **kwargs):
        """Registers every subclass under the class variable :attr:`_registry`"""
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls


class PipelineStep(ABC):
    """
    A PipelineStep object is able to transform some data in a pre-defined way.

    The initializer will set some parameters of the transformation, and then the
    `process_data` function is used to transform an input Data object, returning a copy.
    """

    _registry: Dict[str, Type] = {}
    """Register of all subclasses."""

    def __init__(self):
        self.required_facets = []
        """Specifies a list of facets (such as 'notes' or 'labels') that the passed Data object
        needs to provide."""

    def __init_subclass__(cls, **kwargs):
        """Registers every subclass under the class variable :attr:`_registry`"""
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    def check(self, _) -> Tuple[bool, str]:
        """Test piece of data for certain properties before computing analysis.

        Returns
        -------
        :obj:`bool`
            True if the passed data is eligible.
        :obj:`str`
            Error message in case the passed data is not eligible.
        """
        return True, ""

    def filename_factory(self):
        return self.__class__.__name__

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


PS_TYPES = dict(PipelineStep._registry)
D_TYPES = dict(Data._registry)
