from abc import ABC
from typing import ClassVar, Dict, Iterable, List, Tuple, Type, Union, overload


class Data(ABC):
    """
    This abstract base class unites all classes containing data in some way or another.
    All subclasses are dynamically collected in the class variable :attr:`_registry`.
    """

    _registry: ClassVar[Dict[str, Type]] = {}
    """Register of all subclasses."""

    def __init_subclass__(cls, **kwargs):
        """Registers every subclass under the class variable :attr:`_registry`"""
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    def __init__(self, **kwargs):
        pass

    @classmethod
    @property
    def name(cls) -> str:
        return cls.__name__


class PipelineStep(ABC):
    """
    This abstract base class unites all classes able to transform some data in a pre-defined way.
    All subclasses are dynamically collected in the class variable :attr:`_registry`.

    The initializer will set some parameters of the transformation, and then the
    :meth:`process` method is used to transform an input Data object, returning a copy.


    """

    _registry: ClassVar[Dict[str, Type]] = {}
    """Register of all subclasses."""

    def __init_subclass__(cls, **kwargs):
        """Registers every subclass under the class variable :attr:`_registry`"""
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    def __init__(self, **kwargs):
        pass

    @classmethod
    @property
    def name(cls) -> str:
        return cls.__name__

    def check(self, _) -> Tuple[bool, str]:
        """Test piece of data for certain properties before computing analysis.

        Returns:
            True if the passed data is eligible.
            Error message in case the passed data is not eligible.
        """
        return True, ""

    def filename_factory(self):
        return self.name

    def process(self, data: Data) -> Data:
        """
        Perform a transformation on an input Data object. This should never alter the
        Data or its properties in place, instead returning a copy or view of the input.

        Args:
            data: The data to be transformed. This should not be altered in place.

        Returns:
            A copy of the input Data, potentially transformed in some way defined by this PipelineStep.
        """
        return data

    @overload
    def process_data(self, data: Data) -> Data:
        ...

    @overload
    def process_data(self, data: Iterable[Data]) -> List[Data]:
        ...

    def process_data(
        self, data: Union[Data, Iterable[Data]]
    ) -> Union[Data, List[Data]]:
        """Same as process(), with the difference that an Iterable is accepted."""
        if isinstance(data, Data):
            return self.process(data)
        return [self.process(d) for d in data]
