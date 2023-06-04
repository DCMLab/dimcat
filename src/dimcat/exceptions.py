import logging
from typing import Callable, ClassVar, Dict, Optional

logger = logging.getLogger(__name__)


class DimcatError(Exception):
    nargs2message: ClassVar[Dict[str, str | Callable]] = {0: "Something went wrong."}
    """Mapping the number of arguments passed to the Error to a lambda string constructor accepting that number of
    arguments and generates the error message."""

    def __init__(self, *args, message: Optional[str] = None, **kwargs):
        """Constructor for DimcatError. Pass message="My custom message" to override the default message.

        Args:
            *args:
                If the number of args matches a string constructor in self.nargs2message, that constructor is called.
                Otherwise, a single argument is used as custom message. If no args a given, the default message is used.
            message: Pass message="My custom message" to override the default message and mechanism based on args.
            **kwargs:
        """
        if message is not None:
            msg = message
        else:
            nargs = len(args)
            if nargs in self.nargs2message:
                msg = self.nargs2message[nargs]
                if not isinstance(msg, str):
                    msg = msg(*args)
            else:
                msg = self.nargs2message[0]
        args = (msg,)

        # Call super constructor
        super().__init__(*args)


class BasePathNotDefinedError(DimcatError):
    nargs2message = {
        0: "The base path is not defined.",
    }


class EmptyDatasetError(DimcatError):
    """optional args: (dataset_name,)"""

    nargs2message = {
        0: "The dataset is empty.",
        1: lambda name: f"Dataset {name!r} is empty.",
    }


class EmptyPackageError(DimcatError):
    """optional args: (package_name,)"""

    nargs2message = {
        0: "The package is empty.",
        1: lambda name: f"Package {name!r} is empty.",
    }


class EmptyResourceError(DimcatError):
    """optional args: (resource_name,)"""

    nargs2message = {
        0: "The resource is empty.",
        1: lambda name: f"Resource {name!r} is empty.",
    }


class FeatureUnavailableError(DimcatError):
    """optional args: (feature_name,)"""

    nargs2message = {
        0: "A required feature is not available.",
        1: lambda name: f"Feature {name!r} is not available.",
    }


class NoFeaturesSelectedError(DimcatError):
    nargs2message = {
        0: "No features have been selected.",
    }


class ResourceNotFoundError(DimcatError):
    """optional args: (resource_name, package_name)"""

    nargs2message = {
        0: "Resource not found.",
        1: lambda name: f"Resource {name!r} not found.",
        2: lambda name, package: f"Resource {name!r} not found in {package!r}.",
    }
