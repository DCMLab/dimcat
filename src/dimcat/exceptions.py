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
    """No optional args."""

    nargs2message = {
        0: "The base path is not defined.",
    }


class DuplicateIDError(DimcatError):
    """optional args: (id, facet)"""

    nargs2message = {
        0: "An ID was already in use.",
        1: lambda id: f"The ID {id!r} is already in use.",
        2: lambda id, facet: f"The ID {id!r} is already in use for facet {facet!r}.",
    }


class DuplicatePackageNameError(DimcatError):
    """optional args: (package_name,)"""

    nargs2message = {
        0: "A package with the same name already exists.",
        1: lambda name: f"A package with the name {name!r} already exists.",
    }


class EmptyCatalogError(DimcatError):
    nargs2message = {
        0: "The catalog is empty.",
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


class ExcludedFileExtensionError(DimcatError):
    """optional args: (extension, permissible_extensions)"""

    nargs2message = {
        0: "A file extension is excluded.",
        1: lambda extension: f"File extension {extension!r} is excluded.",
        2: lambda extension, permissible_extensions: f"File extension {extension!r} is excluded. "
        f"Pass one of {permissible_extensions!r}.",
    }


class NoMuseScoreExecutableSpecifiedError(DimcatError):
    """No optional args."""

    nargs2message = {
        0: "No MuseScore executable specified.",
    }


class NoPathsSpecifiedError(DimcatError):
    """No optional args."""

    nargs2message = {
        0: "No valid paths have been specified.",
    }


class ResourceNotProcessableError(DimcatError):
    """optional args: (feature_name,)"""

    nargs2message = {
        0: "Cannot process this feature.",
        1: lambda name: f"Cannot process {name!r}.",
        2: lambda name, step: f"{step!r} cannot process feature {name!r}.",
    }


class FeatureUnavailableError(DimcatError):
    """optional args: (feature_name,)"""

    nargs2message = {
        0: "A required feature is not available.",
        1: lambda name: f"Feature {name!r} is not available.",
    }


class NoFeaturesActiveError(DimcatError):
    """No optional args."""

    nargs2message = {
        0: "No features are currently active and none have been requested. Apply a FeatureExtractor first or "
        "pass specs for at least one feature to be extracted.",
    }


class NoMatchingResourceFoundError(DimcatError):
    """optional args: (config,)"""

    nargs2message = {
        0: "No matching resource found.",
        1: lambda config: f"No matching resource found for {config!r}.",
    }


class PackageNotFoundError(DimcatError):
    """optional args: (package_name,)"""

    nargs2message = {
        0: "Package not found.",
        1: lambda name: f"Package {name!r} not found.",
    }


class ResourceNotFoundError(DimcatError):
    """optional args: (resource_name, package_name)"""

    nargs2message = {
        0: "Resource not found.",
        1: lambda name: f"Resource {name!r} not found.",
        2: lambda name, package: f"Resource {name!r} not found in {package!r}.",
    }
