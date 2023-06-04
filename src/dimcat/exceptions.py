class DimcatError(Exception):
    default_message: str = "Something went wrong."

    def __init__(self, *args, **kwargs):
        """Following frnkstns solution via https://stackoverflow.com/a/49224771"""
        if not args:
            args = (self.default_message,)

        # Call super constructor
        super().__init__(*args, **kwargs)


class EmptyDatasetError(DimcatError):
    default_message = "The dataset is empty."


class EmptyResourceError(DimcatError):
    default_message = "The resource is empty."


class EmptyPackageError(DimcatError):
    default_message = "The package is empty."


class FeatureUnavailableError(DimcatError):
    default_message = "A required feature is not available."


class NoFeaturesSelectedError(DimcatError):
    default_message = "No features selected."


class BasePathNotDefinedError(DimcatError):
    default_message = "The basepath is not defined."
