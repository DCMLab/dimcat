import logging
from pprint import pformat
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
        0: "The basepath is not defined.",
    }


class BaseFilePathMismatchError(DimcatError):
    """optional args: (basepath, filepath)"""

    nargs2message = {
        0: "The (relative) filepath needs to be located beneath the basepath, not above or next to it.",
        1: lambda bp: f"The (relative) filepath needs to be located beneath the basepath {bp!r}, not above or "
        f"next to it.",
        2: lambda bp, fp: f"The (relative) filepath {fp!r} needs to be located beneath the basepath {bp!r}, not above "
        f"or next to it.",
    }


class DataframeIsMissingExpectedColumnsError(DimcatError):
    """optional args: (missing_columns, present_columns)

    Different from ResourceIsMissingFeatureColumnError in that it can be raised by a function that has access only to
    the dataframe, not the resource.
    """

    nargs2message = {
        0: "The dataframe is missing an expected column.",
        1: lambda missing_columns: f"The dataframe is missing the expected column(s) {missing_columns!r}.",
        2: lambda missing_columns, present_columns: f"The dataframe is missing the expected column(s) "
        f"{missing_columns!r}:\n{present_columns!r}.",
    }


class DataframeIncompatibleWithColumnSchemaError(DimcatError):
    """optional args: (resource_name, validation_error, schema_field_names, df_column_names)"""

    nargs2message = {
        0: "The dataframe is incompatible with the column schema.",
        1: lambda name: f"The dataframe is incompatible with the column schema of {name!r}.",
        2: lambda name, validation_error: f"The dataframe is incompatible with the column schema of {name!r}:\n"
        f"{validation_error!r}.",
        3: lambda name, validation_error, schema_field_names: f"The dataframe is incompatible with the column schema "
        f"of {name!r} which specifies the fields {schema_field_names!r}:\n{validation_error!r}.",
        4: lambda name, validation_error, schema_field_names, df_column_names: f"The dataframe with the columns "
        f"{df_column_names!r} is incompatible with the column schema of {name!r} which specifies the fields "
        f"{schema_field_names!r}:\n{validation_error!r}.",
    }


class DatasetNotProcessableError(DimcatError):
    """optional args: (missing,)"""

    nargs2message = {
        0: "Cannot process this Dataset.",
        1: lambda missing: f"Cannot process this Dataset: missing {missing!r}.",
    }


class DuplicateIDError(DimcatError):
    """optional args: (id, facet)"""

    nargs2message = {
        0: "An ID was already in use.",
        1: lambda id: f"The ID {id!r} is already in use.",
        2: lambda id, facet: f"The ID {id!r} is already in use for facet {facet!r}.",
    }


class DuplicateResourceIDsError(DimcatError):
    """optional args: (id_counter,)"""

    nargs2message = {
        0: "Resource IDs are not unique.",
        1: lambda id_counter: f"Several resources have the same ID: {id_counter!r}.",
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


class FeatureIsMissingFormatColumnError(DimcatError):
    """optional args: (feature_name, missing_column(s), format, feature_type)"""

    nargs2message = {
        0: "The feature is missing the column corresponding to this format.",
        1: lambda name: f"Feature {name!r} is missing the column corresponding to this format.",
        2: lambda name, missing_columns: f"Feature {name!r} is missing the column(s) {missing_columns!r} "
        f"corresponding to this format.",
        3: lambda name, missing_columns, format: f"Feature {name!r} is missing the column(s) {missing_columns!r} "
        f"corresponding to {format!r}.",
        4: lambda name, missing_columns, format, feature_type: f"Feature {name!r} of type {feature_type!r} is missing "
        f"the column(s) {missing_columns!r} corresponding to {format!r}.",
    }


class FeatureWithUndefinedValueColumnError(DimcatError):
    """optional args: (feature_name, feature_type"""

    nargs2message = {
        0: "No value_column is defined for this feature.",
        1: lambda name: f"No value_column is defined for feature {name!r}.",
        2: lambda name, feature_type: f"No value_column is defined for feature {name!r} of type {feature_type!r}.",
    }


class FeatureUnavailableError(DimcatError):
    """optional args: (feature_name, getting_from_name)"""

    nargs2message = {
        0: "A required feature is not available.",
        1: lambda name: f"Feature {name!r} is not available.",
        2: lambda name, getting_from_name: f"{getting_from_name!r} does not bring forth the Feature {name!r}. ",
    }


class FilePathNotDefinedError(DimcatError):
    """No optional args."""

    nargs2message = {
        0: "No filepath has been defined.",
    }


class GrouperNotSetUpError(DimcatError):
    """optional args: (grouper_name,)"""

    nargs2message = {
        0: "The grouper has not been setup. Applying it would result in empty features. Set the attribute "
        "'grouped_units'.",
        1: lambda name: f"The {name!r} has not been setup. Applying it would result in empty features. "
        f"Set the attribute 'grouped_units'.",
    }


class InvalidResourcePathError(DimcatError):
    """optional args: (filepath, basepath)"""

    nargs2message = {
        0: "The resource path is invalid.",
        1: lambda filepath: f"The resource path {filepath!r} is invalid.",
        2: lambda filepath, basepath: f"In combination with the basepath {basepath!r}, the resource path {filepath!r} "
        f"is invalid.",
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


class NoFeaturesActiveError(DimcatError):
    """No optional args."""

    nargs2message = {
        0: "No features are currently active and none have been requested. Apply a FeatureExtractor first or "
        "pass specs for at least one feature to be extracted.",
    }


class NoMatchingResourceFoundError(DimcatError):
    """optional args: (config, package_name)"""

    nargs2message = {
        0: "No matching resource found.",
        1: lambda config: f"No matching resource found for {config!r}.",
        2: lambda config, package_name: f"Found no resource in in {package_name!r} that matches {config!r}.",
    }


class PackageDescriptorHasWrongTypeError(DimcatError):
    """optional args: (expected_type, actual_type, name)"""

    nargs2message = {
        0: "The package descriptor resolves to the wrong type.",
        1: lambda expected_type: f"The package descriptor resolves to the wrong type. Expected {expected_type!r}.",
        2: lambda expected_type, actual_type: f"The package descriptor resolves to the wrong type. Expected "
        f"{expected_type!r}, got {actual_type!r}.",
        3: lambda expected_type, actual_type, name: f"The package descriptor for {name!r} resolves to the wrong type. "
        f"Expected {expected_type!r}, got {actual_type!r}.",
    }


class PackageNotFoundError(DimcatError):
    """optional args: (package_name,)"""

    nargs2message = {
        0: "Package not found.",
        1: lambda name: f"Package {name!r} not found.",
    }


class PackageNotFullySerializedError(DimcatError):
    """optional args: (error_message,)"""

    nargs2message = {
        0: "All resources contained in the package have not been serialized.",
        1: lambda error_message: error_message,
    }


class PackageInconsistentlySerializedError(DimcatError):
    """optional args: (package_name, existing_path, missing_path)"""

    nargs2message = {
        0: "The package has been serialized in an inconsistent way, found only ZIP or descriptor, not both.",
        1: lambda package_name: f"The package {package_name!r} has been serialized in an inconsistent way, found "
        f"only ZIP or descriptor, not both.",
        2: lambda package_name, existing_path: f"The package {package_name!r} has been serialized in an "
        f"inconsistent way, expected ZIP and descriptor, found only {existing_path!r}.",
        3: lambda package_name, existing_path, missing_path: f"The package {package_name!r} has been serialized in an "
        f"inconsistent way, expected ZIP and descriptor, found only {existing_path!r} but not {missing_path!r}.",
    }


class PackagePathsNotAlignedError(DimcatError):
    """optional args: (error_message,)"""

    nargs2message = {
        0: "Package paths are not aligned.",
        1: lambda error_message: error_message,
    }


class PotentiallyUnrelatedDescriptorError(DimcatError):
    """optional args: (name, path)"""

    nargs2message = {
        0: "There is a potentially unrelated descriptor on disk. You can load it via .from_descriptor_path().",
        1: lambda name: f"There is a potentially unrelated descriptor for {name!r} on disk. You can load it via "
        f".from_descriptor_path().",
        2: lambda name, path: f"There is a potentially unrelated descriptor for {name!r} on disk. You can "
        f"load it via .from_descriptor_path({path!r}).",
    }


class ResourceAlreadyTransformed(DimcatError):
    """optional args: (name, processor)"""

    nargs2message = {
        0: "Resource has already been processed.",
        1: lambda name: f"Resource {name!r} has already been processed.",
        2: lambda name, processor: f"Resource {name!r} has already been processed by a {processor!r}.",
    }


class ResourceDescriptorHasWrongTypeError(DimcatError):
    """optional args: (expected_type, actual_type, name)"""

    nargs2message = {
        0: "The resource descriptor resolves to the wrong type.",
        1: lambda expected_type: f"The resource descriptor resolves to the wrong type. Expected {expected_type!r}.",
        2: lambda expected_type, actual_type: f"The resource descriptor resolves to the wrong type. Expected "
        f"{expected_type!r}, got {actual_type!r}.",
        3: lambda expected_type, actual_type, name: f"The resource descriptor for {name!r} resolves to the wrong type. "
        f"Expected {expected_type!r}, got {actual_type!r}.",
    }


class ResourceIsMissingCorpusIndexError(DimcatError):
    """optional args: (resource_name, name_of_missing)"""

    nargs2message = {
        0: "The resource is missing a corpus index level.",
        1: lambda name: f"Resource {name!r} is missing a corpus index level.",
        2: lambda name, name_of_missing: f"Resource {name!r} is missing a corpus index level, "
        f"a column named {name_of_missing!r} could not be detected.",
    }


class ResourceIsMissingPieceIndexError(DimcatError):
    """optional args: (resource_name, name_of_missing)"""

    nargs2message = {
        0: "The resource is missing a piece index level.",
        1: lambda name: f"Resource {name!r} is missing a piece index level.",
        2: lambda name, name_of_missing: f"Resource {name!r} is missing a piece index level, "
        f"a column named {name_of_missing!r} could not be detected.",
    }


class ResourceIsFrozenError(DimcatError):
    """optional args: (resource_name, current_basepath, new_basepath)"""

    nargs2message = {
        0: "Resource is frozen, i.e. tied to data stored on disk, so you would need to copy it for the relative paths "
        "ro remain valid.",
        1: lambda name: f"Resource {name!r} is frozen, i.e. tied to data stored on disk, so you would need to copy "
        f"it for the relative paths to remain valid.",
        2: lambda name, current_basepath: f"Resource {name!r} is frozen, i.e. tied to data stored on disk at basepath "
        f"{current_basepath!r}, so you would need to copy it for the relative paths to remain valid.",
        3: lambda name, current_basepath, new_basepath: f"Resource {name!r} is frozen, i.e. tied to data stored on "
        f"disk at basepath {current_basepath!r}. Changing it to {new_basepath!r} would invalidate the "
        f"relative paths. Consider using copy_to_new_location().",
    }


class ResourceIsMisalignedError(ResourceIsFrozenError):
    nargs2message = {
        0: "Package prevents adding resources that cannot be aligned with it without copying them. "
        "Consider using one of the subtypes such as PathPackage or DimcatPackage.",
        1: lambda misaligned_path: f"{misaligned_path!r} is not aligned with the Package and it prevents adding "
        f"misaligned resources. Consider using one of the subtypes such as PathPackage or DimcatPackage.",
        2: lambda misaligned_path, target_path: f"{misaligned_path!r} cannot be aligned with {target_path!r} "
        f"and the Package prevents adding misaligned resources. Consider using one of the subtypes such as "
        f"PathPackage or DimcatPackage.",
        3: lambda misaligned_path, target_path, package_type: f"{misaligned_path!r} cannot be aligned with "
        f"{target_path!r} and the {package_type} prevents adding misaligned resources. Consider using one of "
        f"the subtypes such as PathPackage or DimcatPackage.",
    }


class ResourceIsMissingFeatureColumnError(DimcatError):
    """optional args: (resource_name, missing_column(s), feature_name)"""

    nargs2message = {
        0: "The resource is missing a feature column.",
        1: lambda name: f"Resource {name!r} is missing a feature column.",
        2: lambda name, missing_columns: f"Resource {name!r} is missing the feature column(s) {missing_columns!r}.",
        3: lambda name, missing_columns, feature_name: f"Resource {name!r} is missing the feature column(s) "
        f"{missing_columns!r} for feature {feature_name!r}.",
    }


class ResourceIsPackagedError(ResourceIsFrozenError):
    """optional args: (name, new_path, path_type)"""

    nargs2message = {
        0: "The resource is packaged can cannot store its own descriptor.",
        1: lambda name: f"Resource {name!r} is packaged and its paths cannot be modified.",
        2: lambda name, new_path: f"Resource {name!r} is packaged so {new_path!r} cannot be set.",
        3: lambda name, new_path, path_type: f"Resource {name!r} is packaged so {new_path!r} cannot be set as the new "
        f"{path_type!r}.",
    }


class ResourceNamesNonUniqueError(DimcatError):
    """optional args: (names_or_paths,)"""

    nargs2message = {
        0: "The resulting resource names are not unique.",
        1: lambda names_or_paths: f"Resulting resource names are not unique: {pformat(names_or_paths)}.",
    }


class ResourceNeedsToBeCopiedError(DimcatError):
    """optional args: (path_type, new_path,)"""

    nargs2message = {
        0: "Resource would need copying.",
        1: lambda path_type: f"Cannot set the new {path_type} without copying the resource. Consider using "
        f"copy_to_new_location().",
        2: lambda path_type, new_path: f"Cannot set the new {path_type} to {new_path!r} without copying the "
        f"resource.",
    }


class ResourceNotFoundError(DimcatError):
    """optional args: (resource_name, package_name)"""

    nargs2message = {
        0: "Resource not found.",
        1: lambda name: f"Resource {name!r} not found.",
        2: lambda name, package: f"Resource {name!r} not found in {package!r}.",
    }


class ResourceNotProcessableError(DimcatError):
    """optional args: (resource_name, pipeline_step, resource_type)"""

    nargs2message = {
        0: "Cannot process this Resource.",
        1: lambda name: f"Cannot process {name!r}.",
        2: lambda name, step: f"{step!r} cannot process Resource {name!r}.",
        3: lambda name, step, resource_type: f"{step!r} cannot process Resource {name!r} of type {resource_type!r}.",
    }


class SlicerNotSetUpError(DimcatError):
    """optional args: (slicer_name,)"""

    nargs2message = {
        0: "The slicer has not been setup. Applying it would result in empty features. Set the attribute "
        "'slice_intervals'.",
        1: lambda name: f"The {name!r} has not been setup. Applying it would result in empty features. "
        f"Set the attribute 'slice_intervals'.",
    }
