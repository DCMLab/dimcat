from __future__ import annotations

import inspect
import logging
import os
import shutil
import warnings
import zipfile
from enum import IntEnum, auto
from pathlib import Path
from pprint import pformat
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
)

import frictionless as fl
import marshmallow as mm
import pandas as pd
from dimcat.base import (
    ObjectEnum,
    get_class,
    get_schema,
    get_setting,
    is_default_descriptor_path,
    is_subclass_of,
)
from dimcat.data.base import Data
from dimcat.data.utils import (
    check_descriptor_filename_argument,
    check_rel_path,
    is_default_package_descriptor_path,
    is_default_resource_descriptor_path,
    make_fl_resource,
    make_rel_path,
    store_as_json_or_yaml,
    warn_about_potentially_unrelated_descriptor,
)
from dimcat.dc_exceptions import (
    BaseFilePathMismatchError,
    BasePathNotDefinedError,
    FilePathNotDefinedError,
    InvalidResourcePathError,
    ResourceDescriptorHasWrongTypeError,
    ResourceIsFrozenError,
    ResourceIsPackagedError,
    ResourceNeedsToBeCopiedError,
)
from dimcat.dc_warnings import PotentiallyUnrelatedDescriptorUserWarning
from dimcat.utils import (
    make_valid_frictionless_name,
    make_valid_frictionless_name_from_filepath,
    replace_ext,
    resolve_path,
    treat_basepath_argument,
)
from typing_extensions import Self

try:
    import modin.pandas as mpd

    SomeDataframe: TypeAlias = Union[pd.DataFrame, mpd.DataFrame]
    SomeSeries: TypeAlias = Union[pd.Series, mpd.Series]
    SomeIndex: TypeAlias = Union[pd.MultiIndex, mpd.MultiIndex]
except ImportError:
    # DiMCAT has not been installed via dimcat[modin], hence the dependency is missing
    SomeDataframe: TypeAlias = pd.DataFrame
    SomeSeries: TypeAlias = pd.Series
    SomeIndex: TypeAlias = pd.MultiIndex

if TYPE_CHECKING:
    from .dc import DimcatResource, Feature, Result

logger = logging.getLogger(__name__)
resource_status_logger = logging.getLogger("dimcat.data.resources.ResourceStatus")

D = TypeVar("D", bound=SomeDataframe)
DR = TypeVar("DR", bound="DimcatResource")
F = TypeVar("F", bound="Feature")
R = TypeVar("R", bound="Resource")
Rs = TypeVar("Rs", bound="Result")
S = TypeVar("S", bound=SomeSeries)
IX = TypeVar("IX", bound=SomeIndex)

# region Resource


def reconcile_base_and_file(
    basepath: Optional[str],
    filepath: str,
) -> Tuple[str, str]:
    """

    Args:
        basepath:
        filepath:

    Returns:
        The result is a tuple of an absolute basepath and a relative filepath.if
    """
    assert filepath is not None, "filepath must not be None"
    if not basepath:
        if os.path.isabs(filepath):
            base, file = os.path.split(filepath)
        else:
            base, file = os.getcwd(), filepath
    else:
        if os.path.isabs(filepath):
            base = basepath
            file = make_rel_path(filepath, basepath)
        else:
            base = basepath
            file = filepath
    return resolve_path(base), file


class ResourceStatus(IntEnum):
    """Expresses the status of a class:`Resource` with respect to it being described, valid, and serialized to disk,
    with or without its descriptor file. The enum members have increasing integer values starting with EMPTY == 0.
    Statuses > PATH_ONLY (1) are currently only relevant for DimcatResources. The current status is determined
    by the boolean state of the first three attributes in the table below:

    * is_serialized: True if the resource can be located physically on disk.
    * descriptor_exists: True if a descriptor file (JSON/YAML) is physically present on disk.
    * is_loaded: True if the resource is currently loaded into memory.

    The remaining attributes are derived from the first three and are not used to determine the current status:

    * assumed valid: True if the resource is assumed to be valid, which is the case for all serialized resources.
    * standalone: True if the resource is not part of a package. For "free" (not serialized) resources, it depends
      on the value :attr:`Resource.descriptor_filename` (whether it corresponds to a package or resource descriptor).
    * empty: True if the resource is empty, i.e. it does not data. A DimcatResource that is PATH_ONLY is considered
      empty, whereas a Resource/PathResource is not (they only have status 0 or 1).

    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | ResourceStatus        | is_serialized | descriptor_exists | is_loaded | assumed valid | standalone | empty |
    +=======================+===============+===================+===========+===============+============+=======+
    | EMPTY                 | False         | ?                 | False     |       no      |      ?     |  yes  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | PATH_ONLY             | True          | ?                 | False     |       no      |      ?     |  yes  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | SCHEMA_ONLY           | False         | ?                 | False     |       no      |      ?     |  yes  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | DATAFRAME             | False         | False             | True      |       no      |      ?     |   no  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | VALIDATED             | False         | False             | True      |   guaranteed  |      ?     |   no  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | SERIALIZED            | True          | False             | True      |      yes      |     yes    |   no  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | STANDALONE_LOADED     | True          | True              | True      |      yes      |     yes    |   no  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | PACKAGED_LOADED       | True          | True              | True      |      yes      |     no     |   no  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | STANDALONE_NOT_LOADED | True          | True              | False     |      yes      |     yes    |   no  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | PACKAGED_NOT_LOADED   | True          | True              | False     |      yes      |     no     |   no  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+

    The status of a resource is set at the end of :meth:`Resource.__init__` by
    calling :meth:`Resource._update_status` which, in return calls :meth:`Resource._get_status`.
    """

    EMPTY = 0
    PATH_ONLY = auto()  # only path exists (default in a PathResource)
    SCHEMA_ONLY = auto()  # column_schema available but no dataframe has been loaded
    DATAFRAME = (
        auto()
    )  # dataframe available in memory but not yet validated against the column_schema
    VALIDATED = auto()  # validated dataframe available in memory
    SERIALIZED = (
        auto()
    )  # dataframe serialized to disk but not its descriptor (shouldn't happen) -> can be changed or overwritten
    STANDALONE_LOADED = auto()
    PACKAGED_LOADED = auto()
    STANDALONE_NOT_LOADED = auto()
    PACKAGED_NOT_LOADED = auto()


class ResourceSchema(Data.Schema):
    """Since Resource objects function partially as a wrapper around a frictionless.Resource object, many properties are
    serialized by the means of the frictionless descriptor corresponding to it, which is provided by the
    frictionless library.
    For example, :attr:`resource_name` uses ``.resource.name`` under the hood.
    """

    # class Meta:
    #     ordered = True
    #     unknown = mm.INCLUDE # unknown fields allowed because frictionless descriptors can come with custom metadata

    descriptor_filename = mm.fields.String(allow_none=True, metadata={"expose": False})
    resource = mm.fields.Method(
        serialize="get_frictionless_descriptor",
        deserialize="raw",
        metadata={"expose": False},
    )

    def get_frictionless_descriptor(self, obj: Resource) -> dict:
        descriptor = obj._resource.to_dict()
        return descriptor

    def raw(self, data):
        """Functions as 'deserialize' method for the Schema field 'resource'."""
        return data

    @mm.pre_load
    def unsquash_data_if_necessary(self, data, **kwargs):
        """Data serialized with this schema usually has 'resource' field that contains the frictionless descriptor.
        However, if it has been serialized with the PickleSchema variant, this descriptor has become the top level
        and all other fields have been squashed into it, effectively flattening the dictionary. This method
        reverses this flattening, if necessary.
        """
        if "resource" in data:
            return data
        if isinstance(data, fl.Resource):
            fl_resource = data
        elif "name" not in data:
            # probably manually compiled data
            return data
        else:
            fl_resource = fl.Resource.from_descriptor(data)
        unsquashed_data = {}
        for field_name in self.declared_fields.keys():
            # the frictionless.Resource carries all unknown keys in the 'custom' dictionary
            # we take out those that belong to the schema and leave the rest, which is arbitrary metadata
            field_value = fl_resource.custom.pop(field_name, None)
            if field_value is not None:
                unsquashed_data[field_name] = field_value
        unsquashed_data["resource"] = fl_resource
        return unsquashed_data

    @mm.post_load
    def init_object(self, data, **kwargs):
        if data.get("resource") is not None:
            if not isinstance(data["resource"], fl.Resource):
                data["resource"] = fl.Resource.from_descriptor(data["resource"])
            if "dtype" not in data:
                if data["resource"].schema.fields:
                    data["dtype"] = "DimcatResource"
                else:
                    data["dtype"] = "Resource"
        elif "dtype" not in data:
            # probably manually compiled data
            data["dtype"] = "Resource"
        return super().init_object(data, **kwargs)


class Resource(Data):
    """A Resource is essentially a wrapper around a :obj:`frictionless.Resource` object. Initializing a Resource object
    from a descriptor dispatches to the appropriate subclass, depending on the specified dtype or, if absent,
    to a :class:`DimcatResource` for tabular data and to a :class:`PathResource` for any other.
    """

    @classmethod
    def from_descriptor(
        cls,
        descriptor: dict | fl.Resource,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        **kwargs,
    ) -> Self:
        """Create a Resource from a frictionless descriptor dictionary.

        Args:
            descriptor: Descriptor corresponding to a frictionless resource descriptor.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            basepath:
                Where the file would be serialized and, important for an existing resource, the path against which the
                descriptor's 'filepath' property can be resolved.
            **kwargs: Subclasses can use this method.

        Raises:
            TypeError: If the descriptor is a string or a Path, not a dictionary or a frictionless Resource.
            ResourceDescriptorHasWrongTypeError:
                If the descriptor belongs to a type that is not a subclass of the Resource class to be initialized.

        Returns:

        """
        if isinstance(descriptor, (str, Path)):
            raise TypeError(
                f"This method expects a descriptor dictionary. In order to create a "
                f"{cls.name} from a path, use {cls.__name__}.from_descriptor_path() instead."
            )
        if isinstance(descriptor, fl.Resource):
            fl_resource = descriptor
        else:
            fl_resource = fl.Resource.from_descriptor(descriptor)
        if cls.name == "Resource":
            # dispatch to suitable subclass based on the properties of the descriptor
            if fl_resource.type == "table":
                # descriptor contains tabular data, dispatch to the appropriate DimcatResource type based on the
                # presence or absence of 'dtype' in the descriptor
                descriptor = dict(
                    fl_resource.to_dict(),
                    descriptor_filename=descriptor_filename,
                    basepath=basepath,
                    **kwargs,
                )
                dtype = fl_resource.custom.get("dtype", "DimcatResource")
                Constructor = get_class(dtype)
                try:
                    return Constructor.schema.load(descriptor)
                except mm.ValidationError as e:
                    raise mm.ValidationError(
                        f"Deserializing the descriptor {descriptor!r} with {Constructor.name}.schema failed with \n{e}."
                    ) from e
            # base object to be initialized from a descriptor of non-tabular data, dispatch to PathResource
            return PathResource.from_descriptor(
                descriptor=fl_resource,
                descriptor_filename=descriptor_filename,
                basepath=basepath,
                **kwargs,
            )
        if (dtype := fl_resource.custom.get("dtype")) and not is_subclass_of(
            dtype, cls
        ):
            raise ResourceDescriptorHasWrongTypeError(cls.name, dtype, fl_resource.name)
        # initialize the subclass from the frictionless Resource
        return cls(
            resource=fl_resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            **kwargs,
        )

    @classmethod
    def from_descriptor_path(
        cls,
        descriptor_path: str,
        **kwargs,
    ) -> Self:
        """Create a Resource from a frictionless descriptor file on disk.

        Args:
            descriptor_path: Absolute path where the JSON/YAML descriptor is located.
            basepath:
                If you do not want the folder where the descriptor is located to be treated as basepath,
                you may specify an absolute path higher up within the ``descriptor_path`` to serve as base.
                The resource's filepath will be adapated accordingly, whereas the resource names
                specified in the descriptor will remain the same.
            **kwargs: Subclasses can use this method.
        """
        basepath, descriptor_filename = os.path.split(descriptor_path)
        basepath = resolve_path(basepath)  # could be relative
        if "basepath" in kwargs:
            kw_basepath = kwargs.pop("basepath")
            if kw_basepath != basepath:
                raise ValueError(
                    f"{cls.name}.from_descriptor_path() does not allow for specifying a basepath differnt "
                    f"from the one defined by the descriptor_path."
                )
        fl_resource = fl.Resource.from_descriptor(descriptor_path)
        fl_resource.path = make_rel_path(
            fl_resource.normpath, basepath
        )  # adapt the relative path to the basepath
        return cls.from_descriptor(
            descriptor=fl_resource.to_dict(),
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            **kwargs,
        )

    @classmethod
    def from_filepath(
        cls,
        filepath: str,
        resource_name: Optional[str] = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        **kwargs,
    ) -> Self:
        """Create a Resource from a file on disk, be it a JSON/YAML resource descriptor, or a simple path resource.

        Args:
            filepath: Path pointing to a resource descriptor or a simple path resource.
            resource_name:
                Name of the resource used for retrieving it from a DimcatPackage and as filename when the resource
                is stored to a ZIP file.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            auto_validate:
                By default, the Resource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).
            basepath:
                Basepath to use for the resource. If None, the folder of the ``filepath`` is used.
        """
        if is_default_resource_descriptor_path(filepath):
            return cls.from_descriptor_path(descriptor_path=filepath, **kwargs)
        return cls.from_resource_path(
            resource_path=filepath,
            resource_name=resource_name,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            **kwargs,
        )

    @classmethod
    def from_resource(
        cls,
        resource: Resource,
        descriptor_filename: Optional[str] = None,
        resource_name: Optional[str] = None,
        basepath: Optional[str] = None,
        **kwargs,
    ):
        """Create a Resource from an existing :obj:`Resource`, specifying its name and,
        optionally, at what path it is to be serialized.

        Args:
            resource: An existing :obj:`frictionless.Resource` or a filepath.
            resource_name:
                Name of the resource used for retrieving it from a DimcatPackage and as filename when the resource
                is stored to a ZIP file.
            basepath:
                Lets you change the basepath of the existing resource.
            **kwargs: Subclasses can use this method.
        """
        if not isinstance(resource, Resource):
            raise TypeError(f"Expected a Resource, got {type(resource)!r}.")
        fl_resource = resource.resource
        if fl_resource.path is None:
            # needed because otherwise frictionless complains when asked to make a copy
            fl_resource.path = ""
        new_fl_resource = fl_resource.to_copy()
        resource_kwargs = {
            arg: getattr(resource, arg)
            for arg in resource.schema.fields
            if arg not in ("dtype", "resource")
        }
        if descriptor_filename is not None:
            kwargs["descriptor_filename"] = descriptor_filename
        if basepath is not None:
            kwargs["basepath"] = basepath
        resource_kwargs.update(
            {arg: val for arg, val in kwargs.items() if val is not None}
        )
        new_object = cls(
            resource=new_fl_resource,
            **resource_kwargs,
        )
        if resource_name:
            new_object.resource_name = resource_name
        new_object._corpus_name = resource._corpus_name
        return new_object

    @classmethod
    def from_resource_path(
        cls,
        resource_path: str,
        resource_name: Optional[str] = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        **kwargs,
    ) -> Self:
        """Create a Resource from a file on disk, treating it just as a path even if it's a
        JSON/YAML resource descriptor"""
        if is_default_descriptor_path(resource_path):
            warnings.warn(
                f"You have passed the descriptor path {resource_path!r} to {cls.name}.from_resource_path()"
                f" meaning that the descriptor file itself will be treated like a resource (which could get its own "
                f"descriptor) rather than as the resource it describes. To avoid this warning, instead of Resource use "
                f"either PathResource (to treat the descriptor as a mere path) or DimcatResource (to evaluate the "
                f"descriptor and access the data it describes).",
                SyntaxWarning,
            )
        basepath, resource_path = reconcile_base_and_file(basepath, resource_path)
        fname, extension = os.path.splitext(resource_path)
        if resource_name:
            resource_name = make_valid_frictionless_name(resource_name)
        else:
            resource_name = make_valid_frictionless_name_from_filepath(resource_path)
        options = dict(
            name=resource_name,
            path=resource_path,
            scheme="file",
            format=extension[1:],
        )
        fl_resource = make_fl_resource(**options)
        return cls(
            resource=fl_resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            **kwargs,
        )

    class PickleSchema(ResourceSchema):
        @mm.post_dump()
        def squash_data_for_frictionless(self, data, **kwargs):
            squashed_data = data.pop("resource")
            obj_basepath, desc_basepath = data.get("basepath"), squashed_data.get(
                "basepath"
            )
            if (obj_basepath and desc_basepath) and obj_basepath != desc_basepath:
                # first, reconcile potential discrepancy between basepaths
                # by default, the fields of the resource descriptor are overwritten by the fields of the resource object
                filepath = squashed_data.get("path")
                if os.path.isfile(
                    (obj_normpath := os.path.join(obj_basepath, filepath))
                ):
                    self.logger.error(
                        f"Giving the object's basepath {obj_basepath!r} precedence over the descriptor's "
                        f"basepath ({desc_basepath!r}) because it exists."
                    )
                elif os.path.isfile(os.path.join(desc_basepath, filepath)):
                    del data["basepath"]
                    self.logger.error(
                        f"Using the descriptor's basepath {desc_basepath!r} because the object's basepath "
                        f"would result to the invalid path {obj_normpath!r}."
                    )
                else:
                    raise FileNotFoundError(
                        f"Neither the object's basepath {obj_basepath!r} nor the descriptor's basepath "
                        f"{desc_basepath!r} contain the file {filepath!r}."
                    )
            squashed_data.update(data)
            # the following corresponds to DimcatObject.Schema.validate_dump()
            dtype_schema = get_schema(squashed_data["dtype"])
            report = dtype_schema.validate(squashed_data)
            if report:
                raise mm.ValidationError(
                    f"Dump of {squashed_data['dtype']} created with a {self.name} could not be validated by "
                    f"{dtype_schema.name}."
                    f"\n\nDUMP:\n{pformat(squashed_data, sort_dicts=False)}"
                    f"\n\nREPORT:\n{pformat(report, sort_dicts=False)}"
                )
            return squashed_data

    class Schema(ResourceSchema, Data.Schema):
        pass

    def __init__(
        self,
        resource: fl.Resource = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        **kwargs,
    ):
        """

        Args:
            resource: An existing :obj:`frictionless.Resource`.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            basepath: Where the file would be serialized.
        """
        self.logger.debug(
            f"""
Resource.__init__(
    resource={type(resource)},
    descriptor_filename={descriptor_filename},
    basepath={basepath},
    **kwargs={kwargs},
)"""
        )
        self._status = ResourceStatus.EMPTY
        self._resource: fl.Resource = self._make_empty_fl_resource()
        self._corpus_name: Optional[str] = None
        self._is_valid: Optional[bool] = None
        is_fl_resource = isinstance(resource, fl.Resource)
        if is_fl_resource and basepath is None:
            basepath = resource.basepath
        super().__init__(basepath=basepath)
        if is_fl_resource:
            if resource.path is None:
                resource.path = ""
            self._resource = resource
        elif resource is None:
            pass
        else:
            raise TypeError(
                f"Expected resource to be a frictionless Resource or a file path, got {type(resource)}."
            )
        if self.basepath:
            self._resource.basepath = self.basepath
        if descriptor_filename:
            self._set_descriptor_filename(descriptor_filename)
        self._update_status()
        self.logger.debug(
            f"""
Resource(
    basepath={self.basepath},
    filepath={self.filepath},
    corpus_name={self.get_corpus_name()},
    resource_name={self.resource_name},
    descriptor_filename={self.descriptor_filename},
)"""
        )

    @property
    def basepath(self) -> str:
        return self._basepath

    @basepath.setter
    def basepath(self, basepath: str):
        self.set_basepath(
            basepath=basepath,
            reconcile=False,
        )

    @property
    def corpus_name(self) -> Optional[str]:
        """The name of the corpus this resource belongs to."""
        return self._corpus_name

    @corpus_name.setter
    def corpus_name(self, corpus_name: str):
        valid_name = make_valid_frictionless_name(corpus_name)
        if valid_name != corpus_name:
            self.logger.info(f"Changed {corpus_name!r} name to {valid_name!r}.")
        self._corpus_name = corpus_name

    @property
    def descriptor_filename(self) -> Optional[str]:
        """The path to the descriptor file on disk, relative to the basepath. If you need to fall back to a default
        value, use :meth:`get_descriptor_filename` instead."""
        return self._resource.metadata_descriptor_path

    @descriptor_filename.setter
    def descriptor_filename(self, descriptor_filename: str):
        self.set_descriptor_filename(descriptor_filename)

    @property
    def descriptor_exists(self) -> bool:
        descriptor_path = self.get_descriptor_path()
        if not descriptor_path:
            return False
        return os.path.isfile(descriptor_path)

    @property
    def filepath(self) -> str:
        return self._resource.path

    @filepath.setter
    def filepath(self, filepath: str):
        if os.path.isabs(filepath):
            raise ValueError(f"Filepath must be relative, got {filepath!r}.")
        self._resource.path = filepath

    @property
    def ID(self) -> Tuple[str, str]:
        """The resource's unique ID."""
        if not self.resource_name:
            raise ValueError("Resource name not set.")
        corpus_name = self.get_corpus_name()
        return (corpus_name, self.resource_name)

    @ID.setter
    def ID(self, ID: Tuple[str, str]):
        self.corpus_name, self.resource_name = ID
        self.logger.debug(f"Resource ID updated to {self.ID!r}.")

    @property
    def innerpath(self) -> Optional[str]:
        """The innerpath is the resource's filepath within a zip file."""
        return self._resource.innerpath

    @innerpath.setter
    def innerpath(self, innerpath: str):
        if os.path.isabs(innerpath):
            raise ValueError(f"Inner filepath must be relative, got {innerpath!r}.")
        self._resource.innerpath = innerpath

    @property
    def is_empty(self) -> bool:
        return self._status == ResourceStatus.EMPTY

    @property
    def is_frozen(self) -> bool:
        """Whether the resource is frozen (i.e. it's pointing to data on the disk) or not."""
        return self.basepath is not None and (
            self.resource_exists or self.descriptor_exists
        )

    @property
    def is_loaded(self) -> bool:
        return False

    @property
    def is_valid(self) -> bool:
        """Returns the result of a previous validation or, if the resource has not been validated
        before, do it now."""
        report = self.validate(raise_exception=False, only_if_necessary=True)
        if report is None:
            return True
        return report.valid

    @property
    def is_packaged(self) -> bool:
        """Returns True if the resource is packaged, i.e. its descriptor_filename is the one of
        the :class:`Package` that the resource is a part of. Also means that the resource is passive.
        """
        result = (
            self.descriptor_filename is not None
        ) and is_default_package_descriptor_path(self.descriptor_filename)
        self.logger.debug(
            f"{self.name} {self.resource_name!r} {'is' if result else 'is not'} packaged because its "
            f"descriptor_filename is {self.descriptor_filename!r}."
        )
        return result

    @property
    def is_serialized(self) -> bool:
        """Returns True if the resource is serialized, i.e. it points to a file on disk and, if it
        is a ZIP file, the :attr:`innerpath` is present in that ZIP file."""
        if not self.resource_exists:
            return False
        if self.is_zipped:
            with zipfile.ZipFile(self.normpath) as zip_file:
                return self.innerpath in zip_file.namelist()
        return True

    @property
    def is_zipped(self) -> bool:
        """Returns True if the filepath points to a .zip file."""
        if not self.filepath:
            return False
        return self.filepath.endswith(".zip")

    @property
    def normpath(self) -> str:
        """Absolute path to the serialized or future tabular file. Raises if basepath is not set."""
        if not self.basepath:
            raise BasePathNotDefinedError
        if not self.filepath:
            raise FilePathNotDefinedError
        return os.path.join(self.basepath, self.filepath)

    @property
    def resource(self) -> fl.Resource:
        return self._resource

    @property
    def resource_exists(self) -> bool:
        """Returns True if the resource's normpath exists on disk.
        If the resource :attr:`is_zipped` and you want to check if the :attr:`innerpath` actually
        exists within the ZIP file, use :attr:`is_serialized` instead."""
        try:
            return os.path.isfile(self.normpath)
        except (BasePathNotDefinedError, FilePathNotDefinedError):
            return False

    @property
    def resource_name(self) -> str:
        return self._resource.name

    @resource_name.setter
    def resource_name(self, resource_name: str):
        valid_name = make_valid_frictionless_name(resource_name)
        if valid_name != resource_name:
            self.logger.info(f"Changed {resource_name!r} name to {valid_name!r}.")
        self._resource.name = resource_name
        if not self._resource.path:
            self._resource.path = self.get_innerpath()

    @property
    def status(self) -> ResourceStatus:
        if self._status == ResourceStatus.EMPTY and self._resource.schema.fields:
            self._status = ResourceStatus.SCHEMA_ONLY
            resource_status_logger.debug(
                f"When requesting the status of {self.resource_name}, a column schema was found and the status "
                f"has been changed from {ResourceStatus.EMPTY!r} to {self.status!r}."
            )
        return self._status

    def copy(self) -> Self:
        """Returns a copy of the resource."""
        return self.from_resource(self)

    def copy_to_new_location(
        self,
        basepath: str,
        overwrite: bool = False,
        filepath: Optional[str] = None,
        resource_name: Optional[str] = None,
        descriptor_filename: Optional[str] = None,
    ) -> Self:
        try:
            old_normpath = self.normpath
        except FilePathNotDefinedError:
            raise FilePathNotDefinedError(
                message=f"{self.name} {self.resource_name!r} cannot be copied because it does not have a filepath "
                f"pointing to a resource on disk."
            )
        except BasePathNotDefinedError:
            raise BasePathNotDefinedError(
                message=f"{self.name} {self.resource_name!r} cannot be copied because it does not have a basepath and "
                f"therefore does not point to resource on disk."
            )
        new_innerpath = None
        if filepath:
            check_rel_path(filepath, basepath)
            new_normpath = os.path.join(basepath, filepath)
            if new_normpath.endswith(".zip"):
                if old_normpath.endswith(".zip"):
                    new_innerpath = self.get_innerpath()
                else:
                    new_innerpath = self.filepath
        else:
            new_normpath = os.path.join(basepath, self.filepath)
        if old_normpath == new_normpath:
            new_resource = self.from_resource(
                resource=self,
                descriptor_filename=descriptor_filename,
                basepath=basepath,
            )
            if filepath:
                new_resource.filepath = filepath
            return new_resource
        if os.path.isfile(new_normpath):
            if new_normpath.endswith(".zip"):
                if new_innerpath is None:
                    check_path = self.get_innerpath()
                else:
                    check_path = new_innerpath
                if new_innerpath in zipfile.ZipFile(new_normpath, "r").namelist():
                    if not overwrite:
                        raise FileExistsError(
                            f"{new_normpath}:{check_path} already exists."
                        )
                    else:
                        raise NotImplementedError(
                            f"Would have overwritten {new_normpath}:{check_path} but don't know how."
                        )
            elif not overwrite:
                raise FileExistsError(
                    f"{new_normpath} already exists. You can either set overwrite=True or pass a new filepath."
                )
            else:
                pass
        old_is_zip, new_is_zip = old_normpath.endswith(".zip"), new_normpath.endswith(
            ".zip"
        )
        if not any((old_is_zip, new_is_zip)):
            shutil.copy2(old_normpath, new_normpath)
            source_msg, target_msg = old_normpath, new_normpath
        else:
            old_innerpath = self.get_innerpath()
            if old_normpath.endswith(".zip"):
                source = zipfile.ZipFile(old_normpath, "r").read(old_innerpath)
                source_msg = f"{old_normpath}:{old_innerpath}"
            else:
                source = open(old_normpath, "rb").read()
                source_msg = old_normpath
            if new_normpath.endswith(".zip"):
                with zipfile.ZipFile(new_normpath, "a", zipfile.ZIP_DEFLATED) as target:
                    target.writestr(new_innerpath, source)
                target_msg = f"{new_normpath}:{new_innerpath}"
            else:
                with open(new_normpath, "wb") as target:
                    target.write(source)
                target_msg = new_normpath
        self.logger.info(f"Copied {source_msg} => {target_msg}.")
        fl_resource = self.resource.to_copy()
        new_resource = self.__class__(
            resource=fl_resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
        )
        if resource_name:
            new_resource.resource_name = resource_name
        if new_innerpath:
            new_resource.resource.innerpath = new_innerpath
        if filepath:
            new_resource.filepath = filepath
        if new_resource.filepath.endswith(".zip"):
            new_resource.resource.compression = "zip"
        return new_resource

    def _detach_from_basepath(self):
        self._basepath = None
        self._resource.basepath = None
        self.logger.debug(
            f"Detached {self.resource_name!r} from basepath by setting the property .basepath to None."
        )

    def detach_from_basepath(self):
        self._detach_from_basepath()
        self._update_status()

    def _detach_from_descriptor(self):
        self._resource.metadata_descriptor_path = None
        self.logger.debug(
            f"Detached {self.resource_name!r} from descriptor by setting the property "
            f".descriptor_filename to None."
        )

    def detach_from_descriptor(self):
        self._detach_from_descriptor()
        self._update_status()

    def _detach_from_filepath(self):
        self._resource.path = None
        self.logger.debug(
            f"Detached {self.resource_name!r} from filepath by setting the property .filepath to None."
        )

    def detach_from_filepath(self):
        self._detach_from_filepath()
        self._update_status()

    def get_corpus_name(self) -> str:
        """Returns the value of :attr:`corpus_name` or, if not set, a name derived from the
        resource's filepath.

        Raises:
            ValueError: If neither :attr:`corpus_name` nor :attr:`filepath` are set.
        """

        def return_basepath_name() -> str:
            if self.basepath is None:
                return
            return make_valid_frictionless_name(os.path.basename(self.basepath))

        if self.corpus_name:
            return self.corpus_name
        if not self.filepath:
            return return_basepath_name()
        folder, _ = os.path.split(self.filepath)
        folder = folder.rstrip(os.sep)
        if not folder or folder == ".":
            return return_basepath_name()
        folder_split = folder.split(os.sep)
        if len(folder_split) > 1:
            return make_valid_frictionless_name(folder_split[-1])
        return make_valid_frictionless_name(folder)

    def _get_current_status(self) -> ResourceStatus:
        if self.filepath:
            return ResourceStatus.PATH_ONLY
        return ResourceStatus.EMPTY

    def get_descriptor_filename(
        self,
        set_default_if_missing: bool = False,
    ) -> str:
        """Like :attr:`descriptor_filename` but returning a default value if None.
        If ``set_default_if_missing`` is set to True and no basepath has been set (e.g. during initialization),
        the :attr:`basepath` is permanently set to the  default basepath.
        """
        if self.descriptor_filename:
            return self.descriptor_filename
        if not self.filepath:
            if self.innerpath:
                descriptor_filename = replace_ext(self.innerpath, ".resource.json")
            else:
                descriptor_filename = f"{self.resource_name}.resource.json"
        else:
            descriptor_filename = replace_ext(self.filepath, ".resource.json")
        if set_default_if_missing:
            self.descriptor_filename = descriptor_filename
        return descriptor_filename

    def get_descriptor_path(
        self,
        set_default_if_missing=False,
    ) -> Optional[str]:
        """Returns the path to the descriptor file. If basepath or descriptor_filename are not set, they are set
        permanently to their defaults. If ``create_if_missing`` is set to True, the descriptor file is created if it
        does not exist yet."""
        descriptor_path = os.path.join(
            self.get_basepath(set_default_if_missing=set_default_if_missing),
            self.get_descriptor_filename(set_default_if_missing=set_default_if_missing),
        )
        return descriptor_path

    def get_filepath(
        self,
        set_default_if_missing=False,
    ) -> str:
        """Returns the relative path to the data (:attr:`filepath`) if specified, :attr:`innerpath` otherwise."""
        if not self.filepath:
            innerpath = self.get_innerpath()
            if set_default_if_missing:
                self.filepath = innerpath
            return innerpath
        return self.filepath

    def get_innerpath(
        self,
        set_default_if_missing: bool = False,
    ) -> Optional[str]:
        """Returns the path to the descriptor file."""
        if self.innerpath:
            return self.innerpath
        if not self.is_zipped and self.filepath:
            innerpath = self.filepath
        else:
            if self.resource_name:
                resource_name = self.resource_name
            else:
                resource_name = get_setting("default_resource_name")
            format = self.resource.format
            if format and not resource_name.endswith(format):
                innerpath = f"{resource_name}.{format}"
            else:
                innerpath = resource_name
        if set_default_if_missing:
            self.innerpath = innerpath
        return innerpath

    def get_path_dict(self) -> Dict[str, str]:
        """Returns a dictionary with the paths to the resource's data and descriptor."""
        path_dict = dict(
            basepath=self.basepath,
            filepath=self.filepath,
            innerpath=self.innerpath,
            descriptor_filename=self.descriptor_filename,
            descriptor_path=self.get_descriptor_path(),
        )
        try:
            path_dict["normpath"] = self.normpath
        except (BasePathNotDefinedError, FilePathNotDefinedError):
            path_dict["normpath"] = None
        return path_dict

    def make_descriptor(self) -> dict:
        """Returns a frictionless descriptor for the resource."""
        return self.pickle_schema.dump(self)

    def _make_empty_fl_resource(self):
        """Create an empty frictionless resource object with a minimal descriptor."""
        return make_fl_resource()

    def _set_basepath(
        self,
        basepath: str,
        reconcile: bool = False,
    ) -> None:
        if not self._basepath:
            self._basepath = treat_basepath_argument(basepath, other_logger=self.logger)
            self._resource.basepath = self.basepath
            return
        basepath_arg = resolve_path(basepath)
        if self.basepath == basepath_arg:
            return
        if self.is_frozen:
            if not reconcile:
                raise ResourceIsFrozenError(self.name, self.basepath, basepath_arg)
            # reconcile the current basepath with the new one, which may involve adapting filepath
            if self.resource_exists:
                try:
                    new_filepath = make_rel_path(self.normpath, basepath_arg)
                    self.logger.debug(
                        f"Adapting the current filepath  {self.filepath!r} to the new basepath "
                        f"{basepath_arg!r} by changing it to {new_filepath!r}."
                    )
                    self.filepath = new_filepath
                except BaseFilePathMismatchError:
                    raise ResourceNeedsToBeCopiedError("basepath", basepath_arg)
        if not os.path.isdir(basepath_arg):
            if get_setting("auto_make_dirs"):
                self.logger.debug(f"Creating directory {basepath_arg!r}.")
                os.makedirs(basepath_arg)
            else:
                raise NotADirectoryError(basepath_arg)
        self._basepath = basepath_arg
        self._resource.basepath = basepath_arg
        self.logger.debug(f"Updated basepath to {self.basepath!r}")
        # now check if as a result the resource points to an existing descriptor which might not correspond
        if self.descriptor_exists:
            warnings.warn(
                f"Another descriptor already exists at {self.get_descriptor_path()!r} which may lead to it being "
                f"overwritten.",
                PotentiallyUnrelatedDescriptorUserWarning,
            )

    def set_basepath(
        self,
        basepath: str,
        reconcile: bool = False,
    ) -> None:
        if self.basepath is not None and not reconcile and self.is_packaged:
            raise ResourceIsPackagedError(self.resource_name, basepath, "basepath")
        return self._set_basepath(basepath, reconcile=reconcile)

    def _set_descriptor_filename(
        self,
        descriptor_filename: str,
    ) -> None:
        """

        Args:
            descriptor_filename:

        Raises:
            ValueError: If the descriptor_filename is not a simple filename.
        """
        descriptor_filename = check_descriptor_filename_argument(descriptor_filename)
        if self.descriptor_filename == descriptor_filename:
            return
        self._resource.metadata_descriptor_path = descriptor_filename

    def set_descriptor_filename(
        self,
        descriptor_filename: str,
    ) -> None:
        """

        Args:
            descriptor_filename:

        Raises:
            ValueError: If the descriptor_filename is not a simple filename.
        """
        if self.descriptor_filename == descriptor_filename:
            return
        self._set_descriptor_filename(descriptor_filename)
        if (
            not is_default_package_descriptor_path(descriptor_filename)
            and self.basepath
        ):
            warn_about_potentially_unrelated_descriptor(
                basepath=self.basepath, descriptor_filename=descriptor_filename
            )

    def store_descriptor(
        self, descriptor_path: Optional[str] = None, overwrite=True
    ) -> str:
        """Stores the frictionless descriptor to disk based on the resource's configuration and
        returns its path. Does not modify the resource's :attr:`status`.

        Returns:
            The path to the descriptor file on disk. If None, the default is used.

        Raises:
            InvalidResourcePathError: If the resource's path does not point to an existing file on disk.
        """
        if self.is_packaged:
            raise ResourceIsPackagedError(
                self.resource_name, self.get_descriptor_path()
            )
        if descriptor_path is None:
            descriptor_path = self.get_descriptor_path(set_default_if_missing=True)
        if not overwrite and os.path.isfile(descriptor_path):
            self.logger.info(
                f"Descriptor exists already and will not be overwritten: {descriptor_path}"
            )
            return descriptor_path
        descriptor_dict = self.make_descriptor()
        resource_filepath = descriptor_dict["path"]
        if resource_filepath:
            # check if storing the descriptor would result in a valid normpath
            if resource_basepath := descriptor_dict.get("basepath"):
                resulting_resource_path = os.path.join(
                    resource_basepath, resource_filepath
                )
                if not os.path.isfile(resulting_resource_path):
                    raise InvalidResourcePathError(resource_filepath, resource_basepath)
            else:
                descriptor_location = os.path.dirname(descriptor_path)
                resulting_resource_path = os.path.join(
                    descriptor_location, resource_filepath
                )
                if not os.path.isfile(resulting_resource_path):
                    raise InvalidResourcePathError(
                        resource_filepath, descriptor_location
                    )
        store_as_json_or_yaml(descriptor_dict, descriptor_path)
        self.logger.info(f"{self.name} descriptor written to {descriptor_path}")
        return descriptor_path

    def to_dict(self, pickle: bool = False) -> Dict[str, Any]:
        """Returns a dictionary representation of the resource and stores its descriptor to disk."""
        if not pickle:
            return super().to_dict()
        descriptor_path = self.get_descriptor_path(set_default_if_missing=True)
        descriptor_dict = self.make_descriptor()

        store_as_json_or_yaml(descriptor_dict, descriptor_path)
        return descriptor_dict

    def _update_status(self) -> None:
        status_before = self.status
        self._status = self._get_current_status()
        if self.status != status_before:
            _, filename, lineno, caller, *_ = inspect.stack()[1]
            resource_status_logger.debug(
                f"{self.name}._update_status() was called by {caller}() in {filename!r} ({lineno}).\n"
                f"As a result, the status of {self.resource_name} has been changed {status_before!r} to "
                f"{self.status!r}."
            )

    def validate(
        self,
        raise_exception: bool = False,
        only_if_necessary: bool = False,
    ) -> Optional[fl.Report]:
        """Validate the resource against its descriptor.

        Args:
            raise_exception: (default False) Pass True to raise if the resource is not valid.
            only_if_necessary:
                (default False) Pass True to skip validation if the resource has already been validated or is
                assumed to be valid because it exists on disk.

        Returns:
            None if no validation took place (e.g. because resource is empty or ``only_if_necessary`` was True).
            Otherwise, frictionless report resulting from validating the data against the :attr:`column_schema`.

        Raises:
            FrictionlessException: If the resource is not valid and ``raise_exception`` is True.
        """
        if self.is_empty:
            return
        if only_if_necessary and self._is_valid is not None:
            return
        report = self._resource.validate()
        self._is_valid = True if report is None else report.valid
        if not report.valid:
            errors = [err.message for task in report.tasks for err in task.errors]
            if get_setting("never_store_unvalidated_data") and raise_exception:
                raise fl.FrictionlessException("\n".join(errors))
        return report

    # endregion Resource


class PathResource(Resource):
    """A resource that does not load frictionless descriptors or warns about them as :class:`Resource` would."""

    @classmethod
    def from_filepath(
        cls,
        filepath: str,
        resource_name: Optional[str] = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        **kwargs,
    ) -> Self:
        """Create a Resource from a file on disk, be it a JSON/YAML resource descriptor, or a simple path resource.

        Args:
            filepath: Path pointing to a resource descriptor or a simple path resource.
            resource_name:
                Name of the resource used for retrieving it from a DimcatPackage and as filename when the resource
                is stored to a ZIP file.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            auto_validate:
                By default, the Resource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).
            basepath:
                Basepath to use for the resource. If None, the folder of the ``filepath`` is used.
        """
        return cls.from_resource_path(
            resource_path=filepath,
            resource_name=resource_name,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            **kwargs,
        )

    @classmethod
    def from_resource_path(
        cls,
        resource_path: str,
        resource_name: Optional[str] = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        **kwargs,
    ) -> Self:
        """Create a Resource from a file on disk, treating it just as a path even if it's a
        JSON/YAML resource descriptor."""
        try:
            basepath, resource_path = reconcile_base_and_file(basepath, resource_path)
        except BaseFilePathMismatchError:
            basepath, resource_path = os.path.split(resource_path)
        # The rest of the method is copied from super().from_resource_path
        fname, extension = os.path.splitext(resource_path)
        if resource_name:
            resource_name = make_valid_frictionless_name(resource_name)
        else:
            resource_name = make_valid_frictionless_name_from_filepath(resource_path)
        options = dict(
            name=resource_name,
            path=resource_path,
            scheme="file",
            format=extension[1:],
        )
        fl_resource = make_fl_resource(**options)
        return cls(
            resource=fl_resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            **kwargs,
        )

    def __init__(
        self,
        resource: fl.Resource,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        **kwargs,
    ):
        """

        Args:
            resource: An existing :obj:`frictionless.Resource`.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            basepath: Where the file would be serialized.
        """
        self.logger.debug(
            f"""
PathResource.__init__(
    resource={type(resource)},
    descriptor_filename={descriptor_filename},
    basepath={basepath},
)"""
        )
        if resource is None:
            super().__init__(
                descriptor_filename=descriptor_filename, basepath=basepath, **kwargs
            )
            return
        if not isinstance(resource, fl.Resource):
            raise TypeError(
                f"resource must be of type frictionless.Resource, not {type(resource)}"
            )
        if not resource.path:
            raise ValueError(f"The resource comes without a path: {resource}")
        fl_resource = resource.to_copy()
        if basepath:
            fl_resource.basepath = basepath
        if not fl_resource.normpath:
            fl_resource.basepath = get_setting("default_basepath")
            raise ValueError(f"The resource did not yield a normpath: {fl_resource}.")
        if not os.path.isfile(fl_resource.normpath):
            raise FileNotFoundError(f"Resource does not exist: {fl_resource.normpath}")
        super().__init__(
            resource=fl_resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
        )
        self.logger.debug(
            f"""
Resource(
    basepath={self.basepath},
    filepath={self.filepath},
    corpus_name={self.get_corpus_name()},
    resource_name={self.resource_name},
    descriptor_filename={self.descriptor_filename},
)"""
        )


ResourceSpecs: TypeAlias = Union[Resource, str, Path]


def resource_specs2resource(resource: ResourceSpecs) -> R:
    """Converts a resource specification to a resource.

    Args:
        resource: A resource specification.

    Returns:
        A resource.
    """
    if isinstance(resource, Resource):
        return resource
    if isinstance(resource, (str, Path)):
        return Resource.from_descriptor_path(resource)
    raise TypeError(
        f"Expected a Resource, str, or Path. Got {type(resource).__name__!r}."
    )


class FeatureName(ObjectEnum):
    Annotations = "Annotations"
    Articulation = "Articulation"
    BassNotes = "BassNotes"
    CadenceLabels = "CadenceLabels"
    DcmlAnnotations = "DcmlAnnotations"
    HarmonyLabels = "HarmonyLabels"
    KeyAnnotations = "KeyAnnotations"
    Measures = "Measures"
    Metadata = "Metadata"
    Notes = "Notes"

    def get_class(self) -> Type[F]:
        return get_class(self.name)
