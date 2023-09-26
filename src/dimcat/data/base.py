from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import marshmallow as mm
from dimcat.base import DimcatConfig, DimcatObject, get_pickle_schema, get_setting
from dimcat.dc_exceptions import BaseFilePathMismatchError

logger = logging.getLogger(__name__)


class AbsolutePathStr(str):
    """This is just a string but if it includes the HOME directory, it is represented with a leading '~'."""

    def __repr__(self):
        """If a basepath starts with the (home) directory that "~" resolves to, replace that part with "~"."""
        path = str(self)
        home = os.path.expanduser("~")
        if path.startswith(home):
            path = "~" + path[len(home) :]
        return path


def resolve_path(path) -> Optional[AbsolutePathStr]:
    """Resolves '~' to HOME directory and turns ``path`` into an absolute path.
    This is an identical copy of the function in dimcat.utils.
    """
    if path is None:
        return None
    if isinstance(path, str):
        pass
    elif isinstance(path, Path):
        path = str(path)
    else:
        raise TypeError(f"Expected str or Path, got {type(path)}")
    if "~" in path:
        path = os.path.expanduser(path)
    else:
        path = os.path.abspath(path)
    path = path.rstrip("/\\")
    return AbsolutePathStr(path)


class Data(DimcatObject):
    """
    This base class unites all classes containing data in some way or another.
    """

    @staticmethod
    def treat_new_basepath(
        basepath: str, filepath=None, other_logger=None
    ) -> AbsolutePathStr:
        basepath_arg = resolve_path(basepath)
        if not os.path.isdir(basepath_arg):
            if get_setting("auto_make_dirs"):
                os.makedirs(basepath_arg)
            else:
                raise NotADirectoryError(
                    f"basepath {basepath_arg!r} is not an existing directory."
                )
        if filepath and not os.path.isfile(os.path.join(basepath_arg, filepath)):
            # this would result in a normpath that does not exist
            raise BaseFilePathMismatchError(basepath_arg, filepath)
        if other_logger is None:
            other_logger = logger
        other_logger.debug(f"The basepath been set to {basepath_arg!r}")
        return basepath_arg

    @classmethod
    @property
    def pickle_schema(cls):
        """Returns the (instantiated) PickleSchema singleton object for this class. It is different from the 'noremal'
        Schema in that it stores the tabular data to disk and returns the path to its descriptor.
        """
        return get_pickle_schema(cls.dtype)

    class Schema(DimcatObject.Schema):
        basepath = mm.fields.Str(
            required=False,
            allow_none=True,
            metadata=dict(description="The directory where data would be stored."),
        )

    def __init__(
        self,
        basepath: Optional[str] = None,
    ):
        super().__init__()
        self._basepath = None
        if basepath is not None:
            self.basepath = basepath

    def __str__(self):
        return self.info(return_str=True)

    @property
    def basepath(self) -> str:
        return self._basepath

    @basepath.setter
    def basepath(self, basepath: str):
        self._basepath = self.treat_new_basepath(basepath, other_logger=self.logger)

    def get_basepath(
        self,
        set_default_if_missing: bool = False,
    ) -> str:
        """Get the basepath of the resource. If not specified, the default basepath is returned.
        If ``set_default_if_missing`` is set to True and no basepath has been set (e.g. during initialization),
        the :attr:`basepath` is permanently set to the  default basepath.
        """
        if not self.basepath:
            default_basepath = resolve_path(get_setting("default_basepath"))
            if set_default_if_missing:
                self._basepath = default_basepath
                self.logger.debug(
                    f"The {self.name}'s basepath has been set to the default {default_basepath!r}"
                )
            return default_basepath
        return self.basepath

    def to_config(self, pickle=False) -> DimcatConfig:
        """If ``pickle`` is set to True,"""
        return DimcatConfig(self.to_dict(pickle=pickle))

    def to_dict(self, pickle=False) -> dict:
        if pickle:
            return self.pickle_schema.dump(self)
        return self.schema.dump(self)
