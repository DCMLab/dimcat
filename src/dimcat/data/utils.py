from __future__ import annotations

import json
import os
import warnings
from typing import Optional

import frictionless as fl
import yaml
from dimcat.base import get_setting
from dimcat.dc_exceptions import BaseFilePathMismatchError
from dimcat.dc_warnings import PotentiallyUnrelatedDescriptorUserWarning


def check_descriptor_filename_argument(
    descriptor_filename,
) -> str:
    """Check if the descriptor_filename is a filename  (not path) and warn if it doesn't have the
    extension .json or .yaml.

    Args:
        descriptor_filename:

    Raises:
        ValueError: If the descriptor_filename is absolute.
    """
    subfolder, filepath = os.path.split(descriptor_filename)
    if subfolder not in (".", ""):
        raise ValueError(
            f"descriptor_filename needs to be a filename in the basepath, got {descriptor_filename!r}"
        )
    _, ext = os.path.splitext(filepath)
    if ext not in (".json", ".yaml"):
        warnings.warning(
            f"You've set a descriptor_filename with extension {ext!r} but "
            f"frictionless allows only '.json' and '.yaml'.",
            RuntimeWarning,
        )
    return filepath


def check_rel_path(rel_path, basepath):
    if rel_path.startswith(".."):
        raise ValueError(
            f"{rel_path!r} points outside the basepath {basepath!r} which is not allowed."
        )
    if rel_path.startswith(f".{os.sep}") and len(rel_path) > 2:
        rel_path = rel_path[2:]
    return rel_path


def is_default_package_descriptor_path(filepath: str) -> bool:
    endings = get_setting("package_descriptor_endings")
    if len(endings) == 0:
        warnings.warn(
            "No default file endings for package descriptors are defined in the current settings.",
            RuntimeWarning,
        )
    for ending in endings:
        if filepath.endswith(ending):
            return True
    return False


def is_default_resource_descriptor_path(filepath: str) -> bool:
    endings = get_setting("resource_descriptor_endings")
    if len(endings) == 0:
        warnings.warn(
            "No default file endings for resource descriptors are defined in the current settings.",
            RuntimeWarning,
        )
    for ending in endings:
        if filepath.endswith(ending):
            return True
    return False


def make_rel_path(path: str, start: str):
    """Like os.path.relpath() but ensures that path is contained within start."""
    if not start:
        raise ValueError(f"start must not be empty, but is {start!r}")
    rel_path = os.path.relpath(path, start)
    try:
        return check_rel_path(rel_path, start)
    except ValueError as e:
        raise BaseFilePathMismatchError(start, path) from e


def make_fl_resource(
    name: Optional[str] = None,
    **options,
) -> fl.Resource:
    """Creates a frictionless.Resource by passing the **options to the constructor."""
    new_resource = fl.Resource(**options)
    if name is None:
        new_resource.name = get_setting(
            "default_resource_name"
        )  # replacing the default name "memory"
    else:
        new_resource.name = name
    if "path" not in options:
        new_resource.path = ""
    return new_resource


def warn_about_potentially_unrelated_descriptor(
    basepath: str,
    descriptor_filename: str,
):
    descriptor_path = os.path.join(basepath, descriptor_filename)
    if os.path.isfile(descriptor_path):
        warnings.warn(
            f"Another descriptor already exists at {descriptor_path!r} which may lead to it being "
            f"overwritten.",
            PotentiallyUnrelatedDescriptorUserWarning,
        )


def store_as_json_or_yaml(
    descriptor_dict: dict,
    descriptor_path: str,
    create_dirs: bool = True,
):
    if create_dirs:
        os.makedirs(os.path.dirname(descriptor_path), exist_ok=True)
    if descriptor_path.endswith(".yaml"):
        with open(descriptor_path, "w") as f:
            yaml.dump(descriptor_dict, f)
    elif descriptor_path.endswith(".json"):
        with open(descriptor_path, "w") as f:
            json.dump(descriptor_dict, f, indent=2)
    else:
        raise ValueError(
            f"Descriptor path must end with .yaml or .json: {descriptor_path}"
        )
