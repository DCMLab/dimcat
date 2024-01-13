"""This module is not named test_loggers.py because this causes the registry mechanism to fail when running pytest
on the entire test folder. The problem seems to be linked to importing pkgutil and/or dimcat. This is why the
helper functions are duplicated between loggers.py and enums.py.
"""
import inspect
import logging
import pkgutil
from enum import Enum

import dimcat
import pytest
from dimcat import enums as dimcat_enums

module_logger = logging.getLogger(__name__)


def iter_dimcat_modules():
    for importer, modname, ispkg in pkgutil.walk_packages(
        dimcat.__path__, prefix="dimcat."
    ):
        yield modname, importer.find_module(modname).load_module(modname)


@pytest.fixture(params=iter_dimcat_modules(), ids=lambda x: x[0])
def dimcat_module(request):
    return request.param


def test_enums_module(dimcat_module):
    """Make sure all enums in dimcat are importable from dimcat.enums"""
    modname, mod = dimcat_module
    for name, cls in inspect.getmembers(mod, inspect.isclass):
        if issubclass(cls, Enum):
            cls_name = cls.__name__
            if cls_name in ("Enum", "IntEnum"):
                continue
            test_passes = hasattr(dimcat_enums, cls_name)
            if not test_passes:
                print(f"\nADD {cls_name} TO dimcat.enums")
            assert test_passes
