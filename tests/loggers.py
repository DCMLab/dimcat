import inspect
import logging
import pkgutil

import dimcat
import pytest

module_logger = logging.getLogger(__name__)


def iter_dimcat_modules():
    for importer, modname, ispkg in pkgutil.walk_packages(
        dimcat.__path__, prefix="dimcat."
    ):
        yield modname, importer.find_module(modname).load_module(modname)


@pytest.fixture(params=iter_dimcat_modules(), ids=lambda x: x[0])
def dimcat_module(request):
    return request.param


def test_module_loggers(dimcat_module):
    modname, mod = dimcat_module
    assert hasattr(mod, "module_logger")
    assert hasattr(mod.module_logger, "name")
    assert mod.module_logger.name == modname


def test_class_loggers(dimcat_module):
    modname, mod = dimcat_module
    for name, cls in inspect.getmembers(mod, inspect.isclass):
        if hasattr(cls, "logger"):
            if cls.__module__ != modname:
                continue
            assert hasattr(cls.logger, "name")
            assert cls.logger.name == f"{modname}.{name}"
            print(f"The logger of {cls.name} is {cls.logger.name}")
