from __future__ import annotations

from typing import Iterable, Optional

from dimcat.data.package.base import Package
from dimcat.data.resources.base import Resource


class DimcatPackage(Package):
    def __init__(
        self,
        package_name: str,
        resources: Iterable[Resource] = None,
        basepath: Optional[str] = None,
        descriptor_filepath: Optional[str] = None,
        auto_validate: bool = False,
    ) -> None:
        """

        Args:
            package_name:
                Name of the package that can be used to retrieve it.
            resources:
                An iterable of :class:`Resource` objects to add to the package.
            descriptor_filepath:
                Pass a JSON or YAML filename or relative filepath to override the default (``<package_name>.json``).
                Following frictionless specs it should end on ".datapackage.[json|yaml]".
            basepath:
                The absolute path on the local file system where the package descriptor and all contained resources
                are stored. The filepaths of all included :class:`DimcatResource` objects need to be relative to the
                basepath and DiMCAT does its best to ensure this.
            auto_validate:
                By default, the package is validated everytime a resource is added. Set to False to disable this.
        """
        super().__init__(
            package_name=package_name,
            resources=resources,
            basepath=basepath,
            descriptor_filepath=descriptor_filepath,
            auto_validate=auto_validate,
        )
