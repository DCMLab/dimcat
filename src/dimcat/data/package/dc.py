from __future__ import annotations

from typing import Iterable, Optional

import frictionless as fl
from dimcat.data.package.base import Package, PackageMode
from dimcat.data.resource.base import D, Resource
from dimcat.data.resource.dc import DimcatResource


class DimcatPackage(Package):
    accepted_resource_types = (DimcatResource,)
    default_mode = PackageMode.RECONCILE_SAFELY

    def _verify_creationist_arguments(
        self,
        **kwargs,
    ):
        """Spoiler alert: They are spurious."""
        if not any(kwargs.values()):
            raise ValueError("No arguments were passed to create a resource.")
        if kwargs.get("resource") and kwargs.get("df"):
            raise ValueError("Pass either a resource or a dataframe, not both.")

    def __init__(
        self,
        package_name: str,
        resources: Iterable[Resource] = None,
        basepath: Optional[str] = None,
        descriptor_filename: Optional[str] = None,
        auto_validate: bool = False,
    ) -> None:
        """

        Args:
            package_name:
                Name of the package that can be used to retrieve it.
            resources:
                An iterable of :class:`Resource` objects to add to the package.
            descriptor_filename:
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
            descriptor_filename=descriptor_filename,
            auto_validate=auto_validate,
        )

    def create_and_add_resource(
        self,
        df: Optional[D] = None,
        resource: Optional[Resource | fl.Resource | str] = None,
        resource_name: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
    ) -> None:
        """Adds a resource to the package. Parameters are passed to :class:`DimcatResource`."""
        self._verify_creationist_arguments(df=df, resource=resource)
        if df is not None:
            new_resource = DimcatResource.from_dataframe(
                df=df,
                resource_name=resource_name,
                auto_validate=auto_validate,
                basepath=basepath,
            )
            self.add_resource(new_resource)
            return
        super().create_and_add_resource(
            resource=resource,
            resource_name=resource_name,
            basepath=basepath,
            auto_validate=auto_validate,
        )