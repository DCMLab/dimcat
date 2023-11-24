from __future__ import annotations

from typing import Iterable, Optional

import frictionless as fl
import pandas as pd
from dimcat.base import get_setting
from dimcat.data.packages.base import Package, PackageMode
from dimcat.data.resources.base import D, Resource, SomeDataframe
from dimcat.data.resources.dc import DimcatResource, PieceIndex


class DimcatPackage(Package):
    _accepted_resource_types = (DimcatResource,)
    _default_mode = PackageMode.RECONCILE_SAFELY
    _detects_extensions = get_setting("resource_descriptor_endings")

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
        metadata: Optional[dict] = None,
    ) -> None:
        """

        Args:
            metadata:
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
            metadata:
                Custom metadata to be maintained in the package descriptor.
        """
        super().__init__(
            package_name=package_name,
            resources=resources,
            basepath=basepath,
            descriptor_filename=descriptor_filename,
            auto_validate=auto_validate,
            metadata=metadata,
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

    def get_boolean_resource_table(self) -> SomeDataframe:
        """Returns a table with this package's piece index and one boolean column per resource,
        indicating whether the resource is available for a given piece or not."""
        bool_masks = []
        for resource in self:
            piece_index = resource.get_piece_index()
            if len(piece_index) == 0:
                continue
            bool_masks.append(
                pd.Series(
                    True,
                    dtype="boolean",
                    index=piece_index.index,
                    name=resource.resource_name,
                )
            )
        if len(bool_masks) == 0:
            return pd.DataFrame([], dtype="boolean", index=PieceIndex().index)
        table = pd.concat(bool_masks, axis=1).fillna(False).sort_index()
        table.index.names = ("corpus", "piece")
        table.columns.names = ("resource_name",)
        return table

    def get_piece_index(self) -> PieceIndex:
        """Returns the piece index corresponding to a sorted union of all included resources' indices."""
        IDs = set()
        for resource in self:
            IDs.update(resource.get_piece_index())
        return PieceIndex.from_tuples(sorted(IDs))
