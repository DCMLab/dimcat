from pathlib import Path
from typing import Optional

import frictionless as fl
from dimcat.base import ObjectEnum
from dimcat.data.resource import DimcatResource, Metadata, Resource
from typing_extensions import Self


class MuseScoreFacetName(ObjectEnum):
    MuseScoreFacet = "MuseScoreFacet"
    MuseScoreHarmonies = "MuseScoreHarmonies"
    MuseScoreMeasures = "MuseScoreMeasures"
    MuseScoreNotes = "MuseScoreNotes"


class MuseScoreFacet(DimcatResource):
    """A single facet of a MuseScore package as created by the ms3 MuseScore parsing library. Contains a single TSV
    facet one or several corpora. Naming format ``<name>.<facet>[.tsv]``."""

    _enum_type = MuseScoreFacetName

    @classmethod
    def from_descriptor(
        cls,
        descriptor: dict | Resource,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
    ) -> Self:
        if isinstance(descriptor, (str, Path)):
            raise TypeError(
                f"This method expects a descriptor dictionary. In order to create a "
                f"{cls.name} from a path, use {cls.__name__}.from_descriptor_path() instead."
            )
        if cls.name == "MuseScoreFacet":
            # dispatch to the respective facet based on the resource name
            if isinstance(descriptor, fl.Resource):
                fl_resource = descriptor
            else:
                fl_resource = fl.Resource.from_descriptor(descriptor)
            facet_name2constructor = dict(
                expanded=MuseScoreHarmonies,
                harmonies=MuseScoreHarmonies,
                measures=MuseScoreMeasures,
                metadata=Metadata,
                notes=MuseScoreNotes,
            )
            resource_name = fl_resource.name
            try:
                _, facet_name = resource_name.rsplit(".", 1)
                Klass = facet_name2constructor.get(facet_name)
                if Klass is None:
                    raise NotImplementedError(
                        f"MuseScoreFacet {facet_name} is not implemented."
                    )
            except ValueError:
                if any(
                    resource_name.endswith(f_name) for f_name in facet_name2constructor
                ):
                    Klass = next(
                        klass
                        for f_name, klass in facet_name2constructor.items()
                        if resource_name.endswith(f_name)
                    )
            return Klass.from_descriptor(
                descriptor=descriptor,
                descriptor_filename=descriptor_filename,
                basepath=basepath,
                auto_validate=auto_validate,
                default_groupby=default_groupby,
            )
        return super().from_descriptor(
            descriptor=descriptor,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )


class MuseScoreHarmonies(MuseScoreFacet):
    pass


class MuseScoreMeasures(MuseScoreFacet):
    pass


class MuseScoreNotes(MuseScoreFacet):
    pass
