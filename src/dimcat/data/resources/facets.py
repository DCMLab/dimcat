from pathlib import Path
from typing import Optional  # , ClassVar, Tuple

import frictionless as fl
from dimcat.base import ObjectEnum
from dimcat.data.resources import DimcatResource, FeatureName, Metadata, Resource
from typing_extensions import Self


class MuseScoreFacetName(ObjectEnum):
    MuseScoreFacet = "MuseScoreFacet"
    MuseScoreHarmonies = "MuseScoreHarmonies"
    MuseScoreMeasures = "MuseScoreMeasures"
    MuseScoreNotes = "MuseScoreNotes"


class Facet(DimcatResource):
    """A facet is one aspect of a score that can sensibly ordered and conceived of along the score's timeline. The
    format of a facet depends on the score format and tries to stay as close to the original as possible, using only
    the necessary minimum of standardization. Content an format of a facet define which features can be extracted,
    based on which configuration options.
    """

    extractable_features = None  # : Optional[ClassVar[Tuple[FeatureName, ...]]]


class MuseScoreFacet(Facet):
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
    extractable_features = (FeatureName.HarmonyLabels, FeatureName.KeyAnnotations)


class MuseScoreMeasures(MuseScoreFacet):
    extractable_features = (FeatureName.Measures,)


class MuseScoreNotes(MuseScoreFacet):
    extractable_features = (FeatureName.Notes,)