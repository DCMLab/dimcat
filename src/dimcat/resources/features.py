from __future__ import annotations

import logging
from enum import Enum
from typing import Iterable, List, MutableMapping, Optional, TypeAlias, Union

import frictionless as fl
import marshmallow as mm
from dimcat import DimcatConfig
from dimcat.base import ObjectEnum, is_subclass_of
from dimcat.resources.base import DimcatResource

logger = logging.getLogger(__name__)


class FeatureName(ObjectEnum):
    Notes = "Notes"
    Annotations = "Annotations"
    KeyAnnotations = "KeyAnnotations"
    Metadata = "Metadata"


class Feature(DimcatResource):
    _enum_type = FeatureName


class Metadata(Feature):
    pass


class NotesFormat(str, Enum):
    NAME = "NAME"
    FIFTHS = "FIFTHS"
    MIDI = "MIDI"
    DEGREE = "DEGREE"
    INTERVAL = "INTERVAL"


class Notes(Feature):
    class Schema(Feature.Schema):
        format = mm.fields.Enum(NotesFormat)
        merge_ties = mm.fields.Boolean(
            load_default=True,
            metadata=dict(
                title="Merge tied notes",
                description="If set, notes that are tied together in the score are merged together, counting them "
                "as a single event of the corresponding length. Otherwise, every note head is counted.",
            ),
        )
        weight_grace_notes = mm.fields.Float(
            load_default=0.0,
            validate=mm.validate.Range(min=0.0, max=1.0),
            metadata=dict(
                title="Weight grace notes",
                description="Set a factor > 0.0 to multiply the nominal duration of grace notes which, otherwise, have "
                "duration 0 and are therefore excluded from many statistics.",
            ),
        )

    def __init__(
        self,
        format: NotesFormat = NotesFormat.NAME,
        merge_ties: bool = True,
        weight_grace_notes: float = 0.0,
        resource_name: Optional[str] = None,
        resource: Optional[fl.Resource | str] = None,
        column_schema: Optional[fl.Schema | str] = None,
        basepath: Optional[str] = None,
        filepath: Optional[str] = None,
        descriptor_filepath: Optional[str] = None,
        auto_validate: bool = True,
    ) -> None:
        self._format: NotesFormat = format
        self._weight_grace_notes: float = weight_grace_notes
        super().__init__(
            resource=resource,
            resource_name=resource_name,
            basepath=basepath,
            filepath=filepath,
            column_schema=column_schema,
            descriptor_filepath=descriptor_filepath,
            auto_validate=auto_validate,
        )

    @property
    def format(self) -> NotesFormat:
        return self._format

    @property
    def weight_grace_notes(self) -> float:
        return self._weight_grace_notes


class Annotations(Feature):
    pass


class KeyAnnotations(Annotations):
    pass


FeatureSpecs: TypeAlias = Union[MutableMapping, Feature, FeatureName, str]


def feature_specs2config(feature: FeatureSpecs) -> DimcatConfig:
    """Converts a feature specification into a dimcat configuration.

    Raises:
        TypeError: If the feature cannot be converted to a dimcat configuration.
    """
    if isinstance(feature, DimcatConfig):
        feature_config = feature
    elif isinstance(feature, Feature):
        feature_config = feature.to_config()
    elif isinstance(feature, MutableMapping):
        feature_config = DimcatConfig(feature)
    elif isinstance(feature, str):
        feature_name = FeatureName(feature)
        feature_config = DimcatConfig(dtype=feature_name)
    else:
        raise TypeError(
            f"Cannot convert the {type(feature).__name__} {feature!r} to DimcatConfig."
        )
    if feature_config.options_dtype == "DimcatConfig":
        feature_config = DimcatConfig(feature_config["options"])
    if not is_subclass_of(feature_config.options_dtype, Feature):
        raise TypeError(
            f"DimcatConfig describes a {feature_config.options_dtype}, not a Feature: "
            f"{feature_config.options}"
        )
    return feature_config


def features_argument2config_list(
    features: Optional[FeatureSpecs | Iterable[FeatureSpecs]] = None,
) -> List[DimcatConfig]:
    if features is None:
        return []
    if isinstance(features, (MutableMapping, Feature, FeatureName, str)):
        features = [features]
    configs = []
    for specs in features:
        configs.append(feature_specs2config(specs))
    return configs
