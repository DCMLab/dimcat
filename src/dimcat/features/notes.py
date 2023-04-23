from dataclasses import dataclass

from dimcat.features.base import Feature, FeatureConfig, FeatureName


@dataclass(frozen=True)
class NotesConfig(FeatureConfig):
    _configured_class = FeatureName.Notes


class Notes(Feature):
    _config_type = NotesConfig
