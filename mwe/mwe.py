import json
from dataclasses import asdict

from dimcat.analyzers import Counter, CounterConfig
from dimcat.base import Configuration, DimcatObject
from dimcat.dataset import Dataset
from dimcat.features import FeatureName, NotesConfig


def config2json(config: Configuration, json_path: str) -> None:
    config_dict = asdict(config)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)


def json2dict(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # any of these work as argument for "feature_config"
    feature_name = "nOtEs"  # arbitrary case
    feature_enum = FeatureName(feature_name)
    feature_constructor = feature_enum.get_class()
    default_config = feature_constructor._config_type
    default_config1 = default_config()
    default_config2 = feature_enum.get_config()
    assert default_config1 == default_config2, f"{default_config1} != {default_config2}"
    weighted_config = NotesConfig(weight_grace_notes=0.5)
    assert type(default_config2) == type(
        weighted_config
    ), f"{type(default_config2)} != {type(weighted_config)}"

    analyzer_config1 = CounterConfig(feature_config=weighted_config)
    analyzer1 = DimcatObject.from_config(analyzer_config1)
    analyzer_config2 = analyzer1.config
    assert (
        analyzer_config1 == analyzer_config2
    ), f"{analyzer_config1} != {analyzer_config2}"

    # serialization hadn't been implemented so far, here's a sketch:
    json_path = "counter.json"
    config2json(analyzer_config2, json_path)
    config_dict = json2dict(json_path)
    analyzer_config3 = CounterConfig.from_dict(config_dict)

    assert (
        analyzer_config2 == analyzer_config3
    ), f"{analyzer_config2} != {analyzer_config3}"  # FAILS
    # blind spots:
    # 1. JSON does not contain the type of the pickled object, we have to know which constructor to use
    # 2. the enum values have not been converted back, they remain strings
    # 3. although the string values and enum members compare as equal, this comparison fails because in config2,
    #    feature_config is a NotesConfig, whereas in the restored config 3, it remains a dict (related to 1.)

    analyzer2 = Counter.from_config(
        analyzer_config3
    )  # SHOULD FAIL (because feature_config is not valid)

    dataset = Dataset()
    dataset.load_package("dcml_corpora.datapackage.json")
    analyzed = analyzer2.process(
        dataset
    )  # FAILS because feature_config needs to be hashable but is a dict
