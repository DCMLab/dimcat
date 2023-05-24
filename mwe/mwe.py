from dimcat.analyzers import Counter
from dimcat.base import DimcatConfig
from dimcat.dataset import Dataset
from dimcat.features import FeatureName

if __name__ == "__main__":
    dataset = Dataset()
    dataset.load_package("dcml_corpora.datapackage.json")
    # any of these work as argument for "feature_config"
    feature_name = "nOtEs"  # arbitrary case
    feature_enum = FeatureName(feature_name)
    feature_constructor = feature_enum.get_class()
    F = feature_constructor()
    feature_dict = F.to_dict()
    default_config1 = F.to_config()
    minimal_config = DimcatConfig(dtype=feature_enum)
    report = default_config1.options_schema.validate(minimal_config)
    assert not report
    report = minimal_config.options_schema.validate(default_config1)
    assert not report
    weighted_config = DimcatConfig(minimal_config, weight_grace_notes=0.5)
    assert type(minimal_config) == type(
        weighted_config
    ), f"{type(minimal_config)} != {type(weighted_config)}"

    analyzer0 = Counter(features=weighted_config)
    analyzer_config0 = analyzer0.to_config()
    analyzer1 = analyzer_config0.create()
    analyzer_dict1 = analyzer1.to_dict()
    assert analyzer0.to_dict() == analyzer1.to_dict()
    assert analyzer_config0 == analyzer_dict1

    json_str = analyzer1.to_json()
    analyzer_config1 = DimcatConfig.from_json(json_str)
    analyzer2 = analyzer_config1.create()
    assert analyzer1.to_dict() == analyzer2.to_dict()
    assert analyzer_config0 == analyzer_config1

    json_path = "counter.json"
    analyzer2.to_json_file(json_path)
    analyzer3 = DimcatConfig.from_json_file(json_path).create()
    assert analyzer2.to_dict() == analyzer3.to_dict()

    analyzer_config3 = analyzer3.to_config()
    analyzer4 = Counter.from_config(analyzer_config3)
    assert analyzer3.to_dict() == analyzer4.to_dict()

    analyzed = analyzer4.process(dataset)
