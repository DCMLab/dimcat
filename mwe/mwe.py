import os

from dimcat import Dataset, DimcatConfig
from dimcat.analyzers import Counter
from dimcat.resources.features import FeatureName

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    dataset = Dataset()
    dataset.load_package("dcml_corpora.datapackage.json")
    feature_enum = FeatureName("nOtEs")

    minimal_config = DimcatConfig(dtype=feature_enum)
    weighted_config = DimcatConfig(minimal_config, weight_grace_notes=0.5)

    Notes_constructor = feature_enum.get_class()
    default_config = Notes_constructor().to_config()

    analyzer_config = DimcatConfig(dtype="Counter", features=weighted_config)
    analyzer1 = analyzer_config.create()
    analyzer2 = Counter(features=weighted_config)
    assert analyzer1 == analyzer2, f"{analyzer1.to_dict()} != {analyzer2.to_dict()}"

    analyzed = analyzer1.process(dataset)
    result = analyzed.get_result()
    fig = result.plot()
    print(fig)
