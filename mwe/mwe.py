from dimcat.analyzers import CounterConfig
from dimcat.base import DimcatObject
from dimcat.dataset import Dataset
from dimcat.features import FeatureName

if __name__ == "__main__":
    dataset = Dataset()
    dataset.load_package("dcml_corpora.datapackage.json")
    metadata = dataset.get_feature("metadata")

    selected_feature = FeatureName("KeyAnnotations")
    analyzer_config = CounterConfig(feature_config=selected_feature)
    analyzer = DimcatObject.from_config(analyzer_config)
    analyzed = analyzer.process(dataset)

    result = analyzed.result
    fig = result.plot()
    print(fig)  # fig.show()
