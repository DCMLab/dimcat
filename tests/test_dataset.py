from dimcat.data.datasets.base import Dataset
from dimcat.data.datasets.processed import (
    AnalyzedDataset,
    GroupedAnalyzedDataset,
    GroupedDataset,
)


def test_copying_dataset(dataset_from_single_package):
    new_dataset = Dataset.from_dataset(dataset_from_single_package)
    assert new_dataset == dataset_from_single_package
    assert new_dataset is not dataset_from_single_package
    as_config = dataset_from_single_package.to_config()
    new_dataset = as_config.create()
    assert new_dataset == dataset_from_single_package


def test_processed_dataset(dataset_from_single_package):
    a_dataset = AnalyzedDataset.from_dataset(dataset_from_single_package)
    assert not hasattr(dataset_from_single_package, "get_result")
    assert hasattr(a_dataset, "get_result")
    assert isinstance(a_dataset, AnalyzedDataset)
    assert a_dataset.inputs == dataset_from_single_package.inputs
    assert a_dataset.outputs.has_package("results")

    ag_dataset = GroupedDataset.from_dataset(a_dataset)
    assert isinstance(ag_dataset, GroupedAnalyzedDataset)
    assert isinstance(ag_dataset, GroupedDataset)
    assert isinstance(ag_dataset, AnalyzedDataset)
    assert ag_dataset.inputs == dataset_from_single_package.inputs
