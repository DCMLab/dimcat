import os
from pathlib import Path

from dimcat import Dataset, DimcatConfig, Pipeline
from dimcat.data import resources
from dimcat.data.resources import DimcatIndex
from dimcat.steps import groupers
from dimcat.steps.groupers import CustomPieceGrouper
from dimcat.utils import resolve_path


def resource_names(path):
    return os.sep.join(path.split(os.sep)[-2:])


def load_unittest_corpora():
    here = Path(__file__).parent
    package_path = (
        here / ".." / "unittest_metacorpus" / "unittest_metacorpus.datapackage.json"
    )
    return Dataset.from_package(package_path)


def load_dcml_corpora():
    here = Path(__file__).parent
    package_path = here / ".." / "docs" / "manual" / "dcml_corpora.datapackage.json"
    return Dataset.from_package(package_path)


def load_distant_listening_corpus():
    package_path = resolve_path(
        "~/distant_listening_corpus/distant_listening_corpus.datapackage.json"
    )
    return Dataset.from_package(package_path)


def make_grouper(D):
    piece_index = resources.PieceIndex.from_resource(D.get_metadata())
    grouping = {f"group_{i}": piece_index.sample(3) for i in range(1, 4)}
    return groupers.CustomPieceGrouper.from_grouping(grouping)


def get_piece_groups(dataset):
    input_package = dataset.inputs.get_package()
    piece_index = input_package.get_piece_index()
    n_groups = 4
    grouping = {f"group_{i}": piece_index.sample(i) for i in range(n_groups)}
    grouped_pieces = DimcatIndex.from_grouping(grouping)
    print(len(grouped_pieces))
    return grouped_pieces


if __name__ == "__main__":
    D = load_unittest_corpora()
    analyzer_config = {"dtype": "BigramAnalyzer"}
    feature_config = {"dtype": "Articulation"}
    grouped_pieces = get_piece_groups(D)
    CustomPieceGrouper.schema.load(
        dict(dtype="CustomPieceGrouper", grouped_units=grouped_pieces)
    )
    grouper_config = DimcatConfig(
        dtype="CustomPieceGrouper", grouped_units=grouped_pieces
    )
    pl = Pipeline.from_step_configs(
        [
            dict(dtype="FeatureExtractor", features=[feature_config]),
            analyzer_config,
            grouper_config,
        ]
    )
    print(pl.to_dict())
