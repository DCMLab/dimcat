import os
from pathlib import Path

from dimcat import Dataset
from dimcat.data import resources
from dimcat.steps import groupers
from dimcat.utils import resolve_path


def resource_names(path):
    return os.sep.join(path.split(os.sep)[-2:])


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


if __name__ == "__main__":
    D = load_distant_listening_corpus()
    feature = D.get_feature("keyannotations")
