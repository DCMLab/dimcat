import os
from pathlib import Path

from dimcat import Dataset, DimcatConfig
from dimcat.data import resources
from dimcat.steps import groupers


def resource_names(path):
    return os.sep.join(path.split(os.sep)[-2:])


def load_dcml_corpora():
    global D
    here = Path(__file__).parent
    package_path = here / ".." / "docs" / "manual" / "dcml_corpora.datapackage.json"
    return Dataset.from_package(package_path)


def make_grouper(D):
    piece_index = resources.PieceIndex.from_resource(D.get_metadata())
    grouping = {f"group_{i}": piece_index.sample(3) for i in range(1, 4)}
    return groupers.CustomPieceGrouper.from_grouping(grouping)


if __name__ == "__main__":
    c = DimcatConfig(dtype="Counter", features=dict(dtype="notes", format="banana"))
    D = load_dcml_corpora()
    grouper = make_grouper(D)
    analyzed_dataset = D.apply_steps([grouper, dict(dtype="Counter", features="notes")])
    result = analyzed_dataset.get_result()
    assert result.get_default_groupby()
