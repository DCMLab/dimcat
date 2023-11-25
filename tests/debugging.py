import os

from dimcat import Dataset
from dimcat.steps import groupers

from docs.notebooks.utils import resolve_dir


def resource_names(path):
    return os.sep.join(path.split(os.sep)[-2:])


if __name__ == "__main__":
    package_path = resolve_dir(
        "~/distant_listening_corpus/couperin_concerts/couperin_concerts.datapackage.json"
    )
    D = Dataset.from_package(package_path)
    grouped_D = groupers.HasCadenceAnnotations().process(D)
    grouped_D = groupers.HasCadenceAnnotations().process(D)
