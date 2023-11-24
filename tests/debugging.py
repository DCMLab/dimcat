import os

from dimcat.data import packages


def resource_names(path):
    return os.sep.join(path.split(os.sep)[-2:])


if __name__ == "__main__":
    corpus_path = "/home/laser/git/dimcat/unittest_metacorpus"

    package = packages.Package.from_directory(
        corpus_path,
        package_name="unittest_corpus",
        exclude_re="(?:yml|py)$",  # needed as long as the three corpora contain the .github/workflow
        resource_names=resource_names,
    )
