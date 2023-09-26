import dimcat
from dimcat.steps import analyzers

if __name__ == "__main__":
    package_path = "/home/laser/all_subcorpora/couperin_concerts/couperin_concerts.datapackage.json"
    D = dimcat.Dataset.from_package(package_path)
    _ = D.get_feature("harmonylabels")
    aD = analyzers.Counter("bassnotes").process(D)
    print(aD)
