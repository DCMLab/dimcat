import dimcat

if __name__ == "__main__":
    package_path = "/home/laser/git/dimcat/docs/mwe/dcml_corpora.datapackage.json"
    D = dimcat.Dataset.from_package(package_path)
    keys = D.get_feature("keyannotations")
    keys.load()
