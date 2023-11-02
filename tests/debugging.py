import dimcat
from dimcat import analyzers, groupers

if __name__ == "__main__":
    package_path = "/home/laser/git/dimcat/docs/mwe/dcml_corpora.datapackage.json"
    D = dimcat.Dataset.from_package(package_path)
    grouped_D = groupers.YearGrouper().process(D)
    grouped_notes = grouped_D.get_feature("notes")
    note_proportions = analyzers.Proportions().process(grouped_notes)
