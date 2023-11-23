import dimcat
from dimcat import deserialize_dict

if __name__ == "__main__":
    package_path = "/home/laser/git/dimcat/docs/mwe/dcml_corpora.datapackage.json"
    D = dimcat.Dataset.from_package(package_path)
    pipeline_specs = {
        "dtype": "Pipeline",
        "steps": [
            {
                "dtype": "ModeGrouper",
                "features": [],
                "level_name": "mode",
                "grouped_column": "localkey_mode",
            },
            {
                "dtype": "FeatureExtractor",
                "features": [
                    {"dtype": "DimcatConfig", "options": {"dtype": "BassNotes"}}
                ],
            },
        ],
    }
    PL = deserialize_dict(pipeline_specs)
    processed_D = PL.process(D)
