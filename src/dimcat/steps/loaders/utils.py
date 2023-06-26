import json
import os
from typing import Iterable, Tuple

import ms3
import music21 as m21
import pandas as pd


def store_facets_as_zip(
    facet_df_pairs: Iterable[Tuple[str, pd.DataFrame]],
    zip_path: str,
    overwrite: bool = True,
):
    """Stores the dataframes as <name>.tsv within the given ZIP file."""
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    if os.path.isfile(zip_path):
        if overwrite:
            os.remove(zip_path)
        else:
            raise FileExistsError(
                f"File {zip_path} already exists and overwrite is set to False."
            )
    for facet, df in facet_df_pairs:
        df = ms3.no_collections_no_booleans(df)
        df.to_csv(
            zip_path,
            sep="\t",
            mode="a",
            compression=dict(method="zip", archive_name=f"{facet}.tsv"),
        )


def make_datapackage_descriptor(
    facet_df_pairs: Iterable[Tuple[str, pd.DataFrame]], package_name: str
) -> dict:
    package_descriptor = {"name": package_name, "resources": []}
    for facet, df in facet_df_pairs:
        schema = ms3.get_schema_or_url(facet, df.columns, df.index.names)
        resource_descriptor = ms3.assemble_resource_descriptor(
            name=f"{package_name}.{facet}",
            path=f"{package_name}.zip",
            schema=schema,
            innerpath=f"{facet}.tsv",
        )
        package_descriptor["resources"].append(resource_descriptor)
    return package_descriptor


def store_datapackage(
    facet_df_pairs: Iterable[Tuple[str, pd.DataFrame]],
    name: str,
    directory: str,
    overwrite: bool = True,
) -> str:
    zip_path = os.path.join(directory, f"{name}.zip")
    store_facets_as_zip(facet_df_pairs, zip_path, overwrite=overwrite)
    package_descriptor = make_datapackage_descriptor(facet_df_pairs, name)
    descriptor_path = os.path.join(directory, f"{name}.datapackage.json")
    with open(descriptor_path, "w", encoding="utf-8") as f:
        json.dump(package_descriptor, f)
    return descriptor_path


def get_m21_input_extensions() -> Tuple[str, ...]:
    ext2converter = m21.converter.Converter.getSubConverterFormats()
    return tuple(ext if ext[0] == "." else f".{ext}" for ext in ext2converter.keys())
