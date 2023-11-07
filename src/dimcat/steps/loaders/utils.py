import json
import logging
import os
from typing import Iterable, Iterator, Optional, Tuple

import ms3
import music21 as m21
import pandas as pd
from dimcat.utils import scan_directory

logger = logging.getLogger(__name__)


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
            resource_name=f"{package_name}.{facet}",
            filepath=f"{package_name}.zip",
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
    facet_df_pairs = tuple(facet_df_pairs)
    if len(facet_df_pairs) == 0:
        raise ValueError("Received no data to store as a datapackage.")
    zip_path = os.path.join(directory, f"{name}.zip")
    store_facets_as_zip(facet_df_pairs, zip_path, overwrite=overwrite)
    package_descriptor = make_datapackage_descriptor(facet_df_pairs, name)
    descriptor_path = os.path.join(directory, f"{name}.datapackage.json")
    with open(descriptor_path, "w", encoding="utf-8") as f:
        json.dump(package_descriptor, f)
    return descriptor_path


def get_m21_input_extensions() -> Tuple[str, ...]:
    ext2converter = m21.converter.Converter.getSubConverterFormats()
    extensions = list(ext2converter.keys()) + [".mxl", ".krn"]
    return tuple(ext if ext[0] == "." else f".{ext}" for ext in extensions)


class PathFactory(Iterable[str]):
    def __init__(
        self,
        directory: str,
        extensions: Optional[str | Iterable[str]] = None,
        file_re: Optional[str] = None,
        folder_re: Optional[str] = None,
        exclude_re: str = r"^(\.|_)",
        recursive: bool = True,
        progress: bool = False,
        exclude_files_only: bool = False,
    ):
        """Generator of filtered file paths in ``directory``.

        Args:
          directory: Directory to be scanned for files.
          extensions: File extensions to be included (with or without leading dot). Defaults to all extensions.
          file_re, folder_re:
              Regular expressions for filtering certain file names or folder names.
              The regEx are checked with search(), not match(), allowing for fuzzy search.
          exclude_re:
              Exclude files and folders (unless ``exclude_files_only=True``) containing this regular expression.
              Excludes files starting with a dot or underscore by default, prevent by setting to None or ''.
          recursive: By default, subdirectories are recursively scanned. Pass False to scan only ``dir``.
          progress: Pass True to display the progress (useful for large directories).
          exclude_files_only:
              By default, ``exclude_re`` excludes files and folder. Pass True to exclude only files matching the regEx.

        Yields:
          Full file path.
        """
        self.directory = directory
        self.extensions = extensions
        self.file_re = file_re
        self.folder_re = folder_re
        self.exclude_re = exclude_re
        self.recursive = recursive
        self.progress = progress
        self.exclude_files_only = exclude_files_only

    def __iter__(self) -> Iterator[str]:
        yield from scan_directory(
            directory=self.directory,
            extensions=self.extensions,
            file_re=self.file_re,
            folder_re=self.folder_re,
            exclude_re=self.exclude_re,
            recursive=self.recursive,
            return_tuples=False,
            progress=self.progress,
            exclude_files_only=self.exclude_files_only,
        )
