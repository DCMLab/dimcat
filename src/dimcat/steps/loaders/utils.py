import json
import logging
import os
import re
from typing import Iterable, Iterator, Literal, Optional, Tuple, overload

import ms3
import music21 as m21
import pandas as pd
from dimcat.utils import resolve_path
from tqdm.auto import tqdm

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
            name=f"{package_name}.{facet}",
            path=f"{package_name}.zip",
            schema=schema,
            innerpath=f"{facet}.tsv",
        )
        package_descriptor["resources"].append(resource_descriptor)
    return package_descriptor


def make_extension_regex(extensions: Iterable[str]) -> re.Pattern:
    """Turns file extensions into a regular expression."""
    if isinstance(extensions, str):
        extensions = [extensions]
    else:
        extensions = list(extensions)
    if not extensions:
        return re.compile(".*")
    dot = r"\."
    regex = f"(?:{'|'.join(dot + e.lstrip('.') for e in extensions)})$"
    return re.compile(regex, re.IGNORECASE)


@overload
def scan_directory(
    directory,
    extensions,
    file_re,
    folder_re,
    exclude_re,
    recursive,
    return_tuples: Literal[False],
    progress,
    exclude_files_only,
) -> Iterator[str]:
    ...


@overload
def scan_directory(
    directory,
    extensions,
    file_re,
    folder_re,
    exclude_re,
    recursive,
    return_tuples: Literal[True],
    progress,
    exclude_files_only,
) -> Iterator[Tuple[str, str]]:
    ...


def scan_directory(
    directory: str,
    extensions: Optional[str | Iterable[str]] = None,
    file_re: Optional[str] = None,
    folder_re: Optional[str] = None,
    exclude_re: str = r"^(\.|_)",
    recursive: bool = True,
    return_tuples: bool = False,
    progress: bool = False,
    exclude_files_only: bool = False,
) -> Iterator[str] | Iterator[Tuple[str, str]]:
    """Depth-first generator of filtered file paths in ``directory``.

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
      return_tuples: By default, full file paths are returned. Pass True to return (path, name) tuples instead.
      progress: Pass True to display the progress (useful for large directories).
      exclude_files_only:
          By default, ``exclude_re`` excludes files and folder. Pass True to exclude only files matching the regEx.

    Yields:
      Full file path or, if ``return_tuples=True``, (path, file_name) pairs in random order.
    """
    if file_re is None:
        file_re = r".*"
    if folder_re is None:
        folder_re = r".*"
    extensions_regex = ".*" if extensions is None else make_extension_regex(extensions)

    def traverse(d):
        nonlocal counter

        def check_regex(reg, s, excl=exclude_re):
            try:
                passing = re.search(reg, s) is not None and re.search(excl, s) is None
            except Exception:
                print(reg)
                raise
            return passing

        for dir_entry in os.scandir(d):
            name = dir_entry.name
            path = os.path.join(d, name)
            if dir_entry.is_dir() and (recursive or folder_re != ".*"):
                for res in traverse(path):
                    yield res
            else:
                if pbar is not None:
                    pbar.update()
                if folder_re == ".*":
                    folder_passes = True
                else:
                    folder_path = os.path.dirname(path)
                    if recursive:
                        folder_passes = check_regex(
                            folder_re, folder_path, excl="^$"
                        )  # passes if the folder path matches the regex
                    else:
                        folder = os.path.basename(folder_path)
                        folder_passes = check_regex(
                            folder_re, folder, excl="^$"
                        )  # passes if the folder name itself matches the regex
                    if (
                        folder_passes and not exclude_files_only
                    ):  # True if the exclude_re should also exclude folder names
                        folder_passes = check_regex(
                            folder_re, folder_path
                        )  # is false if any part of the folder path matches exclude_re
                if (
                    dir_entry.is_file()
                    and folder_passes
                    and check_regex(file_re, name)
                    and check_regex(extensions_regex, name)
                ):
                    counter += 1
                    if pbar is not None:
                        pbar.set_postfix({"selected": counter})
                    if return_tuples:
                        yield d, name
                    else:
                        yield path

    if exclude_re is None or exclude_re == "":
        exclude_re = "^$"
    directory = resolve_path(directory)
    counter = 0
    if not os.path.isdir(directory):
        raise NotADirectoryError("Not an existing directory: " + directory)
    pbar = tqdm(desc="Scanning files", unit=" files") if progress else None
    return traverse(directory)


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
    extensions = list(ext2converter.keys()) + [".mxl"]
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
