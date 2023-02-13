from typing import Collection, Iterator, Optional, Tuple, Type, Union

import ms3
from dimcat.dtypes import PathLike, PieceID, PLoader, PPiece


class DcmlLoader(PLoader):
    def __init__(
        self,
        directory: Optional[Union[PathLike, Collection[PathLike]]] = None,
        parse_scores: bool = False,
        parse_tsv: bool = True,
        **kwargs
    ):
        self.parse_scores = parse_scores
        self.parse_tsv = parse_tsv
        self.directories = []
        self.loader = ms3.Parse()
        if isinstance(directory, str):
            directory = [directory]
        if directory is None:
            return
        for d in directory:
            self.add_dir(directory=d, **kwargs)

    def iter_pieces(self) -> Iterator[Tuple[PieceID, PPiece]]:
        for corpus_name, ms3_corpus in self.loader.iter_corpora():
            for fname, piece in ms3_corpus.iter_pieces():
                ID = PieceID(corpus_name, fname)
                yield ID, piece

    def set_loader(self, new_loader: ms3.Parse):
        self.loader = new_loader

    def add_dir(self, directory: PathLike, **kwargs):
        self.directories.append(directory)
        self.loader.add_dir(directory=directory, **kwargs)


def infer_data_loader(directory: str) -> Type[PLoader]:
    return DcmlLoader


if __name__ == "__main__":
    loader = DcmlLoader(directory="~/dcml_corpora")
    assert isinstance(loader, PLoader)
    for ID, piece in loader.iter_pieces():
        assert isinstance(piece, PPiece)
    print("OK")
