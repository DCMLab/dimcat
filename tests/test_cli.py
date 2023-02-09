import os

from dimcat.data import Dataset
from dimcat.cli import get_arg_parser, pcvs


def test_pcvs():
    debussy = "/home/hentsche/debussy"
    parser = get_arg_parser()
    data = Dataset()
    data.data.add_dir(
        directory=debussy, file_re="l000", level="d"
    )  # abbreviating the test works only with ms3 > 1.2.1
    data.data.parse_tsv()
    data.get_indices()
    out = os.path.join(debussy, "pcvs")
    print(out)
    args = parser.parse_args(
        ["pcvs", "-o", out, "-w", "0.5", "-p", "pc", "--fillna", "0.0", "--round", "3"]
    )
    pcvs(data, args)
