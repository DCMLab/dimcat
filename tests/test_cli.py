import os

from dimcat.cli import get_arg_parser, pcvs
from dimcat.data import Dataset
from dimcat.data.loader import DcmlLoader


def test_pcvs():
    debussy = "/home/hentsche/debussy"
    parser = get_arg_parser()
    loader = DcmlLoader(directory=debussy, file_re="l000")
    data = Dataset()
    data.attach_loader(loader)
    out = os.path.join(debussy, "pcvs")
    print(out)
    args = parser.parse_args(
        ["pcvs", "-o", out, "-w", "0.5", "-p", "pc", "--fillna", "0.0", "--round", "3"]
    )
    pcvs(data, args)
