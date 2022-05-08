"""
Commandline interface
"""

import argparse
import logging
import os
import sys

from ms3.cli import check_and_create, check_dir

from .analyzer import ChordSymbolBigrams, ChordSymbolUnigrams
from .data import Corpus
from .writer import TSVwriter

__author__ = "Digital and Cognitive Musicology Lab"
__copyright__ = "École Polytechnique Fédérale de Lausanne"
__license__ = "GPL-3.0-or-later"

_logger = logging.getLogger("DiMCAT")


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "D": logging.DEBUG,
        "I": logging.INFO,
        "W": logging.WARNING,
        "E": logging.ERROR,
        "C": logging.CRITICAL,
    }
    loglevel = levels[loglevel.upper()]
    logformat = "[%(asctime)s] %(name)s-%(levelname)s: %(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def unigrams(corpus):
    """"""
    _logger.debug("Called unigrams...")
    return ChordSymbolUnigrams().process_data(corpus)


def bigrams(corpus):
    """"""
    _logger.debug("Called bigrams...")
    return ChordSymbolBigrams().process_data(corpus)


def get_arg_parser():
    # reusable argument sets
    input_args = argparse.ArgumentParser(add_help=False)
    input_args.add_argument(
        "-d",
        "--dir",
        metavar="DIR",
        default=os.getcwd(),
        type=check_dir,
        help="Folder(s) that will be scanned for corpora to analyze.",
    )
    input_args.add_argument(
        "-o",
        "--out",
        metavar="OUT_DIR",
        type=check_and_create,
        help="""Output directory.""",
    )
    input_args.add_argument(
        "-l",
        "--loglevel",
        metavar="{c, e, w, i, d}",
        default="i",
        help="Choose how many log messages you want to see: "
        "c (none), e, w, i, d (maximum)",
    )

    # main argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
---------------------------------------------------------------------
| Welcome to DiMCAT, the Digital Musicology Corpus Analysis Toolkit |
---------------------------------------------------------------------

The library offers you the following commands. Add the flag -h to one of them to learn about it.
""",
    )
    subparsers = parser.add_subparsers(
        help="The analysis that you want to perform.", dest="action"
    )

    unigrams_parser = subparsers.add_parser(
        "unigrams",
        help="Turn annotation tables into chord counts.",
        parents=[input_args],
    )
    unigrams_parser.set_defaults(func=unigrams)

    bigrams_parser = subparsers.add_parser(
        "bigrams",
        help="Turn annotation tables into counts of chord transitions.",
        parents=[input_args],
    )
    bigrams_parser.set_defaults(func=bigrams)

    return parser


def main(args):
    setup_logging(args.loglevel)
    corpus = Corpus(directory=args.dir)
    if len(corpus.pieces) == 0:
        _logger.error(f"Didn't find anything to analyze here in {args.dir}")
        return
    processed = args.func(corpus)
    if args.out is not None:
        _ = TSVwriter(directory=args.out, suffix=args.func.__name__).process_data(
            processed
        )
    else:
        _logger.info("Successfully analyzed but no output was generated (use --out).")


def run():
    parser = get_arg_parser()
    args = parser.parse_args()
    if "func" not in args:
        parser.print_help()
        return
    main(args)


if __name__ == "__main__":
    run()
