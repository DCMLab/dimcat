"""
Commandline interface
"""

import argparse
import logging
import os
import sys

from ms3.cli import check_and_create, check_dir

from .analyzer import ChordSymbolBigrams, ChordSymbolUnigrams, PitchClassVectors
from .data import Corpus
from .grouper import CorpusGrouper, ModeGrouper, PieceGrouper, YearGrouper
from .pipeline import Pipeline
from .slicer import LocalKeySlicer, NoteSlicer
from .writer import TSVWriter

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


def unigrams(corpus, _):
    """"""
    return ChordSymbolUnigrams().process_data(corpus)


def bigrams(corpus, _):
    """"""
    return ChordSymbolBigrams().process_data(corpus)


def pcvs(corpus, args):
    return PitchClassVectors(
        pitch_class_format=args.pc_format, normalize=args.normalize
    ).process_data(corpus)


def get_arg_parser():
    # re-usable argument set for subcommands
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
        "-g",
        "--groupers",
        metavar="{CorpusGrouper, YearGrouper, ModeGrouper}",
        nargs="+",
        help="""List of slicers to apply before analyzing.""",
    )
    input_args.add_argument(
        "-s",
        "--slicers",
        metavar="{NoteSlicer, LocalKeySlicer}",
        nargs="+",
        help="""List of slicers to apply before analyzing.""",
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

    pcvs_parser = subparsers.add_parser(
        "pcvs",
        help="Turn note lists into pitch class vectors.",
        parents=[input_args],
    )
    pcvs_parser.add_argument(
        "-p",
        "--pc_format",
        help="Defines the type of pitch classes. Can be tpc (default), name, midi, or pc",
        default="tpc",
    )
    pcvs_parser.add_argument(
        "--normalize",
        help="Normalize the pitch class vectors instead of absolute durations in quarters.",
        action="store_true",
    )
    pcvs_parser.set_defaults(func=pcvs)

    return parser


def apply_pipeline(corpus, slicers, groupers):
    if slicers is None:
        slicers = []
    if groupers is None:
        groupers = []
    if len(slicers) == 0 and len(groupers) == 0:
        return corpus
    _logger.info("Assembling pipeline...")
    available_slicers = {"noteslicer": NoteSlicer(), "localkeyslicer": LocalKeySlicer()}
    available_groupers = {
        "corpusgrouper": CorpusGrouper(),
        "piecegrouper": PieceGrouper(),
        "yeargrouper": YearGrouper(),
        "modegrouper": ModeGrouper(),
    }
    slicers = [sl.lower() for sl in slicers]
    groupers = [gr.lower() for gr in groupers]
    if "modegrouper" in groupers and "localkeyslicer" not in slicers:
        slicers.append("localkeyslicer")
        _logger.info(
            "Automatically added LocalKeySlicer() to pipeline which is necessary"
            "for the ModeGrouper() to work."
        )
    slicer_steps = [available_slicers[sl] for sl in slicers]
    grouper_steps = [available_groupers[gr] for gr in groupers]
    steps = slicer_steps + grouper_steps
    _logger.info(f"Result: {steps}")
    pipeline = Pipeline(steps)
    return pipeline.process_data(corpus)


def main(args):
    setup_logging(args.loglevel)
    corpus = Corpus(directory=args.dir)
    if len(corpus.pieces) == 0:
        _logger.error(f"Didn't find anything to analyze here in {args.dir}")
        return
    pre_processed = apply_pipeline(corpus, args.slicers, args.groupers)
    processed = args.func(pre_processed, args)
    if args.out is not None:
        _ = TSVWriter(directory=args.out, prefix=args.func.__name__).process_data(
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
    if args.out is None:
        print(
            "No output directory specified. Type y to use current working directory or any"
            "other key to not write any output."
        )
        answer = input("(y/n)> ")
        if answer.lower() == "y":
            args.out = os.getcwd()
    print(args.__dict__)
    main(args)


if __name__ == "__main__":
    run()
