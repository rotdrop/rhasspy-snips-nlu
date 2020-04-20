"""Command-line interface to rhasspysnips_nlu."""
import argparse
import dataclasses
import logging
import os
import sys
import time
import typing
from pathlib import Path

import jsonlines
import networkx as nx
import rhasspynlu
from snips_nlu import SnipsNLUEngine

from . import train, recognize

_LOGGER = logging.getLogger("rhasspysnips_nlu")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(prog="rhasspysnips_nlu")
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )

    sub_parsers = parser.add_subparsers()
    sub_parsers.required = True
    sub_parsers.dest = "command"

    # -----
    # Train
    # -----
    train_parser = sub_parsers.add_parser(
        "train", help="Train Snips engine from sentences/slots"
    )
    train_parser.set_defaults(func=do_train)

    train_parser.add_argument(
        "--language",
        required=True,
        help="Snips language (de, en, es, fr, it, ja, ko, pt_br, pt_pt, zh)",
    )
    train_parser.add_argument(
        "--sentences",
        required=True,
        action="append",
        default=[],
        help="Path to sentences.ini",
    )
    train_parser.add_argument(
        "--engine-path", required=True, help="Path to save Snips NLU engine"
    )
    train_parser.add_argument("--slots-dir", help="Path to slots directory")
    train_parser.add_argument("--dataset-path", help="Path to save Snips NLU dataset")
    train_parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )

    # ---------
    # Recognize
    # ---------
    recognize_parser = sub_parsers.add_parser(
        "recognize", help="Recognize intent from text"
    )
    recognize_parser.set_defaults(func=do_recognize)
    recognize_parser.add_argument(
        "sentence", nargs="*", default=[], help="Sentences to recognize"
    )
    recognize_parser.add_argument(
        "--engine-path", required=True, help="Path to load Snips NLU engine"
    )
    recognize_parser.add_argument("--slots-dir", help="Path to slots directory")

    # -------------------------------------------------------------------------

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    args.func(args)


# -----------------------------------------------------------------------------


def do_train(args: argparse.Namespace):
    """Train Snips engine from sentences/slots."""

    # Load sentences and slots
    _LOGGER.debug("Loading sentences from %s", args.sentences)
    sentences_dict: typing.Dict[str, str] = {
        sentences_path: open(sentences_path, "r").read()
        for sentences_path in args.sentences
    }

    slots_dict: typing.Dict[str, typing.List[str]] = {}

    if args.slots_dir:
        _LOGGER.debug("Loading slots from %s", args.slots_dir)
        slots_dict = {
            slot_path.name: slot_path.read_text().splitlines()
            for slot_path in Path(args.slots_dir).glob("*")
            if slot_path.is_file()
        }

    train(
        sentences_dict,
        args.language,
        slots_dict=slots_dict,
        engine_path=args.engine_path,
        dataset_path=args.dataset_path,
    )


# -----------------------------------------------------------------------------


def do_recognize(args: argparse.Namespace):
    """Recognize intent from text."""
    _LOGGER.debug("Loading Snips engine from %s", args.engine_path)
    engine = SnipsNLUEngine.from_path(args.engine_path)

    slots_dict: typing.Dict[str, typing.List[str]] = {}

    if args.slots_dir:
        _LOGGER.debug("Loading slots from %s", args.slots_dir)
        slots_dict = {
            slot_path.name: slot_path.read_text().splitlines()
            for slot_path in Path(args.slots_dir).glob("*")
            if slot_path.is_file()
        }

    if args.sentence:
        sentences = args.sentence
    else:
        if os.isatty(sys.stdin.fileno()):
            print("Reading sentences from stdin", file=sys.stderr)

        sentences = sys.stdin

    # Process sentences
    slot_graphs: typing.Dict[str, nx.DiGraph] = {}
    try:
        for sentence in sentences:
            start_time = time.perf_counter()
            recognitions = recognize(
                sentence, engine, slots_dict=slots_dict, slot_graphs=slot_graphs
            )
            end_time = time.perf_counter()

            if recognitions:
                recognition = recognitions[0]
            else:
                recognition = rhasspynlu.Recognition.empty()

            recognition.recognize_seconds = end_time - start_time

            # TODO: Use new entity values
            recognition.tokens = sentence.split()
            recognition.raw_tokens = sentence.split()

            with jsonlines.Writer(sys.stdout) as out:
                out.write(dataclasses.asdict(recognition))

            sys.stdout.flush()
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
