"""Command-line interface to rhasspysnips_nlu."""
import argparse
import dataclasses
import json
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

from . import recognize, train

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
    recognize_parser.add_argument(
        "--json-input", action="store_true", help="Input is JSON instead of plain text"
    )

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
        slots_dir = Path(args.slots_dir)
        if slots_dir.is_dir():
            _LOGGER.debug("Loading slots from %s", args.slots_dir)
            slots_dict = {
                slot_path.name: slot_path.read_text().splitlines()
                for slot_path in slots_dir.glob("*")
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
        slots_dir = Path(args.slots_dir)
        if slots_dir.is_dir():
            _LOGGER.debug("Loading slots from %s", args.slots_dir)
            slots_dict = {
                slot_path.name: slot_path.read_text().splitlines()
                for slot_path in slots_dir.glob("*")
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
            if args.json_input:
                sentence_object = json.loads(sentence)
            else:
                sentence_object = {"text": sentence}

            text = sentence_object["text"]

            start_time = time.perf_counter()
            recognitions = recognize(
                text, engine, slots_dict=slots_dict, slot_graphs=slot_graphs,
            )
            end_time = time.perf_counter()

            if recognitions:
                recognition = recognitions[0]
            else:
                recognition = rhasspynlu.fsticuffs.Recognition.empty()

            recognition.recognize_seconds = end_time - start_time

            recognition.tokens = text.split()

            recognition.raw_text = text
            recognition.raw_tokens = list(recognition.tokens)

            recognition_dict = dataclasses.asdict(recognition)
            for key, value in recognition_dict.items():
                if (key not in sentence_object) or (value is not None):
                    sentence_object[key] = value

            with jsonlines.Writer(sys.stdout) as out:
                # pylint: disable=E1101
                out.write(sentence_object)

            sys.stdout.flush()
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
