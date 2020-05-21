"""Snips NLU training/recognize methods for Rhasspy."""
import io
import logging
import shutil
import tempfile
import typing
from pathlib import Path

import networkx as nx
import rhasspynlu
from rhasspynlu.intent import Entity, Intent, Recognition
from snips_nlu import SnipsNLUEngine
from snips_nlu.dataset import Dataset
from snips_nlu.default_configs import DEFAULT_CONFIGS

_LOGGER = logging.getLogger("rhasspysnips_nlu")

# -----------------------------------------------------------------------------


def train(
    sentences_dict: typing.Dict[str, str],
    language: str,
    slots_dict: typing.Optional[typing.Dict[str, typing.List[str]]] = None,
    engine_path: typing.Optional[typing.Union[str, Path]] = None,
    dataset_path: typing.Optional[typing.Union[str, Path]] = None,
) -> SnipsNLUEngine:
    """Generate Snips YAML dataset from Rhasspy sentences/slots."""
    slots_dict = slots_dict or {}

    _LOGGER.debug("Creating Snips engine for language %s", language)
    engine = SnipsNLUEngine(config=DEFAULT_CONFIGS[language])

    # Parse JSGF sentences
    _LOGGER.debug("Parsing sentences")
    with io.StringIO() as ini_file:
        # Join as single ini file
        for lines in sentences_dict.values():
            print(lines, file=ini_file)
            print("", file=ini_file)

        intents = rhasspynlu.parse_ini(ini_file.getvalue())

    # Split into sentences and rule/slot replacements
    sentences, replacements = rhasspynlu.ini_jsgf.split_rules(intents)

    for intent_sentences in sentences.values():
        for sentence in intent_sentences:
            rhasspynlu.jsgf.walk_expression(
                sentence, rhasspynlu.number_range_transform, replacements
            )

    # Convert to directed graph *without* expanding slots
    # (e.g., $rhasspy/number)
    _LOGGER.debug("Converting to intent graph")
    intent_graph = rhasspynlu.sentences_to_graph(
        sentences, replacements=replacements, expand_slots=False
    )

    # Get start/end nodes for graph
    start_node, end_node = rhasspynlu.jsgf_graph.get_start_end_nodes(intent_graph)
    assert (start_node is not None) and (
        end_node is not None
    ), "Missing start/end node(s)"

    if dataset_path:
        # Use user file
        dataset_file = open(dataset_path, "w+")
    else:
        # Use temporary file
        dataset_file = typing.cast(
            typing.TextIO, tempfile.NamedTemporaryFile(suffix=".yml", mode="w+")
        )
        dataset_path = dataset_file.name

    with dataset_file:
        _LOGGER.debug("Writing YAML dataset to %s", dataset_path)

        # Walk first layer of edges with intents
        for _, intent_node, edge_data in intent_graph.edges(start_node, data=True):
            intent_name: str = edge_data["olabel"][9:]

            # New intent
            print("---", file=dataset_file)
            print("type: intent", file=dataset_file)
            print("name:", quote(intent_name), file=dataset_file)
            print("utterances:", file=dataset_file)

            # Get all paths through the graph (utterances)
            used_utterances: typing.Set[str] = set()
            paths = nx.all_simple_paths(intent_graph, intent_node, end_node)
            for path in paths:
                utterance = []
                entity_name = None
                slot_name = None
                slot_value = None

                # Walk utterance edges
                for from_node, to_node in rhasspynlu.utils.pairwise(path):
                    edge_data = intent_graph.edges[(from_node, to_node)]
                    ilabel = edge_data.get("ilabel")
                    olabel = edge_data.get("olabel")
                    if olabel:
                        if olabel.startswith("__begin__"):
                            slot_name = olabel[9:]
                            entity_name = None
                            slot_value = ""
                        elif olabel.startswith("__end__"):
                            if entity_name == "rhasspy/number":
                                # Transform to Snips number
                                entity_name = "snips/number"
                            elif not entity_name:
                                # Collect actual value
                                assert (
                                    slot_name and slot_value
                                ), f"No slot name or value (name={slot_name}, value={slot_value})"

                                entity_name = slot_name
                                slot_values = slots_dict.get(slot_name)
                                if not slot_values:
                                    slot_values = []
                                    slots_dict[slot_name] = slot_values

                                slot_values.append(slot_value.strip())

                            # Reference slot/entity (values will be added later)
                            utterance.append(f"[{slot_name}:{entity_name}]")

                            # Reset current slot/entity
                            entity_name = None
                            slot_name = None
                            slot_value = None
                        elif olabel.startswith("__source__"):
                            # Use Rhasspy slot name as entity
                            entity_name = olabel[10:]

                    if ilabel:
                        # Add to current slot/entity value
                        if slot_name and (not entity_name):
                            slot_value += ilabel + " "
                        else:
                            # Add directly to utterance
                            utterance.append(ilabel)
                    elif (
                        olabel
                        and (not olabel.startswith("__"))
                        and slot_name
                        and (not slot_value)
                        and (not entity_name)
                    ):
                        slot_value += olabel + " "

                if utterance:
                    utterance_str = " ".join(utterance)
                    if utterance_str not in used_utterances:
                        # Write utterance
                        print("  -", quote(utterance_str), file=dataset_file)
                        used_utterances.add(utterance_str)

            print("", file=dataset_file)

        # Write entities
        for slot_name, values in slots_dict.items():
            if slot_name.startswith("$"):
                # Remove arguments and $
                slot_name = slot_name.split(",")[0][1:]

            # Skip numbers
            if slot_name in {"rhasspy/number"}:
                # Should have been converted already to snips/number
                continue

            # Keep only unique values
            values_set = set(values)

            print("---", file=dataset_file)
            print("type: entity", file=dataset_file)
            print("name:", quote(slot_name), file=dataset_file)
            print("values:", file=dataset_file)

            slot_graph = rhasspynlu.sentences_to_graph(
                {
                    slot_name: [
                        rhasspynlu.jsgf.Sentence.parse(value) for value in values_set
                    ]
                }
            )

            start_node, end_node = rhasspynlu.jsgf_graph.get_start_end_nodes(slot_graph)
            n_data = slot_graph.nodes(data=True)
            for path in nx.all_simple_paths(slot_graph, start_node, end_node):
                words = []
                for node in path:
                    node_data = n_data[node]
                    word = node_data.get("word")
                    if word:
                        words.append(word)

                if words:
                    print("  -", quote(" ".join(words)), file=dataset_file)

            print("", file=dataset_file)

        # ------------
        # Train engine
        # ------------

        if engine_path:
            # Delete existing engine
            engine_path = Path(engine_path)
            engine_path.parent.mkdir(exist_ok=True)

            if engine_path.is_dir():
                # Snips will fail it the directory exists
                _LOGGER.debug("Removing existing engine at %s", engine_path)
                shutil.rmtree(engine_path)
            elif engine_path.is_file():
                _LOGGER.debug("Removing unexpected file at %s", engine_path)
                engine_path.unlink()

        _LOGGER.debug("Training engine")
        dataset_file.seek(0)
        dataset = Dataset.from_yaml_files(language, [dataset_file])
        engine = engine.fit(dataset)

    if engine_path:
        # Save engine
        engine.persist(engine_path)
        _LOGGER.debug("Engine saved to %s", engine_path)

    return engine


# -----------------------------------------------------------------------------


def recognize(
    text: str,
    engine: SnipsNLUEngine,
    slots_dict: typing.Optional[typing.Dict[str, typing.List[str]]] = None,
    slot_graphs: typing.Optional[typing.Dict[str, nx.DiGraph]] = None,
    **parse_args,
) -> typing.List[Recognition]:
    """Recognize intent using Snips NLU."""
    result = engine.parse(text, **parse_args)
    intent_name = result.get("intent", {}).get("intentName")

    if not intent_name:
        # Recognition failure
        return []

    slots_dict = slots_dict or {}
    slot_graphs = slot_graphs or {}

    recognition = Recognition(
        text=text, raw_text=text, intent=Intent(name=intent_name, confidence=1.0)
    )

    # Replace Snips slot values with Rhasspy slot values (substituted)
    for slot in result.get("slots", []):
        slot_name = slot.get("slotName")
        slot_value_dict = slot.get("value", {})
        slot_value = slot_value_dict.get("value")

        entity = Entity(
            entity=slot_name,
            source=slot.get("entity", ""),
            value=slot_value,
            raw_value=slot.get("rawValue", slot_value),
            start=slot["range"]["start"],
            end=slot["range"]["end"],
        )
        recognition.entities.append(entity)

        if (not slot_name) or (not slot_value):
            continue

        slot_graph = slot_graphs.get(slot_name)
        if not slot_graph and (slot_name in slots_dict):
            # Convert slot values to graph
            slot_graph = rhasspynlu.sentences_to_graph(
                {
                    slot_name: [
                        rhasspynlu.jsgf.Sentence.parse(slot_line)
                        for slot_line in slots_dict[slot_name]
                        if slot_line.strip()
                    ]
                }
            )

            slot_graphs[slot_name] = slot_graph

        entity.tokens = slot_value.split()
        entity.raw_tokens = list(entity.tokens)

        if slot_graph:
            # Pass Snips value through graph
            slot_recognitions = rhasspynlu.recognize(entity.tokens, slot_graph)
            if slot_recognitions:
                # Pull out substituted value and replace in Rhasspy entitiy
                new_slot_value = slot_recognitions[0].text
                entity.value = new_slot_value
                entity.tokens = new_slot_value.split()

    return [recognition]


# -----------------------------------------------------------------------------


def quote(s):
    """Surround with quotes for YAML."""
    return f'"{s}"'
