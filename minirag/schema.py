"""Historical entity schema used by the extraction pipeline.

This module is the single source of truth for the entity types and attributes
defined by NDAKG's ``schema.py``:

    BaseEntity + Person, Place, TimeEntity, Event, Artifact, ObjectEntity,
    Concept

The schema is expressed as plain data (no pydantic dependency) so that the
extraction prompt in :mod:`minirag.prompt` and the parser in
:mod:`minirag.operate` cannot drift apart.

Node attributes written to the graph are flat strings; multi-valued attributes
are joined with ``<SEP>`` exactly like ``description`` and ``source_id``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set

GRAPH_FIELD_SEP = "<SEP>"


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

# Attributes every entity carries in addition to entity_name / entity_type /
# description / source_id (which are handled directly by the pipeline).
BASE_ENTITY_ATTRIBUTES: Dict[str, str] = {
    "alternative_names": (
        "Other names, aliases or historical spellings that refer to the SAME "
        "entity. Comma-separated in the model output; stored <SEP>-joined."
    ),
    "time_range": (
        "Temporal bounds associated with the entity, e.g. '1225-1258', "
        "'1428' or 'thế kỷ XIII'. Empty when unknown."
    ),
    "geo_location": (
        "Geographical coordinates or place names associated with the entity."
    ),
}


# The seven sub-typed entity schemas. Keys are the canonical lowercase type
# names; the pipeline stores them uppercased.
ENTITY_TYPES: Dict[str, Dict[str, Any]] = {
    "person": {
        "description": (
            "A historical figure: emperor, general, official, scholar, "
            "religious figure, etc."
        ),
        "attributes": {
            "birth_date": "Birth date or birth year; empty if unknown.",
            "death_date": "Death date or death year; empty if unknown.",
        },
    },
    "place": {
        "description": (
            "A location: capital, citadel, province, battleground, mountain, "
            "river, etc."
        ),
        "attributes": {
            "location_type": (
                "Kind of place, e.g. kinh đô, thành trì, chiến địa, "
                "căn cứ địa, tỉnh."
            ),
        },
    },
    "time": {
        "description": (
            "A temporal entity: a year, a date range, a century or an era."
        ),
        "attributes": {
            "temporal_value": (
                "Normalised temporal value, e.g. '1401-1500' for thế kỷ XV."
            ),
        },
    },
    "event": {
        "description": (
            "A historical episode, battle, campaign, treaty, founding or "
            "other happening."
        ),
        "attributes": {
            "start_time": "Start date or year of the event.",
            "end_time": "End date or year of the event.",
            "location": "Place where the event happened.",
        },
    },
    "artifact": {
        "description": (
            "A historical item, work, document, inscription or other artefact."
        ),
        "attributes": {
            "creator": "Person or group who created the artefact.",
            "time_period": "Period when the artefact was created or used.",
        },
    },
    "object": {
        "description": (
            "A physical historical object that is not naturally an Artifact."
        ),
        "attributes": {},
    },
    "concept": {
        "description": (
            "An abstract ideology, policy, title, institution or social "
            "concept."
        ),
        "attributes": {},
    },
}

DEFAULT_ENTITY_TYPES: List[str] = list(ENTITY_TYPES)


# ---------------------------------------------------------------------------
# Relationship (edge) schema — mirrors NDAKG's relationship record
# ---------------------------------------------------------------------------

# Field order is significant: it is the order used in the model output tuple.
RELATIONSHIP_ATTRIBUTES: Dict[str, str] = {
    "relationship_description": (
        "Explanation of why the source entity and the target entity are "
        "related to each other."
    ),
    "relationship_keywords": (
        "One or more high-level keywords summarising the nature of the "
        "relationship, focusing on concepts or themes."
    ),
    "relationship_strength": (
        "A numeric score (e.g. 1-10) indicating the strength of the "
        "relationship between the two entities."
    ),
}

RELATIONSHIP_FIELDS: List[str] = list(RELATIONSHIP_ATTRIBUTES)

# Type names that models commonly emit but that the schema expresses with a
# different canonical label.
TYPE_ALIASES: Dict[str, str] = {
    "LOCATION": "PLACE",
    "LOCATIONS": "PLACE",
    "PLACES": "PLACE",
    "TIME_ENTITY": "TIME",
    "TIME ENTITY": "TIME",
    "TEMPORAL": "TIME",
    "DATE": "TIME",
    "DATE AND TIME": "TIME",
    "OBJECT_ENTITY": "OBJECT",
    "OBJECT ENTITY": "OBJECT",
    "ARTIFACTS": "ARTIFACT",
    "PERSONS": "PERSON",
    "PEOPLE": "PERSON",
    "EVENTS": "EVENT",
    "CONCEPTS": "CONCEPT",
}

# Common spellings of the attributes above.
ATTRIBUTE_ALIASES: Dict[str, str] = {
    "alternate_names": "alternative_names",
    "alternate_name": "alternative_names",
    "alt_names": "alternative_names",
    "aka": "alternative_names",
    "alias": "alternative_names",
    "aliases": "alternative_names",
    "other_names": "alternative_names",
    "other_name": "alternative_names",
    "name_variants": "alternative_names",
    "time": "time_range",
    "date_range": "time_range",
    "temporal_range": "time_range",
    "geo": "geo_location",
    "geolocation": "geo_location",
    "coordinates": "geo_location",
    "location_geo": "geo_location",
    "birth": "birth_date",
    "birthdate": "birth_date",
    "date_of_birth": "birth_date",
    "born": "birth_date",
    "death": "death_date",
    "deathdate": "death_date",
    "date_of_death": "death_date",
    "died": "death_date",
    "place_type": "location_type",
    "time_value": "temporal_value",
    "start_date": "start_time",
    "start_year": "start_time",
    "begin_time": "start_time",
    "end_date": "end_time",
    "end_year": "end_time",
    "finish_time": "end_time",
    "author": "creator",
    "founder": "creator",
    "era": "time_period",
    "period": "time_period",
    "event_location": "location",
}

_EMPTY_VALUES = {"", "none", "null", "n/a", "na", "unknown", "{}", "[]", "-"}


def _all_type_specific_attributes() -> Set[str]:
    attributes: Set[str] = set()
    for spec in ENTITY_TYPES.values():
        attributes.update(spec.get("attributes", {}))
    return attributes


ALL_TYPE_SPECIFIC_ATTRIBUTES: Set[str] = _all_type_specific_attributes()

# Every attribute the pipeline is allowed to persist on an entity node.
ENTITY_ATTRIBUTE_KEYS: Set[str] = set(BASE_ENTITY_ATTRIBUTES) | ALL_TYPE_SPECIFIC_ATTRIBUTES


def attributes_for_type(entity_type: str) -> Set[str]:
    """Return the attribute names valid for ``entity_type``.

    Unknown types are permissive: any documented attribute is accepted so
    information is not silently discarded when a model invents a type.
    """
    allowed = set(BASE_ENTITY_ATTRIBUTES)
    spec = ENTITY_TYPES.get(str(entity_type or "").strip().lower())
    if spec is None:
        allowed |= ALL_TYPE_SPECIFIC_ATTRIBUTES
    else:
        allowed |= set(spec.get("attributes", {}))
    return allowed


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _strip_outer(value: Any) -> str:
    text = str(value or "").strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    return text


def normalize_entity_type(value: Any) -> str:
    """Canonicalise an entity type label (uppercase, aliases resolved)."""
    text = _strip_outer(value).upper()
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return TYPE_ALIASES.get(text, text)


def normalize_attribute_key(key: Any) -> str:
    """Canonicalise an attribute name (lowercase snake_case, aliases resolved)."""
    text = _strip_outer(key).lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return ATTRIBUTE_ALIASES.get(text, text)


def split_attribute_values(value: Any) -> List[str]:
    """Split a scalar, list or ``<SEP>``-joined value into clean strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items: Iterable[Any] = value
    else:
        items = str(value).split(GRAPH_FIELD_SEP)
    result: List[str] = []
    for item in items:
        cleaned = clean_attribute_value(item)
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def clean_attribute_value(value: Any) -> str:
    """Clean a single attribute value; returns ``""`` for null-ish values."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = sorted({clean_attribute_value(v) for v in value if v is not None})
        return GRAPH_FIELD_SEP.join(p for p in parts if p)
    text = _strip_outer(value)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n,;{}")
    text = _strip_outer(text)
    text = text.strip(" \t\r\n,;")
    if text.lower() in _EMPTY_VALUES:
        return ""
    return text


def merge_attribute_value(*values: Any) -> str:
    """Union several values into a sorted, de-duplicated ``<SEP>`` string."""
    merged: List[str] = []
    for value in values:
        for item in split_attribute_values(value):
            if item not in merged:
                merged.append(item)
    return GRAPH_FIELD_SEP.join(sorted(merged, key=str.lower))


def merge_attribute_dicts(*dicts: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Merge attribute dictionaries, unioning every key's values."""
    merged: Dict[str, str] = {}
    for data in dicts:
        if not data:
            continue
        for key, value in data.items():
            normalised_key = normalize_attribute_key(key)
            if normalised_key not in ENTITY_ATTRIBUTE_KEYS:
                continue
            joined = merge_attribute_value(merged.get(normalised_key, ""), value)
            if joined:
                merged[normalised_key] = joined
    return merged


# ---------------------------------------------------------------------------
# Model-output parsing
# ---------------------------------------------------------------------------

def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON object parsing (json_repair first, then json)."""
    if not text:
        return None
    candidates = [text]
    # Models sometimes wrap the object in quotes or add a trailing marker.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is not None:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            import json_repair

            parsed = json_repair.loads(candidate)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str) and parsed.strip().startswith("{"):
            try:
                parsed = json.loads(parsed)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
        try:
            parsed = json.loads(candidate)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_key_value_pairs(text: str) -> Dict[str, Any]:
    """Fallback for ``key=value; key2=value2`` or ``key: value | ...`` output."""
    def parse_chunks(chunks: Iterable[str]) -> Dict[str, Any]:
        pairs: Dict[str, Any] = {}
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" in chunk:
                key, value = chunk.split("=", 1)
            elif ":" in chunk:
                key, value = chunk.split(":", 1)
            else:
                continue
            key = _strip_outer(key).strip("{} \t\r\n,;")
            value = _strip_outer(value).strip("{} \t\r\n,;")
            pairs[key] = value
        return pairs

    pairs = parse_chunks(re.split(r"[;|\n]+", text))
    if len(pairs) <= 1 and text.count(":") + text.count("=") > 1:
        # Single-quoted JSON without json_repair: split on commas instead.
        pairs = parse_chunks(text.split(","))
    return pairs


def parse_entity_attributes(
    raw: Any,
    entity_type: str = "",
) -> Dict[str, str]:
    """Parse the ``<attributes_json>`` slot of an entity record.

    Accepts a JSON object (possibly repaired), a JSON string, or a fallback
    ``key=value`` list. Keys are canonicalised, values are cleaned, and only
    attributes valid for ``entity_type`` are kept.
    """
    if raw is None:
        return {}

    if isinstance(raw, dict):
        data: Dict[str, Any] = raw
    else:
        text = _strip_outer(raw)
        if not text:
            return {}
        parsed = _parse_json_object(text)
        if parsed is None:
            parsed = _parse_key_value_pairs(text)
        if not isinstance(parsed, dict):
            return {}
        data = parsed

    allowed = attributes_for_type(entity_type)
    attributes: Dict[str, str] = {}
    for key, value in data.items():
        normalised_key = normalize_attribute_key(key)
        if normalised_key not in ENTITY_ATTRIBUTE_KEYS:
            continue
        if normalised_key not in allowed:
            continue
        if isinstance(value, dict):
            # Schema attributes are scalar; ignore nested structures rather
            # than storing their Python repr in the graph.
            continue
        cleaned = clean_attribute_value(value)
        if not cleaned:
            continue
        attributes[normalised_key] = merge_attribute_value(
            attributes.get(normalised_key, ""), cleaned
        )
    return attributes


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_entity_schema() -> str:
    """Render the type/attribute table embedded in the extraction prompt."""
    lines: List[str] = []
    for type_name, spec in ENTITY_TYPES.items():
        attributes: Dict[str, str] = spec.get("attributes", {})
        if attributes:
            attribute_text = "; ".join(
                f"{name} ({description})"
                for name, description in attributes.items()
            )
        else:
            attribute_text = "no type-specific attributes"
        lines.append(
            f"- {type_name.upper()}: {spec['description']} "
            f"Specific attributes: {attribute_text}"
        )
    return "\n".join(lines)


def render_relationship_schema() -> str:
    """Render the relationship-attribute list embedded in the prompt."""
    return "\n".join(
        f"- {name}: {description}"
        for name, description in RELATIONSHIP_ATTRIBUTES.items()
    )
