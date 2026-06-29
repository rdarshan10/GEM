"""Prompts for the Unit 0 chain.

Three jobs, deliberately separated so each can be iterated and scored on its own:

  1. EXTRACT   — pull atomic facts about the tracked attribute(s) out of a messy
                 4k-token chunk. This is the silent-failure step: if extraction is
                 unreliable here, DERIVED_FROM edges will be unreliable later.
  2. RESOLVE   — given the running state for an attribute and a newly extracted
                 fact, decide whether the new fact UPDATES / CONTRADICTS / EXTENDS /
                 is UNRELATED, and what the current value becomes. (Single-hop only
                 in Unit 0 — no graph, just keep-latest-correct.)
  3. ANSWER    — given the resolved current state, answer the benchmark query with
                 just the entity (SubEM wants the gold substring present).

Conflicts in this data are stated WITHOUT explicit negation ("Maria now works at
Google" silently overrides "Maria works at Acme"), so the prompts lean on
chain-of-thought + few-shot rather than keyword spotting.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# 1. Fact extraction
# --------------------------------------------------------------------------- #

EXTRACT_SYSTEM = """You convert factual sentences into (entity, attribute, value) triples.

Each input sentence states ONE relationship: SUBJECT — RELATION — OBJECT.
Map it to:
  entity    = the SUBJECT (who/what the fact is about)
  attribute = a SHORT canonical name for the RELATION (2-4 words, no object in it)
  value     = the OBJECT only (the final name/place/word the relation points to)

Hard rules:
- value is ONLY the object. NEVER fold the relation words into the attribute or leave value empty.
- attribute is short and canonical, so the SAME relation always gets the SAME attribute
  (e.g. always "sport", never "associated with the sport of X").
- exactly ONE triple per sentence. Never merge two sentences into one triple.
- Respond with JSON ONLY, no prose.

Pattern note: many sentences invert as "The RELATION of ENTITY is VALUE".
There, ENTITY is the subject and VALUE is the final name — NOT the whole phrase.
  "The chairperson of Fatah is Mahmoud Abbas." -> entity="Fatah", attribute="chairperson", value="Mahmoud Abbas"

Examples:
  "Thomas Kyd was born in the city of London."        -> {"entity":"Thomas Kyd","attribute":"place of birth","value":"London"}
  "goaltender is associated with the sport of ice hockey." -> {"entity":"goaltender","attribute":"sport","value":"ice hockey"}
  "The author of A Wizard of Earthsea is Ursula K. Le Guin." -> {"entity":"A Wizard of Earthsea","attribute":"author","value":"Ursula K. Le Guin"}
  "The chairperson of Fatah is Mahmoud Abbas."        -> {"entity":"Fatah","attribute":"chairperson","value":"Mahmoud Abbas"}
  "The headquarters of University of Minnesota is located in the city of Minneapolis." -> {"entity":"University of Minnesota","attribute":"headquarters city","value":"Minneapolis"}
  "Nobuhiro Watsuki is famous for Rurouni Kenshin."   -> {"entity":"Nobuhiro Watsuki","attribute":"famous for","value":"Rurouni Kenshin"}
  "rugby union was created in the country of India."  -> {"entity":"rugby union","attribute":"country created in","value":"India"}
  "Victoria Beckham is married to David Beckham."     -> {"entity":"Victoria Beckham","attribute":"spouse","value":"David Beckham"}"""

EXTRACT_USER_TEMPLATE = """Convert every sentence in the passage to a triple.

Return JSON of the exact shape:
{{"facts": [{{"entity": "...", "attribute": "...", "value": "..."}}]}}

Remember: value = the OBJECT only (never empty); attribute = short canonical relation;
one triple per sentence. If there are no factual sentences, return {{"facts": []}}.

PASSAGE:
\"\"\"
{chunk}
\"\"\""""


# --------------------------------------------------------------------------- #
# 2. Single-hop conflict resolution (keep current correct value)
# --------------------------------------------------------------------------- #

RESOLVE_SYSTEM = """You maintain the CURRENT value of an attribute as new facts arrive
over time. Facts arrive in chronological order; a later fact about the same attribute
overrides an earlier one even when it does not explicitly say so.

Classify the new fact against the known current value as exactly one of:
  UPDATES     - same attribute, new value -> replace the current value
  CONTRADICTS - directly negates the current value without giving a replacement
  EXTENDS     - adds detail, current value still holds
  UNRELATED   - different attribute/entity, no interaction

Think step by step, then output the resulting current value.
Respond ONLY with JSON."""

RESOLVE_USER_TEMPLATE = """CURRENT STATE for {entity} / {attribute}: {current_value}

NEW FACT (arrived later): {entity} / {attribute} = {new_value}

Return JSON of the exact shape:
{{"reasoning": "...", "label": "UPDATES|CONTRADICTS|EXTENDS|UNRELATED", "current_value": "..."}}

Rules:
- A later value for the same attribute UPDATES, even with no negation words.
- On CONTRADICTS with no replacement value, set current_value to "UNKNOWN".
- On EXTENDS or UNRELATED, keep current_value unchanged."""


# --------------------------------------------------------------------------- #
# 3. Query answering from resolved state
# --------------------------------------------------------------------------- #

ANSWER_SYSTEM = """You answer a question using ONLY the provided current facts.
The facts already reflect the most recent, resolved values. Answer with the specific
entity/value requested and nothing else — no explanation. If the facts do not contain
the answer, reply with your single best guess from the facts."""

ANSWER_USER_TEMPLATE = """CURRENT FACTS:
{facts_block}

QUESTION: {question}

Answer with just the value:"""
