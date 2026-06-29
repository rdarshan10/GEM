"""Prompts for the two DISTINCT LLM passes the cascade needs.

The plan is emphatic that these are not the same operation, and conflating them is the
original GEM paper's gap:

  classify      — does the new fact CONFLICT with an existing one? (validity interaction)
  derive_links  — does the new fact causally DEPEND on existing ones? (provenance)

A memory can depend on another without conflicting, and conflict without depending.
Keep them apart.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# classify — the conflict primitive everything calls
# --------------------------------------------------------------------------- #

CLASSIFY_SYSTEM = """You compare an EXISTING memory against a NEW statement and decide how
the new statement affects the existing memory's validity. Output exactly one label:

  UPDATES            same attribute, a NEW value is given -> replace the existing value
  CONTRADICTS        new negates existing but gives NO replacement value
  PARTIALLY_UPDATES  existing is partly affected and needs a rewrite, not deletion
  EXTENDS            new adds detail; existing still fully valid
  REPLACES           a procedure/policy is superseded by a new one
  UNRELATED          different attribute / no validity interaction

Critical distinctions:
- SAME ATTRIBUTE IS REQUIRED. A conflict exists ONLY if the new statement gives a different
  value for the SAME attribute of the SAME entity. Two facts merely sharing a name, a place,
  a number, or a topic is NOT a conflict. Do not rewrite a memory just because it mentions a
  word the new statement also mentions.
  In particular: a statement about a person's RESIDENCE (where they live / a move to a city)
  does NOT conflict with a different fact that merely mentions a place for another reason
  (a doctor's location, where a business is, a citizenship/jurisdiction, a favorite venue).
  Those are DIFFERENT attributes -> UNRELATED. Changing where someone lives does not, by
  direct conflict, change any of those other facts.
- If the new statement supplies a new value for the SAME attribute, it is UPDATES (with a
  replacement), NOT CONTRADICTS. CONTRADICTS is only for negation with no new value.
- Judge DIRECT validity interaction only. If the new statement is about a DIFFERENT
  attribute, answer UNRELATED even if the two facts are topically connected or one might
  indirectly depend on the other — indirect dependence is handled elsewhere, not here.
- When a change description says a fact is "no longer reliable / value unknown", a memory
  that was computed FROM that fact is UPDATES with revised_content null (value now unknown);
  one that merely mentioned it but does not depend on its value is UNRELATED.

revised_content:
  - UPDATES / PARTIALLY_UPDATES: the corrected full text of the existing memory
  - if the correct new value is UNKNOWN: null
  - CONTRADICTS / REPLACES / EXTENDS / UNRELATED: null

Examples:
  EXISTING "I live in Bangalore" | NEW "I now live in Mumbai"
    -> UPDATES, revised_content "I live in Mumbai"
  EXISTING "My commute to work is 45 minutes" | NEW "I now live in Mumbai"
    -> UNRELATED   (different attribute; commute's dependence on location is not a DIRECT conflict)
  EXISTING "My commute to work is 45 minutes" | NEW "The fact 'I live in Bangalore' has changed and is now: 'I live in Mumbai'"
    -> UPDATES, revised_content null   (commute was derived from the location; its value is now unknown)
  EXISTING "I wake at 7am to beat the traffic" | NEW "The fact 'My commute to work is 45 minutes' is no longer reliable; its current value is unknown"
    -> PARTIALLY_UPDATES, revised_content "I wake to commute to work (wake time to be reconfirmed)"
  EXISTING "My timezone is IST" | NEW "The fact 'I live in Bangalore' has changed and is now: 'I live in Mumbai'"
    -> UNRELATED   (Mumbai is also IST; timezone does not depend on the city within the same zone)
  EXISTING "My favorite restaurant is in Chicago" | NEW "I have moved to Denver"
    -> UNRELATED   (the restaurant's location is a different attribute from where I live; sharing/
       changing a place does not make these conflict — the restaurant is still in Chicago)
  EXISTING "I hold a Canadian passport" | NEW "I have moved to Seattle"
    -> UNRELATED   (citizenship is a different attribute from residence; moving cities does not
       directly change it)

Think step by step, then respond ONLY with JSON:
{"reasoning": "...", "label": "...", "revised_content": "... or null"}"""

CLASSIFY_USER = """EXISTING memory: {existing}

NEW statement: {new}

Classify the new statement's DIRECT effect on the existing memory."""


# --------------------------------------------------------------------------- #
# derive_links — the causal-dependency (provenance) pass
# --------------------------------------------------------------------------- #

DERIVE_SYSTEM = """A NEW memory has just been created. For EACH candidate existing memory,
decide whether the new memory is causally DERIVED FROM it — i.e. the new memory's truth
DEPENDS ON that candidate.

DECISION RULE — apply to every candidate, one at a time:
  Imagine ONLY that candidate's value changed. Would the new memory then need to be
  re-checked or revised? If YES, it is DERIVED FROM that candidate. If the new memory would
  still hold unchanged, it is NOT. Check every candidate; do not stop at the first match.

A memory often depends on SEVERAL candidates — return ALL that pass the rule. In a dependency
CHAIN (A -> B -> C, each depending on the previous), link the new memory to its MOST DIRECT
cause(s), not only the ultimate root. Walk the chain to its end — the deepest links are the
ones most often missed:
  "my commute is 45 min"            DERIVED FROM "I live in Bangalore"  (move -> commute changes)
  "I wake at 7am to beat traffic"   DERIVED FROM "my commute is 45 min" (commute changes -> wake time changes)
  "my briefing is set for 6:45am, before I wake" DERIVED FROM "I wake at 7am" (wake changes -> briefing time changes)

Do NOT link mere topical relatedness — apply the change-test to be sure:
  "I live in Bangalore" is related to "Bangalore is in India", but India-membership is a fixed
  fact that would not change if you moved, so there is NO dependence.

Return ONLY the candidate ids that pass the change-test. Respond ONLY with JSON:
{"reasoning": "<for each candidate, state whether a change to it would force a revision>",
 "derived_from": ["id", ...]}"""

DERIVE_USER = """NEW memory: {new}

CANDIDATES:
{candidates}

Which candidate ids is the NEW memory causally DERIVED FROM?"""

# Confirmation pass (precision lever): re-check ONE proposed edge in isolation, strictly, to drop
# spurious edges that the bulk derive_links over-proposed. Trades a little recall for precision.
DERIVE_CONFIRM_SYSTEM = """Verify a single proposed dependency. The claim: memory A is DERIVED FROM
memory B — meaning A's validity DEPENDS ON B, so that if B changed, A would need to be re-checked.

Apply the change-test strictly: imagine ONLY B changed. Would A then need revision? Reject mere
topical relatedness or coincidental overlap — only a genuine causal/derivational dependency counts.

Respond ONLY with JSON: {"depends": true/false}"""

DERIVE_CONFIRM_USER = """A (the dependent?): {new}
B (depended on?):    {cand}

Does A genuinely depend on B — would a change to B force A to be re-checked?"""
