"""Smoke test: can Jev make GEM's STALE decision on its own?

One cascade level (Bangalore -> Mumbai) sent as a single Jev call. For each dependent, Jev scores how
affected it is (0-3) and says whether the new value follows from the change; the routing rule from the
Jev plan then picks stop / STALE / rewrite / LLM. Prints Jev's probabilities next to the expected
action so you can eyeball whether the STALE path is safe to take without an LLM.

    put TYPESAFE_API_KEY=... in the repo-root .env (or set it in the environment)
    python stale_eval/jev_smoke.py
"""

import os
import sys
from pathlib import Path

from typesafe_sdk import Noul, Score, TypeSafeClient


def load_dotenv(path: Path = Path(__file__).resolve().parents[1] / ".env") -> None:
    """Minimal KEY=VALUE loader; real environment variables win over the file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip('"\''))

JEV_STOP = 0.9               # P(unaffected) needed to stop the cascade at a fact
UNCERTAIN = (0.35, 0.9)      # max-class confidence in this band -> fall back to the LLM

CHANGE = "The fact 'I live in Bangalore' has changed and is now: 'I live in Mumbai'"
# (dependent, expected action)
DEPENDENTS = [
    ("My commute to work is 45 minutes", "STALE"),
    ("My monthly rent is 30,000 rupees", "STALE"),
    ("My timezone is IST", "STOP"),
    ("I file my taxes in India", "STOP"),
    ("My weekend plans assume I stay in the same city", "STALE"),
    ("I visited Bangalore Palace in 2019", "STOP"),
]

AFFECTED = [
    "Unaffected: the memory is still fully true after the change.",
    "Partly affected: the memory is mostly true but one detail should be reconfirmed.",
    "Value changed: the memory's value is now different or unknown.",
    "Invalid: the memory is no longer true at all.",
]


def route(p_score: dict[int, float], p_known: float) -> str:
    if p_score.get(0, 0.0) >= JEV_STOP:
        return "STOP"
    top = max(p_score.values())
    if UNCERTAIN[0] < top < UNCERTAIN[1] and max(p_score, key=p_score.get) == 0:
        return "LLM"         # leaning unaffected but not sure enough to stop: let the LLM decide
    return "REWRITE" if p_known >= 0.5 else "STALE"


def main() -> int:
    load_dotenv()
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set (add it to .env).")
        return 2

    questions = {}
    for i, _ in enumerate(DEPENDENTS):
        questions[f"d{i}_affected"] = Score(
            instructions=(f"Which specific property of the changed fact does `dependents[{i}]` depend on "
                          "(often a category such as country, time zone or brand, not the exact value), "
                          "and did `change` alter it?"),
            criteria=AFFECTED,
        )
        questions[f"d{i}_value_known"] = Noul(
            instructions=f"If `dependents[{i}]` is affected, does `change` give enough information to state its new value?",
            criteria={"true": "The new value follows from the change.",
                      "false": "The new value is unknown, or the memory is unaffected."},
        )
    state = {
        "change": CHANGE,
        "note": "Each memory in `dependents` was derived from the changed fact.",
        "dependents": [d for d, _ in DEPENDENTS],
    }

    with TypeSafeClient() as client:
        print("models:", ", ".join(m.name for m in client.models.list().models))
        resp = client.system_one(state=state, questions=questions)

    print(f"model={resp.model}  usage={resp.usage.input_tokens} in / {resp.usage.output_tokens} out\n")
    ok = 0
    for i, (fact, expected) in enumerate(DEPENDENTS):
        score = resp.scores[f"d{i}_affected"]
        known = resp.nouls[f"d{i}_value_known"]
        p_known = known.noul
        got = route(dict(score.probabilities), p_known)
        # LLM fallback is safe (just costs a call); STOP on a fact that should go STALE is the miss we fear.
        verdict = "ok" if got == expected else ("safe" if got == "LLM" else "MISS")
        ok += verdict != "MISS"
        probs = " ".join(f"{k}:{v:.2f}" for k, v in sorted(score.probabilities.items()))
        print(f"[{verdict:4}] {fact!r}\n       expected={expected} got={got}  P(affected)={probs}  P(known)={p_known:.2f}")
    print(f"\n{ok}/{len(DEPENDENTS)} without a miss")
    return 0 if ok == len(DEPENDENTS) else 1


if __name__ == "__main__":
    sys.exit(main())
