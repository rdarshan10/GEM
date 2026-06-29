"""Unit 9 — the demo agent: the 90-second watchable moment.

Two agents share the SAME facts and the SAME question. One is backed by GEM (cascade ON);
the other by a flat memory (cascade OFF — resolves the direct conflict but never propagates).
After "I moved to Mumbai", the GEM agent's downstream answer changes correctly (its commute /
schedule went stale and it says so), while the flat agent confidently repeats the now-wrong
Bangalore-based answer because nothing told it those facts depend on the city.

Run:  python -m gem.demo_agent
Needs Ollama reachable (OLLAMA_HOST) with the default model (gpt-oss:120b-cloud).
"""

from __future__ import annotations

import numpy as np

from .engine import GEM, GEMConfig
from .embed import cosine
from .store import Status
from .llm import OllamaClient, LLMConfig

AGENT_SYSTEM = (
    "You are an assistant answering from the user's stored memory. Use ONLY the memory facts "
    "provided. A fact tagged [STALE] or [NEEDS REVIEW] must NOT be trusted for its value — if "
    "it is relevant, tell the user that detail is out of date and needs reconfirming after their "
    "recent change. Be concise (2-3 sentences)."
)


def _retrieve(g: GEM, question: str, k: int = 6):
    """Top-k memories by embedding similarity, INCLUDING stale ones (so the agent can flag
    them). Plain numpy cosine — the demo doesn't need FAISS."""
    q = g.embedder.embed(question)
    scored = [(n, cosine(q, n.embedding)) for n in g.store.all_nodes()
              if n.embedding is not None]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [n for n, _ in scored[:k]]


def _facts_block(nodes) -> str:
    lines = []
    for n in nodes:
        tag = ""
        if n.status != Status.ACTIVE:
            tag = f" [{n.status.value}]"
        elif n.meta.get("needs_review"):
            tag = " [NEEDS REVIEW]"
        lines.append(f"- {n.content}{tag}")
    return "\n".join(lines)


class Agent:
    def __init__(self, llm: OllamaClient, cascade: bool):
        self.gem = GEM(llm=llm, config=GEMConfig(cascade_enabled=cascade))

    def learn(self, fact: str, parents=None):
        return self.gem.ingest(fact, parents=parents, check_conflicts=False)

    def observe(self, fact: str):
        """A new observation that may conflict with / update memory (drives the cascade)."""
        return self.gem.ingest(fact, parents=[])

    def ask(self, question: str) -> str:
        nodes = _retrieve(self.gem, question)
        return self.gem.llm.chat(
            AGENT_SYSTEM, f"MEMORY:\n{_facts_block(nodes)}\n\nQUESTION: {question}"
        )


def build(agent: Agent):
    loc = agent.learn("I live in Bangalore")
    com = agent.learn("My commute to work is 45 minutes", parents=[loc.id])
    wak = agent.learn("I wake at 7am to beat the Bangalore traffic", parents=[com.id])
    agent.learn("My daily briefing is scheduled for 6:45am", parents=[wak.id])
    agent.learn("My timezone is IST", parents=[loc.id])


def main():
    llm = OllamaClient(LLMConfig())
    gem_agent = Agent(llm, cascade=True)
    flat_agent = Agent(llm, cascade=False)
    build(gem_agent)
    build(flat_agent)

    Q = "What time is my daily briefing, and is it still accurate?"

    print("=" * 74)
    print("BEFORE the move — both agents answer the same way")
    print("=" * 74)
    print(f"Q: {Q}\n")
    print(f"  GEM agent : {gem_agent.ask(Q)}\n")
    print(f"  flat agent: {flat_agent.ask(Q)}")

    print("\n" + "=" * 74)
    print('EVENT: user says "I now live in Mumbai"')
    print("=" * 74)
    gem_agent.observe("I now live in Mumbai")
    flat_agent.observe("I now live in Mumbai")
    gem_stale = [n.content for n in gem_agent.gem.store.all_nodes()
                 if n.status != Status.ACTIVE]
    flat_stale = [n.content for n in flat_agent.gem.store.all_nodes()
                  if n.status != Status.ACTIVE]
    print(f"  GEM  marked stale: {gem_stale}")
    print(f"  flat marked stale: {flat_stale}")

    print("\n" + "=" * 74)
    print("AFTER the move — same question, watch the answers diverge")
    print("=" * 74)
    print(f"Q: {Q}\n")
    print(f"  GEM agent : {gem_agent.ask(Q)}\n")
    print(f"  flat agent: {flat_agent.ask(Q)}")
    print("\n(The flat agent still answers from the stale Bangalore-based schedule; the GEM")
    print(" agent knows those facts went stale when the city changed — that's the cascade.)")


if __name__ == "__main__":
    main()
