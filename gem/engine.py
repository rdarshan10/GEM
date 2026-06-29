"""GEM cascade engine — the MVP heart.

Wires the in-memory store, the classify/derive_links primitives, and the embedder into
the operators that carry the novel contribution:

  ingest    — two LLM passes: (1) conflict check vs neighbors, (2) derive_links provenance
  revise    — apply a conflict resolution to a node, then propagate
  propagate — walk DERIVED_FROM dependents, re-check each against the SPECIFIC change,
              recurse; cycle guard + semantic stop + graceful unknown-value handling
  retrieve  — similarity lookup over ACTIVE nodes (salience/decay are post-MVP stubs)

The cascade walks DERIVED_FROM edges ONLY. ASSOCIATED edges are never followed for
invalidation — that's the whole point of typing the edges.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .llm import OllamaClient, LLMConfig, make_llm
from . import classify as C
from . import embed as E
from .store import MemoryStore, Node, Edge, EdgeType, Status, Provenance


@dataclass
class GEMConfig:
    dedup_threshold: float = 0.97     # above this cosine -> merge instead of add
    assoc_threshold: float = 0.45     # above this -> ASSOCIATED edge
    candidate_k: int = 6              # neighbors considered for conflict/derive
    max_depth: int = 6                # safety backstop; semantic stop is the real limiter
    confidence_penalty: float = 0.5   # applied on PARTIALLY_UPDATES / unknown UPDATES
    cascade_enabled: bool = True      # False = flat-memory baseline (resolve direct
                                      # conflicts but never propagate down DERIVED_FROM)
    conservative_invalidation: bool = (
        os.environ.get("GEM_CONSERVATIVE", "").lower() in ("1", "true", "yes"))
    # small-model fail-safe: downgrade destructive SUPERSEDED to recoverable STALE +
    # needs_review, so an over-propagating weak model flags facts for re-confirmation
    # instead of deleting them. Recommended ON for <=7B models, OFF for capable models.
    boundary_samples: int = int(os.environ.get("GEM_BOUNDARY_SAMPLES", "1"))
    boundary_vote_temp: float = 0.5
    cache_decisions: bool = True      # memoize classify on (existing, new) -> skip repeat LLM calls
    conflict_sim_threshold: float = 0.0   # skip the conflict-check LLM call for neighbors below
                                          # this cosine similarity (clearly-unrelated -> can't be a
                                          # conflict). Measured separation: real conflicts >=0.41,
                                          # distractors <=0.32, so ~0.35 cuts cost with no accuracy
                                          # loss. Lets candidate_k be large safely. 0.0 = off.
    escalate: bool = os.environ.get("GEM_ESCALATE", "").lower() in ("1", "true", "yes")
    # escalate=True with a cheap_llm: the cheap model handles the high-volume conflict scan; only
    # DESTRUCTIVE (invalidating) first-pass decisions are confirmed on the capable model. Cuts the
    # expensive-model calls to the destructive minority. (Replaces voting when on.)
    # ESCALATION-triggered self-consistency voting (only when boundary_samples > 1): a single
    # sample decides clear cases; voting is spent ONLY on destructive (invalidating) decisions,
    # where being wrong is costly and a split vote signals uncertainty -> fail-safe. Avg ~1.3x
    # cost, not Nx. The accuracy gain is conditional (model must be >50% per sample); the
    # uncertainty detection is near-unconditional and is the part that earns the cost.


class GEM:
    def __init__(self, llm: OllamaClient | None = None,
                 embedder=None, config: GEMConfig | None = None, store=None,
                 cheap_llm=None):
        # store is duck-typed against MemoryStore's interface; pass a FalkorStore for
        # persistence. Defaults to the zero-dependency in-memory store.
        self.store = store if store is not None else MemoryStore()
        self.llm = llm or make_llm()          # the capable (confirming) model
        self.cheap_llm = cheap_llm            # optional cheap model for escalation's first pass
        self.embedder = embedder or E.default_embedder()
        self.cfg = config or GEMConfig()
        self.trace: list[str] = []    # human-readable log of cascade steps
        self._dec_cache: dict = {}    # (existing, new) -> (label, revised)
        self.stats = {"capable_calls": 0, "cheap_calls": 0, "cache_hits": 0, "sim_skipped": 0}

    def _classify(self, existing: str, new: str):
        """Cost-aware classify: cache + optional escalation. Cheap model first-passes the
        high-volume conflict scan; only DESTRUCTIVE first-pass labels are confirmed on the
        capable model. Falls back to capable-only when escalation isn't configured."""
        key = (existing, new)
        if self.cfg.cache_decisions and key in self._dec_cache:
            self.stats["cache_hits"] += 1
            return self._dec_cache[key]
        if self.cfg.escalate and self.cheap_llm is not None:
            label, revised = C.classify(self.cheap_llm, existing, new)
            self.stats["cheap_calls"] += 1
            if label in C.INVALIDATING:        # only confirm the calls that commit a change
                label, revised = C.classify(self.llm, existing, new)
                self.stats["capable_calls"] += 1
        else:
            label, revised = C.classify(self.llm, existing, new)
            self.stats["capable_calls"] += 1
        if self.cfg.cache_decisions:
            self._dec_cache[key] = (label, revised)
        return label, revised

    def _log(self, msg: str) -> None:
        self.trace.append(msg)

    # --- neighbor candidates -------------------------------------------------- #
    def _neighbors(self, emb) -> list[tuple[Node, float]]:
        return E.search(self.store, emb, k=self.cfg.candidate_k)

    # ------------------------------------------------------------------------- #
    # INGEST — two distinct passes
    # ------------------------------------------------------------------------- #
    def ingest(self, fact: str, *, provenance=Provenance.FACT,
               parents: list[str] | None = None, check_conflicts: bool = True) -> Node:
        """Insert a fact. `parents` lets scenarios pin DERIVED_FROM links explicitly;
        if None, the derive_links LLM pass infers them. `check_conflicts=False` skips the
        conflict-detection pass — used when loading known, conflict-free setup memories so
        only the trigger drives the cascade (keeps scenario setup fast and deterministic)."""
        emb = self.embedder.embed(fact)
        neighbors = self._neighbors(emb)

        # 1. dedup
        if neighbors and neighbors[0][1] > self.cfg.dedup_threshold:
            existing = neighbors[0][0]
            self._log(f"dedup: '{fact}' ~= {existing.id} (sim {neighbors[0][1]:.2f}); merged")
            return existing

        # 2. PASS A — conflict check against close neighbors BEFORE inserting
        actions: list[tuple[Node, C.Label, str | None]] = []
        if check_conflicts:
            for n, sim in neighbors:
                if sim < self.cfg.conflict_sim_threshold:   # clearly unrelated -> skip the LLM
                    self.stats["sim_skipped"] = self.stats.get("sim_skipped", 0) + 1
                    continue
                label, revised = self._classify(n.content, fact)
                if label in C.INVALIDATING:
                    actions.append((n, label, revised))

        # 3. create the node
        node = Node(id=self.store.new_id(), content=fact, embedding=emb,
                    provenance_type=provenance)
        self.store.add_node(node)

        # 4. PASS B — derive_links (causal dependency), distinct from the conflict pass
        if parents is None:
            parent_ids = C.derive_links(self.llm, fact, [n for n, _ in neighbors])
        else:
            parent_ids = parents
        for pid in parent_ids:
            self.store.add_edge(node.id, pid, EdgeType.DERIVED_FROM)
        # ASSOCIATED edges from remaining similar-but-not-parent neighbors
        for n, sim in neighbors:
            if n.id not in parent_ids and sim > self.cfg.assoc_threshold:
                self.store.add_edge(node.id, n.id, EdgeType.ASSOCIATED)

        # 5. fire revision for any conflicts this fact resolved.
        # CRITICAL: one trigger can directly conflict with SEVERAL nodes in the same
        # DERIVED_FROM chain (e.g. a deep chain where every node is relative to the same
        # root). Each conflict must NOT fire an independent cascade: the root's cascade
        # already revises the descendants, so a separate descendant-action re-enters the
        # subchain and re-revises it with a conflicting result (the deep-chain interference
        # bug). Fix: share ONE visited set across all actions and process ancestors first,
        # so a descendant already revised by an upstream cascade is skipped here.
        if actions:
            visited: set = set()
            for (n, label, revised) in self._order_actions_root_first(actions):
                if n.id in visited:
                    self._log(f"ingest conflict: {n.id} already revised by an upstream "
                              f"cascade this ingest; skip re-revision")
                    continue
                self._log(f"ingest conflict: new '{fact}' {label.value} {n.id} '{n.content}'")
                self.revise(n, label, revised, trigger=node, visited=visited)

        return node

    def _order_actions_root_first(self, actions):
        """Order conflict-actions so an ancestor is revised before any of its descendants
        in the action set. Counts, for each action node, how many OTHER action nodes are
        its DERIVED_FROM ancestors; roots (0) sort first. Keeps the cascade single-pass."""
        action_ids = {n.id for (n, _, _) in actions}

        def ancestor_count(node) -> int:
            seen, stack, cnt = set(), [node.id], 0
            while stack:
                cur = stack.pop()
                for p in self.store.derived_from_targets(cur):
                    if p.id in seen:
                        continue
                    seen.add(p.id)
                    if p.id in action_ids:
                        cnt += 1
                    stack.append(p.id)
            return cnt

        return sorted(actions, key=lambda a: ancestor_count(a[0]))

    # ------------------------------------------------------------------------- #
    # REVISE + PROPAGATE — the cascade
    # ------------------------------------------------------------------------- #
    def revise(self, node: Node, label: C.Label, revised_content: str | None, trigger: Node,
               visited: set | None = None):
        # `visited` lets a caller share one cascade-frontier across several revisions of the
        # same observation (see ingest step 5), so overlapping conflicts don't double-revise.
        self._propagate(node, label, revised_content, trigger, depth=0,
                        visited=set() if visited is None else visited)

    def _propagate(self, node: Node, label: C.Label, revised_content: str | None,
                   trigger: Node, depth: int, visited: set, certain: bool = True):
        if node.id in visited:                      # cycle guard
            self._log(f"cycle guard: skip {node.id}")
            return
        visited.add(node.id)
        if depth > self.cfg.max_depth:
            self._log(f"max depth hit at {node.id}")
            return

        old_content = node.content
        self._apply(node, label, revised_content, certain=certain)
        self.store.add_edge(trigger.id, node.id, EdgeType.CONTRADICTS)
        self.store.update_node(node)
        self._log(f"{'  ' * depth}revise {node.id}: {label.value} "
                  f"'{old_content}' -> '{node.content}'"
                  f"{'' if node.status == Status.ACTIVE else f' [{node.status.value}]'}")

        # flat-memory baseline: resolve the direct conflict but never propagate.
        if not self.cfg.cascade_enabled:
            return

        change_desc = self._describe_change(old_content, node, label)

        # walk DERIVED_FROM dependents and recurse
        for dep in self.store.dependents(node.id):
            if dep.status != Status.ACTIVE:
                continue
            # re-check against THIS specific change (divergent-parents rule):
            # is `dep` actually invalidated by what changed in `node`?
            # The dependent has an explicit DERIVED_FROM edge to `node`, so tell the
            # classifier that dependency exists — otherwise it judges surface semantics
            # and prunes genuine dependents whose rationale isn't in their own text.
            # World knowledge can still yield UNRELATED (e.g. timezone unchanged by a
            # same-zone move) — that's the correct semantic stop.
            # Two-step property decomposition: small models prune far more reliably when the
            # boundary judgment is constrained to "which property, did it change?" rather than
            # an open-ended "is this affected?".
            dep_change = (
                f"{change_desc}\n\n"
                f"The memory being checked was derived from that fact (possibly together with "
                f"others). Decide whether THIS change invalidates it, reasoning in two steps:\n"
                f"1. Which SPECIFIC property of the changed fact does this memory actually "
                f"depend on? It is often a CATEGORY, not the exact value — e.g. a location's "
                f"country / broad region / time-zone rather than the precise city; a device's "
                f"brand rather than its model; a role rather than the person.\n"
                f"2. Comparing the old and new values, was that specific property ALTERED, or "
                f"is it PRESERVED? If preserved, answer UNRELATED (the memory still holds). "
                f"Only if the property genuinely changed does the memory need revision."
            )
            dep_certain = True
            if self.cfg.escalate:
                # cheap-model bulk + capable-model confirm on destructive (handles cost)
                dep_label, dep_revised = self._classify(dep.content, dep_change)
            elif self.cfg.boundary_samples > 1:
                # legacy escalation-triggered VOTING: one sample, escalate destructive to N-vote
                dep_label, dep_revised = C.classify(self.llm, dep.content, dep_change)
                if dep_label in C.INVALIDATING:
                    dep_label, dep_revised, dep_certain = C.classify_consistent(
                        self.llm, dep.content, dep_change,
                        samples=self.cfg.boundary_samples,
                        temperature=self.cfg.boundary_vote_temp,
                        seed=(dep_label, dep_revised))
            else:
                dep_label, dep_revised = self._classify(dep.content, dep_change)
            if dep_label == C.Label.UNRELATED:
                self._log(f"{'  ' * (depth + 1)}semantic stop: {dep.id} UNRELATED to change")
                continue
            if not dep_certain:
                self._log(f"{'  ' * (depth + 1)}uncertain ({dep.id}): split vote "
                          f"-> {dep_label.value} routed to review (fail-safe)")
            self._propagate(dep, dep_label, dep_revised, trigger=node,
                            depth=depth + 1, visited=visited, certain=dep_certain)

    def _apply(self, node: Node, label: C.Label, revised_content: str | None,
               *, certain: bool = True):
        # fail-safe: a destructive supersede becomes recoverable STALE + needs_review when
        # EITHER conservative mode is on OR the decision was uncertain (split vote). An
        # uncertain decision also never applies a content rewrite — it flags for review
        # rather than committing a possibly-wrong value.
        soft = self.cfg.conservative_invalidation or not certain
        destructive = Status.STALE if soft else Status.SUPERSEDED
        if label in (C.Label.CONTRADICTS, C.Label.REPLACES):
            node.status = destructive
            if soft:
                node.meta["needs_review"] = True
        elif label == C.Label.UPDATES:
            if revised_content and certain:
                node.content = revised_content
                node.embedding = self.embedder.embed(revised_content)
            else:                                   # unknown value OR uncertain -> stale
                node.status = Status.STALE
                node.confidence *= self.cfg.confidence_penalty
                if not certain:
                    node.meta["needs_review"] = True
        elif label == C.Label.PARTIALLY_UPDATES:
            if revised_content and certain:
                node.content = revised_content
                node.embedding = self.embedder.embed(revised_content)
            node.confidence *= self.cfg.confidence_penalty
            if not certain:
                node.meta["needs_review"] = True

    @staticmethod
    def _describe_change(old_content: str, node: Node, label: C.Label) -> str:
        """Natural-language description of what changed, fed into classify for each
        dependent — so the check is always relative to the SPECIFIC upstream change.
        Crucially, a PARTIALLY_UPDATES rewrite must still signal that the underlying VALUE
        is now uncertain, or the chain breaks when the rewritten text reads as benign."""
        if node.status == Status.SUPERSEDED:
            return f"The fact '{old_content}' has been superseded and is no longer valid."
        if node.status == Status.STALE:
            return (f"The fact '{old_content}' is no longer reliable; "
                    f"its current value is unknown.")
        if label == C.Label.PARTIALLY_UPDATES:
            return (f"The fact '{old_content}' has been partially invalidated; its specific "
                    f"value/timing is now uncertain (revised to: '{node.content}').")
        return f"The fact '{old_content}' has changed and is now: '{node.content}'."

    # ------------------------------------------------------------------------- #
    # RETRIEVE — similarity over ACTIVE nodes (salience/decay are post-MVP)
    # ------------------------------------------------------------------------- #
    def retrieve(self, query: str, k: int = 5) -> list[Node]:
        emb = self.embedder.embed(query)
        ranked = E.search(self.store, emb, k=k)
        return [n for n, _ in ranked]
