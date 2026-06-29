"""Unit 0 — End-to-End Spike on Real Data (the go/no-go gate).

Runs the whole chain — chunk ingestion, fact extraction, single-hop conflict
detection, query answering — against real FactConsolidation (MemoryAgentBench
Conflict_Resolution) data, scored with SubEM exactly the way the benchmark scores it.

Nothing here depends on KuzuDB, FAISS, or the graph store. This module exists to
answer one question before any infrastructure gets built: can a local 7B model
extract facts from messy 4k-token chunks and resolve single-hop conflicts well
enough to approach the ~60% GPT-4o baseline?
"""
