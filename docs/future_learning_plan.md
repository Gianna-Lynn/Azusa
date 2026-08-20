# Future Learning Branch Plan — Not Executed in solution/azusa-v2

This document records future work only. The current `solution/azusa-v2` branch must remain the complete
answer implementation and must not be hollowed into exercises during this phase.

After the answer branch is reviewed and accepted, create learning stages from it:

- **L1 — protocol implementation:** retain detailed protocols, function signatures, examples, and tests;
  hollow local boundary implementations such as validation, serialization, and deterministic helpers.
- **L2 — responsibility organization:** retain data models and black-box contracts; hollow orchestration across
  ingestion, incremental planning, filtering, artifact reuse, and error boundaries.
- **L3 — experiment delivery:** provide only the complete requirements, required artifacts, controlled experiment
  matrix, reporting format, and acceptance boundary; the learner delivers the end-to-end experiment project.

Future integration procedure:

1. tag/freeze the accepted answer commit;
2. derive isolated TODO branches for L1, L2, and L3 without editing the answer branch;
3. verify every TODO branch against answer-derived black-box tests;
4. integrate the selected learning topology into `main` only after review.

No TODO hollowing or `main` integration is performed as part of the current Azusa v2 implementation.
