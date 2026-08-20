# Azusa v2 Course Topology

`solution/azusa-v2` remains the complete answer implementation. `main` carries the corresponding exercises;
the answer branch itself is never hollowed into TODOs.

The learning stages are:

- **L1 — protocol implementation:** retain detailed protocols, function signatures, examples, and tests;
  hollow local boundary implementations such as validation, serialization, and deterministic helpers.
- **L2 — responsibility organization:** retain data models and black-box contracts; hollow orchestration across
  ingestion, incremental planning, filtering, artifact reuse, and error boundaries.
- **L3 — experiment delivery:** provide only the complete requirements, required artifacts, controlled experiment
  matrix, reporting format, and acceptance boundary; the learner delivers the end-to-end experiment project.

Integration procedure:

1. freeze and verify the answer branch;
2. derive the TODO snapshot on a temporary work branch;
3. verify answer green lights and precise course red lights;
4. publish only the final course snapshot to `main`;
5. keep only `main` and `solution/azusa-v2` as long-lived branches.
