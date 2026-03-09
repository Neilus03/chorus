# CHORUS Overview

`chorus/` is the producer side of the repository.

Its responsibilities are:

- preparing or loading scene data
- running the teacher and clustering pipeline
- evaluating scene outputs
- exporting the versioned `training_pack/` contract for downstream consumers

The important architectural rule is that CHORUS should remain generic. It produces supervision packs and evaluation outputs, but it should not adopt consumer-specific naming or assumptions.

The stable handoff from CHORUS to downstream training code is the documented training-pack contract in `docs/training_pack_spec.md`.
