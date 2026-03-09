# CHORUS Monorepo

This repository contains two separate but connected subprojects:

- `chorus/`: the generic CHORUS pipeline that produces scene outputs and training-pack supervision.
- `student/`: the downstream student-training project that consumes the exported training pack as an input contract.

The key boundary between the two is the versioned `training_pack/` format written by CHORUS. That contract is documented in `docs/training_pack_spec.md`.

Additional documentation:

- `docs/chorus_overview.md`
- `docs/student_overview.md`
- `docs/training_pack_spec.md`
