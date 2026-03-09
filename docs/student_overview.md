# Student Overview

`student/` is the consumer side of the repository.

Its responsibilities are expected to include:

- loading exported training packs
- constructing datasets and dataloaders
- defining models, losses, metrics, and training loops
- training downstream models without depending on CHORUS internals

The intended dependency direction is:

```text
chorus/  ->  training_pack/ contract  ->  student/
```

`student/` should consume the versioned pack contract documented in `docs/training_pack_spec.md`, rather than reaching into CHORUS pipeline internals or relying on consumer-specific names embedded inside CHORUS.
