from __future__ import annotations

from chorus.datasets.base import SceneAdapter
from chorus.core.teacher.base import TeacherModel


def run_teacher_stage(
    adapter: SceneAdapter,
    teacher: TeacherModel,
    granularities: list[float],
    frame_skip: int,
) -> list[dict]:
    outputs = []
    for g in granularities:
        outputs.append(teacher.run(adapter=adapter, granularity=g, frame_skip=frame_skip))
    return outputs