from __future__ import annotations

from chorus.common.types import TeacherOutput
from chorus.core.teacher.base import TeacherModel
from chorus.datasets.base import SceneAdapter


def run_teacher_stage(
    adapter: SceneAdapter,
    teacher: TeacherModel,
    granularities: list[float],
    frame_skip: int,
) -> list[TeacherOutput]:
    outputs: list[TeacherOutput] = []
    for granularity in granularities:
        outputs.append(
            teacher.run(
                adapter=adapter,
                granularity=granularity,
                frame_skip=frame_skip,
            )
        )
    return outputs