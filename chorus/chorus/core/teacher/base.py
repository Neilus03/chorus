from __future__ import annotations

from abc import ABC, abstractmethod

from chorus.datasets.base import SceneAdapter


class TeacherModel(ABC):
    @abstractmethod
    def run(self, adapter: SceneAdapter, granularity: float, frame_skip: int) -> dict:
        raise NotImplementedError