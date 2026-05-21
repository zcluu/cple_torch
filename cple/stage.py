from __future__ import annotations

from collections import defaultdict, deque

from .api import CPLEStage


class StageDAGExecutor:
    def order(self, stages: list[CPLEStage]) -> list[CPLEStage]:
        self.validate(stages)
        by_name = {stage.name: stage for stage in stages}
        indegree = {stage.name: 0 for stage in stages}
        children: dict[str, list[str]] = defaultdict(list)
        for stage in stages:
            for dep in stage.depends_on:
                indegree[stage.name] += 1
                children[dep].append(stage.name)
        queue = deque([name for name, degree in indegree.items() if degree == 0])
        result: list[str] = []
        while queue:
            name = queue.popleft()
            result.append(name)
            for child in children[name]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(result) != len(stages):
            raise ValueError("Stage DAG contains a cycle")
        return [by_name[name] for name in result]

    def validate(self, stages: list[CPLEStage]) -> None:
        names = [stage.name for stage in stages]
        if len(names) != len(set(names)):
            raise ValueError("Stage names must be unique")
        known = set(names)
        for stage in stages:
            missing = set(stage.depends_on) - known
            if missing:
                raise ValueError(f"Stage {stage.name} depends on missing stages: {sorted(missing)}")
