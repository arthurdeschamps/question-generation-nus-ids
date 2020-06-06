from abc import ABC
from typing import List, Optional

from knowledge_graph.knowledge_graph_class import Fact


class KnowledgeGraphEngine(ABC):
    def __call__(self, word: str) -> Optional[List[Fact]]:
        raise NotImplementedError()
