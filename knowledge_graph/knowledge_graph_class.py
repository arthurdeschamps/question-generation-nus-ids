from __future__ import annotations
from typing import Dict, Any, List
import networkx as nx
from nltk.corpus import stopwords


class KnowledgeGraph:
    _stopwords = set(stopwords.words('english'))

    def __init__(self, facts: Dict[str, List[Fact]]):
        super(KnowledgeGraph, self).__init__()
        self.graph = KnowledgeGraph._build_graph(facts)

    def get_relevant_facts(self, relevant_to: str) -> Dict[str, List[Fact]]:
        """
        Filters out the facts to only keep those that pertain the given string.
        :param relevant_to: Text to base the filtering on.
        :return: A map between the entities of this graph and their relevant facts.
        """
        relevant_facts = {}
        for node in [n for n, attrs in self.graph.nodes(data=True) if attrs['node_type'] == 'entity']:
            relevant_facts[node] = []
            for n in self.graph.successors(node):
                node_type = nx.get_node_attributes(self.graph, 'node_type')[n]
                if KnowledgeGraph._is_relevant(n, relevant_to) and node_type == 'fact':
                    relevant_facts[node].append(n)
        return relevant_facts

    def __str__(self):
        return str(self.graph)

    @staticmethod
    def _is_relevant(fact: Fact, to: str):
        # We define a fact being relevant to a string if the fact's content has at least one non stop word in common
        # with the string to match against
        relevant_keywords = set(to.split(' ')) & set(fact.content.split(' ')) - KnowledgeGraph._stopwords
        is_relevant = len(relevant_keywords) > 0
        return is_relevant

    @staticmethod
    def _build_graph(facts: Dict[str, List[Fact]]) -> nx.DiGraph:
        graph = nx.DiGraph()
        for entity, entity_facts in facts.items():
            if not graph.has_node(entity):
                graph.add_node(entity, node_type="entity")
            for fact in entity_facts:
                graph.add_node(fact, node_type="fact")
                graph.add_edge(entity, fact)
        return graph


class Fact:

    def __init__(self, entity: str, content: str, extra_information: Dict[str, Any]):
        super(Fact, self).__init__()
        self.entity = entity
        self.content = content
        self.extra = extra_information

    def __str__(self):
        str_repr = f"({self.entity}, {self.content}"
        for k, v in self.extra.items():
            str_repr += f", {k}: {v}"
        return str_repr + ")"
