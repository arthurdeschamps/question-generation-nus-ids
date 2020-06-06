import collections
from typing import Union, List, Iterable

import stanza

from knowledge_graph.engines.google_kg_engine import GoogleKGEngine
from knowledge_graph.engines.kg_engine import KnowledgeGraphEngine
from knowledge_graph.knowledge_graph_class import KnowledgeGraph


class QueryEngine:

    def __init__(self, engine: KnowledgeGraphEngine = GoogleKGEngine()):
        super(QueryEngine, self).__init__()
        assert isinstance(engine, KnowledgeGraphEngine)
        self._ner = stanza.Pipeline(lang='en', processors='tokenize,ner')
        self._engine = engine

    def retrieve_knowledge_graph(self, text_source: Union[str, Iterable[str]]) -> KnowledgeGraph:
        if isinstance(text_source, str):
            return self.retrieve_knowledge_graph(self._get_named_entities(text_source))
        elif isinstance(text_source, collections.abc.Iterable):
            facts = {}
            for entity in iter(text_source):
                entity_facts = self._query(entity)
                if entity_facts is not None:
                    facts[entity] = entity_facts
            return KnowledgeGraph(facts)

    def _get_named_entities(self, paragraph: str) -> List[str]:
        return [entity.text for entity in self._ner(paragraph).ents]

    def _query(self, word: str):
        return self._engine(word)


# Example usage
if __name__ == '__main__':
    context = "A procedural consequence of the establishment of the Scottish Parliament is that " \
              "Scottish MPs sitting in the UK House of Commons are able to vote on domestic " \
              "legislation that applies only to England, Wales and Northern Ireland \u2013 whilst " \
              "English, Scottish, Welsh and Northern Irish Westminster MPs are unable to vote on " \
              "the domestic legislation of the Scottish Parliament"
    question = "What consequence of establishing the Scottish Parliament applies to Scottish MPs sitting in the " \
               "UK House of Commons?"

    query_engine = QueryEngine()
    kg = query_engine.retrieve_knowledge_graph(context)
    relevant_facts = kg.get_relevant_facts(question)
    for entity, results in relevant_facts.items():
        print(f"Entity: {entity}")
        for knowledge in results:
            print(str(knowledge))
        print()
