from logging import error, warning
from typing import Optional, List
import requests
from defs import GKG_API_KEY_FILEPATH, GKG_SERVICE_URL
from knowledge_graph.engines.kg_engine import KnowledgeGraphEngine
from knowledge_graph.knowledge_graph_class import Fact


class GoogleKGEngine(KnowledgeGraphEngine):
    api_key = open(GKG_API_KEY_FILEPATH).read()
    service_url = GKG_SERVICE_URL

    def __init__(self, *args, **kwargs):
        super(GoogleKGEngine, self).__init__(*args, **kwargs)

    def __call__(self, word: str) -> Optional[List[Fact]]:
        http_response = requests.get(GKG_SERVICE_URL, params=GoogleKGEngine._build_query(word))
        if http_response.ok:
            try:
                result = http_response.json()
                list_key = "itemListElement"
                if list_key in result and isinstance(result[list_key], list):
                    facts = [kl for kl in (GoogleKGEngine._parse_single_result(r['result']) for r in result[list_key])
                             if kl is not None]
                    if len(facts) > 0:
                        return facts
            except ValueError as e:
                error("Couldn't parse HTTP response as JSON.")
                error(e)
        else:
            error(http_response.text)
        return None

    @staticmethod
    def _parse_single_result(result):
        try:
            return Fact(
                entity=result['name'],
                content=result['detailedDescription']['articleBody'],
                extra_information={
                    'short_description': result['description'] if 'description' in result else "",
                    'entity_type': result['@type']
                }
            )
        except KeyError:
            pass
        return None

    @staticmethod
    def _build_query(keyword: str, limit=10, indent=True):
        return {
            "query": keyword,
            "limit": limit,
            "indent": indent,
            "key": GoogleKGEngine.api_key
        }
