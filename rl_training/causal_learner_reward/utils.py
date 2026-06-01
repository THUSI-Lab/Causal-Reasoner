
import json
import re
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)






def normalize_text(s: str) -> str:

    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    return re.sub(r'[^\w\s]', '', s.lower()).strip()


def _tokenize_for_matching(text: str) -> Set[str]:

    text = text.lower().strip()
    if not text:
        return set()

    return set(w for w in re.split(r'[\s\W]+', text) if w)


def token_overlap(pred: str, gold: str) -> float:

    pred_tokens = set(normalize_text(pred).split())
    gold_tokens = set(normalize_text(gold).split())
    if not gold_tokens:
        return 0.0
    return len(pred_tokens & gold_tokens) / len(gold_tokens)


def _contains_keyword(text: str, keywords: Set[str]) -> bool:

    text_lower = text.lower()
    tokens = _tokenize_for_matching(text)
    for kw in keywords:
        if ' ' in kw or '-' in kw:

            if kw in text_lower:
                return True
        else:

            if kw in tokens:
                return True
    return False






def parse_model_json(solution_str: str) -> Optional[Any]:

    if not solution_str or not solution_str.strip():
        return None


    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", solution_str)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass


    try:
        return json.loads(solution_str.strip())
    except json.JSONDecodeError:
        pass






    for pattern in [r'\{[^{}]*\}', r'\[[\s\S]*?\]', r'\{[\s\S]*?\}', r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
        m2 = re.search(pattern, solution_str)
        if m2:
            try:
                return json.loads(m2.group(0))
            except json.JSONDecodeError:
                continue

    return None


def safe_json_loads(s: str, default: Any = None) -> Any:

    if not isinstance(s, str):
        return s if s is not None else default
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return default






def extract_entities_from_facts(facts: List[Dict]) -> Set[str]:

    entities = set()

    entity_keys = {
        'entity', 'object', 'subject', 'name',
        'patient', 'tool', 'item',
        'target', 'agent', 'receiver',                   
        'artifact', 'resource', 'actor'                      
    }

    for fact in facts:
        if not isinstance(fact, dict):
            continue
        for key, value in fact.items():
            if key.lower() in entity_keys or key.lower().endswith(('_entity', '_object', '_target', '_agent')):
                if isinstance(value, str) and value.strip():
                    entities.add(normalize_text(value))

    return entities
