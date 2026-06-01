from __future__ import annotations

import json
from typing import Any, Dict, List

from causal_trace_task_prompt_registry import CAUSAL_TRACE_SYSTEM_PROMPT, task_requirement


def build_trace_messages(row: Dict[str, Any], task_name: str) -> List[Dict[str, Any]]:
    question, answer = extract_qa(row)
    meta = row.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    source_context = row.get("llm_fields") or row.get("source_context") or meta.get("llm_fields") or meta.get("source_context") or meta
    user_text = (
        f"TASK: {task_name}\n"
        f"TASK_REQUIREMENT: {task_requirement(task_name)}\n\n"
        f"SOURCE_CONTEXT:\n{json.dumps(source_context, ensure_ascii=False, indent=2)}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"TARGET_ANSWER:\n{answer}"
    )
    return [
        {"role": "system", "content": CAUSAL_TRACE_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def extract_qa(row: Dict[str, Any]) -> tuple[str, str]:
    question = str(row.get("question") or "")
    answer = str(row.get("answer") or "")
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for msg in conversations:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("from") or msg.get("role") or "").lower()
            value = str(msg.get("value") or msg.get("content") or "")
            if role in {"human", "user"} and not question:
                question = value
            if role in {"gpt", "assistant"} and not answer:
                answer = value
    return question, answer


def attach_trace(row: Dict[str, Any], trace: str) -> Dict[str, Any]:
    out = dict(row)
    question, answer = extract_qa(out)
    conversations = out.get("conversations")
    traced_answer = f"<think>{trace.strip()}</think>\n{answer}"
    if isinstance(conversations, list):
        new_conversations = []
        replaced = False
        for msg in conversations:
            if isinstance(msg, dict) and str(msg.get("from") or msg.get("role") or "").lower() in {"gpt", "assistant"} and not replaced:
                msg = dict(msg)
                if "value" in msg:
                    msg["value"] = traced_answer
                else:
                    msg["content"] = traced_answer
                replaced = True
            new_conversations.append(msg)
        out["conversations"] = new_conversations
    else:
        out["question"] = question
        out["answer"] = traced_answer
    meta = out.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    meta["causal_trace_added"] = True
    out["meta"] = meta
    return out
