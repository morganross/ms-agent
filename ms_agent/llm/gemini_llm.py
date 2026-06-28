# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from ms_agent.llm.openai_llm import OpenAI
from ms_agent.utils.constants import get_service_config
from omegaconf import DictConfig


def _normalize_gemini_tool_call_messages(
        messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for message in messages:
        if message.get('role') == 'assistant' and message.get('tool_calls'):
            message = dict(message)
            message['content'] = None
        normalized.append(message)
    return normalized


def _sanitize_gemini_tool_name(name: str, used: set[str]) -> str:
    sanitized = re.sub(r'[^A-Za-z0-9_]', '_', name)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_') or 'tool'
    if not re.match(r'[A-Za-z_]', sanitized[0]):
        sanitized = f'tool_{sanitized}'
    if sanitized not in used and len(sanitized) <= 64:
        used.add(sanitized)
        return sanitized

    digest = hashlib.sha1(name.encode('utf-8')).hexdigest()[:8]
    base = sanitized[:55].rstrip('_') or 'tool'
    candidate = f'{base}_{digest}'
    counter = 1
    while candidate in used:
        suffix = f'_{counter}'
        candidate = f'{base[:64 - len(suffix)]}{suffix}'
        counter += 1
    used.add(candidate)
    return candidate


def _gemini_tool_name_maps(
        tools: Optional[List[Dict[str, Any]]]) -> Tuple[Dict[str, str], Dict[
            str, str]]:
    original_to_safe: Dict[str, str] = {}
    safe_to_original: Dict[str, str] = {}
    used: set[str] = set()
    for tool in tools or []:
        function = tool.get('function') if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        original = function.get('name')
        if not original:
            continue
        safe = _sanitize_gemini_tool_name(str(original), used)
        original_to_safe[str(original)] = safe
        safe_to_original[safe] = str(original)
    return original_to_safe, safe_to_original


def _ensure_openai_tools(llm: OpenAI,
                         tools: Optional[List[Dict[str, Any]]]
                         ) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    if all(isinstance(tool, dict) and tool.get('type') == 'function'
           and isinstance(tool.get('function'), dict) for tool in tools):
        return copy.deepcopy(tools)
    return copy.deepcopy(llm.format_tools(tools))


def _rewrite_tool_names_in_messages(messages: List[Dict[str, Any]],
                                    name_map: Dict[str, str]
                                    ) -> List[Dict[str, Any]]:
    rewritten = copy.deepcopy(messages)
    for message in rewritten:
        for tool_call in message.get('tool_calls') or []:
            function = tool_call.get('function')
            if not isinstance(function, dict):
                continue
            name = function.get('name')
            if name in name_map:
                function['name'] = name_map[name]
    return rewritten


def _rewrite_tool_names_in_tools(tools: Optional[List[Dict[str, Any]]],
                                 name_map: Dict[str, str]
                                 ) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    rewritten = copy.deepcopy(tools)
    for tool in rewritten:
        function = tool.get('function')
        if not isinstance(function, dict):
            continue
        name = function.get('name')
        if name in name_map:
            function['name'] = name_map[name]
    return rewritten


def _restore_tool_names_in_completion(completion: Any,
                                      name_map: Dict[str, str]) -> Any:
    if not name_map:
        return completion
    for choice in getattr(completion, 'choices', []) or []:
        message = getattr(choice, 'message', None)
        for tool_call in getattr(message, 'tool_calls', None) or []:
            function = getattr(tool_call, 'function', None)
            name = getattr(function, 'name', None)
            if name in name_map:
                function.name = name_map[name]
    return completion


class Gemini(OpenAI):

    def __init__(self, config: DictConfig):
        super().__init__(
            config,
            base_url=config.llm.get('gemini_base_url')
            or get_service_config('gemini').base_url,
            api_key=config.llm.get('gemini_api_key'))
        self._apply_gemini_defaults()

    def _apply_gemini_defaults(self):
        model = str(self.model or '').lower()
        if 'gemini-2.5' not in model or 'pro' in model:
            return
        self.args.setdefault('reasoning_effort', 'none')

    def _format_input_message(self, messages):
        formatted_messages = super()._format_input_message(messages)
        return _normalize_gemini_tool_call_messages(formatted_messages)

    def _call_llm(self,
                  messages: List[Any],
                  tools: Optional[List[Dict[str, Any]]] = None,
                  **kwargs) -> Any:
        messages = self._format_input_message(messages)
        tools = _ensure_openai_tools(self, tools)
        original_to_safe, safe_to_original = _gemini_tool_name_maps(tools)
        messages = _rewrite_tool_names_in_messages(messages, original_to_safe)
        tools = _rewrite_tool_names_in_tools(tools, original_to_safe)

        is_streaming = kwargs.get('stream', False)
        stream_options_config = self.args.get('stream_options', {})
        if is_streaming and stream_options_config.get('include_usage', True):
            kwargs.setdefault('stream_options', {})['include_usage'] = True
        if tools and 'tool_choice' not in kwargs:
            kwargs['tool_choice'] = 'auto'

        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools, **kwargs)
        if not is_streaming:
            _restore_tool_names_in_completion(completion, safe_to_original)
        return completion
