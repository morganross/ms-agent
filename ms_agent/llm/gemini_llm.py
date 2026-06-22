# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict, List

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
