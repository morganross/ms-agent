from ms_agent.llm.gemini_llm import (
    _gemini_tool_name_maps,
    _normalize_gemini_tool_call_messages,
    _rewrite_tool_names_in_messages,
    _rewrite_tool_names_in_tools,
)


def test_gemini_tool_call_messages_drop_synthetic_assistant_content() -> None:
    messages = [
        {
            'role': 'assistant',
            'content': 'Let me do a tool calling.',
            'tool_calls': [{'id': 'call-1'}],
        },
        {
            'role': 'tool',
            'content': '{"status":"ok"}',
            'tool_call_id': 'call-1',
        },
        {
            'role': 'assistant',
            'content': 'Done.',
        },
    ]

    normalized = _normalize_gemini_tool_call_messages(messages)

    assert normalized[0]['content'] is None
    assert normalized[0]['tool_calls'] == [{'id': 'call-1'}]
    assert normalized[1] is messages[1]
    assert normalized[2] is messages[2]
    assert messages[0]['content'] == 'Let me do a tool calling.'


def test_gemini_tool_names_are_sanitized_and_rewritten() -> None:
    tools = [{
        'type': 'function',
        'function': {
            'name': 'web_search---searchbox_search',
            'description': 'Search',
            'parameters': {'type': 'object'},
        },
    }]
    messages = [{
        'role': 'assistant',
        'content': None,
        'tool_calls': [{
            'id': 'call-1',
            'type': 'function',
            'function': {
                'name': 'web_search---searchbox_search',
                'arguments': '{}',
            },
        }],
    }]

    original_to_safe, safe_to_original = _gemini_tool_name_maps(tools)
    rewritten_tools = _rewrite_tool_names_in_tools(tools, original_to_safe)
    rewritten_messages = _rewrite_tool_names_in_messages(
        messages, original_to_safe)

    assert original_to_safe == {
        'web_search---searchbox_search': 'web_search_searchbox_search'
    }
    assert safe_to_original == {
        'web_search_searchbox_search': 'web_search---searchbox_search'
    }
    assert rewritten_tools[0]['function'][
        'name'] == 'web_search_searchbox_search'
    assert rewritten_messages[0]['tool_calls'][0]['function'][
        'name'] == 'web_search_searchbox_search'
    assert tools[0]['function']['name'] == 'web_search---searchbox_search'
