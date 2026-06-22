from ms_agent.llm.gemini_llm import _normalize_gemini_tool_call_messages


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
