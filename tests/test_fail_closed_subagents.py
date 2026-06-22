import pytest

from ms_agent.agent.llm_agent import LLMAgent
from ms_agent.llm.openai_llm import _ensure_supported_finish_reason
from ms_agent.llm.utils import Message
from ms_agent.tools.agent_tool import AgentTool


class _Spec:
    output_mode = 'final_message'
    max_output_chars = 1000
    tool_name = 'searcher_tool'


class _FakeToolManager:

    def __init__(self, response):
        self._response = response

    async def parallel_call_tool(self, _tool_calls):
        return [self._response]


def test_ensure_supported_finish_reason_rejects_malformed_function_call() -> None:
    with pytest.raises(RuntimeError, match='unsupported finish_reason'):
        _ensure_supported_finish_reason(
            'function_call_filter: MALFORMED_FUNCTION_CALL')


def test_agent_tool_format_output_rejects_blank_terminal_assistant_message(
) -> None:
    tool = AgentTool.__new__(AgentTool)
    messages = [Message(role='assistant', content='')]
    with pytest.raises(RuntimeError, match='finished without a final assistant message'):
        tool._format_output(messages, _Spec())


@pytest.mark.asyncio
async def test_llm_agent_after_tool_call_rejects_blank_assistant_without_tools(
) -> None:
    agent = LLMAgent.__new__(LLMAgent)
    agent.tag = 'unit-test-agent'
    agent.runtime = type('Runtime', (), {'should_stop': False})()
    agent.loop_callback = _noop_loop_callback

    with pytest.raises(RuntimeError, match='empty assistant response'):
        await agent.after_tool_call([Message(role='assistant', content='')])


@pytest.mark.asyncio
async def test_llm_agent_parallel_tool_call_rejects_failed_subagent_tool(
) -> None:
    agent = LLMAgent.__new__(LLMAgent)
    agent.tool_manager = _FakeToolManager(
        'Tool calling failed: {"tool_name":"agent_tools---searcher_tool"}, details: boom'
    )
    agent.log_output = lambda *_args, **_kwargs: None

    messages = [
        Message(
            role='assistant',
            content='',
            tool_calls=[{
                'id': 'call-1',
                'tool_name': 'agent_tools---searcher_tool',
                'arguments': '{}',
                'type': 'function',
                'index': 0,
            }],
        )
    ]

    with pytest.raises(RuntimeError, match='Sub-agent tool agent_tools---searcher_tool failed'):
        await agent.parallel_tool_call(messages)


@pytest.mark.asyncio
async def test_llm_agent_parallel_tool_call_rejects_empty_subagent_output(
) -> None:
    agent = LLMAgent.__new__(LLMAgent)
    agent.tool_manager = _FakeToolManager('')
    agent.log_output = lambda *_args, **_kwargs: None

    messages = [
        Message(
            role='assistant',
            content='',
            tool_calls=[{
                'id': 'call-1',
                'tool_name': 'agent_tools---searcher_tool',
                'arguments': '{}',
                'type': 'function',
                'index': 0,
            }],
        )
    ]

    with pytest.raises(RuntimeError, match='returned empty output'):
        await agent.parallel_tool_call(messages)


async def _noop_loop_callback(_point, _messages):
    return None
