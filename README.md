# toolstream

Typed streaming SDK for LLM tool-calling loops over Azure OpenAI / AI Gateway APIs.

## Install

```
pip install toolstream
```

Requires Python >= 3.11. Single runtime dependency: httpx.

## Quick start

```python
import asyncio
from toolstream import AsyncSession, SessionConfig, Text, ToolUse, StepFinish, Result

config = SessionConfig(
    model="gpt-4o",
    api_key="sk-your-api-key",
    base_url="https://your-gateway.example.com/v1",
    system_prompt="You are a helpful assistant.",
)

async def main():
    async with AsyncSession(config) as session:
        async for event in session.send("What is 2 + 2?"):
            if isinstance(event, Text):
                print(event.text, end="")
            elif isinstance(event, ToolUse):
                print(f"[tool: {event.tool}] {event.status}")
            elif isinstance(event, StepFinish):
                print(f"\n({event.input_tokens} in, {event.output_tokens} out, ${event.cost:.4f})")
            elif isinstance(event, Result):
                print(f"Done in {event.steps} step(s), total ${event.total_cost:.4f}")

asyncio.run(main())
```

A synchronous wrapper is also available:

```python
from toolstream import SyncSession, SessionConfig, Text, Result

config = SessionConfig(
    model="gpt-4o",
    api_key="sk-your-api-key",
    base_url="https://your-gateway.example.com/v1",
    system_prompt="You are a helpful assistant.",
)

with SyncSession(config) as session:
    for event in session.send("Hello"):
        if isinstance(event, Text):
            print(event.text, end="")
```

## Tool registration

Decorate functions with `@tool` to make them callable by the LLM. Type annotations on parameters are used to generate the JSON schema automatically.

```python
from toolstream import tool, collect_tools, SessionConfig, AsyncSession

@tool()
def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city."""
    return f"22 degrees {units} in {city}"

config = SessionConfig(
    model="gpt-4o",
    api_key="sk-your-api-key",
    base_url="https://your-gateway.example.com/v1",
    system_prompt="You can look up weather.",
    tools=collect_tools(my_tools_module),
)
```

Use `inject` to pass dependencies that the LLM should not see or fill:

```python
from toolstream import tool, ToolContext
from dataclasses import dataclass

@dataclass
class AppContext(ToolContext):
    db_connection: object

@tool(inject=["ctx"])
def query_db(ctx: AppContext, sql: str) -> str:
    """Run a read-only SQL query."""
    return str(ctx.db_connection.execute(sql))
```

Injected parameters are excluded from the schema sent to the LLM and resolved at call time via the `tool_context` on `SessionConfig`.

## Agent definitions

Agents are defined in `.agent.json` files:

```json
{
    "name": "summarizer",
    "version": "1.0",
    "description": "Summarizes documents",
    "prompt_template": "Summarize the following {format} document:\n\n{content}",
    "tools": [
        {"name": "word_count"}
    ],
    "model": "gpt-4o-mini"
}
```

Load and invoke an agent:

```python
from toolstream import load_agent, invoke_agent, SessionConfig, Text

definition = load_agent("agents/summarizer.agent.json")

base_config = SessionConfig(
    model="gpt-4o",
    api_key="sk-your-api-key",
    base_url="https://your-gateway.example.com/v1",
    system_prompt="",  # overridden by agent prompt_template
)

async with invoke_agent(
    definition,
    base_config,
    variables={"format": "markdown", "content": doc_text},
    available_tools={"word_count": word_count_fn},
) as session:
    async for event in session.send("Go"):
        if isinstance(event, Text):
            print(event.text, end="")
```

Use `discover_agents()` to find all `.agent.json` files in a project directory, explicit paths, or installed Python packages:

```python
from toolstream import discover_agents

agents = discover_agents(cwd=".", packages=["my_agents_pkg"])
```

## Event types

All events are frozen dataclasses. Use `isinstance` checks to handle them.

| Event | Description |
|-------|-------------|
| `StepStart` | A new completion step has begun. Carries `session_id`, `message_id`, `timestamp`. |
| `Text` | A chunk of streamed text from the model. |
| `ToolUse` | A tool was called and returned a result. Includes `tool` name, `input`, `output`, `status`. |
| `StepFinish` | A completion step ended. Carries token counts (`input_tokens`, `output_tokens`, `reasoning_tokens`, `cache_read_tokens`, `cache_write_tokens`), `cost`, and `reason` (`"stop"` or `"tool-calls"`). |
| `Error` | An error occurred. Carries `name`, `message`, and `data` dict. |
| `Result` | Final summary after all steps complete. Carries `total_input_tokens`, `total_output_tokens`, `total_cost`, `steps`. |

## ToolContext

`ToolContext` is a dataclass base class for dependency injection into tools. Subclass it with whatever your tools need, then pass an instance via `SessionConfig.tool_context`:

```python
from dataclasses import dataclass
from toolstream import ToolContext, SessionConfig

@dataclass
class MyContext(ToolContext):
    api_client: object
    user_id: str

config = SessionConfig(
    model="gpt-4o",
    api_key="sk-your-api-key",
    base_url="https://your-gateway.example.com/v1",
    system_prompt="...",
    tool_context=MyContext(api_client=client, user_id="u-123"),
)
```

Tools that declare `inject=["ctx"]` will receive the context object as their `ctx` parameter at call time. The injected parameters are hidden from the LLM's schema.
