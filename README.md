# gpt_agents.py

🚨 **Highlight: Now with native Anthropic Claude support!**

**Current Version:** 0.1.5

[![PyPI version](https://img.shields.io/pypi/v/gpt-agents-py.svg)](https://pypi.org/project/gpt-agents-py/)

> **Unique:** This is a single-file, multi-agent framework for LLMs—everything is implemented in one core file with no dependencies for maximum clarity and hackability. See the main implementation here: [`gpt_agents.py`](https://github.com/jameswdelancey/gpt_agents.py/blob/main/gpt_agents_py/gpt_agents.py)

A minimal, modular Python framework for building and running multi-agent LLM workflows with tool use, validation, and orchestration. 

## Project Overview

gpt_agents.py provides abstractions for:
- **Agent**: An LLM-powered entity with a goal, backstory, tasks, and tools.
- **Task**: A unit of work for an agent, with a description and expected output.
- **Tool**: A callable function the agent can use, with a name, description, and argument schema.
- **Organization**: A group of agents working together, passing results between them.

The framework manages LLM prompting, tool execution, validation, and robust control flow to maximize correct answers. See [`tests/test_integration.py`](https://github.com/jameswdelancey/gpt_agents.py/blob/main/tests/test_integration.py) for advanced usage and testing patterns.

## Requirements
- Python 3.11+ (standard library only)
- [OpenAI API Key](https://platform.openai.com/account/api-keys) in `api_key.json` at the project root:
  ```json
  { "openai": "sk-..." }
  ```
- **(New!) [Anthropic API Key](https://console.anthropic.com/settings/keys) also supported:**
  ```json
  { "anthropic": "sk-ant-..." }
  ```
  Add both keys to `api_key.json` if you want to use both providers.

## Installation

```bash
git clone <repository-url>
cd gpt_agents.py
```

No pip or poetry installation required for runtime. This project has **zero runtime dependencies** and uses only the Python standard library.

For development (linting, formatting, type checking, testing), install dev dependencies:
```bash
pip install black isort mypy flake8
```
Or, if using Poetry:
```bash
poetry install --with dev
```
## Usage

### Customizing LLM API Calls

You can override how LLM API requests are handled by creating your own subclass of `LLMCallerBase` and registering it globally. This makes it easy to integrate with any LLM provider, add logging, or mock responses for testing.

```python
from gpt_agents_py.gpt_agents import LLMCallerBase, set_llm_caller, call_llm, Message

class MyCustomLLMCaller(LLMCallerBase):
    def prepare_llm_response(self, messages, api_key="api_key"):
        # Implement your own LLM API logic here
        self._response_text = "This is a mock response!"
        self._tokens_used = 0

# Register your custom LLM caller
set_llm_caller(MyCustomLLMCaller())

# Now all LLM calls use your logic
response = call_llm([Message(role="user", content="Hello!")])
print(response)
```

### Integrating Other LLM Providers

To use a different LLM backend (such as Azure OpenAI, Anthropic, Cohere, Groq, or a local model), simply implement your `LLMCallerBase` subclass with the API logic for that provider and set it via `set_llm_caller`. This approach gives you full control over request formatting, authentication, and response handling.


### Anthropic Claude Integration

Native support for Anthropic Claude models is included! To use Claude, just set the LLM caller to `AnthropicLLMCaller`:

```python
from gpt_agents_py.extensions.anthropic_llm_caller import AnthropicLLMCaller
from gpt_agents_py.gpt_agents import set_llm_caller

set_llm_caller(AnthropicLLMCaller())
```

See `examples/anthropic_basic_usage.py` for a complete working example, including tracing and logging. Make sure your `api_key.json` includes your Anthropic key (see Requirements above).

### Running the example

Runnable demos are located in the `examples/` directory. To run the basic usage example:


```bash
python examples/basic_usage.py                               # Normal mode
python examples/basic_usage.py --debug                       # Step-through debug mode for agent reasoning
python examples/basic_usage.py --trace                       # Log all LLM prompts and responses to file.txt
python examples/basic_usage.py --trace --trace-filename mylog.txt   # Log LLM traces to custom file
python examples/basic_usage.py --debug --trace               # Combine step-through and trace logging
```

- The `--debug` flag enables step-through mode, which pauses execution at each agent reasoning step for inspection.
- The `--trace` flag logs all LLM prompts and responses to a file (default: `file.txt`).
- The `--trace-filename` flag lets you specify a custom trace log file (used with `--trace`).
- Both flags can be combined.

### Customizing Prompts at Runtime

You can override any prompt template used by the agent at runtime without modifying the source code. This is useful for adapting the agent's behavior, tone, or instructions for different use cases.

Use the `set_prompt_value` function from `gpt_agents.py`:

```python
from gpt_agents_py.gpt_agents import set_prompt_value

# Example: Change the system instruction prompt
set_prompt_value('instruction_prompt', 'You are a helpful assistant. Always explain your reasoning.')
```

See the `Prompts` NamedTuple in `gpt_agents.py` for all available prompt keys you can override.

- By default, both are off and the trace filename is `file.txt`.

### Running Tests

Integration tests are provided using Python's `unittest` framework:

```bash
python -m unittest discover tests
```

Development checks (run from project root):
```bash
black --check gpt_agents_py examples tests
isort --check gpt_agents_py examples tests
flake8 gpt_agents_py examples tests
mypy gpt_agents_py examples tests
```

### Continuous Integration

This project uses [GitHub Actions](https://github.com/jameswdelancey/gpt_agents.py/blob/main/.github/workflows/python-app.yml) to automatically lint, type-check, and run tests on pushes and pull requests to `main`, across Python 3.11 and 3.12.

### Example: Defining Agents, Tools, and Running an Organization

```python
from gpt_agents_py.gpt_agents import Agent, Organization, Task, Tool, organization_executor

def population_tool(args):
    country = args.get("country")
    return {"france": 67000000, "germany": 83000000}.get(country.lower(), "unknown")

def calculator_tool(args):
    op = args["operation"]
    a, b = float(args["a"]), float(args["b"])
    if op == "add": return str(a + b)
    if op == "subtract": return str(a - b)
    if op == "multiply": return str(a * b)
    if op == "divide": return str(a / b)
    return "unknown"

agent = Agent(
    role="Researcher",
    goal="Find population and add 1000",
    backstory="Expert in demographics.",
    tasks=[Task(name="GetPop", description="Get France's population", expected_output="67000000")],
    tools=[
        Tool(name="PopulationTool", description="Returns country population", args_schema="country:str", func=population_tool),
        Tool(name="Calculator", description="Performs arithmetic", args_schema="operation:str, a:str, b:str", func=calculator_tool)
    ]
)
org = Organization(agents=[agent])
result = organization_executor(org)
print(result)
```

See [`gpt_agents_py/examples.py`](https://github.com/jameswdelancey/gpt_agents.py/blob/main/gpt_agents_py/examples.py) for a more complete demo.

## Testing

Integration tests are provided in the `tests/` directory. LLM calls are mocked for deterministic testing, and real tool logic is exercised.

To run all tests:

```bash
python -m unittest discover tests
```

See [`tests/test_integration.py`](https://github.com/jameswdelancey/gpt_agents.py/blob/main/tests/test_integration.py) for an example of integration testing with mocked LLM calls.

## Contributing

Contributions are welcome! If you have suggestions, bug reports, feature requests, or would like to submit a pull request, please open an issue or PR on GitHub. All contributions—large or small—are appreciated.

### Contributors and Forks

- [gpt_agents.py](https://github.com/jameswdelancey/gpt_agents.py) | [James Delancey](https://github.com/jameswdelancey) Example


## License

MIT
