# gpt_agents_py | James Delancey | MIT License
import json
import os
import sys
import unittest
from typing import cast
from unittest.mock import Mock, mock_open, patch

# Patch before importing the module that loads api_key.json
mock_api_key_content = '{"openai": "sk-test"}'
patcher = patch("gpt_agents_py.gpt_agents.open", mock_open(read_data=mock_api_key_content))
patcher.start()
sys.modules["gpt_agents_py.gpt_agents"].__dict__["open"] = open  # Ensure open is patched in the module namespace

from gpt_agents_py import (  # noqa E402
    Agent,
    Message,
    Organization,
    OrganizationConclusion,
    Task,
    Tool,
    organization_executor,
)


def population_tool(args: dict[str, str]) -> str:
    country = args.get("country")
    fake_populations = {
        "france": 67000000,
        "germany": 83000000,
        "spain": 47000000,
        "italy": 60000000,
    }
    if country is None:
        raise ValueError("Argument 'country' is required.")
    return str(fake_populations[country.lower()])


def calculator_tool(args: dict[str, str]) -> str:
    op = args.get("operation")
    a_val = args.get("a")
    b_val = args.get("b")
    if a_val is None or b_val is None:
        raise ValueError("Arguments 'a' and 'b' are required.")
    a = float(a_val)
    b = float(b_val)
    if op == "add":
        return str(a + b)
    elif op == "subtract":
        return str(a - b)
    elif op == "multiply":
        return str(a * b)
    elif op == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return str(a / b)
    raise ValueError(f"Unknown operation: {op}")


def weather_tool(args: dict[str, str]) -> str:
    city = args.get("city")
    fake_weather = {
        "paris": "Cloudy, 18C",
        "berlin": "Sunny, 21C",
        "madrid": "Rainy, 15C",
        "rome": "Sunny, 24C",
    }
    if city is None:
        raise ValueError("Argument 'city' is required.")
    return fake_weather[city.lower()]


@patch("gpt_agents_py.gpt_agents.call_llm")
class TestIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = Agent(
            role="Calculator",
            goal="Return the answer to everything",
            backstory="Knows the answer is always 42.",
            tasks=[Task(name="Ultimate", description="What is the answer to life, the universe, and everything?", expected_output="42")],
            tools=[
                Tool(
                    name="population",
                    description="Returns the population of a given country.",
                    args_schema="{country: string}",
                    func=population_tool,
                ),
            ],
        )
        self.org = Organization(agents=[self.agent])

    def test_organization_executor_with_mocked_llm(self, mock_llm: Mock) -> None:
        # Use real LLM outputs from file.txt for the agent and validator turns
        mock_llm.side_effect = [
            """I will use the population tool to get the population of France.\nAction: population\nAction Input: {"country": "France"}""",
            """```
Thought: I now know the final answer
Final Answer: The population of France is 67,000,000
```""",
            """Thought: The output directly states the number representing France's population as 67,000,000 with no extra information or context provided.\nFinal Answer: yes""",
            """```
Thought: I should use the population tool to get the population of Germany.
Action: population
Action Input: {"country": "Germany"}
```""",
            """```
Thought: I now know the final answer.
Final Answer: The population of Germany is 83,000,000.
```""",
            """Thought: The output directly states the number representing Germany's population as 83,000,000 with no extra information or context provided.\nFinal Answer: yes""",
            """```
Thought: I now know the final answer
Final Answer: The population of France is 67,000,000 and the population of Germany is 83,000,000. The difference in population between Germany and France is 16,000,000.
```""",
            """Thought: The output provides the correct summary of the population of France and Germany, including the difference as required.\nFinal Answer: yes""",
        ]
        result = organization_executor(self.org)
        self.assertIsNotNone(result)
        result = cast(OrganizationConclusion, result)
        self.assertIn("Final Answer: The population of France is 67,000,000", result.final_conclusion.output)

    def test_organization_executor_real_tool(self, mock_llm: Mock) -> None:
        # Use real LLM outputs from file.txt for the agent and validator turns
        mock_llm.side_effect = [
            """I will use the population tool to get the population of France.\nAction: population\nAction Input: {"country": "France"}""",
            """```
Thought: I now know the final answer
Final Answer: The population of France is 67,000,000
```""",
            """Thought: The output directly states the number representing France's population as 67,000,000 with no extra information or context provided.\nFinal Answer: yes""",
            """```
Thought: I should use the population tool to get the population of Germany.
Action: population
Action Input: {"country": "Germany"}
```""",
            """```
Thought: I now know the final answer.
Final Answer: The population of Germany is 83,000,000.
```""",
            """Thought: The output directly states the number representing Germany's population as 83,000,000 with no extra information or context provided.\nFinal Answer: yes""",
            """```
Thought: I now know the final answer
Final Answer: The population of France is 67,000,000 and the population of Germany is 83,000,000. The difference in population between Germany and France is 16,000,000.
```""",
            """Thought: The output provides the correct summary of the population of France and Germany, including the difference as required.\nFinal Answer: yes""",
        ]
        result = organization_executor(self.org)
        self.assertIsNotNone(result)
        result = cast(OrganizationConclusion, result)
        self.assertIn("Final Answer: The population of France is 67,000,000", result.final_conclusion.output)


def test_capture_llm_prompts() -> None:
    org = Organization(
        agents=[
            Agent(
                role="Test Agent",
                goal="Test Goal",
                backstory="Test Backstory",
                tasks=[Task(description="Test Task", expected_output="Test Output", name="Test Task")],
                tools=[],
            )
        ]
    )
    outputs_path = "llm_prompts.jsonl"

    def capture_llm(messages: list[Message]) -> str:
        # Save each prompt as JSON
        with open(outputs_path, "a") as f:
            f.write(json.dumps([m.content for m in messages]) + "\n")
        # Return a dummy value (will need to be replaced with real expected output)
        return "MOCKED"

    if os.path.exists(outputs_path):
        os.remove(outputs_path)
    with patch("gpt_agents_py.gpt_agents.call_llm", side_effect=capture_llm):
        try:
            organization_executor(org)
        except Exception:
            pass  # Ignore errors, just want to capture prompts


if __name__ == "__main__":
    unittest.main()
