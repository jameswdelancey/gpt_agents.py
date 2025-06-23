# gpt_agents_py | James Delancey | MIT License
import argparse
import logging

from gpt_agents_py.gpt_agents import (  # set_debug_mode,; set_trace_mode,
    Agent,
    Organization,
    Task,
    Tool,
    log_json,
    organization_executor,
    set_debug_mode,
    set_trace_mode,
)

logging.basicConfig(level=logging.DEBUG, format="[%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s", datefmt="%m%d %H:%M:%S")


# Demo tools
def population_tool(args: dict[str, str]) -> str:
    if not isinstance(args, dict):
        raise ValueError(f"Arguments must be a dictionary. You gave: {type(args).__name__}")
    country = args.get("country")
    if not isinstance(country, str) or not country:
        raise ValueError(f"Argument 'country' must be a non-empty string. You gave: {country!r}")
    country_lc = country.lower()
    fake_populations = {
        "france": 67000000,
        "germany": 83000000,
        "spain": 47000000,
        "italy": 60000000,
    }
    if country_lc not in fake_populations:
        raise ValueError(f"Unknown country: {country!r}. Expected one of: {list(fake_populations.keys())}")
    return str(fake_populations[country_lc])


def calculator_tool(args: dict[str, str]) -> str:
    if not isinstance(args, dict):
        raise ValueError(f"Arguments must be a dictionary. You gave: {type(args).__name__}")
    op = args.get("operation")
    a = args.get("a")
    b = args.get("b")
    if op not in {"add", "subtract", "multiply", "divide"}:
        raise ValueError(f"Argument 'operation' must be one of: add, subtract, multiply, divide. You gave: {op!r}")
    if a is None or b is None:
        raise ValueError(f"Arguments 'a' and 'b' must be provided. You gave: a={a!r}, b={b!r}")
    try:
        a_num = float(a)
        b_num = float(b)
    except Exception:
        raise ValueError(f"Arguments 'a' and 'b' must be convertible to numbers. You gave: a={a!r}, b={b!r}")
    if op == "add":
        return str(a_num + b_num)
    elif op == "subtract":
        return str(a_num - b_num)
    elif op == "multiply":
        return str(a_num * b_num)
    elif op == "divide":
        if b_num == 0:
            raise ValueError("Division by zero. You gave: b=0")
        return str(a_num / b_num)
    raise ValueError(f"Unknown operation: {op!r}")


def weather_tool(args: dict[str, str]) -> str:
    city = args.get("city")
    if not isinstance(city, str) or not city:
        raise ValueError(f"Argument 'city' must be a non-empty string. You gave: {city!r}")
    city_lc = city.lower()
    fake_weather = {
        "paris": "Cloudy, 18C",
        "berlin": "Sunny, 21C",
        "madrid": "Rainy, 15C",
        "rome": "Sunny, 24C",
    }
    matches = [k for k in fake_weather if city_lc in k or k in city_lc]
    if len(matches) == 1:
        return fake_weather[matches[0]]
    elif len(matches) > 1:
        raise ValueError(f"Ambiguous city name '{city}'. Matches: {matches}. You gave: {city!r}")
    else:
        raise ValueError(f"Unknown city: {city!r}. Expected one of: {list(fake_weather.keys())}")


def convert_tool(args: dict[str, str]) -> str:
    value = args.get("value")
    unit = args.get("unit")
    if value is None or unit is None:
        raise ValueError(f"Arguments 'value' and 'unit' must be provided as strings. You gave: value={value!r}, unit={unit!r}. Example: value='32', unit='C' or unit='F'.")
    try:
        value_num = float(value)
    except Exception:
        raise ValueError(f"Argument 'value' must be convertible to float. You gave: value={value!r}")
    if unit.upper() == "F":
        return str(round((value_num - 32) * 5 / 9, 2))
    elif unit.upper() == "C":
        return str(round((value_num * 9 / 5) + 32, 2))
    else:
        raise ValueError(f"Argument 'unit' must be 'F' or 'C'. You gave: unit={unit!r}")


population_tools = [
    Tool(
        name="population",
        description="Returns the population of a given country.",
        args_schema="{country: string}",
        func=population_tool,
    ),
]
calculator_tools = [
    Tool(
        name="calculator",
        description="Performs a calculation on two numbers. Operation can be add, subtract, multiply, or divide.",
        args_schema="{operation: string, a: number, b: number}",
        func=calculator_tool,
    ),
]
weather_tools = [
    Tool(
        name="weather",
        description="Returns the current weather for a given city.",
        args_schema="{city: string}",
        func=weather_tool,
    ),
    Tool(
        name="convert",
        description="Converts temperatures between Celsius and Fahrenheit.",
        args_schema="{value: string, unit: string}",
        func=convert_tool,
    ),
]

agent1 = Agent(
    role="Population Analyst",
    goal="Summarize the population difference between France and Germany.",
    backstory="Expert in demographic analysis.",
    tasks=[
        Task(
            description="Get the population of France.",
            expected_output="A complete sentence representing France's population.",
            name="France Population",
            llm_messages=[],
        ),
        Task(
            description="Get the population of Germany.",
            expected_output="A complete sentence representing Germany's population.",
            name="Germany Population",
            llm_messages=[],
        ),
        Task(
            description="Summarize the population findings for both France and Germany, stating the population of each and the difference between them.",
            expected_output="A summary sentence or paragraph that includes the population of France, the population of Germany, and the difference between them.",
            name="Population Summary",
            llm_messages=[],
        ),
    ],
    tools=population_tools + calculator_tools,
)

agent2 = Agent(
    role="Weather Specialist",
    goal="Get and convert temperatures for France and Germany.",
    backstory="Expert in meteorology and temperature conversions.",
    tasks=[
        Task(
            description="Get the weather for Paris, France.",
            expected_output="A complete sentence representing the weather in Paris.",
            name="Paris Weather",
            llm_messages=[],
        ),
        Task(
            description="Convert the temperature in Paris to Fahrenheit.",
            expected_output="A complete sentence representing the temperature in Fahrenheit.",
            name="Paris Temperature Conversion",
            llm_messages=[],
        ),
        Task(
            description="Get the weather for Berlin, Germany.",
            expected_output="A complete sentence representing the weather in Berlin.",
            name="Berlin Weather",
            llm_messages=[],
        ),
        Task(
            description="Convert the temperature in Berlin to Fahrenheit.",
            expected_output="A complete sentence representing the temperature in Fahrenheit.",
            name="Berlin Temperature Conversion",
            llm_messages=[],
        ),
        Task(
            description="Summarize the weather findings for both Paris and Berlin, referencing their temperatures in both Celsius and Fahrenheit.",
            expected_output="A summary sentence or paragraph about the weather in both cities, using the actual temperatures found.",
            name="Weather Summary",
            llm_messages=[],
        ),
    ],
    tools=weather_tools,
)

agent3 = Agent(
    role="Travel Brief Writer",
    goal="Write a clear, concise travel brief based on all available data.",
    backstory="Professional travel writer who synthesizes information for travelers.",
    tasks=[
        Task(
            description=(
                "Using all the data from previous agents, write a clear and helpful travel brief for someone considering a trip to France and Germany. "
                "Your summary MUST explicitly mention the population difference between France and Germany, and the actual temperatures for Paris and Berlin in both Celsius and Fahrenheit as provided by previous agents. "
                "Do not invent new numbers; use only the facts given."
            ),
            expected_output="A travel brief that references population and weather insights, using the actual numbers and temperatures from previous agents.",
            name="Travel Brief Summary",
            llm_messages=[],
        ),
    ],
    tools=[],
)

org = Organization(agents=[agent1, agent2, agent3])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a basic gpt_agents example. Use --debug to enable step-through mode for agent reasoning.")
    parser.add_argument("--debug", action="store_true", help="Enable step-through debug mode for agent reasoning.")
    parser.add_argument("--trace", action="store_true", help="Enable LLM trace logging for agent reasoning.")
    parser.add_argument("--trace-filename", type=str, default="file.txt", help="Filename for LLM trace log (used with --trace). Default: file.txt")
    args = parser.parse_args()
    set_debug_mode(args.debug)
    set_trace_mode(args.trace, filename=args.trace_filename)
    examples()


def examples() -> None:
    final_result = organization_executor(org)
    log_json(logging.DEBUG, "Organization Results:", final_result)
    print("Organization Results:")
    print(final_result.final_conclusion.output if final_result else "No final conclusion")


if __name__ == "__main__":
    main()
