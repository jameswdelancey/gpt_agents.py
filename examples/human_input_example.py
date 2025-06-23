from examples.basic_usage import calculator_tools, population_tools
from gpt_agents_py.gpt_agents import Agent, Task, agent_executor


def main() -> None:
    # Dummy tool for demonstration
    def echo_tool(args: dict[str, str]) -> str:
        return f"Echo: {args.get('text', '')}"

    # echo = Tool(
    #     name="Echo",
    #     description="Echoes your input to yourself.",
    #     args_schema="{ 'text': 'string' }",
    #     func=echo_tool,
    # )

    # Task that requires human input
    agent = Agent(
        role="Population Analyst",
        goal="Summarize the population difference between France and Germany.",
        backstory="Expert in demographic analysis.",
        tasks=[
            Task(
                description="Get the population of France.",
                expected_output="A complete sentence representing France's population.",
                name="France Population",
                llm_messages=[],
                require_human_input=True,
            ),
        ],
        tools=population_tools + calculator_tools,
        disable_validation=True,
        disable_summary=True,
    )

    result = agent_executor(agent, agent_conclusions=[])
    print("\nFinal Task Conclusion:")
    print("Input:\n", result.input)
    print("Output:\n", result.output)


if __name__ == "__main__":
    main()
