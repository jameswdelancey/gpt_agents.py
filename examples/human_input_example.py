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
    human_task = Task(
        name="ChatWithHuman",
        description=(
            "Engage in a helpful, thoughtful conversation with the human user. "
            "Always begin your response with 'Thought:' to show your reasoning, "
            "and clearly prefix your actual answer for the human with 'Final Answer:'. "
            "Carefully consider and address the user's questions below. "
            "Wait for further human input before ending the conversation."
        ),
        expected_output="A multi-turn conversation with the human, concluded only when the human requests to end.",
        require_human_input=True,
        disable_validation=True,
        llm_messages=[],
    )

    agent = Agent(
        role="Conversational Agent",
        goal="Provide interactive, friendly, and helpful assistance to the human in real time.",
        backstory=(
            "You are an attentive and approachable assistant, skilled at holding interactive conversations. "
            "Your style is kind, clear, and patient. You always show your reasoning with a 'Thought:' before giving your answer."
        ),
        tasks=[human_task],
        tools=[],
    )

    result = agent_executor(agent, agent_conclusions=[])
    print("\nFinal Task Conclusion:")
    print("Input:\n", result.input)
    print("Output:\n", result.output)


if __name__ == "__main__":
    main()
