# gpt_agents_py | James Delancey | MIT License
import json
import logging
import os
import re
import time
import traceback
from enum import Enum
from typing import Callable, List, NamedTuple, NewType, Optional, Set


class Prompts(NamedTuple):
    role_playing_template: str
    tools_template: str
    no_tools_template: str
    tool_not_found_prompt: str
    action_input_not_dict_prompt: str
    tool_call_failed_prompt: str
    action_input_parse_failed_prompt: str
    tool_no_output_prompt: str
    validation_system_prompt: str
    validation_user_prompt: str
    validation_retry_prompt: str
    instruction_prompt: str
    current_task_prompt: str
    context_explanation_prompt: str
    missing_thought_prompt: str
    thought_required_prompt: str
    tool_retry_prompt: str
    validation_feedback_prompt: str
    retry_failed_validation_prompt: str
    coaching_prompt: str
    force_final_answer_prompt: str
    retry_failed_validation_prompt_2: str
    summary_task_description_prompt: str


logging.basicConfig(level=logging.INFO, format="[%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s", datefmt="%m%d %H:%M:%S")


def log_json(level: int, msg: str, obj: object) -> None:
    """
    Log a JSON object at the given level with pretty printing and newlines rendered.
    If the object is a NamedTuple, convert it to a dict. If not JSON serializable, use type name as placeholder.
    """

    def convert(o: object) -> object:
        # Recursively convert NamedTuples to dicts
        if isinstance(o, tuple) and hasattr(o, "_asdict"):
            return {k: convert(v) for k, v in o._asdict().items()}
        elif isinstance(o, list):
            return [convert(i) for i in o]
        elif isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        return o

    def fallback(o: object) -> str:
        return f"<{type(o).__name__}>"

    s = json.dumps(convert(obj), indent=2, default=fallback).replace("\\n", "\n")
    if os.environ.get("TERM", "") and hasattr(logging, "StreamHandler"):
        # Use ANSI green for msg if vty/terminal detected
        msg_colored = f"\033[92m{msg}\033[0m"
    else:
        msg_colored = msg
    logging.log(level, f"{msg_colored} {s}", stacklevel=2)


# Regex to extract Thought, Action, and Action Input
AGENT_ACTION_REGEX = re.compile(r"Thought:\s*(?P<thought>.*?)\nAction:\s*(?P<action>.*?)\nAction Input:\s*(?P<action_input>\{.*?\})(?:\n|$)", re.DOTALL)

# Global debug mode
DEBUG_MODE = False  # Set to True to enable step-through debugging
TRACE_LLM = False
TRACE_LLM_FILENAME = "file.txt"


def set_debug_mode(enabled: bool) -> None:
    """
    Enable or disable debug mode globally for gpt_agents.
    """
    global DEBUG_MODE
    DEBUG_MODE = enabled


def get_debug_mode() -> bool:
    """
    Get the current value of the global DEBUG_MODE flag.
    """
    return DEBUG_MODE


def get_trace_llm() -> bool:
    """
    Get the current value of the global TRACE_LLM flag.
    """
    return TRACE_LLM


def get_trace_llm_filename() -> str:
    """
    Get the current value of the global TRACE_LLM_FILENAME.
    """
    return TRACE_LLM_FILENAME


def set_trace_mode(enabled: bool, filename: str = "file.txt") -> None:
    """
    Enable or disable LLM trace logging globally for gpt_agents, and set the trace filename.
    """
    global TRACE_LLM, TRACE_LLM_FILENAME
    TRACE_LLM = enabled
    TRACE_LLM_FILENAME = filename
    if os.path.exists(filename):
        os.remove(filename)


def debug_step(msg: str | None = None) -> None:
    """
    If DEBUG_MODE is enabled, print a debug message and wait for user input to continue.
    """
    if DEBUG_MODE:
        if msg:
            print(f"[DEBUG STEP] {msg}")
        else:
            print("[DEBUG STEP]")
        input("Press Enter to continue...")


# Regex to extract Thought and Final Answer
AGENT_FINAL_REGEX = re.compile(r"Thought:\s*(?P<thought>.*?)\nFinal Answer:\s*(?P<final_answer>.*?)(?:\n*$|$)", re.DOTALL)

# Regex to extract just Thought
AGENT_THOUGHT_ONLY_REGEX = re.compile(r"Thought:\s*(?P<thought>.*?)(?:\n|$)", re.DOTALL)


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extracts the 'Final Answer' from the given text using regex.
    Returns the answer as a string with trailing whitespace stripped, or None if not found.
    """
    match = re.search(r"Final Answer:\s*(?P<final_answer>.*?)(?:\n*$|$)", text, re.DOTALL)
    if match:
        return match.group("final_answer")
    return text


TOTAL_TOKENS = 0  # Global variable to track total tokens from OpenAI response
MODEL = "gpt-3.5-turbo"  # OpenAI model to use

# String templates for agent prompts
PROMPTS = Prompts(
    role_playing_template="""
You are {role}, a highly capable professional with the following background: {backstory}

Your primary goal is: {goal}

Approach each task with strategic thinking, clarity, and professionalism. Leverage your expertise and unique perspective to deliver the best possible outcome. Communicate your reasoning clearly and act with confidence and precision.
""",
    tools_template="""
You are an expert at leveraging external tools to solve complex tasks. You should always prefer using the provided tools for any reasoning, calculation, or information retrieval, even if you believe you know the answer. Provided tools are reliable, accurate, and designed to help you achieve the best results.

{tools}

When a tool is available, do not attempt to solve any part of the task internally—always use the most relevant tool for each step. Your responses should reflect a clear, step-by-step approach, using tools as your primary method of reasoning and action.

⚠️ STRICT FORMAT REQUIREMENT ⚠️
You MUST ALWAYS use the following format in your response. If you deviate from this format in any way, your session will be immediately terminated and your output will be rejected.

---
```
Thought: <your reasoning about what to do next>
Action: <the action to take, only one name of [{tool_names}], just the name, exactly as it's written>
Action Input: <the input to the action, just a simple JSON object, enclosed in curly braces, using " to wrap keys and values>
```
---

- "Thought:" MUST always be present before every "Action:" and "Action Input:". You may NOT omit it.
- You are strictly forbidden from performing any calculations yourself. All arithmetic (add, subtract, multiply, divide) must be performed using the calculator tool. If you need to compute a difference, you must call the calculator tool with the appropriate arguments.
- Do NOT include an Observation until you have received the result from the system.
- Once all necessary information is gathered, return the following format:

---
```
Thought: I now know the final answer
Final Answer: <the final answer to the original input question>
```
---
""",
    no_tools_template="""
Approach the following task with strategic thinking, clarity, and professionalism. Communicate your reasoning clearly and deliver the best possible outcome based on your expertise and knowledge.

You must use the following format in your response:

---
Thought: <your reasoning about what to do next>
Final Answer: <your complete and most helpful answer to the original input question>
---

You MUST use this format exactly—your job depends on it!
""",
    tool_not_found_prompt="""
Thought: I tried to use the tool '{action}', but it doesn't exist.
Action: None
Action Input: {{}}
Observation: The tool '{action}' is not available. You must use one of the following tools: {tool_list}. Please try again using one of the available tools.
""",
    action_input_not_dict_prompt="""
Thought: I attempted to use the tool '{tool_name}', but the Action Input is not a dictionary.
Action: {tool_name}
Action Input: {action_input_str}
Observation: Action Input must be a JSON dictionary. Please provide a valid JSON dictionary for this tool.
""",
    tool_call_failed_prompt="""
Thought: I attempted to use the tool '{tool_name}' with input {action_input}, but it failed.
Action: {tool_name}
Action Input: {action_input_json}
Observation: Tool call failed. Reason: {exception}. Expected input schema: {tool_inputs}. Please review your tool input and try again.
""",
    action_input_parse_failed_prompt="""
Thought: I attempted to use the tool '{tool_name}', but failed to parse the Action Input.
Action: {tool_name}
Action Input: {action_input_str}
Observation: Failed to parse Action Input as JSON. Error: {exception}. Please provide a valid JSON dictionary for this tool.
""",
    tool_no_output_prompt="""
Thought: I attempted to use the tool '{tool_name}', but it returned no or invalid output.
Action: {tool_name}
Action Input: {action_input_json}
Observation: Tool returned no or invalid output. Please check your input and try again.
""",
    validation_system_prompt="""
You are a careful, critical validator. Your ONLY job is to check whether the output below fulfills the expected description exactly and directly—no assumptions, no extra reasoning.

Respond ONLY with:
---
Thought: <short reasoning>
Final Answer: yes

---
OR
---
Thought: <short reasoning>
Final Answer: no

---
Do NOT return anything but 'yes' or 'no' as the Final Answer. End with a line break.
""",
    validation_user_prompt="""
Output to check:
{final_answer}

Expected description:
{expected_output}

Does the output fulfill the expected description, with no missing or extra information? Think step by step, but reply ONLY with a Final Answer of 'yes' or 'no' as instructed above.
""",
    validation_retry_prompt="""
---
Thought: I see that my previous answer did not fulfill the task requirements.
The expected output was: '{expected_output}'.
My previous answer was: '{final_answer}'.
Validation agent responded with: '{result_final_answer}'.
Please try again, making sure to provide an answer that directly matches the expected output.
---
""",
    instruction_prompt="""
ONLY answer the following task. Do NOT attempt to solve the overall agent goal or reference other tasks.
""",
    current_task_prompt="""
Current Task: {task_description}

Begin! This is VERY important to you, use the tools available and give your best Final Answer, your job depends on it!

Thought:
""",
    context_explanation_prompt="""
You are provided with context from previous steps. Use this context to answer the current task. Do NOT repeat any tool calls or calculations already completed unless explicitly instructed. Summarize, rephrase, or synthesize the context as needed to best answer the current task.
""",
    missing_thought_prompt="""
Please provide a Thought with your Action and Action Input or a Thought with your Final Answer.
""",
    thought_required_prompt="""
Thought is required for this action.
""",
    tool_retry_prompt="""
The previous tool invocation failed with the following error: {exception}. Please try again.
""",
    validation_feedback_prompt="""
IMPORTANT: Your previous answer did NOT pass validation. Please review the expected output and try again.

Expected output: {expected_output}
Previous answer: {previous_answer}
Validation result: {validation_result}

Please provide a new answer that directly matches the expected output.
""",
    retry_failed_validation_prompt="""
IMPORTANT: Your previous answer did NOT pass validation.
Read the feedback below and try again, giving a Final Answer that directly fulfills the expected output.
{exception}
""",
    coaching_prompt="""
You have only provided a Thought. To proceed, you must now either:
- Use a tool by specifying an Action and Action Input, or
- Provide your best possible Final Answer.
Please do not repeat your previous thought. Continue by taking a concrete action or giving a final answer.
""",
    force_final_answer_prompt="""Now it's time you MUST give your absolute best final answer. You'll ignore all previous instructions, stop using any tools, and just return your absolute BEST Final answer.""",
    retry_failed_validation_prompt_2="""
IMPORTANT: Your previous answer did NOT pass validation.
Read the feedback below and try again, giving a Final Answer that directly fulfills the expected output.
{exception}
""",
    summary_task_description_prompt="Use the following information to achieve the agent's goal ({goal})",
)


def set_prompt_value(key: str, new_value: str) -> None:
    global PROMPTS

    def _extract_placeholders(template: str) -> Set[str]:
        return set(re.findall(r"\{(\w+)\}", template or ""))

    if not hasattr(PROMPTS, key):
        raise KeyError(f"PROMPTS has no key '{key}'")
    old_value = getattr(PROMPTS, key)
    old_placeholders = _extract_placeholders(old_value)
    new_placeholders = _extract_placeholders(new_value)
    if old_placeholders != new_placeholders:
        raise ValueError(f"Placeholder mismatch for '{key}'. Existing placeholders: {sorted(old_placeholders)}, new placeholders: {sorted(new_placeholders)}")
    PROMPTS = PROMPTS._replace(**{key: new_value})


class MessageType(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(NamedTuple):
    role: MessageType
    content: str


class OpenAIMessage(NamedTuple):
    role: str
    content: str


class Task(NamedTuple):
    name: str
    description: str
    expected_output: str


class Tool(NamedTuple):
    name: str
    description: str
    args_schema: str
    func: Callable[[dict[str, str]], str]


class Agent(NamedTuple):
    role: str
    goal: str
    backstory: str
    tasks: List[Task]
    tools: List[Tool]


class Organization(NamedTuple):
    agents: list[Agent]


class ToolConclusion(NamedTuple):
    input: str
    output: str


class TaskConclusion(NamedTuple):
    input: str
    output: str


class ValidationConclusion(NamedTuple):
    input: str
    output: str


class StepState(NamedTuple):
    phase: str
    agent: Agent
    task: Task
    result: TaskConclusion
    context: list[Message]
    llm_messages: list[Message]
    task_idx: int
    task_conclusions: list[TaskConclusion]


class AgentConclusion(NamedTuple):
    agent: Agent
    input: str
    output: str
    task_conclusions: list[TaskConclusion]


class OrganizationConclusion(NamedTuple):
    final_conclusion: TaskConclusion
    agent_conclusions: list[AgentConclusion]
    context: list[Message]


LLMResponseText = NewType("LLMResponseText", str)


class LLMCallerBase:
    """
    Base class for LLM API callers. Prepares, executes, and tracks LLM responses and token usage.
    Includes retry and error handling logic.
    """

    def __init__(self) -> None:
        self._response_text: Optional[LLMResponseText] = None
        self._tokens_used: Optional[int] = None

    def prepare_llm_response(self, messages: list["Message"], api_key: str = "api_key") -> None:
        import urllib.error
        import urllib.request

        url = "https://api.openai.com/v1/chat/completions"
        model = "gpt-3.5-turbo"
        payload = {"model": model, "messages": [{"role": m.role.value, "content": m.content} for m in messages]}
        key = load_api_keys()[api_key]
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        log_json(logging.INFO, "LLM Payload:", payload)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        retries = 5
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = resp.read().decode("utf-8")
                    resp_json = json.loads(resp_data)
                    log_json(logging.DEBUG, "LLM Raw Response:", resp_json)
                    assert "error" not in resp_json, f"OpenAI API returned error: {resp_json['error']}"
                    content = resp_json["choices"][0]["message"]["content"]
                    assert isinstance(content, str), "OpenAI response content is not a string"
                    total_tokens = resp_json["usage"]["total_tokens"]
                    assert isinstance(total_tokens, int), "OpenAI response total_tokens is not an int"
                    self._response_text = LLMResponseText(content)
                    self._tokens_used = total_tokens
                    if get_trace_llm():
                        try:
                            with open(get_trace_llm_filename(), "a") as f:
                                f.write("\n\n==================== TRACEBACK ====================\n")
                                f.write("".join(traceback.format_list(traceback.extract_stack()[-6:-2])) + "\n\n")
                                f.write("\n==================== CALL_LLM INPUT ====================\n")
                                for m in messages:
                                    f.write(f"Role: {m.role.value}\nContent: {m.content}\n")
                                f.write("\n==================== CALL_LLM OUTPUT ====================\n")
                                f.write(content + "\n\n")
                        except Exception as log_exc:
                            logging.error(f"Failed to write LLM mock data: {log_exc}")
                    return
            except urllib.error.HTTPError as e:
                error_content = e.read().decode("utf-8")
                log_json(logging.DEBUG, "LLM HTTPError raw content:", error_content)
                try:
                    error_json = json.loads(error_content)
                    log_json(logging.ERROR, "LLM HTTPError:", {"status": e.code, "reason": e.reason, "error": error_json})
                except Exception:
                    log_json(logging.ERROR, "LLM HTTPError (unparsable JSON):", {"status": e.code, "reason": e.reason, "error": error_content})
            except TimeoutError as e:
                log_json(logging.ERROR, "LLM recoverable network error:", {"type": type(e).__name__, "message": str(e)})
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                raise Exception(f"Network error after retries: {e}")
            except Exception as e:
                log_json(logging.ERROR, "LLM Unexpected error:", {"type": type(e).__name__, "message": str(e)})
                raise Exception(f"Unexpected error: {e}")
        raise Exception("LLM API call failed after all retries")

    def get_llm_response(self) -> Optional[LLMResponseText]:
        return self._response_text

    def get_llm_tokens_used(self) -> Optional[int]:
        return self._tokens_used


_DEFAULT_LLM_CALLER: LLMCallerBase = LLMCallerBase()


def set_llm_caller(llm_caller: LLMCallerBase) -> None:
    global _DEFAULT_LLM_CALLER
    _DEFAULT_LLM_CALLER = llm_caller


def call_llm(messages: list["Message"]) -> LLMResponseText:
    """
    Sends a list of Message objects to a language model (LLM) API and returns the assistant's response content as LLMResponseText (NewType).
    The transport can be swapped by replacing the global _DEFAULT_LLM_CALLER.
    """
    _DEFAULT_LLM_CALLER.prepare_llm_response(messages)
    resp = _DEFAULT_LLM_CALLER.get_llm_response()
    if resp is None:
        raise Exception("No response from LLM API")
    return LLMResponseText(resp)


def load_api_keys() -> dict[str, str]:
    """
    Load API keys from api_key.json at the project root.
    Returns a dict of keys. Raises FileNotFoundError if missing.
    Ensures the result is of type dict[str, str] at runtime.
    """
    api_keys_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api_key.json")
    with open(api_keys_path, "r") as f:
        api_keys = json.load(f)
    if not isinstance(api_keys, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in api_keys.items()):
        raise TypeError("api_key.json must contain a dict[str, str]")
    log_json(logging.DEBUG, "Loaded API_KEYS:", {k: (v[:6] + "..." if isinstance(v, str) and len(v) > 10 else v) for k, v in api_keys.items()})
    return api_keys


def tool_executor(action: str, action_input_str: str, tools: list[Tool], s: str) -> ToolConclusion:
    """
    Executes a tool action parsed from an LLM output, given a regex match for action/thought/action_input,
    the list of available tools, and the in-progress LLM output string.
    Returns a ToolConclusion with the original input JSON string and the output string.
    Handles tool lookup, input parsing, function execution, and error reporting.
    All exceptions are phrased as prompts to be given back to the LLM.
    """
    tool = next((t for t in tools if t.name == action), None)
    if not tool:
        tool_list = json.dumps([t.name for t in tools])
        prompt = PROMPTS.tool_not_found_prompt.format(action=action, tool_list=tool_list)
        logging.error(prompt)
        raise Exception(prompt)

    try:
        action_input = json.loads(action_input_str)
        logging.debug(f"Parsed action_input: {action_input}")
        if not isinstance(action_input, dict):
            prompt = PROMPTS.action_input_not_dict_prompt.format(tool_name=tool.name, action_input_str=action_input_str)
            logging.error(prompt)
            raise Exception(prompt)
    except Exception as e:
        prompt = PROMPTS.action_input_parse_failed_prompt.format(tool_name=tool.name if tool else action, action_input_str=action_input_str, exception=e)
        logging.error(prompt)
        raise Exception(prompt)

    try:
        logging.info(f"Executing tool '{tool.name}' with input {action_input}")
        result = tool.func(action_input)
        logging.debug(f"Tool '{tool.name}' execution result: {result}")
    except Exception as e:
        tool_inputs = getattr(tool, "args_schema", "{}")
        prompt = PROMPTS.tool_call_failed_prompt.format(
            tool_name=tool.name, action_input=action_input, action_input_json=json.dumps(action_input), exception=e, tool_inputs=tool_inputs
        )
        logging.error(prompt)
        raise Exception(prompt)

    if not result:
        prompt = PROMPTS.tool_no_output_prompt.format(tool_name=tool.name, action_input_json=json.dumps(action_input))
        logging.error(prompt)
        raise Exception(prompt)

    log_json(logging.INFO, "Tool executed successfully:", {"action": action, "input": action_input, "result": result})
    result = s.rstrip() + f"\nObservation: {result}\n"
    info = f"Tool '{tool.name}' succeeded with input {action_input}."
    log_json(logging.INFO, "Tool conclusion:", {"input": info, "output": result})
    return ToolConclusion(input=info, output=result)


def validation_executor(final_answer: str, task: Task) -> ValidationConclusion:
    """
    Validates a final answer string against the Task's expected_output.
    Returns ValidationConclusion on success.
    Raises Exception with validation prompt and result if validation fails.
    """
    system_prompt = PROMPTS.validation_system_prompt
    validation_prompt = PROMPTS.validation_user_prompt.format(final_answer=final_answer, expected_output=task.expected_output)
    val_llm_messages = [
        Message(role=MessageType.SYSTEM, content=system_prompt),
        Message(role=MessageType.USER, content=validation_prompt),
    ]
    llm_response = call_llm(val_llm_messages)
    llm_response_text = str(llm_response)
    extracted_answer = extract_final_answer(llm_response_text)
    result_final_answer = extracted_answer.lower().strip() if extracted_answer is not None else ""

    passed = result_final_answer == "yes"
    log_json(logging.DEBUG, "Validation finished", {"validation_prompt": validation_prompt, "result_final_answer": result_final_answer, "passed": passed})
    debug_step(f"Validation result for task: {task.name} - {passed} (yes vs {result_final_answer!r})")
    if not passed:
        retry_prompt = PROMPTS.validation_retry_prompt.format(expected_output=task.expected_output, final_answer=final_answer, result_final_answer=result_final_answer)
        log_json(logging.WARNING, "Validation failed:", {"task": task.name, "expected": task.expected_output, "actual": final_answer})
        raise Exception(retry_prompt)
    return ValidationConclusion(input=validation_prompt, output=result_final_answer)


def task_executor(agent: Agent, task: Task, task_conclusions: list[TaskConclusion]) -> TaskConclusion:
    """
    Executes a single task for the agent, orchestrating LLM interaction, tool usage, and answer validation.
    The core control flow is:
      1. Construct system and user prompts with agent persona, tools, task, and any prior context.
      2. Iteratively interact with the LLM:
         - If LLM provides a final answer, validate it and return success if valid.
         - If LLM attempts a tool call, execute tool and supply result as context to LLM.
         - If LLM only provides a thought, coach it to take action or answer.
      3. Retry as needed, including forced attempts to nudge the LLM to answer.
      4. Raise if no valid answer is obtained after all attempts.
    This layered loop ensures robust handling of tool errors, ambiguous outputs, and stubborn LLM behavior, maximizing the chance of a valid answer.
    """
    tools = agent.tools
    # --- Build system prompt with tool information if available ---
    if tools:
        tool_descriptions = "\n".join(f"- {t.name}: {t.description} (args: {t.args_schema})" for t in tools)
        tool_names = ", ".join(t.name for t in tools)
        system_content = PROMPTS.tools_template.format(tools=tool_descriptions, tool_names=tool_names)
    else:
        system_content = PROMPTS.no_tools_template

    # --- Build user prompt with agent persona, task, and prior context ---
    user_content = PROMPTS.role_playing_template.format(role=agent.role, goal=agent.goal, backstory=agent.backstory)
    prompt_parts = [user_content]
    prompt_parts.append(PROMPTS.instruction_prompt)
    if task_conclusions:
        # Explain how context should be used, encourage synthesis rather than repetition
        prompt_parts.append(PROMPTS.context_explanation_prompt)
        # Show the context messages as input
        for tc in task_conclusions:
            prompt_parts.append(f"{tc.input}\n{tc.output}\n")
    prompt_parts.append(PROMPTS.current_task_prompt.format(task_description=task.description))
    full_user_prompt = "\n\n".join(prompt_parts)

    # Initialize LLM message chain (can be mutated across retries)
    llm_messages = [
        Message(role=MessageType.SYSTEM, content=system_content),
        Message(role=MessageType.USER, content=full_user_prompt),
    ]

    max_attempts = 5  # Allow several attempts for normal LLM/task interaction
    extra_attempts = 2  # Allow a couple forced attempts if LLM gets stuck
    for attempt in range(max_attempts):
        try:
            # Query LLM
            llm_response = call_llm(llm_messages)
            llm_response_text = str(llm_response)
            llm_messages.append(Message(role=MessageType.ASSISTANT, content=llm_response_text.strip()))

            # --- Parse LLM output ---
            final_match = AGENT_FINAL_REGEX.search(llm_response_text)
            action_match = AGENT_ACTION_REGEX.search(llm_response_text)
            thought_only_match = AGENT_THOUGHT_ONLY_REGEX.search(llm_response_text)
            if not thought_only_match:
                llm_messages.append(Message(role=MessageType.USER, content=PROMPTS.missing_thought_prompt))
                continue
            debug_step(f"LLM response parsing: {llm_response_text}\nFinal Match: {final_match}\nAction Match: {action_match}\nThought Only Match: {thought_only_match}")

            if action_match:
                # LLM wants to use a tool. Try to execute and supply result as new context.
                log_json(
                    logging.DEBUG,
                    "LLM response parsing:",
                    {
                        "attempt": attempt,
                        "Thought": action_match.group("thought").strip(),
                        "Action": action_match.group("action").strip(),
                        "Action Input": action_match.group("action_input").strip(),
                    },
                )
                try:
                    tool_conclusion = tool_executor(
                        action_match.group("action").strip(),
                        action_match.group("action_input").strip(),
                        tools,
                        llm_response_text,
                    )
                    # Inform LLM of tool outcome as new message
                    llm_messages.append(Message(role=MessageType.USER, content=f"{tool_conclusion.input}\n{tool_conclusion.output}"))
                except Exception as e:
                    # Tool failed: prompt LLM to try again
                    llm_messages.append(
                        Message(
                            role=MessageType.USER,
                            content=PROMPTS.tool_retry_prompt.format(exception=str(e)),
                        )
                    )
                continue  # Continue to next LLM round
            elif final_match:
                # LLM gave a Final Answer. Validate it.
                log_json(
                    logging.INFO,
                    "LLM response parsing:",
                    {
                        "attempt": attempt,
                        "Thought": final_match.group("thought").strip(),
                        "Final Answer": final_match.group("final_answer").strip(),
                    },
                )
                for _ in range(3):  # Allow several retries if validation fails
                    try:
                        debug_step(f"Validating final answer for task: {task.name}")
                        validation_executor(final_match.group("final_answer").strip(), task)
                        return TaskConclusion(
                            input=f"Task Name: {task.name}\nTask Description: {task.description}\nTask Expected Output: {task.expected_output}",
                            output=f"Final Answer: {final_match.group('final_answer').strip()}",
                        )
                    except Exception as e:
                        # Give feedback to LLM and request a better answer
                        retry_prompt = PROMPTS.retry_failed_validation_prompt.format(exception=str(e))
                        llm_messages.append(Message(role=MessageType.USER, content=retry_prompt))
                        llm_response = call_llm(llm_messages)
                        llm_response_text = str(llm_response)
                        llm_messages.append(Message(role=MessageType.ASSISTANT, content=llm_response_text.strip()))
                        final_match = AGENT_FINAL_REGEX.search(llm_response_text)
                        if not final_match:
                            raise Exception("Validation failed: No valid final answer. RESET_TASK")
                else:
                    # Too many invalid answers
                    raise Exception("Validation failed after retries. RESET_TASK")

            elif thought_only_match:
                # LLM is indecisive, nudge it to take action or answer
                log_json(
                    logging.DEBUG,
                    "LLM response parsing:",
                    {
                        "attempt": attempt,
                        "Thought": thought_only_match.group("thought").strip(),
                    },
                )
                llm_messages.append(Message(role=MessageType.USER, content=PROMPTS.coaching_prompt))
                continue

        except Exception as e:
            # Catch all unexpected errors, log and continue attempt loop
            log_json(logging.WARNING, "LLM/Tool error during attempt:", {"attempt": attempt, "error": str(e)})
            continue

    # --- Fallback: force LLM to answer if all else fails ---
    for force_attempt in range(extra_attempts):
        try:
            llm_messages.append(
                Message(
                    role=MessageType.USER,
                    content=PROMPTS.force_final_answer_prompt,
                )
            )
            llm_response = call_llm(llm_messages)
            llm_response_text = str(llm_response)
            final_match = AGENT_FINAL_REGEX.search(llm_response_text)
            if final_match:
                log_json(
                    logging.INFO,
                    "LLM response parsing:",
                    {
                        "attempt": max_attempts + force_attempt,
                        "Thought": final_match.group("thought").strip(),
                        "Final Answer": final_match.group("final_answer").strip(),
                    },
                )
                for _ in range(3):
                    try:
                        debug_step(f"Validating final answer for task: {task.name}")
                        validation_executor(final_match.group("final_answer").strip(), task)
                        return TaskConclusion(
                            input=f"Task Name: {task.name}\nTask Description: {task.description}\nTask Expected Output: {task.expected_output}",
                            output=final_match.group("final_answer").strip(),
                        )
                    except Exception as e:
                        retry_prompt = PROMPTS.retry_failed_validation_prompt_2.format(exception=str(e))
                        llm_messages.append(Message(role=MessageType.USER, content=retry_prompt))
                        llm_response = call_llm(llm_messages)
                        llm_response_text = str(llm_response)
                        llm_messages.append(Message(role=MessageType.ASSISTANT, content=llm_response_text.strip()))
                        final_match = AGENT_FINAL_REGEX.search(llm_response_text)
                        if not final_match:
                            raise Exception("Validation failed: No valid final answer. RESET_TASK")
        except Exception as e:
            log_json(logging.WARNING, "Forced attempt error:", {"force_attempt": force_attempt, "error": str(e)})
            continue

    # After all attempts, fail: no valid answer could be obtained
    raise Exception("Validation failed: No valid final answer. RESET_TASK")


def agent_executor(agent: Agent, agent_conclusions: list[AgentConclusion]) -> AgentConclusion:
    """
    Executes all tasks for the agent sequentially.
    - Builds context from previous AgentConclusion.task_conclusions if provided.
    - Handles retry logic for task failures and validation.
    Inserts each TaskConclusion at the beginning of the task_conclusions list.
    Returns an AgentConclusion containing all results.
    """
    # Build context from previous agent_conclusions' task_conclusions
    task_conclusions: list[TaskConclusion] = []
    for ac in agent_conclusions:
        task_conclusions.extend(ac.task_conclusions)

    for task in agent.tasks:
        max_retries = 3
        result = None
        for attempt in range(max_retries):
            try:
                log_json(logging.DEBUG, "agent_executor.task_attempt", {"agent": agent.role, "task": task.name, "attempt": attempt + 1})
                result = task_executor(agent=agent, task=task, task_conclusions=task_conclusions)
                task_conclusions.append(result)
                log_json(logging.DEBUG, "agent_executor.task_result", {"agent": agent.role, "task": task.name, "attempt": attempt + 1, "result": result})
                break
            except Exception as e:
                msg = str(e)
                if "RESET_TASK" in msg:
                    log_json(
                        logging.INFO,
                        "agent_executor.reset_task",
                        {"agent": agent.role, "task": task.name, "attempt": attempt + 1, "error": msg},
                    )
                    continue
                else:
                    log_json(
                        logging.INFO,
                        "agent_executor.task_exception",
                        {"agent": agent.role, "task": task.name, "attempt": attempt + 1, "error": msg},
                    )
                    raise
    summary_task = Task(
        name="Summary",
        description=PROMPTS.summary_task_description_prompt.format(goal=agent.goal),
        expected_output=agent.goal,
    )
    summary = task_executor(
        agent=Agent(
            role="Summary",
            goal="Summarize the results of the previous tasks and provide a final answer.",
            backstory="",
            tasks=[summary_task],
            tools=[],
        ),
        task=summary_task,
        task_conclusions=task_conclusions,
    )
    return AgentConclusion(agent=agent, input=summary.input, output=summary.output, task_conclusions=task_conclusions)


def organization_executor(org: Organization) -> Optional[OrganizationConclusion]:
    """
    Executes each agent in the organization in order, passing all previous agents' conclusions as context to the next.
    Returns an OrganizationConclusion with the final output from the last agent and all agent conclusions.
    Optionally supports a callback after each agent.
    """
    agent_conclusions: list[AgentConclusion] = []
    for agent in org.agents:
        agent_conclusion = agent_executor(agent=agent, agent_conclusions=agent_conclusions)
        agent_conclusions.append(agent_conclusion)
    for agent_conclusion in agent_conclusions[::-1]:
        return OrganizationConclusion(
            final_conclusion=agent_conclusion.task_conclusions[-1],
            agent_conclusions=agent_conclusions,
            context=[],
        )
    return None


def main() -> None:
    try:
        logging.error("Import from this file similar to the examples.py file to use this library.")
    except Exception as e:
        logging.exception(f"Error: {e}")


if __name__ == "__main__":
    main()
