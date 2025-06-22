# gpt_agents_py | James Delancey | MIT License
import logging

from examples.basic_usage import org
from gpt_agents_py.extensions.anthropic_llm_caller import AnthropicLLMCaller
from gpt_agents_py.gpt_agents import (  # set_debug_mode,; set_trace_mode,
    log_json,
    organization_executor,
    set_llm_caller,
)

if __name__ == "__main__":
    # set_trace_mode(True)
    # set_debug_mode(True)
    set_llm_caller(AnthropicLLMCaller())

    final_result = organization_executor(org)
    log_json(logging.DEBUG, "Organization Results:", final_result)
    print("Organization Results:")
    print(final_result.final_conclusion.output if final_result else "No final conclusion")
