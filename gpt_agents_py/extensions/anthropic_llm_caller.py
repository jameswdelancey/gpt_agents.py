# gpt_agents_py | James Delancey | MIT License
import collections
import json
import logging
import threading
import time
import traceback
from typing import List

from gpt_agents_py.gpt_agents import (
    LLMCallerBase,
    LLMResponseText,
    Message,
    get_trace_llm,
    get_trace_llm_filename,
    load_api_keys,
    log_json,
)


class AnthropicLLMCaller(LLMCallerBase):
    """
    LLMCaller for Anthropic models (e.g., Claude Sonnet).
    Implements a rate limiter: max 20 queries per minute (QPM).
    """

    _rate_limit_lock = threading.Lock()
    _rate_limit_window = 60  # seconds
    _rate_limit_max = 10
    _rate_limit_times: collections.deque[float] = collections.deque(maxlen=_rate_limit_max)

    def _acquire_rate_limit(self) -> None:
        with self._rate_limit_lock:
            now = time.time()
            # Remove timestamps older than 60 seconds
            while self._rate_limit_times and now - self._rate_limit_times[0] > self._rate_limit_window:
                self._rate_limit_times.popleft()
            if len(self._rate_limit_times) >= self._rate_limit_max:
                # Sleep until the earliest timestamp is out of window
                sleep_time = self._rate_limit_window - (now - self._rate_limit_times[0])
                if sleep_time > 0:
                    log_json(logging.INFO, "Anthropic LLM rate limit exceeded, sleeping for:", {"sleep_time": sleep_time})
                    time.sleep(sleep_time)
                # After sleep, clean up old timestamps
                now = time.time()
                while self._rate_limit_times and now - self._rate_limit_times[0] > self._rate_limit_window:
                    self._rate_limit_times.popleft()
            self._rate_limit_times.append(time.time())

    def prepare_llm_response(self, messages: List["Message"], api_key: str = "anthropic") -> None:
        self._acquire_rate_limit()
        import urllib.error
        import urllib.request

        # Load API keys
        try:
            keys = load_api_keys()
            key = keys[api_key]
        except (FileNotFoundError, KeyError):
            raise BaseException(f"Anthropic API key '{api_key}' not found in api_keys.json.")

        url = "https://api.anthropic.com/v1/messages"
        model = "claude-3-7-sonnet-latest"
        # Anthropic expects the first 'system' message as a top-level 'system' field, not in the messages list
        system_prompt = None
        filtered_messages = []
        for m in messages:
            if m.role.value == "system" and system_prompt is None:
                system_prompt = m.content
            else:
                filtered_messages.append({"role": m.role.value, "content": m.content})
        payload = {
            "model": model,
            "max_tokens": 4096,
            "messages": filtered_messages,
        }
        if system_prompt is not None:
            payload["system"] = system_prompt

        headers = {
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }
        log_json(logging.INFO, "Anthropic LLM Payload:", payload)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        retries = 5
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = resp.read().decode("utf-8")
                    resp_json = json.loads(resp_data)
                    log_json(logging.DEBUG, "Anthropic LLM Raw Response:", resp_json)
                    assert "error" not in resp_json, f"Anthropic API returned error: {resp_json['error']}"
                    # Defensive check for empty or missing content
                    if not resp_json.get("content") or not isinstance(resp_json["content"], list) or not resp_json["content"]:
                        log_json(logging.ERROR, "Anthropic LLM API returned empty content", {"response": resp_json})
                        raise Exception("You must provide a non-empty string as the response content.")
                    content = resp_json["content"][0]["text"]
                    assert isinstance(content, str), "Anthropic response content is not a string"
                    self._response_text = LLMResponseText(content)
                    # Anthropic does not always return token usage, so set to None or extract if present
                    self._tokens_used = resp_json.get("usage", {}).get("output_tokens")
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
                            logging.error(f"Failed to write LLM trace data: {log_exc}")
                    return
            except urllib.error.HTTPError as e:
                error_content = e.read().decode("utf-8")
                log_json(logging.DEBUG, "Anthropic LLM HTTPError raw content:", error_content)
                try:
                    error_json = json.loads(error_content)
                    log_json(logging.ERROR, "Anthropic LLM HTTPError:", {"status": e.code, "reason": e.reason, "error": error_json})
                except Exception:
                    log_json(logging.ERROR, "Anthropic LLM HTTPError (unparsable JSON):", {"status": e.code, "reason": e.reason, "error": error_content})
            except TimeoutError as e:
                log_json(logging.ERROR, "Anthropic LLM recoverable network error:", {"type": type(e).__name__, "message": str(e)})
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                raise Exception(f"Network error after retries: {e}")
            except Exception as e:
                log_json(logging.ERROR, "Anthropic LLM Unexpected error:", {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc(limit=3)})
                raise Exception(f"Unexpected error: {e}")
        raise Exception("Anthropic LLM API call failed after all retries")
