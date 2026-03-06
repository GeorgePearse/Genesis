from typing import List, Union, Optional, Dict
import random
from pydantic import BaseModel
from .client import get_client_llm
from .models.pricing import (
    CLAUDE_MODELS,
    OPENAI_MODELS,
    DEEPSEEK_MODELS,
    GEMINI_MODELS,
    BEDROCK_MODELS,
    REASONING_OAI_MODELS,
    REASONING_CLAUDE_MODELS,
    REASONING_DEEPSEEK_MODELS,
    REASONING_GEMINI_MODELS,
    REASONING_AZURE_MODELS,
    REASONING_BEDROCK_MODELS,
    OPENROUTER_MODELS,
)
from .models import (
    query_anthropic,
    query_openai,
    query_deepseek,
    query_gemini,
    QueryResult,
)
import logging
import time

logger = logging.getLogger(__name__)


THINKING_TOKENS = {
    "auto": 4096,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "max": 16384,
}


def sample_batch_kwargs(
    num_samples: int,
    model_names: Union[List[str], str] = "gpt-4o-mini-2024-07-18",
    temperatures: Union[List[float], float] = 0.0,
    max_tokens: Union[List[int], int] = 4096,
    reasoning_efforts: Union[List[str], str] = "",
    model_sample_probs: Optional[List[float]] = None,
    unique_filter: bool = False,
):
    """Sample a dictionary of kwargs for a given model."""
    all_kwargs = []
    attempts = 0
    max_attempts = num_samples * 10  # Prevent infinite loops

    while len(all_kwargs) < num_samples and attempts < max_attempts:
        kwargs_dict = sample_model_kwargs(
            model_names=model_names,
            temperatures=temperatures,
            max_tokens=max_tokens,
            reasoning_efforts=reasoning_efforts,
            model_sample_probs=model_sample_probs,
        )

        if unique_filter:
            if kwargs_dict not in all_kwargs:
                all_kwargs.append(kwargs_dict)
        else:
            all_kwargs.append(kwargs_dict)

        attempts += 1

    if len(all_kwargs) < num_samples:
        logger.info(
            f"Could not generate {num_samples} unique kwargs combinations "
            f"after {max_attempts} attempts"
        )
        logger.info(f"Returning {len(all_kwargs)} unique kwargs combinations.")

    return all_kwargs


def sample_model_kwargs(
    model_names: Union[List[str], str] = "gpt-4o-mini-2024-07-18",
    temperatures: Union[List[float], float] = 0.0,
    max_tokens: Union[List[int], int] = 4096,
    reasoning_efforts: Union[List[str], str] = "",
    model_sample_probs: Optional[List[float]] = None,
):
    """Sample a dictionary of kwargs for a given model."""
    # Make all inputs lists
    if isinstance(model_names, str):
        model_names = [model_names]
    if isinstance(temperatures, float):
        temperatures = [temperatures]
    if isinstance(max_tokens, int):
        max_tokens = [max_tokens]
    if isinstance(reasoning_efforts, str):
        reasoning_efforts = [reasoning_efforts]

    kwargs_dict = {}
    # perform model sampling if list provided
    if model_sample_probs is not None:
        if len(model_sample_probs) != len(model_names):
            raise ValueError(
                "model_sample_probs must have the same length as model_names"
            )
        if not abs(sum(model_sample_probs) - 1.0) < 1e-9:
            raise ValueError("model_sample_probs must sum to 1")
        kwargs_dict["model_name"] = random.choices(
            model_names, weights=model_sample_probs, k=1
        )[0]
    else:
        kwargs_dict["model_name"] = random.choice(model_names)

    # perform temperature sampling if list provided
    # set temperature to 1.0 for reasoning models
    if kwargs_dict["model_name"] in (
        REASONING_OAI_MODELS
        + REASONING_CLAUDE_MODELS
        + REASONING_DEEPSEEK_MODELS
        + REASONING_GEMINI_MODELS
        + REASONING_AZURE_MODELS
        + REASONING_BEDROCK_MODELS
    ):
        kwargs_dict["temperature"] = 1.0
    else:
        kwargs_dict["temperature"] = random.choice(temperatures)

    # perform reasoning effort sampling if list provided
    # set max_completion_tokens for OAI reasoning models
    if kwargs_dict["model_name"] in (REASONING_OAI_MODELS + REASONING_AZURE_MODELS):
        kwargs_dict["max_output_tokens"] = random.choice(max_tokens)
        r_effort = random.choice(reasoning_efforts)
        if r_effort != "auto":
            kwargs_dict["reasoning"] = {"effort": r_effort}

    if kwargs_dict["model_name"] in (REASONING_GEMINI_MODELS):
        kwargs_dict["max_tokens"] = random.choice(max_tokens)
        r_effort = random.choice(reasoning_efforts)
        # Always enable thinking if effort is auto or specified
        t = THINKING_TOKENS.get(r_effort, 4096)
        thinking_tokens = t if t < kwargs_dict["max_tokens"] else 1024
        kwargs_dict["extra_body"] = {
            "extra_body": {
                "google": {
                    "thinking_config": {
                        "thinking_budget": thinking_tokens,
                        "include_thoughts": True,
                    }
                }
            }
        }

    elif kwargs_dict["model_name"] in (
        REASONING_CLAUDE_MODELS + REASONING_BEDROCK_MODELS
    ):
        kwargs_dict["max_tokens"] = min(random.choice(max_tokens), 16384)
        r_effort = random.choice(reasoning_efforts)

        # Enable thinking tokens
        t = THINKING_TOKENS.get(r_effort, 4096)
        thinking_tokens = t if t < kwargs_dict["max_tokens"] else 1024

        # sample only from thinking tokens that are valid
        kwargs_dict["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_tokens,
        }

    else:
        if (
            kwargs_dict["model_name"] in CLAUDE_MODELS
            or kwargs_dict["model_name"] in BEDROCK_MODELS
            or kwargs_dict["model_name"] in REASONING_CLAUDE_MODELS
            or kwargs_dict["model_name"] in REASONING_BEDROCK_MODELS
            or kwargs_dict["model_name"] in DEEPSEEK_MODELS
            or kwargs_dict["model_name"] in REASONING_DEEPSEEK_MODELS
            or kwargs_dict["model_name"].startswith("openrouter/")
        ):
            kwargs_dict["max_tokens"] = random.choice(max_tokens)
        else:
            kwargs_dict["max_output_tokens"] = random.choice(max_tokens)

    return kwargs_dict


def query(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[BaseModel] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    tools: Optional[List[Dict]] = None,
    tool_map: Optional[Dict[str, callable]] = None,
    **kwargs,
) -> QueryResult:
    """Query the LLM."""
    client, model_name_processed = get_client_llm(
        model_name, structured_output=output_model is not None
    )
    if model_name.startswith("openrouter/"):
        # OpenRouter uses the OpenAI-compatible client
        query_fn = query_openai
        # We need to pass the processed model name (e.g., anthropic/claude-3-5-sonnet)
        # to the query function, but get_client_llm already returns it.
        # However, query_openai expects the model_name argument to be passed to client.chat.completions.create
        # so we should use the one returned by get_client_llm.
        model_name = model_name_processed
    elif model_name in CLAUDE_MODELS.keys() or "anthropic" in model_name:
        query_fn = query_anthropic
    elif model_name in OPENAI_MODELS.keys():
        query_fn = query_openai
    elif model_name in DEEPSEEK_MODELS.keys():
        query_fn = query_deepseek
    elif model_name in GEMINI_MODELS.keys():
        query_fn = query_gemini
    else:
        raise ValueError(f"Model {model_name} not supported.")

    start_time = time.time()

    # Loop for tool calling
    max_tool_iterations = 5
    current_msg_history = list(msg_history)
    current_msg = msg
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for i in range(max_tool_iterations):
        result = query_fn(
            client,
            model_name,
            current_msg,
            system_msg,
            current_msg_history,
            output_model,
            model_posteriors=model_posteriors,
            tools=tools,
            **kwargs,
        )

        # Accumulate costs and tokens
        total_cost += result.cost
        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens

        # Check for tool calls
        if result.tool_calls and tool_map:
            # Execute tools
            tool_outputs = []
            for tool_call in result.tool_calls:
                function_name = tool_call.get("name") or tool_call.get(
                    "function", {}
                ).get("name")
                function_args = tool_call.get("input") or tool_call.get(
                    "function", {}
                ).get("arguments")
                call_id = tool_call.get("id")

                if isinstance(function_args, str):
                    try:
                        import json

                        function_args = json.loads(function_args)
                    except:
                        pass

                if function_name in tool_map:
                    try:
                        logger.info(
                            f"Executing tool {function_name} with args {function_args}"
                        )
                        output = tool_map[function_name](**function_args)
                    except Exception as e:
                        output = {"error": str(e)}
                else:
                    output = {"error": f"Tool {function_name} not found"}

                tool_outputs.append(
                    {
                        "tool_call_id": call_id,
                        "output": str(output),
                        "name": function_name,
                    }
                )

            # Prepare history for next iteration
            # The query_fn already appended the assistant message with tool calls to new_msg_history in result
            current_msg_history = result.new_msg_history

            # Add tool outputs to history
            for tool_output in tool_outputs:
                if "anthropic" in model_name or model_name in CLAUDE_MODELS:
                    current_msg_history.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_output["tool_call_id"],
                                    "content": tool_output["output"],
                                }
                            ],
                        }
                    )
                else:
                    # OpenAI format
                    current_msg_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_output["tool_call_id"],
                            "name": tool_output["name"],
                            "content": tool_output["output"],
                        }
                    )

            # For the next iteration, msg is empty because context is in history
            # But query_fn signature expects a msg.
            # For Anthropic, we just continue conversation.
            # For OpenAI, we send new history.
            # We need to adjust query_fn to handle empty msg if history has it?
            # actually query_fn appends msg to history. If msg is empty string, it appends empty user message which might be weird.
            # But we can just pass "Continue" or similar if needed, or rely on the fact that we updated history.
            # Actually, let's look at query_fn again. It does `new_msg_history = msg_history + [{"role": "user", "content": msg}]`
            # We don't want to add a user message if we are just returning tool outputs.
            # We need to pass the *updated* history as msg_history, and an empty msg?
            # Or better, we modify query logic to just use the updated history.

            # Since we can't easily modify query_fn signature behaviour without breaking things,
            # we will pass a dummy prompt like "continue" or rely on the fact that for tool use loops,
            # the "user" response IS the tool output.
            # Wait, `query_fn` adds a user message with `msg`.
            # If we pass `msg=""`, it adds an empty user message.
            # Correct flow for tool use:
            # 1. User: msg
            # 2. Assistant: tool_call
            # 3. Tool: tool_result
            # 4. Assistant: final response

            # In our loop:
            # Iter 0: msg passed as `msg`. `msg_history` is initial history.
            # `query_fn` constructs `[...history, user:msg]`. Returns `assistant:tool_call`.
            # We get `result.new_msg_history` which is `[...history, user:msg, assistant:tool_call]`.
            # We append `tool:tool_result` to this.
            # Iter 1: We want to get the next assistant response.
            # If we call `query_fn` again, we need to pass the updated history.
            # But `query_fn` enforces adding a user message `msg`.
            # We cannot use `query_fn` as is for the loop step if it forces a new user message.
            # We need to change `query_fn` to allow `msg` to be None/skip if history is sufficient?
            # Or we hack it by passing the tool output as the `msg` for the next turn?

            # For OpenAI:
            # messages = [system] + history + [user:msg]
            # If we want to send tool outputs, they are "tool" role messages.
            # So `msg` is not appropriate for tool outputs.

            # We need to modify `query_fn` to handle this scenario or implement the loop logic inside `query_fn`.
            # But `query` wraps `query_fn`.

            # Let's modify `query` to NOT use `query_fn` for subsequent iterations, but call client directly?
            # That duplicates logic.

            # Better: Modify `query_fn` to accept `msg=None` and skip adding it if so.

            current_msg = None
            # We will modify query_fn to handle msg=None

        else:
            # No tool calls, we are done
            break

    end_time = time.time()

    # Update result costs/tokens to reflect total
    result.cost = total_cost
    result.input_tokens = total_input_tokens
    result.output_tokens = total_output_tokens

    # Log to ClickHouse
    try:
        from genesis.utils.clickhouse_logger import ch_logger

        log_messages = []
        if system_msg:
            log_messages.append({"role": "system", "content": system_msg})
        if msg_history:
            log_messages.extend(msg_history)
        if msg:
            log_messages.append({"role": "user", "content": msg})

        ch_logger.log_llm_interaction(
            model=model_name,
            messages=log_messages,
            response=result.content if result else "None",
            cost=result.cost
            if result and hasattr(result, "cost") and result.cost
            else 0.0,
            execution_time=end_time - start_time,
            metadata=kwargs,
            thought=result.thought if result and hasattr(result, "thought") else "",
        )
    except Exception as e:
        logger.warning(f"Failed to log to ClickHouse: {e}")

    return result
