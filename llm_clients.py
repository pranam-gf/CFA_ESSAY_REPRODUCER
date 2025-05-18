"""
Functions for interacting with various Language Model APIs.
"""
import time
import json
import logging
import re
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from google import genai
from google.genai import types
from writerai import Writer
from groq import Groq
import anthropic
from mistralai.client import MistralClient
import string
import writerai
import tiktoken
import time 
import random
from . import config
import os
import httpx

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY = 1 
RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    RateLimitError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    Exception 
)

def _estimate_tokens_tiktoken(text: str, encoding_name: str = "cl100k_base") -> int | None:
    """Estimates token count for a given text using tiktoken.
    Args:
        text: The text to estimate tokens for.
        encoding_name: The tiktoken encoding to use (e.g., "cl100k_base").   
    Returns:
        Estimated token count or None if estimation fails.
    """
    if not text: 
        return 0
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(text))
        return num_tokens
    except Exception as e:
        logger.warning(f"Could not estimate tokens using tiktoken (encoding: {encoding_name}): {e}")
        return None

def get_llm_response(prompt: str, model_config: dict, is_json_response_expected: bool = False) -> dict | None:
    """
    Sends a prompt to the specified LLM API and parses the response.
    Measures response time and attempts to extract token counts.

    Args:
        prompt: The prompt string to send to the LLM.
        model_config: Dictionary containing model type, ID, parameters.
        is_json_response_expected: If True, attempts to parse the entire response as JSON.
                                   If False (default), extracts a single letter A,B,C,D.

    Returns:
        A dictionary containing {'response_content': <parsed_response_or_letter>,
                                'raw_response_text': <full_response_text>,
                                'response_time', 'input_tokens', 'output_tokens'},
        or None if the API call failed or response parsing failed critically.
        'response_content' will be a dict if is_json_response_expected is True and parsing succeeds,
        otherwise it will be the extracted letter or "X" on failure.
    """
    model_type = model_config.get("type")
    model_id = model_config.get("model_id")
    parameters = model_config.get("parameters", {}).copy()
    config_id = model_config.get("config_id", model_id)

    if model_type == "openai" and is_json_response_expected and parameters.get("response_format", {}).get("type") == "json_object":
        if "json" not in prompt.lower():
             logger.warning(f"OpenAI model {config_id} called with response_format=json_object, but 'json' not found in prompt. This might lead to API errors.")
    elif model_type == "openai":
         parameters.pop('response_format', None)

    logger.info(f"Sending prompt to {model_type} model: {config_id} (JSON Expected: {is_json_response_expected})")
    start_time = time.time()
    input_tokens = None
    output_tokens = None
    response_text_for_error = "N/A"
    elapsed_time = 0
    for attempt in range(MAX_RETRIES):
        try:
            if model_type == "openai":
                if not config.OPENAI_API_KEY:
                    logger.error(f"Missing OpenAI API key for model {config_id}.")
                    return {"error_message": "Missing OpenAI API key", "response_time": 0}
                openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
                
                openai_params = parameters.copy()
                if is_json_response_expected and openai_params.get("response_format", {}).get("type") == "json_object":
                     pass
                else:
                    openai_params.pop('response_format', None)
                try:
                    api_response = openai_client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        **openai_params
                    )
                    response_text_for_error = api_response.choices[0].message.content.strip() if api_response.choices and api_response.choices[0].message else ""
                    if hasattr(api_response, 'usage') and api_response.usage:
                        input_tokens = api_response.usage.prompt_tokens
                        output_tokens = api_response.usage.completion_tokens
                    else: 
                        logger.warning(f"Token usage data not found in OpenAI response for {config_id}. Setting to None.")
                except Exception as e:
                    elapsed_time = time.time() - start_time
                    logger.error(f"OpenAI API Error for {config_id} after {elapsed_time:.2f}s: {e}. Raw response snippet: {response_text_for_error[:100]}...")
                    error_details = {"type": "APIError", "message": str(e)}
                    if hasattr(e, 'status_code'):
                        error_details["status_code"] = e.status_code
                    if hasattr(e, 'body') and e.body:
                        error_details["body"] = e.body
                    return {"error_message": f"OpenAI API Error: {str(e)}", "response_time": elapsed_time, "details": error_details, "raw_response_text": response_text_for_error}

            elif model_type == "xai":
                if not config.XAI_API_KEY:
                    logger.error(f"Missing xAI API key for model {config_id}.")
                    return {"error_message": "Missing xAI API key", "response_time": 0}
                xai_client = OpenAI(
                    api_key=config.XAI_API_KEY,
                    base_url="https://api.x.ai/v1",
                )
                xai_params = parameters.copy()
                xai_params.pop('response_format', None)
                
                try:
                    api_response = xai_client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        **xai_params
                    )
                    response_text_for_error = api_response.choices[0].message.content.strip() if api_response.choices and api_response.choices[0].message else ""
                    if hasattr(api_response, 'usage') and api_response.usage:
                        input_tokens = api_response.usage.prompt_tokens
                        output_tokens = api_response.usage.completion_tokens
                    else: 
                        logger.warning(f"Token usage data not found in xAI response for {config_id}. Setting to None.")
                except Exception as e:
                    elapsed_time = time.time() - start_time
                    logger.error(f"xAI API Error for {config_id} after {elapsed_time:.2f}s: {e}. Raw response snippet: {response_text_for_error[:100]}...")
                    error_details = {"type": "APIError", "message": str(e)}
                    if hasattr(e, 'status_code'):
                        error_details["status_code"] = e.status_code
                    if hasattr(e, 'body') and e.body:
                        error_details["body"] = e.body
                    return {"error_message": f"xAI API Error: {str(e)}", "response_time": elapsed_time, "details": error_details, "raw_response_text": response_text_for_error}

            elif model_type == "gemini":
                if not config.GEMINI_API_KEY:
                    logger.error(f"Missing Gemini API key for model {config_id}.")
                    return {"error_message": "Missing Gemini API key", "response_time": 0}
                
                gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)

                gemini_params_from_config = parameters.copy()
                
                gen_config_args = {
                    k: v for k, v in gemini_params_from_config.items() 
                    if k in ["temperature", "top_p", "top_k", "max_output_tokens", "candidate_count", "stop_sequences"] and v is not None
                }
                
                thinking_budget_value = gemini_params_from_config.get("thinking_budget")
                final_api_call_config = None

                if thinking_budget_value is not None:
                    try:
                        thinking_config_obj = types.ThinkingConfig(thinking_budget=thinking_budget_value)
                        final_api_call_config = types.GenerateContentConfig(
                            **gen_config_args,
                            thinking_config=thinking_config_obj
                        )
                        logger.info(f"Gemini model {config_id}: Using GenerateContentConfig with ThinkingConfig (budget: {thinking_budget_value}) and other args: {gen_config_args}")
                    except AttributeError as e:
                        logger.warning(
                            f"AttributeError when creating ThinkingConfig/GenerateContentConfig for Gemini model {config_id}: {e}. "
                            f"This suggests 'types.ThinkingConfig' or 'types.GenerateContentConfig' is not available. "
                            f"The 'thinking_budget' will be ignored. Falling back to plain dict for config."
                        )
                        final_api_call_config = gen_config_args
                    except Exception as e_other_config:
                        logger.warning(
                            f"Unexpected error ({type(e_other_config).__name__}) when setting up GenerateContentConfig with ThinkingConfig for {config_id}: {e_other_config}. "
                            f"Falling back to plain dict for config."
                        )
                        final_api_call_config = gen_config_args
                else:
                    final_api_call_config = gen_config_args if gen_config_args else None 
                    logger.info(f"Gemini model {config_id}: No 'thinking_budget' provided. Config: {final_api_call_config}")

                logger.debug(f"Gemini prompt for {config_id}:\\n{prompt[:200]}...")
                logger.info(f"Gemini API call config for {config_id}: {final_api_call_config}")
                
                try:
                    api_response = gemini_client.models.generate_content(
                        model=model_id, 
                        contents=[prompt],
                        config=final_api_call_config
                    )
                    logger.debug(f"Gemini raw api_response object for {config_id}: {api_response}")
                    
                    if hasattr(api_response, 'text'):
                        raw_text_from_api = api_response.text 
                        if raw_text_from_api is not None:
                            response_text_for_error = raw_text_from_api.strip()
                            logger.info(f"Extracted text for {config_id} via api_response.text. Length: {len(response_text_for_error)}. Snippet: '{response_text_for_error[:100]}...'")
                        else:
                            response_text_for_error = ""
                            logger.warning(f"api_response.text is None for {config_id}. Will check for blocking. Defaulting response text to empty string.")
                    else:
                        response_text_for_error = ""
                        logger.warning(f"api_response.text attribute missing for {config_id}. Defaulting response text to empty string.")

                    if hasattr(api_response, 'prompt_feedback') and api_response.prompt_feedback:
                        block_reason = getattr(api_response.prompt_feedback, 'block_reason', None)
                        if block_reason:
                            safety_ratings_details_list = getattr(api_response.prompt_feedback, 'safety_ratings', [])
                            safety_ratings_str = "; ".join([str(rating) for rating in safety_ratings_details_list])
                            
                            error_message_on_block = f"Gemini content blocked: {block_reason}. Details: {safety_ratings_str}"
                            logger.error(f"Error for {config_id}: {error_message_on_block}")
                            return {"error_message": error_message_on_block,
                                    "response_time": time.time() - start_time,
                                    "details": {"type": "ContentBlocked", "reason": str(block_reason), "safety_ratings": safety_ratings_str},
                                    "raw_response_text": response_text_for_error 
                                   }
                    
                    if not response_text_for_error:
                         finish_reason_from_candidate = "Unknown"
                         if hasattr(api_response, 'candidates') and api_response.candidates and \
                            len(api_response.candidates) > 0 and hasattr(api_response.candidates[0], 'finish_reason'):
                             finish_reason_from_candidate = str(api_response.candidates[0].finish_reason)
                         logger.warning(f"Gemini response text is empty for {config_id}. Finish reason (from candidate, if available): {finish_reason_from_candidate}. This is expected if MAX_TOKENS is hit before output, or model chose to output nothing.")

                    if hasattr(api_response, 'usage_metadata') and api_response.usage_metadata:
                        usage_meta = api_response.usage_metadata
                        input_tokens = getattr(usage_meta, 'prompt_token_count', None)
                        
                        thoughts_tokens = getattr(usage_meta, 'thoughts_token_count', 0) or 0
                        candidates_tokens = getattr(usage_meta, 'candidates_token_count', 0) or 0
                        
                        output_tokens = thoughts_tokens + candidates_tokens

                        if thoughts_tokens > 0:
                            logger.info(f"Gemini usage for {config_id}: Input={input_tokens}, Candidates={candidates_tokens}, Thoughts={thoughts_tokens}, Calculated Output (Candidates+Thoughts)={output_tokens}")
                        else:
                            logger.info(f"Gemini usage for {config_id}: Input={input_tokens}, Candidates={candidates_tokens} (No thoughts tokens reported), Calculated Output={output_tokens}")

                        if output_tokens == 0 and input_tokens is not None and hasattr(usage_meta, 'total_token_count') and usage_meta.total_token_count is not None:
                            inferred_output_tokens = usage_meta.total_token_count - input_tokens
                            if inferred_output_tokens >= 0:
                                output_tokens = inferred_output_tokens
                                logger.warning(f"Gemini for {config_id}: Output tokens (candidates + thoughts) inferred from total_token_count as {output_tokens} because direct candidate/thoughts tokens were zero/missing.")
                            else:
                                logger.warning(f"Gemini for {config_id}: Cannot infer output tokens as total_token_count ({usage_meta.total_token_count}) < prompt_token_count ({input_tokens}).")
                        elif output_tokens == 0 and (thoughts_tokens == 0 and candidates_tokens == 0):
                             logger.warning(f"Gemini for {config_id}: Calculated output_tokens is 0. Direct thoughts_tokens and candidates_tokens were also 0. Total_token_count fallback not applicable or also resulted in 0.")

                    else: 
                        logger.warning(f"Gemini token count not found via usage_metadata for {config_id}. Setting input/output tokens to None.")
                        input_tokens = None
                        output_tokens = None

                except Exception as e:
                    elapsed_time = time.time() - start_time
                    current_response_text_snippet = response_text_for_error[:100] if 'response_text_for_error' in locals() and response_text_for_error else "N/A"
                    logger.error(f"Gemini API Error for {config_id} after {elapsed_time:.2f}s: {e}. Raw response snippet: {current_response_text_snippet}...", exc_info=True)
                    return {"error_message": str(e), "response_time": elapsed_time, "raw_response_text": response_text_for_error if 'response_text_for_error' in locals() else "Error before text extraction", "details": {"type": type(e).__name__, "unexpected": True}}

            elif model_type == "writer":
                if not config.WRITER_API_KEY:
                    logger.error(f"Missing Writer API key for model {config_id}.")
                    return {"error_message": "Missing Writer API key", "response_time": 0}
                writer_client = Writer(api_key=config.WRITER_API_KEY)
                writer_params = parameters.copy()
                writer_params.pop('response_format', None) 

                try:
                    api_response = writer_client.chat.chat(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_id,
                        **writer_params
                    )
                    
                    response_text_for_error = ""
                    finish_reason = None
                    input_tokens = None
                    output_tokens = None

                    if api_response.choices and api_response.choices[0].message:
                        response_text_for_error = api_response.choices[0].message.content.strip()
                        if hasattr(api_response.choices[0], 'finish_reason') and api_response.choices[0].finish_reason:
                            finish_reason = api_response.choices[0].finish_reason
                            logger.info(f"Writer API finish_reason for {config_id}: {finish_reason}")
                    else:
                        logger.warning(f"Writer API response for {config_id} did not contain expected choices/message structure.")

                    if hasattr(api_response, 'usage') and api_response.usage is not None:
                        prompt_tokens_from_api = getattr(api_response.usage, 'prompt_tokens', None)
                        completion_tokens_from_api = getattr(api_response.usage, 'completion_tokens', None)
                        
                        if prompt_tokens_from_api is not None and completion_tokens_from_api is not None:
                            input_tokens = prompt_tokens_from_api
                            output_tokens = completion_tokens_from_api
                            logger.info(f"Retrieved token counts from Writer API response for {config_id}: In={input_tokens}, Out={output_tokens}")
                        else:
                            logger.warning(f"Writer API usage object present for {config_id}, but prompt_tokens ({prompt_tokens_from_api}) or completion_tokens ({completion_tokens_from_api}) is None or missing. Will attempt estimation if needed.")
                    else:
                        logger.warning(f"Writer API response for {config_id} did not contain 'usage' object or it was None. Will attempt estimation if needed.")
                    
                    if input_tokens is None or output_tokens is None:
                        logger.warning(f"Writer token count not fully available from API for {config_id} (API In: {input_tokens}, API Out: {output_tokens}). Attempting local estimation for missing values.")
                        
                        if input_tokens is None:
                            estimated_input = _estimate_tokens_tiktoken(prompt)
                            if estimated_input is not None:
                                input_tokens = estimated_input
                                logger.info(f"Estimated input tokens for {config_id} using tiktoken: {input_tokens}")
                            else:
                                logger.warning(f"Failed to estimate input tokens for {config_id}.")
                        
                        if output_tokens is None:
                            if response_text_for_error:
                                estimated_output = _estimate_tokens_tiktoken(response_text_for_error)
                                if estimated_output is not None:
                                    output_tokens = estimated_output
                                    logger.info(f"Estimated output tokens for {config_id} using tiktoken: {output_tokens}")
                                else:
                                    logger.warning(f"Failed to estimate output tokens for {config_id}.")
                            else: 
                                output_tokens = 0 
                                logger.info(f"No response text to estimate output tokens for {config_id}; setting output_tokens to 0 as it was not provided by API.")
                    
                    if finish_reason == "length":
                        current_elapsed_time = time.time() - start_time
                        error_explanation = f"Writer API: Output truncated due to length limit (max_tokens: {writer_params.get('max_tokens')}). Finish reason: {finish_reason}."
                        logger.warning(error_explanation + f" ({config_id})")
                        
                        parsed_content_for_error_case = None
                        if is_json_response_expected:
                            try:
                                if response_text_for_error:
                                    json_match = re.search(r"```json\\s*([\\s\\S]*?)\\s*```|({.*?})|(\\[.*?\\])", response_text_for_error, re.DOTALL)
                                    if json_match:
                                        json_str = next(g for g in json_match.groups() if g is not None)
                                        parsed_content_for_error_case = json.loads(json_str)
                                    else:
                                        parsed_content_for_error_case = json.loads(response_text_for_error)
                                    logger.info(f"Successfully parsed (truncated by length) JSON from Writer response for {config_id}.")
                                if not parsed_content_for_error_case:
                                    raise json.JSONDecodeError("No JSON content found or empty after truncation", response_text_for_error or "", 0)
                            except json.JSONDecodeError:
                                logger.warning(f"Could not parse truncated (by length) Writer response as JSON for {config_id}. Defaulting to error structure.")
                                parsed_content_for_error_case = {"answer": "X", "explanation": error_explanation, "error_message": error_explanation}
                        else:
                            parsed_content_for_error_case = {"answer": "X", "explanation": error_explanation}

                        return {
                            "response_json": parsed_content_for_error_case if is_json_response_expected else None,
                            "response_content": parsed_content_for_error_case,
                            "raw_response_text": response_text_for_error,
                            "response_time": current_elapsed_time,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "error_message": error_explanation,
                            "details": {"type": "TruncatedResponse", "finish_reason": finish_reason}
                        }
                except writerai.APIError as e_writer_api:
                    current_elapsed_time = time.time() - start_time
                    logger.error(f"Writer API Error for {config_id} after {current_elapsed_time:.2f}s: {e_writer_api}. Raw response snippet: {response_text_for_error[:100]}...")
                    error_details = {"type": "WriterAPIError", "message": str(e_writer_api)}
                    if hasattr(e_writer_api, 'status_code'): 
                        error_details["status_code"] = e_writer_api.status_code
                    if hasattr(e_writer_api, 'body') and e_writer_api.body: 
                         error_details["body"] = str(e_writer_api.body)[:500]
                    return {
                        "error_message": f"Writer API Error: {str(e_writer_api)}", 
                        "response_time": current_elapsed_time, 
                        "details": error_details, 
                        "raw_response_text": response_text_for_error
                    }
                
            elif model_type == "groq":
                if not config.GROQ_API_KEY:
                    logger.error(f"Missing Groq API key for model {config_id}.")
                    return {"error_message": "Missing Groq API key", "response_time": 0}
                groq_client = OpenAI(
                    api_key=config.GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1",
                )
                groq_params = parameters.copy()
                if is_json_response_expected and groq_params.get("response_format", {}).get("type") == "json_object":
                    if "json" not in prompt.lower():
                        logger.warning(f"Groq model {config_id} called with response_format=json_object, but 'json' not found in prompt.")
                else:
                     groq_params.pop('response_format', None)

                api_response = groq_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    **groq_params
                )
                response_text_for_error = api_response.choices[0].message.content.strip() if api_response.choices and api_response.choices[0].message else ""
                if hasattr(api_response, 'usage') and api_response.usage:
                    input_tokens = api_response.usage.prompt_tokens
                    output_tokens = api_response.usage.completion_tokens
                else:
                    logger.warning(f"Groq token count not found in usage object for {config_id}. Setting to None.")

            elif model_type == "mistral_official":
                if not config.MISTRAL_API_KEY:
                    logger.error(f"Missing Mistral API key for model {config_id}.")
                    return {"error_message": "Missing Mistral API key", "response_time": 0}
                mistral_client = MistralClient(api_key=config.MISTRAL_API_KEY)
                
                mistral_params = parameters.copy()
                
                mistral_params.pop('response_format', None)

                api_response = mistral_client.chat(
                    model=model_id,
                    messages=[anthropic.types.MessageParam(role="user", content=prompt)], 
                    **mistral_params
                )
                response_text_for_error = api_response.choices[0].message.content.strip() if api_response.choices and api_response.choices[0].message else ""
                if hasattr(api_response, 'usage') and api_response.usage:
                    input_tokens = api_response.usage.prompt_tokens
                    output_tokens = api_response.usage.completion_tokens
                else:
                    logger.warning(f"Mistral token count not found in usage object for {config_id}. Estimating.")
                    input_tokens = _estimate_tokens_tiktoken(prompt) 
                    output_tokens = _estimate_tokens_tiktoken(response_text_for_error)

            elif model_type == "anthropic":
                if not config.ANTHROPIC_API_KEY:
                    logger.error(f"Missing Anthropic API key for model {config_id}.")
                    return {"error_message": "Missing Anthropic API key", "response_time": 0}
                
                api_key_name = "ANTHROPIC_API_KEY"
                anthropic_api_key = os.getenv(api_key_name)
                if not anthropic_api_key:
                    raise ValueError(f"{api_key_name} not found in environment variables.")
                
                anthropic_client = anthropic.Anthropic(
                    api_key=anthropic_api_key,
                    timeout=httpx.Timeout(timeout=60.0 * 60.0) 
                )
                
                anthropic_params = parameters.copy()
                anthropic_params.pop('response_format', None) 
                
                if 'max_tokens' not in anthropic_params:
                    anthropic_params['max_tokens'] = 4096 
                    logger.warning(f"'max_tokens' not specified for Anthropic model {config_id}, defaulting to {anthropic_params['max_tokens']}.")

                accumulated_text = ""
                current_input_tokens = None
                current_output_tokens = None

                try:
                    with anthropic_client.messages.stream(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        **anthropic_params
                    ) as stream:
                        for event in stream:
                            if event.type == "message_start":
                                if event.message.usage and hasattr(event.message.usage, 'input_tokens'):
                                    current_input_tokens = event.message.usage.input_tokens
                                    logger.info(f"Anthropic stream: input_tokens from message_start: {current_input_tokens}")
                            elif event.type == "content_block_delta":
                                if event.delta.type == "text_delta":
                                    accumulated_text += event.delta.text
                            elif event.type == "message_delta":
                                if event.usage and hasattr(event.usage, 'output_tokens'):
                                    current_output_tokens = event.usage.output_tokens 
                                    logger.debug(f"Anthropic stream: cumulative output_tokens from message_delta: {current_output_tokens}")
                            elif event.type == "message_stop":
                                logger.info(f"Anthropic stream: message_stop event received.")
                                final_message_obj_on_stop = stream.get_final_message()
                                if final_message_obj_on_stop and final_message_obj_on_stop.usage:
                                    if current_input_tokens is None and hasattr(final_message_obj_on_stop.usage, 'input_tokens'):
                                        current_input_tokens = final_message_obj_on_stop.usage.input_tokens
                                        logger.info(f"Anthropic stream: input_tokens from get_final_message (on stop): {current_input_tokens}")
                                    if hasattr(final_message_obj_on_stop.usage, 'output_tokens'):
                                        current_output_tokens = final_message_obj_on_stop.usage.output_tokens
                                        logger.info(f"Anthropic stream: output_tokens from get_final_message (on stop): {current_output_tokens}")
                                elif current_input_tokens is None or current_output_tokens is None:
                                     logger.warning(f"Anthropic stream for {config_id}: Could not definitively get all token counts from message_stop/get_final_message. Input: {current_input_tokens}, Output: {current_output_tokens}")

                            elif event.type == "error":
                                logger.error(f"Anthropic stream error for {config_id}: {event.error}")
                                
                                error_details = {"type": event.error.get("type", "stream_error"), "message": event.error.get("message", "Unknown stream error")}
                                return {
                                    "error_message": f"Anthropic Stream Error: {event.error.get('type', 'unknown')}",
                                    "response_time": time.time() - start_time,
                                    "details": error_details,
                                    "raw_response_text": accumulated_text,
                                    "input_tokens": current_input_tokens,
                                    "output_tokens": current_output_tokens
                                }
                    
                    response_text_for_error = accumulated_text.strip()
                    
                    if current_input_tokens is None or current_output_tokens is None:
                        logger.warning(f"Anthropic stream for {config_id}: One or both token counts still None after stream. Input: {current_input_tokens}, Output: {current_output_tokens}. Consider if stream completed fully.")

                    logger.info(f"Anthropic stream completed for {config_id}. Final Input: {current_input_tokens}, Final Output: {current_output_tokens}. Response length: {len(response_text_for_error)}")

                except anthropic.APIStatusError as e:
                    elapsed_time = time.time() - start_time
                    logger.error(f"Anthropic API Status Error for {config_id} after {elapsed_time:.2f}s: {e.status_code} - {e.message}. Raw response snippet: {response_text_for_error[:100]}...")
                    error_details = {"type": "APIStatusError", "message": str(e.message), "status_code": e.status_code}
                    if hasattr(e, 'response') and e.response and hasattr(e.response, 'text'):
                         error_details["body"] = e.response.text[:500] 
                    return {"error_message": f"Anthropic API Status Error: {e.status_code} - {e.message}", "response_time": elapsed_time, "details": error_details, "raw_response_text": response_text_for_error}
                except anthropic.APIConnectionError as e:
                    elapsed_time = time.time() - start_time
                    logger.error(f"Anthropic API Connection Error for {config_id} after {elapsed_time:.2f}s: {e}. Raw response snippet: {response_text_for_error[:100]}...")
                    return {"error_message": f"Anthropic API Connection Error: {str(e)}", "response_time": elapsed_time, "details": {"type": "APIConnectionError"}, "raw_response_text": response_text_for_error}
                except anthropic.RateLimitError as e:
                    elapsed_time = time.time() - start_time
                    logger.error(f"Anthropic Rate Limit Error for {config_id} after {elapsed_time:.2f}s: {e}. Raw response snippet: {response_text_for_error[:100]}...")
                    return {"error_message": f"Anthropic Rate Limit Error: {str(e)}", "response_time": elapsed_time, "details": {"type": "RateLimitError"}, "raw_response_text": response_text_for_error}
                except anthropic.APIError as e: 
                    elapsed_time = time.time() - start_time
                    logger.error(f"Anthropic API Error for {config_id} after {elapsed_time:.2f}s: {e}. Raw response snippet: {response_text_for_error[:100]}...")
                    error_details = {"type": "APIError", "message": str(e)}
                    if hasattr(e, 'status_code'): 
                        error_details["status_code"] = e.status_code
                    if hasattr(e, 'body') and e.body: 
                         error_details["body"] = str(e.body)[:500]
                    return {"error_message": f"Anthropic API Error: {str(e)}", "response_time": elapsed_time, "details": error_details, "raw_response_text": response_text_for_error}

            else:
                logger.error(f"Unsupported model type: {model_type} for {config_id}")
                return {"error_message": f"Unsupported model type: {model_type}", "response_time": 0}

            elapsed_time = time.time() - start_time
            logger.info(f"Response from {config_id} received in {elapsed_time:.2f}s. Raw: '{response_text_for_error[:100]}...'")
            break 

        except RETRYABLE_EXCEPTIONS as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {config_id} ({type(e).__name__}). Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                elapsed_time = time.time() - start_time
                logger.error(f"API call to {config_id} failed after {MAX_RETRIES} attempts and {elapsed_time:.2f}s: {e}. Raw response snippet: {response_text_for_error[:100]}...", exc_info=True)
                return {"error_message": str(e), "response_time": elapsed_time, "raw_response_text": response_text_for_error, "details": {"type": type(e).__name__}}
        except Exception as e: 
            elapsed_time = time.time() - start_time
            logger.error(f"API call to {config_id} failed with unexpected error after {elapsed_time:.2f}s: {e}. Raw response snippet: {response_text_for_error[:100]}...", exc_info=True)
            return {"error_message": str(e), "response_time": elapsed_time, "raw_response_text": response_text_for_error, "details": {"type": type(e).__name__, "unexpected": True}}

    parsed_content_for_return = None

    if is_json_response_expected:
        try:
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```|({.*?})|(\[.*?\])", response_text_for_error, re.DOTALL)
            if json_match:
                json_str = next(g for g in json_match.groups() if g is not None)
                parsed_content_for_return = json.loads(json_str)
                logger.info(f"Successfully parsed JSON from response for {config_id}.")
            else:
                parsed_content_for_return = json.loads(response_text_for_error)
                logger.info(f"Successfully parsed entire response as JSON for {config_id}.")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response for {config_id}. Raw: '{response_text_for_error[:200]}...'")
            return {
                "response_json": {"answer": "X", "explanation": "JSON parsing failed"},
                "response_content": "X",
                "error_message": "JSON parsing failed",
                "raw_response_text": response_text_for_error,
                "response_time": elapsed_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
    else:
        cleaned_answer = "X"
        if response_text_for_error:
            search_text = response_text_for_error[-100:]
            logger.info(f"Extracting answer SOLELY from the last 100 characters of response for {config_id}. Snippet: '{search_text}'")

            final_answer_match = re.search(r"Final Answer:\s*([A-D])", search_text, re.IGNORECASE | re.MULTILINE)
            if final_answer_match:
                cleaned_answer = final_answer_match.group(1).upper()
                logger.info(f"Extracted final answer '{cleaned_answer}' for {config_id} from explicit 'Final Answer: [A-D]' in trailing 100 chars.")
            else:
                escaped_punctuation = re.escape(string.punctuation)
                match = re.search(
                    rf"(?:^|\s|[{escaped_punctuation}])([A-D])(?:$|\s|[{escaped_punctuation}])",
                    search_text,
                    re.MULTILINE
                )
                if match:
                    cleaned_answer = match.group(1).upper()
                    logger.info(f"Extracted single letter answer '{cleaned_answer}' for {config_id} from delimited pattern in trailing 100 chars.")
                else:
                    fallback_match = re.search(r"([A-D])", search_text)
                    if fallback_match:
                        cleaned_answer = fallback_match.group(1).upper()
                        logger.warning(f"Used fallback regex to extract single letter answer '{cleaned_answer}' for {config_id} from any occurrence in trailing 100 chars.")
                    else:
                        logger.error(f"Could not extract single letter answer for {config_id} from trailing 100 chars: '{search_text}'. Marking as X.")
        else:
            logger.error(f"Empty or no usable response text received for {config_id}. Marking as X.")
        
        parsed_content_for_return = {"answer": cleaned_answer, "explanation": ""}

    final_result = {
        
        "response_json": parsed_content_for_return if is_json_response_expected else None,
        
        "response_content": parsed_content_for_return, 
        "raw_response_text": response_text_for_error,
        "response_time": elapsed_time,
        "input_tokens": current_input_tokens if model_type == "anthropic" else input_tokens,
        "output_tokens": current_output_tokens if model_type == "anthropic" else output_tokens
    }
    if 'error_message' in (parsed_content_for_return or {}):
        final_result['error_message'] = parsed_content_for_return['error_message']

    return final_result