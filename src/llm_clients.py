"""
Functions for interacting with various Language Model APIs.
"""
import time
import json
import logging
import re
import boto3
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
import google.genai as genai
from google.genai import types
from writerai import Writer
from groq import Groq
import anthropic
from mistralai import Mistral
import writerai
import tiktoken

from . import config

logger = logging.getLogger(__name__)

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

def _get_tokens_from_headers(headers: dict, input_key: str, output_key: str) -> tuple[int | None, int | None]:
    """Safely extracts token counts from response headers."""
    input_val_str = headers.get(input_key)
    output_val_str = headers.get(output_key)
    
    input_t = None
    if input_val_str and input_val_str.isdigit():
        input_t = int(input_val_str)
        
    output_t = None
    if output_val_str and output_val_str.isdigit():
        output_t = int(output_val_str)
    if input_t is not None and output_t is not None:
        return input_t, output_t
    return None, None

def get_llm_response(prompt: str, model_config: dict, is_json_response_expected: bool = False) -> dict | None:
    """
    Sends a prompt to the specified LLM API and parses the response.
    Measures response time and attempts to extract token counts.

    Args:
        prompt: The prompt string to send to the LLM.
        model_config: Dictionary containing model type, ID, parameters.
        is_json_response_expected: If True, attempts to parse the entire response as JSON.
                                   If False (default), returns the processed full text response.

    Returns:
        A dictionary containing {'response_content': <parsed_response>,
                                'raw_response_text': <full_raw_response_text>,
                                'response_time': <response_time_seconds>,
                                'input_tokens': <input_token_count_or_None>,
                                'output_tokens': <output_token_count_or_None>},
        or None if the API call failed before a response structure could be formed (e.g., missing API key).
        'response_content':
            - If 'is_json_response_expected' is True and JSON parsing succeeds: a dictionary.
            - If 'is_json_response_expected' is True and JSON parsing fails: the string "X".
            - If 'is_json_response_expected' is False: a string containing the processed full text.
        An 'error_message' key may be present if issues occurred during the call or parsing.
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

    try:
        if model_type == "bedrock":
            if not config.AWS_ACCESS_KEY_ID or not config.AWS_SECRET_ACCESS_KEY:
                logger.error(f"Missing AWS credentials for Bedrock model {config_id}.")
                return {"error_message": "Missing AWS credentials", "response_time": 0}
            bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=config.AWS_REGION,
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
            )
            accept = 'application/json'
            contentType = 'application/json'

            if "anthropic" in model_id:
                messages = [{"role": "user", "content": prompt}]
                body_params = {
                    "messages": messages,
                    "anthropic_version": parameters.get("anthropic_version", "bedrock-2023-05-31"),
                    **{k: v for k, v in parameters.items() if k not in ["anthropic_version"]}
                }
                body_params.pop("response_format", None)
                body = json.dumps(body_params)

                model_identifier = model_config.get("inference_profile_arn") if model_config.get("use_inference_profile", False) else model_id
                api_response = bedrock_client.invoke_model(
                    body=body, modelId=model_identifier, accept=accept, contentType=contentType
                )
                response_body = json.loads(api_response.get('body').read())
                response_text_for_error = response_body.get('content', [{}])[0].get('text', '')
                api_headers = api_response.get('ResponseMetadata', {}).get('HTTPHeaders', {})
                input_tokens, output_tokens = _get_tokens_from_headers(
                    api_headers,
                    'x-amzn-bedrock-input-token-count',
                    'x-amzn-bedrock-output-token-count'
                )
                if input_tokens is None or output_tokens is None:
                    logger.debug(f"Token counts not found/incomplete in headers for Bedrock Anthropic {config_id}. Trying response body.")
                    usage_from_body = response_body.get('usage', {})
                    input_tokens = usage_from_body.get('input_tokens')
                    output_tokens = usage_from_body.get('output_tokens')
                    if input_tokens is None or output_tokens is None:
                        logger.warning(f"Token counts not found in body for Bedrock Anthropic {config_id}. Setting to None.")
                else:
                    logger.debug(f"Retrieved token counts from headers for Bedrock Anthropic {config_id}.")

            elif "mistral" in model_id or "meta" in model_id:
                bedrock_params = parameters.copy()
                bedrock_params.pop("response_format", None)

                if "meta" in model_id:  
                    if 'max_tokens' in bedrock_params:
                        max_tokens_value = bedrock_params.pop('max_tokens')
                        if max_tokens_value:
                            bedrock_params['max_gen_len'] = max_tokens_value
                
                body = json.dumps({"prompt": prompt, **bedrock_params})
                api_response = bedrock_client.invoke_model(
                    body=body, modelId=model_id, accept=accept, contentType=contentType
                )
                response_body = json.loads(api_response.get('body').read())
                api_headers = api_response.get('ResponseMetadata', {}).get('HTTPHeaders', {})

                if "mistral" in model_id:
                    response_text_for_error = response_body.get('outputs', [{}])[0].get('text', '')
                    input_tokens, output_tokens = _get_tokens_from_headers(
                        api_headers,
                        'x-amzn-bedrock-input-token-count',
                        'x-amzn-bedrock-output-token-count' 
                    )
                    if input_tokens is None or output_tokens is None:
                        logger.debug(f"Token counts not found/incomplete in headers for Bedrock Mistral {config_id}. Trying response body.")
                        usage_from_body = response_body.get('usage', {})
                        input_tokens = usage_from_body.get('prompt_token_count') 
                        output_tokens = usage_from_body.get('completion_token_count') 
                        if input_tokens is None or output_tokens is None: 
                            input_tokens = usage_from_body.get('input_tokens')
                            output_tokens = usage_from_body.get('output_tokens')
                        if input_tokens is None or output_tokens is None:
                             logger.warning(f"Token counts not found in body for Bedrock Mistral {config_id}. Setting to None.")
                    else:
                        logger.debug(f"Retrieved token counts from headers for Bedrock Mistral {config_id}.")

                elif "meta" in model_id:
                    response_text_for_error = response_body.get('generation', '')
                    input_tokens, output_tokens = _get_tokens_from_headers(
                        api_headers,
                        'x-amzn-bedrock-input-token-count',
                        'x-amzn-bedrock-output-token-count'
                    )
                    if input_tokens is None or output_tokens is None:
                        logger.debug(f"Token counts not found/incomplete in headers for Bedrock Meta {config_id}. Trying response body.")
                        input_tokens = response_body.get('prompt_token_count')
                        output_tokens = response_body.get('generation_token_count') 
                        if input_tokens is None or output_tokens is None: 
                            usage_from_body = response_body.get('usage', {})
                            input_tokens = usage_from_body.get('input_tokens')
                            output_tokens = usage_from_body.get('output_tokens')
                        if input_tokens is None or output_tokens is None:
                            logger.warning(f"Token counts not found in body for Bedrock Meta {config_id}. Setting to None.")
                    else:
                        logger.debug(f"Retrieved token counts from headers for Bedrock Meta {config_id}.")
            
            else:
                logger.error(f"Unsupported Bedrock model provider for {config_id}")
                return {"error_message": f"Unsupported Bedrock model: {config_id}", "response_time": 0}

        elif model_type == "anthropic": 
            if not config.ANTHROPIC_API_KEY:
                logger.error(f"Missing Anthropic API key for model {config_id}.")
                return {"error_message": "Missing Anthropic API key", "response_time": 0}
            
            anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            anthropic_params = parameters.copy()
            anthropic_params.pop('response_format', None) 
            
            if 'max_tokens' not in anthropic_params:
                anthropic_params['max_tokens'] = 4096 
                logger.warning(f"'max_tokens' not specified for Anthropic model {config_id}, defaulting to {anthropic_params['max_tokens']}.")
            if 'timeout' not in anthropic_params:
                anthropic_params['timeout'] = 1800.0 
                logger.info(f"Setting explicit timeout of {anthropic_params['timeout']}s for Anthropic call {config_id} as no timeout was specified in config and using non-streaming.")
            else:
                logger.info(f"Using timeout {anthropic_params['timeout']}s from config for Anthropic call {config_id}.")

            try:
                api_response = anthropic_client.messages.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    **anthropic_params
                )
                response_text_for_error = api_response.content[0].text if api_response.content else ""
                
                if api_response.usage:
                    input_tokens = api_response.usage.input_tokens
                    output_tokens = api_response.usage.output_tokens
                else:
                    logger.warning(f"Token usage data not found in Anthropic response for {config_id}. Setting to None.")

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

        elif model_type == "mistral_official": 
            if not config.MISTRAL_API_KEY:
                logger.error(f"Missing Mistral API key for model {config_id}.")
                return {"error_message": "Missing Mistral API key", "response_time": 0}
            
            mistral_client = Mistral(api_key=config.MISTRAL_API_KEY)
            
            mistral_params = parameters.copy()
            mistral_params.pop('response_format', None) 

            try:
                chat_response = mistral_client.chat.complete(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    **mistral_params
                )
                response_text_for_error = chat_response.choices[0].message.content if chat_response.choices else ""
                
                if chat_response.usage:
                    input_tokens = chat_response.usage.prompt_tokens
                    output_tokens = chat_response.usage.completion_tokens
                else:
                    logger.warning(f"Token usage data not found in Mistral response for {config_id}. Setting to None.")

            except Exception as e: 
                elapsed_time = time.time() - start_time
                logger.error(f"Mistral API Error for {config_id} after {elapsed_time:.2f}s: {e}. Raw response snippet: {response_text_for_error[:100]}...")
                error_details = {"type": "MistralAPIError", "message": str(e)}
                if isinstance(e, NotImplementedError) and "This client is deprecated" in str(e):
                    error_details["message"] = f"Mistral client version is deprecated. {str(e)}"
                return {"error_message": f"Mistral API Error: {error_details['message']}", "response_time": elapsed_time, "details": error_details, "raw_response_text": response_text_for_error}

        elif model_type == "openai":
            if not config.OPENAI_API_KEY:
                logger.error(f"Missing OpenAI API key for model {config_id}.")
                return {"error_message": "Missing OpenAI API key", "response_time": 0}
            openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            
            openai_params = parameters.copy()
            if is_json_response_expected and openai_params.get("response_format", {}).get("type") == "json_object":
                 pass
            else:
                openai_params.pop('response_format', None)

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

        elif model_type == "gemini":
            if not config.GEMINI_API_KEY:
                logger.error(f"Missing Gemini API key for model {config_id}.")
                return {"error_message": "Missing Gemini API key", "response_time": 0}
            
            gemini_params_from_config = parameters.copy()
            effective_model_id = model_id if model_id.startswith("models/") else f"models/{model_id}"
            logger.info(f"Using effective model_id for Gemini: {effective_model_id}")

            try:
                
                logger.info(f"Attempting Gemini API call for {config_id} using genai.Client() pattern.")
                gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)

                gen_content_config_args = {
                    k: v for k, v in gemini_params_from_config.items()
                    if k in ["temperature", "top_p", "top_k", "max_output_tokens", "candidate_count", "stop_sequences", "system_instruction"] and v is not None
                }
        
                if "safety_settings" in gemini_params_from_config and gemini_params_from_config["safety_settings"] is not None:
                    logger.info(f"Gemini model {config_id} (Client API): safety_settings found in params. Note: Passing them via GenerateContentConfig is SDK version dependent.")
                thinking_budget_value = gemini_params_from_config.get("thinking_budget")
                final_generate_content_config_obj = None

                current_gen_content_config_dict_for_builder = gen_content_config_args.copy()

                if thinking_budget_value is not None:
                    if "flash" in model_id.lower(): 
                        try:
                            thinking_config_obj = types.ThinkingConfig(thinking_budget=thinking_budget_value)
                            current_gen_content_config_dict_for_builder['thinking_config'] = thinking_config_obj
                            logger.info(f"Gemini model {config_id} (Client API): Preparing GenerateContentConfig with ThinkingConfig (budget: {thinking_budget_value}).")
                        except AttributeError:
                            logger.warning(
                                f"Gemini model {config_id} (Client API): types.ThinkingConfig is not available or cannot be assigned to "
                                f"GenerateContentConfig args in the current SDK version. 'thinking_budget' will be ignored for GenerateContentConfig."
                            )
                        except Exception as e_thinking_config:
                            logger.warning(
                                f"Gemini model {config_id} (Client API): Error setting up ThinkingConfig for GenerateContentConfig "
                                f"(type: {type(e_thinking_config).__name__}, msg: {e_thinking_config}). 'thinking_budget' will be ignored."
                            )
                    else:
                        logger.info(f"Gemini model {config_id} (Client API): 'thinking_budget' parameter is present but model_id '{model_id}' does not appear to be a 'flash' model. Ignoring 'thinking_budget'.")
                
                if current_gen_content_config_dict_for_builder:
                    try:
                        final_generate_content_config_obj = types.GenerateContentConfig(**current_gen_content_config_dict_for_builder)
                        logger.info(f"Gemini model {config_id} (Client API): Using GenerateContentConfig with args: {current_gen_content_config_dict_for_builder}")
                    except TypeError as te_gen_content:
                        logger.warning(
                            f"Gemini model {config_id} (Client API): TypeError creating GenerateContentConfig: {te_gen_content}. "
                            f"Attempting without problematic args (e.g., thinking_config if it was the cause)."
                        )
                        if 'thinking_config' in current_gen_content_config_dict_for_builder: 
                            del current_gen_content_config_dict_for_builder['thinking_config']
                        if current_gen_content_config_dict_for_builder:
                            final_generate_content_config_obj = types.GenerateContentConfig(**current_gen_content_config_dict_for_builder)
                            logger.info(f"Gemini model {config_id} (Client API): Using fallback GenerateContentConfig with args: {current_gen_content_config_dict_for_builder}")
                        else:
                            final_generate_content_config_obj = None 
                            logger.info(f"Gemini model {config_id} (Client API): No specific GenerateContentConfig args after fallback. Using SDK defaults.")
                    except Exception as e_gen_content_config:
                        logger.warning(
                           f"Gemini model {config_id} (Client API): Unexpected error creating GenerateContentConfig (type: {type(e_gen_content_config).__name__}, msg: {e_gen_content_config}). "
                           f"SDK will use default GenerateContentConfig."
                       )
                        final_generate_content_config_obj = None
                else:
                    logger.info(f"Gemini model {config_id} (Client API): No specific GenerateContentConfig args. Using SDK defaults.")
                    final_generate_content_config_obj = None

                
                
                
                
                api_response = gemini_client.models.generate_content(
                    model=effective_model_id,
                    contents=[prompt], 
                    config=final_generate_content_config_obj
                    
                    
                    
                )
                logger.info(f"Gemini API call for {config_id} using genai.Client() pattern succeeded.")

            except (AttributeError, TypeError) as e_sdk_pattern:
                logger.warning(f"Gemini API call for {config_id} with genai.Client() pattern failed ({type(e_sdk_pattern).__name__}: {e_sdk_pattern}). Falling back to genai.GenerativeModel() pattern.")
                
                
                
                gemini_model_instance = genai.GenerativeModel(effective_model_id) 

                gen_config_args = {
                    k: v for k, v in gemini_params_from_config.items() 
                    if k in ["temperature", "top_p", "top_k", "max_output_tokens", "candidate_count", "stop_sequences"] and v is not None
                }
                
                safety_settings_fallback = gemini_params_from_config.get("safety_settings")
                tools_fallback = gemini_params_from_config.get("tools")
                tool_config_fallback = gemini_params_from_config.get("tool_config")

                thinking_budget_value = gemini_params_from_config.get("thinking_budget")
                final_generation_config_obj_fallback = None

                current_gen_config_dict_for_builder_fallback = gen_config_args.copy()
                if thinking_budget_value is not None:
                    if "flash" in model_id.lower(): 
                        try:
                            thinking_config_obj_fallback = types.ThinkingConfig(thinking_budget=thinking_budget_value)
                            current_gen_config_dict_for_builder_fallback['thinking_config'] = thinking_config_obj_fallback
                            logger.info(f"Gemini model {config_id} (Fallback): Preparing GenerationConfig with ThinkingConfig (budget: {thinking_budget_value}).")
                        except AttributeError:
                            logger.warning(
                                f"Gemini model {config_id} (Fallback): types.ThinkingConfig is not available or cannot be assigned to GenerationConfig args "
                                f"in the current SDK version for fallback. 'thinking_budget' will be ignored."
                            )
                        except Exception as e_thinking_config_fallback:
                            logger.warning(
                                f"Gemini model {config_id} (Fallback): Error setting up ThinkingConfig (type: {type(e_thinking_config_fallback).__name__}, msg: {e_thinking_config_fallback}). "
                                f"'thinking_budget' will be ignored."
                            )
                    else:
                        logger.info(f"Gemini model {config_id} (Fallback): 'thinking_budget' parameter is present but model_id '{model_id}' does not appear to be a 'flash' model. Ignoring 'thinking_budget'.")
                
                if current_gen_config_dict_for_builder_fallback:
                    try:
                        final_generation_config_obj_fallback = types.GenerationConfig(**current_gen_config_dict_for_builder_fallback)
                        logger.info(f"Gemini model {config_id} (Fallback): Using GenerationConfig with args: {current_gen_config_dict_for_builder_fallback}")
                    except TypeError as te_fallback:
                        logger.warning(
                            f"Gemini model {config_id} (Fallback): TypeError creating GenerationConfig: {te_fallback}. "
                            f"Attempting without problematic args."
                        )
                        if 'thinking_config' in current_gen_config_dict_for_builder_fallback:
                            del current_gen_config_dict_for_builder_fallback['thinking_config']
                        if current_gen_config_dict_for_builder_fallback:
                            final_generation_config_obj_fallback = types.GenerationConfig(**current_gen_config_dict_for_builder_fallback)
                            logger.info(f"Gemini model {config_id} (Fallback): Using fallback GenerationConfig with args: {current_gen_config_dict_for_builder_fallback}")
                        else:
                            final_generation_config_obj_fallback = None
                            logger.info(f"Gemini model {config_id} (Fallback): No specific generation config args after fallback. Using SDK defaults.")
                    except Exception as e_gen_config_fallback:
                        logger.warning(
                           f"Gemini model {config_id} (Fallback): Unexpected error creating GenerationConfig (type: {type(e_gen_config_fallback).__name__}, msg: {e_gen_config_fallback}). "
                           f"SDK will use default GenerationConfig."
                       )
                        final_generation_config_obj_fallback = None
                else:
                    logger.info(f"Gemini model {config_id} (Fallback): No specific generation config args. Using SDK defaults for GenerationConfig.")
                    final_generation_config_obj_fallback = None
                
                api_response = gemini_model_instance.generate_content(
                    contents=[prompt],
                    generation_config=final_generation_config_obj_fallback,
                    safety_settings=safety_settings_fallback,
                    tools=tools_fallback,
                    tool_config=tool_config_fallback
                )
                logger.info(f"Gemini API call for {config_id} using genai.GenerativeModel() fallback succeeded.")
            
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
                                json_match = re.search(r"```json\s*([\s\S]*?)\s*```|({.*?})|(\[.*?\])", response_text_for_error, re.DOTALL)
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

        elif model_type == "sagemaker":
            if not config.AWS_ACCESS_KEY_ID or not config.AWS_SECRET_ACCESS_KEY:
                logger.error(f"Missing AWS credentials for SageMaker model {config_id}.")
                return {"error_message": "Missing AWS credentials", "response_time": 0}
            sagemaker_client = boto3.client(
                service_name='sagemaker-runtime',
                region_name=config.AWS_REGION,
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
            )
            sagemaker_params = parameters.copy()
            sagemaker_params.pop('response_format', None)

            body_payload = {"inputs": prompt, "parameters": sagemaker_params}
            body = json.dumps(body_payload)
            endpoint_name = model_id
            
            api_response = sagemaker_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=body
            )
            response_body_bytes = api_response['Body'].read()
            try:
                response_body_str = response_body_bytes.decode('utf-8')
                response_body = json.loads(response_body_str)
                if isinstance(response_body, list) and len(response_body) > 0 and isinstance(response_body[0], dict):
                    response_text_for_error = response_body[0].get('generated_text', str(response_body[0]))
                elif isinstance(response_body, dict):
                    response_text_for_error = response_body.get('generated_text', 
                                                               response_body.get('text', 
                                                               response_body.get('answer', 
                                                               response_body.get('completion', 
                                                               response_body.get('generation', str(response_body))))))
                else:
                    response_text_for_error = str(response_body)
            except (json.JSONDecodeError, UnicodeDecodeError) as decode_err:
                logger.error(f"SageMaker response for {config_id} was not valid JSON or UTF-8: {decode_err}. Raw bytes: {response_body_bytes[:100]}")
                response_text_for_error = response_body_bytes.decode('latin-1', errors='replace')

            input_tokens = None
            output_tokens = None
            logger.warning(
                f"SageMaker token count is not reliably available. Token counts for {config_id} will be 'None'."
            )
        else:
            logger.error(f"Unsupported model type: {model_type} for model {config_id}")
            return {"error_message": f"Unsupported model type: {model_type}", "response_time": 0}

        elapsed_time = time.time() - start_time
        logger.info(f"Response from {config_id} received in {elapsed_time:.2f}s. Raw: '{response_text_for_error[:100]}...'")

    except APIError as e:
        elapsed_time = time.time() - start_time
        logger.error(f"OpenAI API Error for {config_id} after {elapsed_time:.2f}s: {e}. Raw response snippet: {response_text_for_error[:100]}...")
        error_details = {"type": "APIError", "message": str(e)}
        if hasattr(e, 'status_code'):
            error_details["status_code"] = e.status_code
        if hasattr(e, 'body') and e.body:
            error_details["body"] = e.body
        return {"error_message": f"OpenAI API Error: {str(e)}", "response_time": elapsed_time, "details": error_details, "raw_response_text": response_text_for_error}
    except APIConnectionError as e:
        elapsed_time = time.time() - start_time
        logger.error(f"OpenAI API Connection Error for {config_id} after {elapsed_time:.2f}s: {e}. Raw response snippet: {response_text_for_error[:100]}...")
        return {"error_message": f"OpenAI API Connection Error: {str(e)}", "response_time": elapsed_time, "details": {"type": "APIConnectionError"}, "raw_response_text": response_text_for_error}
    except RateLimitError as e:
        elapsed_time = time.time() - start_time
        logger.error(f"OpenAI Rate Limit Error for {config_id} after {elapsed_time:.2f}s: {e}. Raw response snippet: {response_text_for_error[:100]}...")
        return {"error_message": f"OpenAI Rate Limit Error: {str(e)}", "response_time": elapsed_time, "details": {"type": "RateLimitError"}, "raw_response_text": response_text_for_error}
    except Exception as e:
        elapsed_time = time.time() - start_time
        
        if model_type == "bedrock" and "botocore.exceptions" in str(type(e)):
             logger.error(f"Bedrock Client Error for {config_id} after {elapsed_time:.2f}s: {e}. Raw response snippet: {response_text_for_error[:100]}...", exc_info=True)
             return {"error_message": f"Bedrock Client Error: {str(e)}", "response_time": elapsed_time, "details": {"type": "BedrockClientError", "original_exception": str(type(e))}, "raw_response_text": response_text_for_error}
        
        logger.error(f"API call to {config_id} failed after {elapsed_time:.2f}s: {e}. Raw response snippet: {response_text_for_error[:100]}...", exc_info=True)
        return {"error_message": str(e), "response_time": elapsed_time, "raw_response_text": response_text_for_error}

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
            
            logger.error(f"Failed to parse JSON response for {config_id}. Raw: '{response_text_for_error}'") 
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
        if response_text_for_error and response_text_for_error.strip():
            processed_text = response_text_for_error.strip()
            processed_text = re.sub(r'\s+', ' ', processed_text)
            parsed_content_for_return = processed_text
            logger.info(f"Processed full text response for {config_id}: '{processed_text[:100]}...'")
        else:
            logger.warning(f"Empty or no usable response text received for {config_id}. Setting response_content to empty string.")
            parsed_content_for_return = ""

    final_result = {
        
        "response_json": parsed_content_for_return if is_json_response_expected else None,
        
        "response_content": parsed_content_for_return, 
        "raw_response_text": response_text_for_error,
        "response_time": elapsed_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }
    if 'error_message' in (parsed_content_for_return or {}):
        final_result['error_message'] = parsed_content_for_return['error_message']

    return final_result