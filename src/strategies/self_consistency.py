"""
Self-consistency prompting strategy.
"""
import logging
import random
import re
import sys
import threading
from collections import Counter
import time
from typing import Dict, Any, List, Optional, Callable
import copy

from .. import llm_clients 
from ..utils import ui_utils
from ..utils.prompt_utils import parse_question_data
from ..prompts.cot import ESSAY_COT_PROMPT, format_essay_cot_prompt
from .. import config
from ..utils.text_utils import clean_llm_answer_for_similarity

logger = logging.getLogger(__name__)

def generate_prompt_for_cot_strategy(entry: dict, cot_template: str) -> str:
    """Generates a CoT prompt for a given question entry using the standardized parser."""
    parsed_data = parse_question_data(entry)
    
    return cot_template.format(
        vignette=parsed_data.get('vignette', ''),
        question_stem=parsed_data.get('question_stem', parsed_data.get('question', '')),
        option_a=parsed_data.get('options_dict', {}).get('A', 'Option A not provided'),
        option_b=parsed_data.get('options_dict', {}).get('B', 'Option B not provided'),
        option_c=parsed_data.get('options_dict', {}).get('C', 'Option C not provided')
    )

def run_self_consistency_strategy(
    parsed_questions: List[Dict[str, Any]],
    model_config: Dict[str, Any],
    n_samples: int,
    cot_template: str, 
    format_cot_prompt_func: Callable,
    processing_animation: ui_utils.LoadingAnimation,
    current_model_config_id: str
) -> Dict[str, Any]:
    """
    Runs the self-consistency strategy adapted for essay generation.
    Generates n_samples essays using a CoT prompt and selects the first successful one.
    All samples are stored.

    Args:
        parsed_questions: List of parsed question data.
        model_config: Configuration for the LLM.
        n_samples: Number of essay samples to generate.
        cot_template: The CoT prompt template string (e.g., ESSAY_COT_PROMPT).
        format_cot_prompt_func: Function to format the CoT prompt (e.g., format_essay_cot_prompt).
        processing_animation: Instance of LoadingAnimation for UI updates.
        current_model_config_id: The config_id of the model being processed.

    Returns:
        A dictionary containing results, token counts, time, etc.
    """
    results = []
    total_input_tokens_overall = 0
    total_output_tokens_overall = 0
    total_time_taken_overall = 0
    overall_successful_generations = 0

    num_questions = len(parsed_questions)

    for i, question_data in enumerate(parsed_questions):
        if processing_animation and hasattr(processing_animation, 'message'):
            processing_animation.message = f"Processing with {current_model_config_id} (SC-N{n_samples}): {i+1}/{num_questions} questions"

        try:
            prompt = format_cot_prompt_func(question_data, cot_template)
        except KeyError as e:
            logger.error(f"Missing key for prompt formatting in SC question_data for question {i}: {e}. Data: {question_data}")
            result_item = {
                **question_data,
                "prompt": "ERROR: SC Prompt formatting failed due to missing key.",
                "llm_answer": "ERROR: SC Prompt formatting failed.",
                "raw_llm_answer": "",
                "cleaned_llm_answer": "",
                "all_cot_samples": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "response_time": 0,
                "model_id": model_config.get("model_id"),
                "config_id": model_config.get("config_id"),
                "strategy": f"self_consistency_essay_n{n_samples}",
                "error": f"SC Prompt formatting failed: Missing key {e}"
            }
            results.append(result_item)
            continue

        result_item = {
            **question_data,
            "prompt": prompt,
            "llm_answer": "",
            "raw_llm_answer": "",
            "cleaned_llm_answer": "",
            "all_cot_samples": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "response_time": 0,
            "model_id": model_config.get("model_id"),
            "config_id": model_config.get("config_id"),
            "strategy": f"self_consistency_essay_n{n_samples}",
            "error": None
        }
        
        sample_responses_text = []
        current_question_input_tokens = 0
        current_question_output_tokens = 0
        current_question_response_time = 0
        first_successful_answer_raw = None
        first_successful_answer_cleaned = None
        at_least_one_sample_generated_successfully = False

        for sample_num in range(n_samples):
            logger.info(f"Generating sample {sample_num + 1}/{n_samples} for question {i+1} with {current_model_config_id} (SC)")
            try:
                llm_response_data = llm_clients.get_llm_response(
                    prompt=prompt, 
                    model_config=model_config,
                    is_json_response_expected=False
                )

                if llm_response_data:
                    current_question_response_time += llm_response_data.get("response_time", 0)
                    current_question_input_tokens += llm_response_data.get("input_tokens", 0)
                    current_question_output_tokens += llm_response_data.get("output_tokens", 0)
                    
                    raw_sample_answer_text = llm_response_data.get("response_content", "")
                    sample_responses_text.append(raw_sample_answer_text)

                    if not llm_response_data.get("error_message") and raw_sample_answer_text.strip():
                        if not at_least_one_sample_generated_successfully:
                            first_successful_answer_raw = raw_sample_answer_text
                            first_successful_answer_cleaned = clean_llm_answer_for_similarity(raw_sample_answer_text)
                            at_least_one_sample_generated_successfully = True
                    else:
                        error_msg_sample = llm_response_data.get("error_message", "Empty response or unspecified error in sample")
                        logger.warning(f"Sample {sample_num+1} for question {i+1} (SC) failed or was empty: {error_msg_sample}")
                        if not result_item["error"]:
                            result_item["error"] = f"Error in sample {sample_num+1}: {error_msg_sample}"
                else:
                    logger.error(f"LLM call returned None for sample {sample_num+1}, question {i+1} (SC)")
                    sample_responses_text.append("ERROR: LLM call returned None")
                    if not result_item["error"]:
                        result_item["error"] = f"Error in sample {sample_num+1}: LLM call returned None"
            except Exception as e:
                logger.error(f"Exception during sample {sample_num+1} for question {i+1} (SC): {e}", exc_info=True)
                sample_responses_text.append(f"ERROR: {str(e)}")
                if not result_item["error"]:
                     result_item["error"] = f"Exception in sample {sample_num+1}: {str(e)}"
        
        result_item["all_cot_samples"] = sample_responses_text
        result_item["input_tokens"] = current_question_input_tokens
        result_item["output_tokens"] = current_question_output_tokens
        result_item["response_time"] = current_question_response_time
        total_input_tokens_overall += current_question_input_tokens
        total_output_tokens_overall += current_question_output_tokens
        total_time_taken_overall += current_question_response_time

        if at_least_one_sample_generated_successfully:
            result_item["llm_answer"] = first_successful_answer_cleaned
            result_item["raw_llm_answer"] = first_successful_answer_raw
            result_item["cleaned_llm_answer"] = first_successful_answer_cleaned
            result_item["answer_length"] = len(first_successful_answer_cleaned) if first_successful_answer_cleaned else 0
            result_item["error"] = None
            overall_successful_generations += 1
        elif not result_item["error"]:
            result_item["llm_answer"] = "ERROR: No successful samples generated."
            result_item["raw_llm_answer"] = ""
            result_item["cleaned_llm_answer"] = ""
            result_item["answer_length"] = 0
            result_item["error"] = "No successful samples generated after all attempts."
        
        
        if result_item.get("error") and "answer_length" not in result_item:
            result_item["answer_length"] = 0
            if not result_item["llm_answer"]:
                 result_item["llm_answer"] = f"ERROR: {result_item['error']}"
            if not result_item["raw_llm_answer"]: result_item["raw_llm_answer"] = ""
            if not result_item["cleaned_llm_answer"]: result_item["cleaned_llm_answer"] = ""

        results.append(result_item)

    avg_latency_overall = total_time_taken_overall / num_questions if num_questions > 0 else 0

    return {
        "results": results,
        "total_input_tokens": total_input_tokens_overall,
        "total_output_tokens": total_output_tokens_overall,
        "total_tokens": total_input_tokens_overall + total_output_tokens_overall,
        "total_time_taken_seconds": total_time_taken_overall,
        "average_latency_ms": avg_latency_overall * 1000,
        "successful_generations": overall_successful_generations,
        "total_questions": num_questions
    } 