"""
Default processing strategy: process each question once.
"""
import logging
import sys
import threading
import re 
from .. import llm_clients 
from ..utils import ui_utils
from ..utils.prompt_utils import parse_question_data
from ..prompts.default import BASIC_ESSAY_PROMPT 
from .. import config
import time
from typing import Dict, Any, List
from ..utils.ui_utils import LoadingAnimation
from ..utils.text_utils import clean_llm_answer_for_similarity

logger = logging.getLogger(__name__)

def run_default_strategy(
    parsed_questions: List[Dict[str, Any]], 
    model_config: Dict[str, Any],
    processing_animation: LoadingAnimation,
    current_model_config_id: str
) -> Dict[str, Any]:
    """
    Runs the default essay generation strategy for a given model and list of questions.

    Args:
        parsed_questions: List of parsed question data.
        model_config: Configuration for the LLM.
        processing_animation: Instance of LoadingAnimation for UI updates.
        current_model_config_id: The config_id of the model being processed.

    Returns:
        A dictionary containing results, total tokens, total time, etc.
    """
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_time_taken = 0
    successful_generations = 0
    num_questions = len(parsed_questions)

    for i, question_data in enumerate(parsed_questions):
        if processing_animation and hasattr(processing_animation, 'message'):
            processing_animation.message = f"Processing with {current_model_config_id} (Default Essay): {i+1}/{num_questions} questions"
        try:
            prompt = BASIC_ESSAY_PROMPT.format(**question_data)
        except KeyError as e:
            logger.error(f"Missing key for prompt formatting in question_data for question {i}: {e}. Data: {question_data}")
            result_item = {
                **question_data, 
                "prompt": "ERROR: Prompt formatting failed due to missing key.",
                "llm_answer": "ERROR: Prompt formatting failed.",
                "raw_llm_answer": "",
                "cleaned_llm_answer": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "response_time": 0,
                "model_id": model_config.get("model_id"),
                "config_id": model_config.get("config_id"),
                "strategy": "default_essay",
                "error": f"Prompt formatting failed: Missing key {e}"
            }
            results.append(result_item)
            continue 
    
        result_item = {
            **question_data, 
            "prompt": prompt,
            "llm_answer": "", 
            "raw_llm_answer": "",
            "cleaned_llm_answer": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "response_time": 0,
            "model_id": model_config.get("model_id"),
            "config_id": model_config.get("config_id"),
            "strategy": "default_essay",
            "error": None
        }

        try:
            llm_response_data = llm_clients.get_llm_response(
                prompt=prompt, 
                model_config=model_config,
                is_json_response_expected=False 
            )

            if llm_response_data and not llm_response_data.get("error_message"):
                raw_answer = llm_response_data.get("response_content", "")
                cleaned_answer = clean_llm_answer_for_similarity(raw_answer)
                
                result_item.update({
                    "llm_answer": cleaned_answer, 
                    "raw_llm_answer": raw_answer,
                    "cleaned_llm_answer": cleaned_answer,
                    "input_tokens": llm_response_data.get("input_tokens"),
                    "output_tokens": llm_response_data.get("output_tokens"),
                    "response_time": llm_response_data.get("response_time"),
                    "answer_length": len(cleaned_answer) 
                })
                successful_generations += 1
                if llm_response_data.get("input_tokens") is not None: total_input_tokens += llm_response_data.get("input_tokens")
                if llm_response_data.get("output_tokens") is not None: total_output_tokens += llm_response_data.get("output_tokens")
                if llm_response_data.get("response_time") is not None: total_time_taken += llm_response_data.get("response_time")
            else:
                error_message = llm_response_data.get("error_message", "Unknown error") if llm_response_data else "LLM call failed (returned None)"
                logger.error(f"Error with {model_config.get('config_id')} (Default) for question {question_data.get('question_id', i)}: {error_message}")
                result_item.update({
                    "llm_answer": f"ERROR: {error_message}",
                    "raw_llm_answer": "",
                    "cleaned_llm_answer": "",
                    "error": error_message,
                    "input_tokens": llm_response_data.get("input_tokens") if llm_response_data else 0,
                    "output_tokens": llm_response_data.get("output_tokens") if llm_response_data else 0,
                    "response_time": llm_response_data.get("response_time") if llm_response_data else 0,
                    "answer_length": 0 
                })
                if llm_response_data and llm_response_data.get("response_time") is not None:
                    total_time_taken += llm_response_data.get("response_time")
        
        except Exception as e:
            logger.error(f"Strategy level exception for {model_config.get('config_id')} (Default) question {question_data.get('question_id', i)}: {e}", exc_info=True)
            result_item.update({
                "llm_answer": f"ERROR: {str(e)}",
                "raw_llm_answer": "",
                "cleaned_llm_answer": "",
                "error": str(e),
                "answer_length": 0 
            })

        results.append(result_item)

    avg_latency = total_time_taken / num_questions if num_questions > 0 else 0
    
    return {
        "results": results,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_time_taken_seconds": total_time_taken,
        "average_latency_ms": avg_latency * 1000, 
        "successful_generations": successful_generations,
        "total_questions": num_questions
    } 