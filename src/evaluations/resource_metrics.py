"""
Functions for calculating resource usage metrics from LLM run results.
Includes latency, token counts, cost, etc.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional


from . import cost_evaluation

logger = logging.getLogger(__name__)

def calculate_resource_usage(results_data: list[dict], model_config_info: Optional[Dict[str, Any]] = None) -> dict:
    """
    Calculates resource usage metrics (latency, tokens, cost) from LLM run results.

    Args:
        results_data: A list of dictionaries, where each represents a question's
                      result and may contain 'response_time', 'input_tokens',
                      'output_tokens', 'answer_length', and 'cost'.
        model_config_info: Optional dictionary containing model configuration details,
                           like 'type' and 'config_id', for cost calculation.

    Returns:
        A dictionary containing aggregated resource metrics:
        'average_latency_ms', 'total_input_tokens', 'total_output_tokens',
        'total_tokens', 'avg_answer_length', 'total_cost'.
    """
    if not results_data:
        logger.warning("No results data provided. Cannot calculate resource usage metrics.")
        return {
            "average_latency_ms": None,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "avg_answer_length": 0.0,
            "total_cost": 0.0
        }
    
    valid_times = [
        r.get('response_time') for r in results_data 
        if isinstance(r, dict) and r.get('response_time') is not None
    ]
    avg_time_per_question_sec = np.mean(valid_times) if valid_times else None
    average_latency_ms = avg_time_per_question_sec * 1000 if avg_time_per_question_sec is not None else None
    
    valid_in_tokens = [
        r.get('input_tokens') for r in results_data 
        if isinstance(r, dict) and r.get('input_tokens') is not None
    ]
    total_in_tokens = sum(valid_in_tokens) if valid_in_tokens else 0
    
    valid_out_tokens = [
        r.get('output_tokens') for r in results_data 
        if isinstance(r, dict) and r.get('output_tokens') is not None
    ]
    total_out_tokens = sum(valid_out_tokens) if valid_out_tokens else 0
    
    total_tokens = total_in_tokens + total_out_tokens
    
    answer_lengths = [
        r.get('answer_length', 0) for r in results_data 
        if isinstance(r, dict) 
    ]
    num_results_for_len = len(answer_lengths)
    avg_answer_len = np.mean(answer_lengths) if num_results_for_len > 0 else 0.0
    
    valid_results_for_cost = [r for r in results_data if isinstance(r, dict)]
    cost_dict = cost_evaluation.calculate_model_cost(valid_results_for_cost, model_config_info)

    resource_metrics = {
        "average_latency_ms": average_latency_ms,
        "total_input_tokens": total_in_tokens,
        "total_output_tokens": total_out_tokens,
        "total_tokens": total_tokens,
        "avg_answer_length": avg_answer_len,
        "total_cost": cost_dict.get('total_cost', 0.0)
    }
    
    return resource_metrics
