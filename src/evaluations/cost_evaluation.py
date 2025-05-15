import logging
from typing import List, Dict, Any, Optional
from src.configs.model_pricing import (
    get_pricing,
    get_openai_pricing,
    get_anthropic_pricing,
    get_gemini_pricing,
    get_groq_pricing,
    get_mistral_pricing as get_mistral_official_pricing,
    get_writer_pricing as get_writer_palmyra_pricing,
    get_grok_pricing
)

logger = logging.getLogger(__name__)

def calculate_model_cost(results_data: List[Dict[str, Any]], model_config_item: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates the estimated cost for API calls based on token usage for various model providers.

    Args:
        results_data: A list of dictionaries, where each dictionary contains
                      results from a single LLM call, potentially including
                      'input_tokens' and 'output_tokens'.
        model_config_item: The configuration dictionary for the model run,
                           used to check model type and ID.

    Returns:
        A dictionary containing the total estimated cost, e.g., {'total_cost': 0.123}.
        Returns {'total_cost': 0.0} if the model type is not supported, pricing is missing,
        or token data is unavailable.
    """
    total_cost = 0.0
    model_type = model_config_item.get("type")
    model_id = model_config_item.get("model_id", "")
    config_id_used = model_config_item.get("config_id", model_id)

    if not model_id:
         logger.error("Missing model_id from model_config_item for cost calculation.")
         return {"total_cost": 0.0}
         
    pricing = None
    price_per_input_token = 0.0
    price_per_output_token = 0.0

    pricing = get_pricing(model_type, model_id)

    if pricing:
        if model_type == "openai":
            price_per_input_token = pricing.get("input", 0.0)
            price_per_output_token = pricing.get("output", 0.0)
        else:
            price_per_input_token = pricing.get("prompt_tokens_cost_per_million", 0.0) / 1_000_000
            price_per_output_token = pricing.get("completion_tokens_cost_per_million", 0.0) / 1_000_000
    else:
        logger.warning(f"Skipping cost calculation for run using model ID '{model_id}' (type: {model_type}) as pricing is not defined.")
        return {"total_cost": 0.0}

    if price_per_input_token == 0.0 and price_per_output_token == 0.0:
        logger.warning(f"Both input and output token prices are zero for model '{model_id}' (type: {model_type}). Cost will be zero.")
        return {"total_cost": 0.0}
    
    num_items_missing_tokens = 0
    for item in results_data:
        input_tokens = item.get('input_tokens')
        output_tokens = item.get('output_tokens')

        if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
            item_cost = (input_tokens * price_per_input_token) + (output_tokens * price_per_output_token)
            total_cost += item_cost
        else:
            num_items_missing_tokens += 1
            

    if num_items_missing_tokens > 0 and len(results_data) > 0 : 
        logger.warning(f"Cost calculation for {model_id} (type: {model_type}) might be incomplete. Missing token data for {num_items_missing_tokens}/{len(results_data)} items.")

    logger.info(f"Estimated cost for {model_type} model {model_id}: ${total_cost:.6f}")
    return {"total_cost": total_cost} 


def calculate_total_cost_from_aggregated_tokens(
    total_input_tokens: Optional[int],
    total_output_tokens: Optional[int],
    model_id_for_pricing: str,
    model_type_for_pricing: str
) -> Dict[str, float]:
    """
    Calculates the estimated total cost based on aggregated input and output token counts.

    Args:
        total_input_tokens: Total number of input tokens for the run.
        total_output_tokens: Total number of output tokens for the run.
        model_id_for_pricing: The specific model ID to use for fetching pricing.
        model_type_for_pricing: The type of the model (e.g., 'openai', 'anthropic').

    Returns:
        A dictionary containing the total estimated cost, e.g., {'total_cost': 0.123}.
        Returns {'total_cost': 0.0} if pricing is missing or token data is incomplete.
    """
    if total_input_tokens is None or total_output_tokens is None:
        logger.warning(f"Cannot calculate cost for {model_id_for_pricing}: Missing aggregated token counts.")
        return {"total_cost": 0.0}

    pricing = get_pricing(model_type_for_pricing, model_id_for_pricing)
    price_per_input_token = 0.0
    price_per_output_token = 0.0

    if pricing:
        if model_type_for_pricing == "openai": 
            price_per_input_token = pricing.get("input", 0.0)
            price_per_output_token = pricing.get("output", 0.0)
        else: 
            price_per_input_token = pricing.get("prompt_tokens_cost_per_million", 0.0) / 1_000_000
            price_per_output_token = pricing.get("completion_tokens_cost_per_million", 0.0) / 1_000_000
    else:
        logger.warning(f"Skipping cost calculation for model ID '{model_id_for_pricing}' (type: {model_type_for_pricing}) as pricing is not defined.")
        return {"total_cost": 0.0}

    if price_per_input_token == 0.0 and price_per_output_token == 0.0:
        logger.warning(f"Both input and output token prices are zero for model '{model_id_for_pricing}' (type: {model_type_for_pricing}). Cost will be zero.")
    cost = (total_input_tokens * price_per_input_token) + (total_output_tokens * price_per_output_token)
    
    logger.info(f"Calculated total cost for {model_type_for_pricing} model {model_id_for_pricing} (Aggregated): ${cost:.6f}")
    return {"total_cost": cost} 