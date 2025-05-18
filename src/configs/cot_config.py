"""
Model configurations tailored for Chain-of-Thought (CoT) prompting strategies.
"""

ALL_MODEL_CONFIGS_COT = [    
    {
        "config_id": "gemini-2.5-pro-cot",
        "type": "gemini",
        "model_id": "gemini-2.5-pro-preview-05-06",
        "prompt_strategy_type": "COHERENT_CFA_COT", 
        "parameters": {
            "temperature": 0.5, 
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 65536
        }
    },
    {
        "config_id": "gemini-2.5-flash-cot",
        "type": "gemini",
        "model_id": "gemini-2.5-flash-preview-04-17",
        "prompt_strategy_type": "COHERENT_CFA_COT", 
        "parameters": {
            "temperature": 0.5, 
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 65536,
            "thinking_budget": 24576
        }
    },
    {
        "config_id": "claude-3.7-sonnet-cot",
        "type": "anthropic",
        "model_id": "claude-3-7-sonnet-20250219",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 64000
        }
    },
    {
        "config_id": "claude-3.5-sonnet-cot",
        "type": "anthropic",
        "model_id": "claude-3-5-sonnet-20241022",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "claude-3.5-haiku-cot",
        "type": "anthropic",
        "model_id": "claude-3-5-haiku-20241022",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "mistral-large-official-cot",
        "type": "mistral_official",
        "model_id": "mistral-large-latest",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 4096 
        }
    },
    {
        "config_id": "codestral-latest-cot",
        "type": "mistral_official",
        "model_id": "codestral-latest",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 32768 
        }
    },
    {
        "config_id": "gpt-4o-cot",
        "type": "openai",
        "model_id": "gpt-4o",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5
            
        }
    },
    {
        "config_id": "gpt-o3-cot",
        "type": "openai",
        "model_id": "o3", 
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            # "max_tokens": 1500,
            "response_format": {"type": "json_object"}
        }
    },
    {
        "config_id": "gpt-o4-mini-cot",
        "type": "openai",
        "model_id": "o4-mini", 
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            # "max_tokens": 1000,
            "response_format": {"type": "json_object"}
        }
    },
    {
        "config_id": "gpt-4.1-cot",
        "type": "openai",
        "model_id": "gpt-4.1-2025-04-14",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 32768
            
        }
    },
    {
        "config_id": "gpt-4.1-mini-cot",
        "type": "openai",
        "model_id": "gpt-4.1-mini-2025-04-14",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 32768
            
        }
    },
    {
        "config_id": "gpt-4.1-nano-cot",
        "type": "openai",
        "model_id": "gpt-4.1-nano-2025-04-14",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "grok-3-mini-beta-cot-high-effort",
        "type": "xai",
        "model_id": "grok-3-mini-beta",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.1,
            "reasoning_effort": "high"
        }
    },
    {
        "config_id": "grok-3-mini-beta-cot-low-effort",
        "type": "xai",
        "model_id": "grok-3-mini-beta",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.1,
            "reasoning_effort": "low"
        }
    },
    {
        "config_id": "grok-3-cot",
        "type": "xai",
        "model_id": "grok-3-latest",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 16384
        }
    },

    
    {
        "config_id": "palmyra-fin-cot",
        "type": "writer",
        "model_id": "palmyra-fin",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5
        }
    },
    {
        "config_id": "deepseek-r1-cot",
        "type": "groq",
        "model_id": "deepseek-r1-distill-llama-70b", 
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.6,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "groq-llama-4-maverick",
        "type": "groq",
        "model_id": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "groq-llama-4-scout",
        "type": "groq",
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "groq-llama-guard-4",
        "type": "groq",
        "model_id": "meta-llama/Llama-Guard-4-12B",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 128
        }
    },
    {
        "config_id": "groq-llama3.3-70b-cot",
        "type": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "groq-llama3.1-8b-instant-cot",
        "type": "groq",
        "model_id": "llama-3.1-8b-instant",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    }
] 