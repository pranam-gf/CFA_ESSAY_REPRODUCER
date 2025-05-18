"""
Model configurations specifically for the Self-Discover strategy.
Initially mirrors default configs, can be tuned later.
"""

SELF_DISCOVER_CONFIGS = [
    {
    "config_id": "gemini-2.5-pro-self-discover",
    "type": "gemini",
    "model_id": "gemini-2.5-pro-preview-05-06",
    "parameters": {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 65536,
        "thinking_budget": 24576
    }
    },
    {
    "config_id": "gemini-2.5-flash-self-discover",
    "type": "gemini",
    "model_id": "gemini-2.5-flash-preview-04-17",
    "parameters": {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 65536,
        "thinking_budget": 24576
    }
    },
    {
        "config_id": "claude-3.5-haiku-self-discover",
        "type": "anthropic",
        "model_id": "claude-3-5-haiku-20241022",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "mistral-large-official-self-discover",
        "type": "mistral_official",
        "model_id": "mistral-large-latest",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 4096
        }
    },
    {
        "config_id": "codestral-latest-self-discover",
        "type": "mistral_official",
        "model_id": "codestral-latest",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 32768 
        }
    },
    {
        "config_id": "gpt-4o-self-discover",
        "type": "openai",
        "model_id": "gpt-4o",
        "parameters": {
            "temperature": 0.1
        }
    },
    
    {
        "config_id": "gpt-o3-self-discover",
        "type": "openai",
        "model_id": "o3",
        "parameters": {
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
    },
    {
        "config_id": "gpt-o4-mini-self-discover",
        "type": "openai",
        "model_id": "o4-mini",
        "parameters": {
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
    },
    {
        "config_id": "gpt-4.1-self-discover",
        "type": "openai",
        "model_id": "gpt-4.1-2025-04-14",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "gpt-4.1-mini-self-discover",
        "type": "openai",
        "model_id": "gpt-4.1-mini-2025-04-14",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "gpt-4.1-nano-self-discover",
        "type": "openai",
        "model_id": "gpt-4.1-nano-2025-04-14",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 32768
        }
    },
    
    {
        "config_id": "grok-3-mini-beta-self-discover-high-effort",
        "type": "xai",
        "model_id": "grok-3-mini-beta",
        "parameters": {
            "temperature": 0.1,
            "reasoning_effort": "high"
        }
    },
    {
        "config_id": "grok-3-mini-beta-self-discover-low-effort",
        "type": "xai",
        "model_id": "grok-3-mini-beta",
        "parameters": {
            "temperature": 0.1,
            "reasoning_effort": "low"
        }
    },
    {
        "config_id": "grok-3-self-discover",
        "type": "xai",
        "model_id": "grok-3-latest",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 16384
        }
    },
    
    {
    "config_id": "gemini-2.5-pro-self-discover",
    "type": "gemini",
    "model_id": "gemini-2.5-pro-exp-03-25",
    "parameters": {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 65536
    }
    },
    {
    "config_id": "gemini-2.5-flash-self-discover",
    "type": "gemini",
    "model_id": "gemini-2.5-flash-preview-04-17",
    "parameters": {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 65536,
        "thinking_budget": 4096 #24576
    }
    },
        {
        "config_id": "palmyra-fin-self-discover",
        "type": "writer",
        "model_id": "palmyra-fin",
        "parameters": {
            "temperature": 0.1
        }
    },
    {
        "config_id": "deepseek-r1-self-discover",
        "type": "groq",
        "model_id": "deepseek-r1-distill-llama-70b", 
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
        "config_id": "groq-llama3.3-70b-self-discover",
        "type": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "groq-llama3.1-8b-instant-self-discover",
        "type": "groq",
        "model_id": "llama-3.1-8b-instant",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    }
    
] 