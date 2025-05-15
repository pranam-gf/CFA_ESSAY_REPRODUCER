"""
Default model configurations for general essay generation strategies.
"""

ALL_MODEL_CONFIGS = [
    
    {
    "config_id": "claude-3.7-sonnet",
    "type": "anthropic",
    "model_id": "claude-3-7-sonnet-20250219",
    "parameters": {
        "temperature": 0.1,
        "top_p": 0.999,
        "top_k": 250,
        "max_tokens": 64000
    }
    },
    {
        "config_id": "claude-3.5-sonnet",
        "type": "anthropic",
        "model_id": "claude-3-5-sonnet-20241022",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "claude-3.5-haiku",
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
        "config_id": "mistral-large-official",
        "type": "mistral_official",
        "model_id": "mistral-large-latest",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 4096
        }
    },
    {
        "config_id": "codestral-latest-official",
        "type": "mistral_official",
        "model_id": "codestral-latest",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "palmyra-fin-default",
        "type": "writer",
        "model_id": "palmyra-fin",
        "parameters": {
            "temperature": 0.0,
            "max_tokens": 4096
        }
    },
    {
        "config_id": "gpt-4o",
        "type": "openai",
        "model_id": "gpt-4o",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 8192
        }
    },
    
    {
        "config_id": "o3-mini",
        "type": "openai",
        "model_id": "o3-mini-2025-01-31",
        "parameters": {
            "temperature": 1.0,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "o4-mini",
        "type": "openai",
        "model_id": "o4-mini-2025-04-16",
        "parameters": {
            "temperature": 1.0,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "gpt-4.1",
        "type": "openai",
        "model_id": "gpt-4.1-2025-04-14",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "gpt-4.1-mini",
        "type": "openai",
        "model_id": "gpt-4.1-mini-2025-04-14",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "gpt-4.1-nano",
        "type": "openai",
        "model_id": "gpt-4.1-nano-2025-04-14",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 32768
        }
    },
    
    {
        "config_id": "grok-3",
        "type": "xai",
        "model_id": "grok-3-latest",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 16384
        }
    },
    {
        "config_id": "grok-3-mini-beta-high-effort",
        "type": "xai",
        "model_id": "grok-3-mini-beta",
        "parameters": {
            "temperature": 0.1,
            "reasoning_effort": "high",
            "max_tokens": 8192
        }
    },
    {
        "config_id": "grok-3-mini-beta-low-effort",
        "type": "xai",
        "model_id": "grok-3-mini-beta",
        "parameters": {
            "temperature": 0.1,
            "reasoning_effort": "low",
            "max_tokens": 8192
        }
    },
    
    {
    "config_id": "gemini-2.5-pro",
    "type": "gemini",
    "model_id": "gemini-2.5-pro-exp-03-25",
    "parameters": {
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 65536
    }
    },
    {
    "config_id": "gemini-2.5-flash",
    "type": "gemini",
    "model_id": "gemini-2.5-flash-preview-04-17",
    "parameters": {
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 65536
    }
    },
    {
        "config_id": "deepseek-r1",
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
        "config_id": "groq-llama3.3-70b",
        "type": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "groq-llama3.1-8b-instant",
        "type": "groq",
        "model_id": "llama-3.1-8b-instant",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    },
] 