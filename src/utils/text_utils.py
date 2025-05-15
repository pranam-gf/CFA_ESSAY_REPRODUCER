import re

def clean_llm_answer_for_similarity(text: str) -> str:
    """
    Cleans an LLM-generated essay answer to make it more suitable for 
    similarity comparison against a plain text reference.
    Removes common markdown, prefixes, and placeholders LaTeX.

    Args:
        text: The raw LLM-generated answer string.

    Returns:
        A cleaned version of the text.
    """
    if not text:
        return ""

    cleaned_text = re.sub(r"^\*\*Answer:\*\*\s*", "", text, flags=re.IGNORECASE)    
    cleaned_text = re.sub(r"^\s*---\s*$", "", cleaned_text, flags=re.MULTILINE) 
    cleaned_text = re.sub(r"\s*---\s*", " ", cleaned_text) 
    cleaned_text = re.sub(r"###\s+", "", cleaned_text)
    cleaned_text = re.sub(r"\*\*(.*?)\*\*", r"\\1", cleaned_text)    
    cleaned_text = re.sub(r"\*(.*?)\*", r"\\1", cleaned_text)
    cleaned_text = re.sub(r"\\\[.*?\\\]", "[FORMULA]", cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r"\$\$.*?\$\$", "[FORMULA]", cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    return cleaned_text 