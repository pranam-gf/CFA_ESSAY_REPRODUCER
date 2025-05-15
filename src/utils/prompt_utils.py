"""
Utility functions for processing and formatting prompt data.
"""
from typing import Dict, Any

def parse_question_data(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses the raw essay question data into standardized components for prompt formatting.

    Args:
        question_data: The dictionary for a single question item.
                       Expected keys: 'folder', 'vignette', 'question', 'explanation'.

    Returns:
        A dictionary containing:
        'folder': The topic/category of the question.
        'vignette': The vignette text.
        'question': The essay question text.
        'explanation': The reference/model answer.
    """
    folder = question_data.get('folder', 'No folder/topic provided.')
    vignette = question_data.get('vignette', 'No vignette provided.')
    question_text = question_data.get('question', 'No question text provided.')
    explanation = question_data.get('explanation', 'No explanation provided.')

    return {
        "folder": folder,
        "vignette": vignette,
        "question": question_text,
        "explanation": explanation,
    } 