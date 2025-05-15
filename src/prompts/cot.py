"""
Stores Chain-of-Thought (CoT) related prompt templates.
"""
from ..utils.prompt_utils import parse_question_data 

COHERENT_CFA_COT = """
You are a Chartered Financial Analyst (CFA) charterholder.  Your task is to answer one multiple‐choice question from the CFA curriculum.  Follow these steps:

1. Restate the question stem in your own words.
2. Think through it step by step, showing your reasoning (use bullet points if helpful).
3. Evaluate each of the choices provided in the Options section, noting why each could be right or wrong.
4. Conclude by selecting the single best answer. First, provide a one-sentence justification for your choice. Then, on a new, separate line, write "Final Answer: [LETTER]", where [LETTER] is the capital letter of your chosen option (e.g., "Final Answer: A"). This "Final Answer: [LETTER]" line must be the absolute last line of your response.

Vignette:
{vignette}

Question Stem:
{question_stem}

Options:
A: {option_a}
B: {option_b}
C: {option_c}

Answer:
"""

SELF_CONSISTENCY_INSTRUCTIONS = """
Repeat the above chain-of-thought prompt N times (with temperature > 0) to get multiple independent reasoning traces.  
Finally, perform a majority vote on the selected letters to pick your final answer.
"""

ESSAY_COT_PROMPT = """
**Topic:** {folder}

**Vignette:**
{vignette}

**Question:**
{question}

**Instructions for Chain-of-Thought Essay Construction:**

You are a Chartered Financial Analyst (CFA) charterholder. Your task is to construct a comprehensive, well-structured essay answer to the question above. Follow these steps carefully:

1.  **Understand the Core Request:** Briefly rephrase the main objective of the question in your own words. What key information or analysis is being sought?
2.  **Identify Key Information & Concepts:** Based on the vignette, topic, and question, list the critical pieces of information, CFA curriculum concepts, formulas, or analytical frameworks that will be relevant to constructing your answer.
3.  **Outline Your Essay Structure:** Before writing, create a brief bullet-point outline of how you will structure your essay. This should include the main sections or arguments you will present.
4.  **Step-by-Step Elaboration:** Following your outline, elaborate on each point.
    *   If calculations are needed, show your work clearly, explaining each step and the components of any formulas used.
    *   If discussing concepts, define them and explain their relevance to the question.
    *   Ensure your reasoning is logical and directly supported by the information in the vignette where applicable.
5.  **Synthesize and Conclude:** Briefly summarize your main points and provide a concluding statement that directly answers the question.
6.  **Review (Self-Correction):** Quickly review your drafted essay for clarity, accuracy, completeness, and conciseness. Ensure it directly addresses all parts of the question.

**Begin Essay Construction (following the steps above):**

[Your detailed, step-by-step constructed essay answer here]
"""

def format_essay_cot_prompt(question_data: dict) -> str:
    """
    Formats the ESSAY_COT_PROMPT with the specific question details.

    Args:
        question_data: Dictionary for a single essay question item,
                       expected to be parsed by `parse_question_data`
                       (containing 'folder', 'vignette', 'question').

    Returns:
        Formatted prompt string.
    """
    return ESSAY_COT_PROMPT.format(
        folder=question_data.get('folder', 'N/A'),
        vignette=question_data.get('vignette', 'N/A'),
        question=question_data.get('question', 'N/A')
    ) 