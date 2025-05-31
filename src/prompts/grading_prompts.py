"""
Prompts for LLM-based grading of essay answers.
"""
GRADING_SYSTEM_PROMPT = """
You are an expert CFA exam grader. Your task is to evaluate a generated answer to a CFA essay question based on the provided original question and a reference model answer.
You must provide a numerical score from 1 to 10, where 1 is very poor and 10 is excellent.
You must also provide a brief justification for your score, highlighting the strengths and weaknesses of the generated answer, especially in comparison to the reference model answer.

Your response MUST be in the following JSON format:
{
  "score": <integer_score_from_1_to_10>,
  "justification": "<your_brief_justification_here>"
}
"""

GRADING_USER_PROMPT_TEMPLATE = """
Original Question:
{question}

Reference Model Answer (for your guidance as a grader):
{reference_answer}

Generated Answer to Evaluate:
{generated_answer}

Please provide your evaluation in the specified JSON format, including a 'score' (an integer from 1 to 10) and a 'justification' (a string).
"""
# Add this to src/prompts/grading_prompts.py

CFA_LEVEL_III_EFFICIENT_GRADING_SYSTEM_PROMPT = """
You are a strict CFA Level III examination grader. You must follow the provided grading details EXACTLY - no partial credit beyond what is explicitly specified.

**CRITICAL GRADING REQUIREMENTS:**
- You MUST return ONLY a single integer between {min_score} and {max_score}
- Maximum possible score: {max_score}
- Minimum possible score: {min_score}
- NEVER exceed the maximum score under any circumstances
- Follow the grading details below EXACTLY - they specify the ONLY ways to earn points

**STRICT GRADING PROTOCOL:**
- Award points ONLY if the grading details criteria are met EXACTLY as specified
- Do NOT award partial credit unless explicitly mentioned in the grading details
- Do NOT give points for "close enough" answers - criteria must be met precisely
- Do NOT award points for effort, methodology, or partial understanding unless the grading details specify this
- If the grading details say "2 points for X", the student must demonstrate X completely to get those 2 points
- If multiple criteria exist (e.g., "2 points for A, 2 points for B"), each must be met independently

**QUESTION ASKED:**
{question}

**CONTEXT/VIGNETTE (if applicable):**
{vignette}

**GRADING DETAILS (FOLLOW EXACTLY):**
{answer_grading_details}

**CORRECT ANSWER/EXPLANATION (for reference only):**
{correct_answer}

**STUDENT'S ANSWER:**
{generated_answer}

**RESPONSE:** Return only the integer score based on strict adherence to the grading details above.
"""

def get_full_cfa_level_iii_efficient_grading_prompt(
    question: str,
    vignette: str,
    answer_grading_details: str,
    correct_answer: str,
    student_answer: str,
    min_score: int,
    max_score: int
) -> str:
    """Generate the full CFA Level III efficient grading prompt with all context."""
    return CFA_LEVEL_III_EFFICIENT_GRADING_SYSTEM_PROMPT.format(
        question=question,
        vignette=vignette,
        answer_grading_details=answer_grading_details,
        correct_answer=correct_answer,
        generated_answer=student_answer,
        min_score=min_score,
        max_score=max_score
    )

def format_grading_prompt(question_text: str, reference_answer_text: str, generated_answer_text: str) -> tuple[str, str]:
    """
    Formats the system and user prompts for the LLM grader.

    Args:
        question_text: The original essay question.
        reference_answer_text: The reference/model answer.
        generated_answer_text: The LLM-generated answer to be graded.

    Returns:
        A tuple containing the system prompt and the formatted user prompt.
    """
    user_prompt = GRADING_USER_PROMPT_TEMPLATE.format(
        question=question_text,
        reference_answer=reference_answer_text,
        generated_answer=generated_answer_text
    )
    return GRADING_SYSTEM_PROMPT, user_prompt

def get_full_structured_grading_prompt(question_text: str, reference_answer_text: str, generated_answer_text: str) -> str:
    """
    Combines the system and user prompt into a single string, suitable for models
    that don't have a separate system prompt input, or for easier logging.
    The system prompt's instructions about JSON output are critical.

    Args:
        question_text: The original essay question.
        reference_answer_text: The reference/model answer.
        generated_answer_text: The LLM-generated answer to be graded.
    
    Returns:
        A single string combining the system and user prompts.
    """
    system_prompt, user_prompt = format_grading_prompt(question_text, reference_answer_text, generated_answer_text)
    return f"{system_prompt}\n\n{user_prompt}" 