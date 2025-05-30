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
CFA_LEVEL_III_EFFICIENT_GRADING_SYSTEM_PROMPT = """
You are tasked with grading essay answers from the CFA Level 3 examination. You will be supplied with the correct answer/explanation, the grading details (where to assign marks) and the student's answer. Return ONLY a numeric value from {min_score} to {max_score} indicating the number of marks the student should receive.
The maximum possible score for this specific question is {max_score}.
The minimum possible score for this specific question is {min_score}.
Here is the correct answer/explanation:
{correct_answer}
Here are the answer grading details:
{answer_grading_details}
Here is the student's answer:
{generated_answer}
"""

def get_full_cfa_level_iii_efficient_grading_prompt(answer_grading_details: str, student_answer: str, min_score: int = 0, max_score: int = 10, correct_answer: str = "") -> str:
    """
    Creates a CFA Level III efficient grading prompt that returns only a numerical score.
    
    Args:
        answer_grading_details: The grading criteria/rubric
        student_answer: The answer to be graded
        min_score: Minimum possible score (default 0)
        max_score: Maximum possible score (default 10)
        correct_answer: The reference/model answer (optional)
        
    Returns:
        Complete prompt for the LLM grader
    """
    return CFA_LEVEL_III_EFFICIENT_GRADING_SYSTEM_PROMPT.format(
        answer_grading_details=answer_grading_details,
        generated_answer=student_answer,
        min_score=min_score,
        max_score=max_score,
        correct_answer=correct_answer or "Not provided"
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