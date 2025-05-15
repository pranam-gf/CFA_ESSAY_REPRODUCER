"""
Prompt template for the Self-Discover reasoning strategy.

Guides the LLM to first outline a reasoning structure before solving the problem.
Inspired by: https://arxiv.org/abs/2402.14310v1
"""
from ..utils.prompt_utils import parse_question_data

SELF_DISCOVER_PROMPT_TEMPLATE = """\
**Task:** Solve the following multiple-choice question by first devising a reasoning structure using the Self-Discover method.

**Context/Vignette:**
{vignette}

**Question Stem:**
{question_stem}

**Options:**
A: {option_a}
B: {option_b}
C: {option_c}

**Self-Discover Reasoning Process:**

**1. Select Reasoning Modules:**
   Identify and list the core reasoning modules or types of thinking needed to solve this specific question. Examples: Causal Reasoning, Definition Understanding, Calculation, Comparison, Rule Application, Concept Identification, etc.

**2. Adapt Modules to the Problem:**
   For each selected module, briefly explain how it specifically applies to this question and the given options. Outline the steps you will take within each module.

**3. Implement Reasoning Structure:**
   Execute the plan outlined above step-by-step.
   [Model generates reasoning steps here]

**4. Final Answer:**
   Based on the reasoning, critically evaluate the options and provide the final answer.
   **IMPORTANT**: Conclude your response with the final answer choice letter (A, B, C, or D) on a new line, formatted exactly as: `The final answer is: **[Option Letter]**` (e.g., `The final answer is: **B**`). Do not include any other text after this final line.
   [Model provides final answer letter here in the specified format]

**Begin Reasoning:**

[Your reasoning structure and step-by-step solution here]

**Final Answer:** [Correct Option Letter]
"""

ESSAY_SELF_DISCOVER_PROMPT_TEMPLATE = """\
**Task:** Construct a comprehensive essay answer to the following question by first devising a reasoning structure using the Self-Discover method.

**Topic:** {folder}

**Context/Vignette:**
{vignette}

**Question:**
{question}

**Self-Discover Essay Construction Process:**

**1. Select Reasoning Modules for Essay Construction:**
   Identify and list the core reasoning modules or types of thinking needed to construct a thorough essay for this specific question. Examples:
   -   **Problem Deconstruction:** Breaking down the question into smaller, manageable parts.
   -   **Information Extraction:** Identifying key facts, data, and constraints from the vignette.
   -   **Concept Application:** Determining relevant financial theories, models, or formulas (e.g., valuation methods, risk analysis, portfolio management principles).
   -   **Calculation & Quantitative Analysis:** Performing necessary calculations and interpreting their results.
   -   **Qualitative Analysis & Argumentation:** Developing logical arguments, discussing implications, and justifying conclusions.
   -   **Structuring & Synthesis:** Organizing the analysis into a coherent essay structure.

**2. Adapt Modules to the Essay Task:**
   For each selected module, briefly explain how it specifically applies to constructing the essay for this question. Outline the key steps or considerations for each module in relation to the essay.
   For example:
   -   *Problem Deconstruction:* What are the sub-questions or core components the essay must address?
   -   *Information Extraction:* What specific data points from the vignette are crucial for each part of the essay?
   -   *Concept Application:* Which specific CFA curriculum concepts will form the backbone of the analysis? How will they be linked?

**3. Outline the Essay Structure (based on adapted modules):**
   Develop a high-level outline for your essay. This should detail the main sections (e.g., Introduction, Analysis of Factor X, Calculation of Y, Discussion of Implications, Conclusion) and the key points to be covered in each.

**4. Implement Reasoning and Write the Essay:**
   Execute the plan outlined above. Write the essay step-by-step, following your structured outline. Ensure you:
   -   Clearly explain your reasoning for each step or argument.
   -   Show all calculations if required, detailing the inputs and formulas.
   -   Integrate information from the vignette and the topic appropriately.
   -   Maintain a logical flow and clear, professional language suitable for a CFA context.

**Begin Essay Construction (following the Self-Discover process):**

[Model generates its reasoning structure (Modules, Adaptation, Outline) and then the full essay here]

**Full Essay Answer:**
[The complete essay answer should be placed here]
"""

def format_self_discover_prompt(question_data: dict, use_essay_format: bool = True) -> str:
    """
    Formats the Self-Discover prompt (either MCQ or Essay) with the specific question details,
    using the standardized parsing utility.

    Args:
        question_data: Dictionary for a single question item.
        use_essay_format: If True, uses ESSAY_SELF_DISCOVER_PROMPT_TEMPLATE.
                          Otherwise, uses SELF_DISCOVER_PROMPT_TEMPLATE (for MCQs).

    Returns:
        Formatted prompt string.
    """
    parsed_data = parse_question_data(question_data)

    if use_essay_format:
        return ESSAY_SELF_DISCOVER_PROMPT_TEMPLATE.format(
            folder=parsed_data.get('folder', 'N/A'),
            vignette=parsed_data.get('vignette', 'N/A'),
            question=parsed_data.get('question', 'N/A')
        )
    else: 
        return SELF_DISCOVER_PROMPT_TEMPLATE.format(
            vignette=parsed_data['vignette'],
            question_stem=parsed_data['question_stem'],
            option_a=parsed_data['options_dict'].get('A', 'Option A not provided'),
            option_b=parsed_data['options_dict'].get('B', 'Option B not provided'),
            option_c=parsed_data['options_dict'].get('C', 'Option C not provided')
        )

def generate_prompt_for_self_discover_strategy(entry: dict) -> str:
    """Generates the Self-Discover prompt for the LLM using the standardized parser.
       This version defaults to MCQ for now, assuming it might still be used by an unmodified MCQ strategy path.
       For essays, call format_self_discover_prompt(entry, use_essay_format=True) directly.
    """
    return format_self_discover_prompt(entry, use_essay_format=False) 