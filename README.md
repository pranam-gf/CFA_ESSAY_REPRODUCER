<p align="center">
  <a href="https://www.goodfin.com/" target="_blank">
    <img src="img/gf_logo.svg" alt="GoodFin Logo" width="200" style="background-color:white;"/>
  </a>
</p>

# CFA Essay Question Answer Generator & LLM Benchmark

This project provides a robust framework for generating and evaluating essay-style answers to Chartered Financial Analyst (CFA) program questions using a diverse set of Large Language Models (LLMs). It features an interactive command-line interface (CLI) with loading animations and progress indicators, facilitating a comprehensive benchmarking workflow. The primary goal is to assess LLM capabilities in generating high-quality, contextually relevant financial essays and to provide a reproducible research platform.

## Essay Generation and Evaluation Process

The core pipeline is designed for rigorous benchmarking:

1.  **Data Ingestion:** Loads CFA essay questions from `data/updated_data.json`. Each entry is expected to contain `folder` (topic), `vignette` (context/scenario), `question` (the essay prompt), and `explanation` (a reference/gold-standard answer).
2.  **Interactive Configuration:** Users select from a range of LLMs and prompting strategies (e.g., direct generation, Chain-of-Thought, Self-Discover) via the CLI.
3.  **Essay Generation:** The system dispatches questions (formatted with vignette, question text, and topic) to the chosen LLMs, which then generate essay-style responses.
4.  **Comprehensive Evaluation:** Generated essays undergo a multi-faceted evaluation:
    *   **LLM Self-Grading:** The generating LLM (or a designated evaluation LLM) assesses its own output against the reference `explanation` based on a structured rubric, providing a score and justification.
    *   **Semantic Similarity:**
        *   **Cosine Similarity:** TF-IDF vectors are used to calculate the cosine similarity between the generated essay and the reference `explanation`.
        *   **ROUGE Scores:** ROUGE-L (Precision, Recall, F1-measure) is calculated to assess overlap with the reference answer.
    *   **Resource & Cost Tracking:** Input/output tokens, processing latency, and estimated API costs are meticulously recorded for each generation.
5.  **Results Persistence:** All generated essays, detailed evaluation scores, performance metrics, and configuration metadata are saved to JSON files in the `results/` directory for later analysis and reproducibility. Detailed per-item results are also saved.
6.  **Visualization & Reporting:** A suite of plots is generated to compare and analyze model/strategy performance (see "Visualizations and Analysis" section). A summary report is also printed to the console.

## Benchmark Overview

This LLM benchmark evaluates state-of-the-art language models on their proficiency in tackling CFA **essay questions**. It measures the quality of generated content, semantic alignment with reference answers, and various efficiency metrics.

<table>
<thead>
<tr>
<th align="center">Benchmark Component</th>
<th align="center">Details</th>
<th align="center">Approx. Count</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center"><strong>Models Evaluated</strong></td>
<td>Includes a diverse set such as Claude-3 series (Sonnet, Haiku), Mistral series (Large, Codestral), Palmyra-fin, GPT series (GPT-4o, GPT-4.1, -mini, -nano), Grok series, Gemini series (2.5 Pro, 2.5 Flash), Deepseek-R1, Llama series (Llama-4, Llama-3.3, Llama-3.1), and others. Models are configurable in <code>src/configs/default_config.py</code>.</td>
<td align="center"><strong>20+</strong></td>
</tr>
<tr>
<td align="center"><strong>Prompting Strategies</strong></td>
<td>Default Essay (single pass), Self-Consistency Essay (N=3, N=5 samples using CoT), Self-Discover Essay. All strategies are adapted for essay generation.</td>
<td align="center"><strong>4</strong></td>
</tr>
<tr>
<td align="center"><strong>Key Evaluation Metrics</strong></td>
<td>LLM Self-Graded Score (1-10), Cosine Similarity, ROUGE-L (Precision, Recall, F1-measure), Average Latency (ms), Total Input/Output Tokens, Average Answer Length, Total API Cost ($), Total Run Time (s).</td>
<td align="center"><strong>10+</strong></td>
</tr>
<tr>
<td align="center"><strong>Core Visualizations</strong></td>
<td>Distribution of Cosine Similarity Scores (KDE), Cosine Similarity vs. Self-Evaluation Score (Scatter + Regression), Error Heatmap (Cosine vs. Self-Eval Binned), LLM Self-Grading Calibration Curve, Radar Chart of Evaluation Dimensions. Additional plots include Model/Strategy comparisons for all key metrics, trade-off scatter plots (e.g., Score vs. Latency, Score vs. Cost), and detailed per-item result analysis.</td>
<td align="center"><strong>15+</strong></td>
</tr>
</tbody>
</table>

This benchmark framework facilitates in-depth analysis of over 20 LLMs across multiple advanced prompting strategies, all tailored for essay generation. It measures a rich set of performance and quality metrics, and produces a wide array of visualizations to support rigorous evaluation of LLM capabilities on CFA essay questions.

## Project Structure

The project is organized to promote modularity and ease of extension:

```
CFA_MCQ_REPRODUCER/  # Project Root (Note: actual name might vary, e.g., CFA_ESSAY_REPRODUCER)
├── data/
│   ├── updated_data.json      # Input essay question data (vignette, question, topic, explanation).
│   └── answer_grading_details.json # CFA Level III grading criteria for re-grading utility.
├── img/
│   └── *.png                  # Images for README.
├── results/
│   ├── comparison_charts/     # Directory for all generated plots.
│   ├── evaluated_results_*.json # Per-item detailed evaluation results for each run.
│   └── response_data_*.json   # Raw JSON outputs from LLM strategies.
│   └── model_warnings.log     # Log file for warnings encountered during model runs.
├── src/
│   ├── __init__.py
│   ├── config.py              # Global settings, API key names, default paths.
│   ├── llm_clients.py         # LLM API interaction layer.
│   ├── configs/               # Model configurations and parameters.
│   │   ├── __init__.py
│   │   └── default_config.py  # Primary configuration for models and their parameters.
│   ├── evaluations/           # Evaluation metric calculation modules.
│   │   ├── __init__.py
│   │   ├── essay_evaluation.py  # Cosine similarity, ROUGE, LLM self-grading.
│   │   └── resource_metrics.py  # Token counting, latency calculation.
│   │   └── cost_evaluation.py   # API cost estimation.
│   ├── prompts/               # LLM prompt templates and formatting functions.
│   │   ├── __init__.py
│   │   ├── cot.py               # Chain-of-Thought prompts for essays.
│   │   ├── self_discover.py     # Self-Discover prompts for essays.
│   │   ├── default.py           # Basic essay prompts.
│   │   └── grading_prompts.py   # CFA Level III grading prompts for re-grading.
│   ├── strategies/            # Different prompting strategy implementations.
│   │   ├── __init__.py
│   │   ├── default.py
│   │   ├── self_consistency.py
│   │   └── self_discover.py
│   ├── utils/                 # Utility functions.
│   │   ├── __init__.py
│   │   ├── ui_utils.py        # CLI animations and colored output.
│   │   ├── prompt_utils.py    # Prompt generation and data parsing utilities.
│   │   ├── text_utils.py      # Text cleaning functions for LLM answers.
│   │   └── regrade_with_gpt4_1.py # Unbiased re-grading utility (see below).
│   │   ├── analyze_reasoning_vs_nonreasoning_scores.py # Analysis script.
│   │   ├── analyze_opensource_vs_closedsource.py # Analysis script.
│   │   ├── generate_model_category_summary.py # Analysis script.
│   │   └── generate_advanced_analysis.py # Analysis script.
│   ├── main.py                # Main script to orchestrate the pipeline.
│   └── plotting.py            # Script to generate all evaluation plots.
├── .env                       # Local environment variables (API keys). Not version controlled.
├── .env.example               # Example .env file structure.
├── requirements.txt           # Python package dependencies.
└── README.md                  # This file.
```

## Setup

1.  **Clone the repository (if applicable).**

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Keys:**
    Copy `.env.example` to `.env` in the project root directory.
    Add your API keys to this `.env` file. The script will load these variables.

    ```env
    # Example .env content
    OPENAI_API_KEY="your_openai_api_key"
    ANTHROPIC_API_KEY="your_anthropic_api_key"
    GEMINI_API_KEY="your_gemini_api_key"
    # ... add other keys as needed (Mistral, Groq, AWS, etc.)
    ```

5.  **Prepare Input Data:**
    - Ensure your essay question data file, `updated_data.json`, is located in the `data/` directory.
    - **Required Data Structure:** `updated_data.json` must be a JSON list, where each object represents an essay question and includes:
        - `folder`: (string) The topic or category (e.g., "Portfolio Management").
        - `vignette`: (string) The context or scenario for the question.
        - `question`: (string) The essay question prompt.
        - `explanation`: (string) The reference or model answer for evaluation.
        - (Optional but recommended) `question_hash` or a unique `question_id`.

## Running the Pipeline

Navigate to the project root directory in your terminal. Run the main script as a module:

```bash
python3 -m src.main
```

This will initiate the following workflow:
- Load essay question data from `data/updated_data.json` (with loading animation).
- **Prompt you to interactively select LLM models and essay generation strategies.**
- Process each question with the chosen configurations, displaying real-time progress updates.
- Evaluate generated essays using semantic similarity (Cosine, ROUGE-L), LLM self-grading, and resource metrics.
- Display informative status messages (success, error, info, warning) with color coding.
- Save generated essays, detailed evaluation scores, and metadata to uniquely named JSON files in the `results/` directory.
- Generate a comprehensive set of plots in `results/comparison_charts/`.
- Print a formatted summary table of the run's performance to the console.

### Interactive UI Features

The CLI provides an enhanced user experience:
1.  **Loading Animations**: For operations like data loading, model processing, and results saving.
2.  **Per-Question Progress Indicators**: During LLM processing (e.g., `Processing with gpt-4o (Default Essay): [15/50] questions ⠋`).
3.  **Colored Console Output**: For clear distinction of success (✓ green), error (✗ red), info (ℹ blue), and warning (⚠ yellow) messages.
4.  **Formatted Results Summary**: A detailed table summarizing key metrics for each model/strategy combination at the end of the run.

## Unbiased Essay Re-grading Utility

### Addressing Self-Grading Bias

**Critical Issue**: In the initial benchmark design, each LLM was grading its own generated essays, potentially introducing significant bias where models might systematically over-rate or under-rate their own outputs.

**Solution**: The `src/utils/regrade_with_gpt4_1.py` utility addresses this bias by using **GPT-4.1 as a consistent, independent grader** across all models and strategies. This approach reproduces the unbiased evaluation methodology from the [JP Morgan CFA paper](https://aclanthology.org/2024.emnlp-industry.80.pdf) (Mahfouz et al., 2024), ensuring fair comparison across different LLMs.

### Key Features:
- **Consistent Grader**: Uses GPT-4.1 with identical grading prompts for all essays
- **CFA Level III Grading System**: Implements proper CFA grading rubrics with detailed scoring criteria
- **Automatic Data Matching**: Matches generated essays to grading criteria using folder + position indexing
- **Backup & Safety**: Creates backups before modifying result files
- **Comprehensive Logging**: Tracks score changes, processing statistics, and potential issues

### Usage:

```bash
# Dry run to preview changes (recommended first)
python -m src.utils.regrade_with_gpt4_1 --strategy default_essay --dry-run

# Re-grade a specific strategy
python -m src.utils.regrade_with_gpt4_1 --strategy self_consistency_essay_n3 --execute

# Re-grade all essay strategies
python -m src.utils.regrade_with_gpt4_1 --strategy all --execute
```

### Requirements:
- **OpenAI API Key**: GPT-4.1 access required (set `OPENAI_API_KEY` in `.env`)
- **Grading Data**: Requires `data/answer_grading_details.json` with CFA grading criteria
- **Generated Results**: Works with existing `evaluated_results_*.json` files from any strategy
- **Backup Files**: Original files preserved as `.json.backup`

This utility is essential for reproducing the rigorous, unbiased evaluation standards used in academic LLM research and enables fair comparison of model capabilities across the benchmark.

## Post-Hoc Analysis Scripts

After running the main pipeline (`python3 -m src.main`) and potentially the re-grading utility, several scripts are available in `src/utils/` to perform more detailed analysis on the generated `overall_model_performance_summary.csv` and `reasoning_vs_nonreasoning_analysis.csv` files located in `results/essay_analysis_charts/`.

Ensure you have run `python3 -m src.utils.main_csv_reproducer` to generate the `overall_model_performance_summary.csv` before running these analysis scripts.

### 1. Reasoning vs. Non-Reasoning Analysis

This script categorizes models and compares their performance based on whether they are considered "reasoning" or "non-reasoning" models.
- **Script:** `src/utils/analyze_reasoning_vs_nonreasoning_scores.py`
- **Input:** `results/essay_analysis_charts/overall_model_performance_summary.csv`
- **Output:** 
    - `results/essay_analysis_charts/reasoning_vs_nonreasoning_analysis.csv` (detailed breakdown)
    - Console output with summary statistics and t-test results.
- **Usage:**
  ```bash
  python -m src.utils.analyze_reasoning_vs_nonreasoning_scores
  ```

### 2. Open-Source vs. Closed-Source Analysis

This script categorizes models based on whether they are open-source or closed-source and compares their performance.
- **Script:** `src/utils/analyze_opensource_vs_closedsource.py`
- **Input:** `results/essay_analysis_charts/overall_model_performance_summary.csv`
- **Output:**
    - `results/essay_analysis_charts/opensource_vs_closedsource_detailed_analysis.csv`
    - `results/essay_analysis_charts/opensource_vs_closedsource_category_summary.csv`
    - Console output with summary statistics and t-test results.
- **Usage:**
  ```bash
  python -m src.utils.analyze_opensource_vs_closedsource
  ```
  (Note: The list of open-source models is currently hardcoded within the script.)

### 3. Model Category Performance Summary

This script generates a summary of performance metrics grouped by model category (Reasoning vs. Non-Reasoning).
- **Script:** `src/utils/generate_model_category_summary.py`
- **Input:** `results/essay_analysis_charts/reasoning_vs_nonreasoning_analysis.csv`
- **Output:**
    - `results/essay_analysis_charts/model_category_performance_summary.csv`
    - Console output with a detailed breakdown.
- **Usage:**
  ```bash
  python -m src.utils.generate_model_category_summary
  ```

### 4. Advanced Analysis Generation

This script calculates more advanced aggregate statistics, efficiency metrics, and model type comparisons.
- **Script:** `src/utils/generate_advanced_analysis.py`
- **Input:** `results/all_results_reproduced.csv` (Note: This is different from other analysis scripts; ensure it's generated by `src/utils/main_csv_reproducer.py`)
- **Output:**
    - `results/advanced_essay_analysis/model_essay_aggregate_stats.csv`
    - `results/advanced_essay_analysis/strategy_essay_aggregate_stats.csv`
    - `results/advanced_essay_analysis/essay_efficiency_metrics.csv`
    - `results/advanced_essay_analysis/model_type_essay_aggregate_stats.csv`
    - Console output logging the process.
- **Usage:**
  ```bash
  python -m src.utils.generate_advanced_analysis
  ```
  (Note: This script relies on model type definitions in `src/configs/default_config.py`.)

These scripts provide deeper insights into the benchmark results, allowing for nuanced comparisons and analysis beyond the primary plotting outputs.

## Configuration

-   **LLM Models & Parameters:**
    -   The primary configuration file for defining models, their API identifiers, types (mapping to `llm_clients.py`), and specific parameters (e.g., `temperature`, `max_output_tokens`, `thinking_budget` for Gemini, `reasoning_effort` for Grok) is `src/configs/default_config.py`.
    -   Users can add, remove, or modify model entries in this file to customize the benchmark.
    -   Example model entry in `src/configs/default_config.py`:
        ```python
        {
            "config_id": "gemini-2.5-pro-flash-budgeted", # Descriptive ID for this run
            "type": "gemini",                             # Maps to a client in llm_clients.py
            "model_id": "gemini-1.5-flash-001",           # Actual API model ID
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_output_tokens": 2048, # Suitable for essays
                "thinking_budget": 1000    # Example for specific models
            }
        }
        ```
-   **API Keys & Global Settings:** While API keys are best managed via the `.env` file, other global settings like default data/results paths or retry configurations can be found in `src/config.py`.
-   **Prompt Templates:**
    -   General essay prompts are in `src/prompts/default.py`.
    -   Chain-of-Thought essay prompts are in `src/prompts/cot.py` (template `ESSAY_COT_PROMPT`, formatter `format_essay_cot_prompt`).
    -   Self-Discover essay prompts are in `src/prompts/self_discover.py` (template `ESSAY_SELF_DISCOVER_PROMPT_TEMPLATE`, formatter `format_self_discover_prompt`).
    These can be modified to tailor the LLM's approach to essay generation.

## Visualizations and Analysis

The plotting module (`src/plotting.py`) generates a rich suite of visualizations saved in `results/comparison_charts/`, enabling thorough analysis of LLM performance:

📊 **1. Distribution of Cosine Similarity Scores**
   - **Type:** Kernel Density Estimate (KDE) Plot.
   - **Why:** Shows the semantic similarity distribution of generated answers compared to reference answers. Helps visualize consistency and central tendency of similarity scores across models/strategies.
   - **Implementation:** X-axis: Cosine similarity score (0-1); Y-axis: Density; Hue/Color: LLM model; Facets: Prompting strategy.

📈 **2. Correlation: Cosine Similarity vs. Self-Evaluation Scores**
   - **Type:** Scatter Plot with Regression Line.
   - **Why:** Measures the alignment between automated semantic similarity (cosine) and the LLM's own assessment of its answer quality.
   - **Implementation:** X-axis: Cosine similarity; Y-axis: Self-evaluation score (1–10); Points: Individual answers; Annotations: Pearson correlation coefficient (r) and p-value.

🔍 **3. Error Heatmap: Cosine Similarity vs. Self-Evaluation**
   - **Type:** 2D Heatmap.
   - **Why:** Visualizes mismatches or agreements between cosine similarity and self-evaluation scores by showing the density of answers in binned categories. Highlights areas where automated metrics and LLM judgment diverge.
   - **Implementation:** X-axis: Binned Cosine similarity; Y-axis: Binned Self-evaluation score; Color: Number of instances per bin.

🧠 **4. LLM Self-Grading Calibration Curve**
   - **Type:** Line Plot (Calibration Curve).
   - **Why:** Evaluates how well an LLM's self-assigned confidence scores (self-grades) correlate with an objective performance metric (e.g., cosine similarity or ROUGE-L F1).
   - **Implementation:** X-axis: Self-evaluation score bin (1–10); Y-axis: Mean of the chosen performance metric for that bin; Shaded Area: Confidence intervals.

📌 **5. Radar Chart of Evaluation Dimensions**
   - **Type:** Radar / Spider Plot.
   - **Why:** Compares models or model-strategy combinations across multiple normalized evaluation dimensions (e.g., cosine similarity, self-grade, ROUGE-L F1, ROUGE-L Precision, ROUGE-L Recall).
   - **Implementation:** Each axis: An evaluation metric (normalized to 0-1); Each polygon: A model-strategy pair.

**Additional Aggregated Plots:**
-   Bar charts comparing models and strategies across all key metrics (e.g., average scores, latency, cost).
-   Scatter plots illustrating trade-offs (e.g., average ROUGE-L F1 vs. average latency; average ROUGE-L F1 vs. total cost).
-   Specific comparisons for Self-Consistency (N=3 vs. N=5) performance.

This comprehensive suite of visualizations allows researchers and developers to dissect model performance from various angles, understand metric correlations, identify outliers, and make informed decisions based on empirical evidence.