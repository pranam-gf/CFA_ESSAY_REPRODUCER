import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

RESULTS_BASE_PATH = "../../results/essay_analysis_charts/"

def categorize_models_by_source(model_name: str, open_source_list: list) -> str:
    """Categorizes a model as Open-Source or Closed-Source."""
    if model_name in open_source_list:
        return "Open-Source"
    return "Closed-Source"

def analyze_model_performance_by_source(csv_file_path: str, open_source_models: list):
    """
    Analyzes model performance by categorizing models into Open-Source vs. Closed-Source.

    Args:
        csv_file_path (str): Path to the overall model performance summary CSV.
        open_source_models (list): A list of model names identified as open-source.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"❌ Error: Could not find file {csv_file_path}")
        print("Please ensure the file exists and try again.")
        return

    df['source_type'] = df['model'].apply(lambda x: categorize_models_by_source(x, open_source_models))

    detailed_output_path = f"{RESULTS_BASE_PATH}opensource_vs_closedsource_detailed_analysis.csv"
    df.to_csv(detailed_output_path, index=False)
    print(f"💾 Detailed analysis saved to: {detailed_output_path}")

    metrics_to_aggregate = {
        'self_grade_score_sum': ['count', 'mean', 'std', 'min', 'max'],
        'cosine_similarity_mean': ['mean', 'std'],
        'rouge_l_f1measure_mean': ['mean', 'std'],
        'avg_api_cost': ['mean', 'std'],
        'avg_latency_ms': ['mean', 'std']
    }
    
    valid_metrics_to_aggregate = {k: v for k, v in metrics_to_aggregate.items() if k in df.columns}

    if not valid_metrics_to_aggregate:
        print("❌ Error: None of the specified metrics for aggregation are present in the CSV file.")
        return

    category_stats = df.groupby('source_type').agg(valid_metrics_to_aggregate).round(4)

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE ANALYSIS: OPEN-SOURCE vs. CLOSED-SOURCE")
    print("=" * 80)
    
    print("\n📊 MODEL CATEGORIZATION:")
    open_source_in_data = df[df['source_type'] == 'Open-Source']['model'].unique()
    closed_source_in_data = df[df['source_type'] == 'Closed-Source']['model'].unique()
    print(f"• Open-Source Models Found ({len(open_source_in_data)}): {sorted(list(open_source_in_data))}")
    print(f"• Closed-Source Models Found ({len(closed_source_in_data)}): {sorted(list(closed_source_in_data))}")

    print("\n📈 SUMMARY STATISTICS (Means):")
    for metric in valid_metrics_to_aggregate.keys():
        if metric in category_stats.columns.levels[0]:
            open_mean = category_stats.loc['Open-Source', (metric, 'mean')] if 'Open-Source' in category_stats.index else np.nan
            closed_mean = category_stats.loc['Closed-Source', (metric, 'mean')] if 'Closed-Source' in category_stats.index else np.nan
            print(f"• Avg {metric}:")
            print(f"  - Open-Source: {open_mean:.4f}")
            print(f"  - Closed-Source: {closed_mean:.4f}")
            if pd.notna(open_mean) and pd.notna(closed_mean) and closed_mean != 0:
                 print(f"  - Difference (Open - Closed): {(open_mean - closed_mean):.4f}")
                 print(f"  - Ratio (Open / Closed): {(open_mean / closed_mean):.2f}x")
            elif pd.notna(open_mean) and pd.notna(closed_mean) and closed_mean == 0 and open_mean != 0:
                 print(f"  - Difference (Open - Closed): {(open_mean - closed_mean):.4f}")
                 print(f"  - Ratio (Open / Closed): Inf (Closed source mean is 0)")


    print("\n📋 DETAILED CATEGORY STATISTICS:")
    print(category_stats)

    summary_output_path = f"{RESULTS_BASE_PATH}opensource_vs_closedsource_category_summary.csv"
    flat_category_stats = category_stats.copy()
    flat_category_stats.columns = ['_'.join(col).strip() for col in flat_category_stats.columns.values]
    flat_category_stats.to_csv(summary_output_path)
    print(f"\n💾 Category summary saved to: {summary_output_path}")

    if 'self_grade_score_sum' in df.columns:
        open_source_scores = df[df['source_type'] == 'Open-Source']['self_grade_score_sum'].dropna()
        closed_source_scores = df[df['source_type'] == 'Closed-Source']['self_grade_score_sum'].dropna()

        if len(open_source_scores) > 1 and len(closed_source_scores) > 1:
            t_stat, p_value = ttest_ind(open_source_scores, closed_source_scores, equal_var=False) # Welch's t-test
            print(f"\n📊 STATISTICAL ANALYSIS (Self-Grade Score Sum):")
            print(f"• T-test statistic: {t_stat:.4f}")
            print(f"• P-value: {p_value:.6f}")
            print(f"• Statistical significance (p < 0.05): {'Yes' if p_value < 0.05 else 'No'}")
        else:
            print("\n📊 STATISTICAL ANALYSIS (Self-Grade Score Sum): Not enough data for t-test (need >1 sample in each group).")
    else:
        print("\n📊 STATISTICAL ANALYSIS (Self-Grade Score Sum): 'self_grade_score_sum' column not found.")


if __name__ == "__main__":
    OPEN_SOURCE_MODEL_NAMES = [
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "deepseek-r1-distill-llama-70b"
    ]
    input_csv_path = f"{RESULTS_BASE_PATH}overall_model_performance_summary.csv"
    
    analyze_model_performance_by_source(input_csv_path, OPEN_SOURCE_MODEL_NAMES)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80) 