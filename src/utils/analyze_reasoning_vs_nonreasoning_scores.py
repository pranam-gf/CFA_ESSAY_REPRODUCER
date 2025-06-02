import pandas as pd
import numpy as np
from scipy.stats import ttest_ind 
CSV_BASE_PATH = "../../results/essay_analysis_charts/"
def categorize_models():
    """
    Categorize models into reasoning vs non-reasoning based on their capabilities and design.
    Updated to match the categorization from analyze_essay_results.py
    
    Returns:
        tuple: (reasoning_models, non_reasoning_models) - sets of model names
    """    
    reasoning_models = {
        'gemini-2.5-pro',  
        'gemini-2.5-flash',  
        'o3-mini',  
        'o4-mini',  
        'grok-3',  
        'grok-3-mini-beta-high-effort',  
        'deepseek-r1',   
    }
    non_reasoning_models = {
        'claude-opus-4',  
        'claude-sonnet-4',  
        'claude-3.7-sonnet',  
        'grok-3-mini-beta-low-effort',  
        'groq-llama-4-maverick', 'groq-llama-4-scout', 'groq-llama3.3-70b', 'groq-llama3.1-8b-instant',  
        'claude-3.5-haiku', 'claude-3.5-sonnet',  
        'gpt-4o', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',  
        'mistral-large-official',  
        'palmyra-fin-default',  
        'codestral-latest-official',  
        'groq-llama-guard-4'  
    }
    
    return reasoning_models, non_reasoning_models

def analyze_model_performance(csv_file_path):
    """
    Analyze model performance by reasoning capability.
    Args:
        csv_file_path (str): Path to the overall model performance summary CSV
    """    
    df = pd.read_csv(csv_file_path)
    reasoning_models, non_reasoning_models = categorize_models()
    df['model_category'] = df['model'].apply(
        lambda x: 'Reasoning' if x in reasoning_models else 'Non-Reasoning'
    )
    unrecognized_models = set(df['model']) - reasoning_models - non_reasoning_models
    if unrecognized_models:
        print(f"⚠️  Unrecognized models (will be treated as Non-Reasoning): {unrecognized_models}")
        df.loc[df['model'].isin(unrecognized_models), 'model_category'] = 'Non-Reasoning'
    
    category_stats = df.groupby('model_category').agg({
        'self_grade_score_sum': ['count', 'mean', 'std', 'min', 'max'],
        'self_grade_score_normalized': ['mean', 'std', 'min', 'max'],
        'cosine_similarity_mean': ['mean', 'std'],
        'rouge_l_f1measure_mean': ['mean', 'std']
    }).round(4)
    print("=" * 80)
    print("SELF-GRADE SCORE ANALYSIS: REASONING vs NON-REASONING MODELS")
    print("=" * 80)
    print("\n📊 MODEL CATEGORIZATION:")
    print(f"• Reasoning Models ({len(reasoning_models)}): {sorted(reasoning_models)}")
    print(f"• Non-Reasoning Models ({len(non_reasoning_models)}): {sorted(non_reasoning_models)}")
    
    print("\n📈 SUMMARY STATISTICS:")
    reasoning_df = df[df['model_category'] == 'Reasoning']
    non_reasoning_df = df[df['model_category'] == 'Non-Reasoning']
    
    reasoning_avg = reasoning_df['self_grade_score_sum'].mean()
    non_reasoning_avg = non_reasoning_df['self_grade_score_sum'].mean()
    
    print(f"• Reasoning Models Average Score: {reasoning_avg:.2f}")
    print(f"• Non-Reasoning Models Average Score: {non_reasoning_avg:.2f}")
    print(f"• Difference (Reasoning - Non-Reasoning): {reasoning_avg - non_reasoning_avg:.2f}")
    if non_reasoning_avg != 0: 
        print(f"• Percentage Improvement: {((reasoning_avg - non_reasoning_avg) / non_reasoning_avg * 100):.1f}%")
    else:
        print("• Percentage Improvement: N/A (Non-Reasoning Average Score is 0)")
    print("\n📋 DETAILED CATEGORY STATISTICS:")
    print(category_stats)
    
    print("\n🏆 TOP 5 PERFORMERS BY CATEGORY:")
    
    print("\nReasoning Models (by self_grade_score_sum):")
    top_reasoning = reasoning_df.nlargest(5, 'self_grade_score_sum')[['model', 'self_grade_score_sum', 'best_strategy']]
    for idx, (_, row) in enumerate(top_reasoning.iterrows(), 1):
        print(f"  {idx}. {row['model']}: {row['self_grade_score_sum']} (Strategy: {row['best_strategy']})")
    
    print("\nNon-Reasoning Models (by self_grade_score_sum):")
    top_non_reasoning = non_reasoning_df.nlargest(5, 'self_grade_score_sum')[['model', 'self_grade_score_sum', 'best_strategy']]
    for idx, (_, row) in enumerate(top_non_reasoning.iterrows(), 1):
        print(f"  {idx}. {row['model']}: {row['self_grade_score_sum']} (Strategy: {row['best_strategy']})")
    reasoning_scores = reasoning_df['self_grade_score_sum'].dropna().values 
    non_reasoning_scores = non_reasoning_df['self_grade_score_sum'].dropna().values 
    
    if len(reasoning_scores) > 1 and len(non_reasoning_scores) > 1: 
        t_stat, p_value = ttest_ind(reasoning_scores, non_reasoning_scores, equal_var=False) 
        print(f"\n📊 STATISTICAL ANALYSIS:")
        print(f"• T-test statistic: {t_stat:.4f}")
        print(f"• P-value: {p_value:.6f}")
        print(f"• Statistical significance (p < 0.05): {'Yes' if p_value < 0.05 else 'No'}")
    else:
        print("\n📊 STATISTICAL ANALYSIS: Not enough data for t-test (need >1 sample in each group with non-NaN scores).")

    output_file = f'{CSV_BASE_PATH}reasoning_vs_nonreasoning_analysis.csv'
    df_output = df[['model', 'model_category', 'self_grade_score_sum', 'self_grade_score_normalized', 
                    'best_strategy', 'cosine_similarity_mean', 'rouge_l_f1measure_mean']].copy()
    df_output.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    return df, reasoning_avg, non_reasoning_avg

if __name__ == "__main__":    
    input_csv_path = f"{CSV_BASE_PATH}overall_model_performance_summary.csv"
    try:
        df, reasoning_avg, non_reasoning_avg = analyze_model_performance(input_csv_path)
 
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file {input_csv_path}")
        print("Please ensure the file exists and try again.")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc() 