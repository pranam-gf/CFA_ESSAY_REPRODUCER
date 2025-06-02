import pandas as pd
import numpy as np
RESULTS_BASE_PATH = "../../results/essay_analysis_charts/"

def generate_model_category_performance_summary():
    """
    Generate model category performance summary from the reasoning vs non-reasoning analysis.
    Reproduces the model_category_performance_summary.csv format.
    """
    MAX_POSSIBLE_SCORE = 149    
    input_file = f"{RESULTS_BASE_PATH}reasoning_vs_nonreasoning_analysis.csv"
    df = pd.read_csv(input_file)
    
    print("🔍 GENERATING MODEL CATEGORY PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"📊 Input data: {len(df)} models")
    print(f"📊 Categories: {df['model_category'].unique()}")
    
    category_stats = df.groupby('model_category').agg({
        'self_grade_score_sum': ['mean', 'max', 'min', 'std', 'count'],
        'cosine_similarity_mean': ['mean', 'std', 'count'],
        'rouge_l_f1measure_mean': ['mean', 'std', 'count']
    }).reset_index()
    
    category_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in category_stats.columns.values]    
    column_mapping = {
        'model_category': 'model_category',
        'self_grade_score_sum_mean': 'self_grade_score_sum_avg_across_models',
        'self_grade_score_sum_max': 'self_grade_score_sum_max_across_models',
        'self_grade_score_sum_min': 'self_grade_score_sum_min_across_models',
        'self_grade_score_sum_std': 'self_grade_score_sum_std_across_models',
        'self_grade_score_sum_count': 'models_in_category_count',
        'cosine_similarity_mean_mean': 'cosine_similarity_mean',
        'cosine_similarity_mean_std': 'cosine_similarity_std',
        'cosine_similarity_mean_count': 'cosine_similarity_count',
        'rouge_l_f1measure_mean_mean': 'rouge_l_f1measure_mean',
        'rouge_l_f1measure_mean_std': 'rouge_l_f1measure_std',
        'rouge_l_f1measure_mean_count': 'rouge_l_f1measure_count'
    }
    
    category_stats = category_stats.rename(columns=column_mapping)
    
    category_stats['self_grade_score_sum_avg_normalized'] = (
        category_stats['self_grade_score_sum_avg_across_models'] / MAX_POSSIBLE_SCORE
    ) * 100
    
    category_stats['self_grade_score_sum_max_normalized'] = (
        category_stats['self_grade_score_sum_max_across_models'] / MAX_POSSIBLE_SCORE
    ) * 100    
    numeric_columns = [
        'self_grade_score_sum_avg_across_models', 'self_grade_score_sum_max_across_models',
        'self_grade_score_sum_min_across_models', 'self_grade_score_sum_std_across_models',
        'self_grade_score_sum_avg_normalized', 'self_grade_score_sum_max_normalized',
        'cosine_similarity_mean', 'cosine_similarity_std',
        'rouge_l_f1measure_mean', 'rouge_l_f1measure_std'
    ]
    
    for col in numeric_columns:
        if col in category_stats.columns:
            category_stats[col] = category_stats[col].round(6)    
    category_stats = category_stats.sort_values(
        by='self_grade_score_sum_avg_across_models', 
        ascending=False
    )
    
    expected_columns = [
        'model_category',
        'self_grade_score_sum_avg_across_models',
        'self_grade_score_sum_max_across_models',
        'self_grade_score_sum_min_across_models',
        'self_grade_score_sum_std_across_models',
        'models_in_category_count',
        'self_grade_score_sum_avg_normalized',
        'self_grade_score_sum_max_normalized',
        'cosine_similarity_mean',
        'cosine_similarity_std',
        'cosine_similarity_count',
        'rouge_l_f1measure_mean',
        'rouge_l_f1measure_std',
        'rouge_l_f1measure_count'
    ]
    
    available_columns = [col for col in expected_columns if col in category_stats.columns]
    category_stats = category_stats[available_columns]
    output_file = f"{RESULTS_BASE_PATH}model_category_performance_summary.csv"
    category_stats.to_csv(output_file, index=False)
    
    print(f"\n✅ Generated model category performance summary:")
    print(category_stats.to_string(index=False))
    
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n📈 DETAILED BREAKDOWN:")
    for _, row in category_stats.iterrows():
        category = row['model_category']
        avg_score = row['self_grade_score_sum_avg_across_models']
        max_score = row['self_grade_score_sum_max_across_models']
        min_score = row['self_grade_score_sum_min_across_models']
        count = row['models_in_category_count']
        avg_normalized = row['self_grade_score_sum_avg_normalized']
        
        print(f"\n🔸 {category} Models ({int(count)} models):")
        print(f"   • Average Score: {avg_score:.2f} ({avg_normalized:.1f}% of max)")
        print(f"   • Score Range: {min_score:.0f} - {max_score:.0f}")
        print(f"   • Standard Deviation: {row['self_grade_score_sum_std_across_models']:.2f}")
        
        if 'cosine_similarity_mean' in row and pd.notna(row['cosine_similarity_mean']):
            print(f"   • Avg Cosine Similarity: {row['cosine_similarity_mean']:.4f}")
        if 'rouge_l_f1measure_mean' in row and pd.notna(row['rouge_l_f1measure_mean']):
            print(f"   • Avg ROUGE-L F1: {row['rouge_l_f1measure_mean']:.4f}")
    
    return category_stats

if __name__ == "__main__":
    try:
        summary = generate_model_category_performance_summary()
        print("\n" + "=" * 60)
        print("✅ MODEL CATEGORY SUMMARY GENERATION COMPLETE!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find input file - {str(e)}")
        print(f"Please ensure {RESULTS_BASE_PATH}reasoning_vs_nonreasoning_analysis.csv exists.")
    except Exception as e:
        print(f"❌ Error during generation: {str(e)}")
        import traceback
        traceback.print_exc() 