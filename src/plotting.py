"""
Functions for generating comparison charts for model performance.
"""
import logging
import os
import pandas as pd
import numpy as np 
from . import config
from .utils import ui_utils
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from typing import Dict, List, Union, Optional, Any
import matplotlib.cm as cm
from pathlib import Path
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)
INSPIRED_PALETTE = ["#4C72B0", "#DD8452", "#8C8C8C", "#595959", "#9370DB", "#57A057"]

sns.set_theme(
    style="ticks", 
    palette=INSPIRED_PALETTE, 
    font="sans-serif" 
)
plt.rcParams.update({
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], 
    "axes.labelsize": 11, 
    "axes.titlesize": 13, 
    "font.size": 11,      
    "legend.fontsize": 10,
    "xtick.labelsize": 10, 
    "ytick.labelsize": 10, 
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": False, 
    # "grid.color": "#E0E0E0", 
    "axes.edgecolor": "#333333", 
    "axes.linewidth": 1.2, 
    "axes.titlepad": 15, 
    "figure.facecolor": "white", 
    "savefig.facecolor": "white", 
    "xtick.direction": "out", 
    "ytick.direction": "out", 
    "xtick.major.size": 5, 
    "ytick.major.size": 5, 
    "xtick.major.width": 1.2, 
    "ytick.major.width": 1.2, 
    "xtick.minor.size": 3,  
    "ytick.minor.size": 3,  
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.bottom": True, 
    "ytick.left": True,   
    # "text.usetex": False, 
})



def _wrap_labels(ax, width, break_long_words=False):
    """Wraps labels on an axes object."""
    labels = []
    ticks = ax.get_xticks()
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=0)


def _safe_get(data_dict: Dict[str, Any], keys: List[str], default: Optional[Any] = None) -> Optional[Any]:
    for key in keys:
        if key in data_dict:
            return data_dict[key]
    return default


def _prepare_plot_data(all_model_run_summaries: dict) -> pd.DataFrame | None:
    """
    Prepares data from the summary dictionary into a pandas DataFrame suitable for plotting.

    Args:
        all_model_run_summaries: A dictionary containing summaries of model runs.
                                 Expected structure: {model_id: {strategy_name: {metrics...}}}

    Returns:
        A pandas DataFrame with columns like 'Model', 'Strategy', 'Metric', 'Value',
        or None if the input is empty or malformed.
    """
    plot_data = []
    
    numerical_metrics = [
        
        'accuracy', 'f1_score', 'precision', 'recall', 
        
        'avg_cosine_similarity', 'avg_self_grade_score',
        'avg_rouge_l_precision', 'avg_rouge_l_recall', 'avg_rouge_l_f1measure',
        
        'average_latency_ms', 'total_cost', 
        'total_input_tokens', 'total_output_tokens', 'total_tokens', 
        'avg_answer_length', 'total_run_time_s'
    ]

    if not all_model_run_summaries:
        logger.warning("No model run summaries provided for plotting.")
        return None

    for model_id, strategies in all_model_run_summaries.items():
        if not isinstance(strategies, dict):
            logger.warning(f"Expected dictionary of strategies for model '{model_id}', got {type(strategies)}. Skipping.")
            continue
        for strategy_name, combined_metrics in strategies.items():
            if not isinstance(combined_metrics, dict):
                logger.warning(f"Expected dictionary of combined metrics for model '{model_id}', strategy '{strategy_name}', got {type(combined_metrics)}. Skipping.")
                continue

            if "error" in combined_metrics:
                logger.warning(f"Run for model '{model_id}', strategy '{strategy_name}' encountered an error: {combined_metrics['error']}. Skipping metrics.")
                continue

            for metric_name in numerical_metrics:
                value = combined_metrics.get(metric_name)
                if isinstance(value, (int, float)):
                     plot_data.append({
                        'Model': model_id,
                        'Strategy': strategy_name, 
                        'Metric': metric_name,
                        'Value': float(value)
                    })
                elif value is not None:
                     logger.debug(f"Metric '{metric_name}' for {model_id}/{strategy_name} has non-numeric value '{value}'. Skipping for numerical plot.")


    if not plot_data:
        logger.warning("No valid data points extracted for plotting.")
        return None

    df = pd.DataFrame(plot_data)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df.dropna(subset=['Value'], inplace=True) 

    if df.empty:
        logger.warning("DataFrame is empty after processing and cleaning. No plots will be generated.")
        return None

    logger.info(f"Prepared DataFrame for plotting with {len(df)} valid rows.")
    return df

def _get_strategy_type(strategy_name: str) -> str:
    """Extracts the base strategy type from the full strategy name."""
    if "Self-Consistency" in strategy_name:
        return "SC-CoT"
    elif "Self-Discover" in strategy_name:
        return "Self-Discover"
    elif "Default" in strategy_name:
        return "Default"
    else:
        return "Unknown"

def _get_strategy_param(strategy_name: str) -> str:
    """Extracts the parameter (e.g., N=3) from the strategy name, if present."""
    if "N=3" in strategy_name:
        return "N=3"
    elif "N=5" in strategy_name:
        return "N=5"
    else:
        return "" 

def _plot_metric_by_strategy_comparison(df: pd.DataFrame, output_dir: Union[str, Path], metric: str):
    """Generates grouped bar chart comparing key strategies for each model using Seaborn."""
    metric_df = df[df['Metric'] == metric].copy()
    if metric_df.empty:
        logger.info(f"Skipping '{metric}' by strategy comparison plot: No data found for this metric.")
        return
    
    strategies_to_compare_explicit = [
        s for s in df['Strategy'].unique() 
        if "Default" in s or "Self-Discover" in s or ("Self-Consistency CoT" in s and "N=3" in s)
    ]
    if not strategies_to_compare_explicit:
         logger.info(f"Skipping '{metric}' by strategy comparison plot: No standard strategies found.")
         return

    df_comp = metric_df[metric_df['Strategy'].isin(strategies_to_compare_explicit)].copy()

    if df_comp.empty or df_comp['Model'].nunique() < 1:
        included_models = df_comp['Model'].unique().tolist() if not df_comp.empty else []
        logger.info(f"Skipping '{metric}' by strategy comparison plot: Not enough data for comparison across models. Found models: {included_models}")
        return
    
    metric_title = metric.replace('_', ' ').title()
    title = f'{metric_title} Comparison Across Strategies and Models'
    num_models = df_comp['Model'].nunique()
    num_strategies = df_comp['Strategy'].nunique()
    plt.figure(figsize=(max(10, num_models * 2), 6 + num_strategies * 0.5))
    current_palette = INSPIRED_PALETTE[:num_strategies] if num_strategies > 0 else INSPIRED_PALETTE[:1]
    ax = sns.barplot(data=df_comp, x='Model', y='Value', hue='Strategy', palette=current_palette)

    for p in ax.patches:
        height = p.get_height()
        try:
            if height == 0:
                label_text = '0'
            elif abs(height) < 0.001 and height != 0:
                label_text = f'{height:.2e}'
            elif abs(height) < 1:
                label_text = f'{height:.3f}'
            elif abs(height) < 100:
                label_text = f'{height:.2f}'
            else:
                label_text = f'{int(round(height))}'
        except TypeError:
            label_text = "N/A"

        ax.text(p.get_x() + p.get_width() / 2.,
                height + (ax.get_ylim()[1] * 0.01), 
                label_text,
                ha='center', 
                va='bottom',
                fontsize=plt.rcParams["font.size"] * 0.8)

    ax.set_xlabel("Model") 
    ax.set_ylabel(metric_title) 
    ax.set_title(title, fontsize=plt.rcParams["axes.titlesize"], pad=plt.rcParams["axes.titlepad"], loc='left', fontweight='bold')
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, color='lightgray')
    ax.set_axisbelow(True)

    if metric.lower() in ['accuracy', 'f1_score', 'precision', 'recall'] or \
       'rate' in metric.lower() or 'percentage' in metric.lower() or 'score' in metric.lower():
        current_max_val = df_comp['Value'].max() if not df_comp.empty else 1.0
        current_min_val = df_comp['Value'].min() if not df_comp.empty else 0.0
        plot_min_y = 0 if current_min_val >= 0 else current_min_val * 1.1 
        if metric.lower() in ['accuracy', 'f1_score', 'precision', 'recall']:            
            plot_max_y = max(1.05, current_max_val * 1.05 if current_max_val > 0 else 1.05)
        else:
            plot_max_y = current_max_val * 1.1 if current_max_val > 0 else 0.1 
            if current_max_val == 0 and current_min_val == 0 : plot_max_y = 0.1 
        ax.set_ylim(bottom=plot_min_y, top=plot_max_y)
    else:
        
        if not df_comp.empty and df_comp['Value'].min() >= 0:
            ax.set_ylim(bottom=0, top=df_comp['Value'].max() * 1.1 if df_comp['Value'].max() > 0 else None)
        elif not df_comp.empty:
            ax.set_ylim(top=df_comp['Value'].max() * 1.1 if df_comp['Value'].max() > 0 else None) 

    
    plt.legend(title='Strategy', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.xticks(rotation=45, ha="right", fontsize=plt.rcParams["xtick.labelsize"] * 0.9) 
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False, trim=False) 
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    chart_filename = output_path / f"comparison_{metric.lower()}_by_model_and_strategy.png" 
    plt.savefig(chart_filename, dpi=plt.rcParams["savefig.dpi"], bbox_inches='tight')
    plt.close()
    logger.info(f"Saved chart: {chart_filename}")

def _plot_sc_comparison(df: pd.DataFrame, output_dir: Union[str, Path], metric: str = 'avg_rouge_l_f1measure'):
    """
    Generates a bar chart comparing Self-Consistency (N=3 vs N=5) against Default
    for a specific metric (defaulting to ROUGE-L F1).
    """
    metric_df = df[df['Metric'] == metric].copy()
    if metric_df.empty:
        logger.info(f"Skipping SC '{metric}' comparison plot: No data found for this metric.")
        return

    metric_df['strategy_type'] = metric_df['Strategy'].apply(_get_strategy_type)
    metric_df['strategy_param'] = metric_df['Strategy'].apply(_get_strategy_param)
    metric_df['base_model_id'] = metric_df['Model']
    df_comp = metric_df[metric_df['strategy_type'] == 'SC-CoT'].copy()

    comparable_params = df_comp[df_comp['strategy_param'].str.contains(r'N=\d+', regex=True)]['strategy_param'].unique()
    if len(comparable_params) < 2:
        logger.info(f"Skipping SC '{metric}' comparison plot: Need results for at least two different N values (e.g., N=3 and N=5). Found: {comparable_params}")
        return
    
    df_comp = df_comp[df_comp['strategy_param'].isin(comparable_params)].copy()

    metric_title = metric.replace('_', ' ').title()
    title = f'Self-Consistency CoT {metric_title}: Comparison by Samples (N)'

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_comp, x='base_model_id', y='Value', hue='strategy_param',
                     errorbar=None) 

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8, padding=3)

    ax.set_xlabel("Base Model")
    ax.set_ylabel(metric_title)
    ax.set_title(title, loc='left', fontweight='bold') 

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylim(bottom=0)
    if metric == 'accuracy' or metric == 'f1_score' or metric == 'precision' or metric == 'recall':
         plt.ylim(top=max(1.05, df_comp['Value'].max() * 1.1 if not df_comp.empty else 1.05))
    else:
         plt.ylim(top=df_comp['Value'].max() * 1.15 if not df_comp.empty else None)
    
    ax.legend(title='Samples (N)', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 

    filename_base = f'comparison_sc_{metric}_n_samples'
    output_path = Path(output_dir) / f'{filename_base}.png'
    try:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {output_path}: {e}")
        plt.close()

def _plot_scatter_tradeoff(df: pd.DataFrame, output_dir: Union[str, Path], metric_y: str, metric_x: str):
    """Generates scatter plot showing a trade-off between two metrics using Seaborn."""
    try:
        df_pivot = df.pivot_table(index=['Model', 'Strategy'], columns='Metric', values='Value').reset_index()
    except Exception as e:
        logger.error(f"Failed to pivot DataFrame for scatter plot ({metric_y} vs {metric_x}): {e}. Columns: {df.columns}, Metrics: {df['Metric'].unique()}", exc_info=True)
        return

    if metric_x not in df_pivot.columns or metric_y not in df_pivot.columns:
        logger.warning(f"Skipping {metric_y} vs {metric_x} plot: One or both metrics not found after pivoting. Available: {df_pivot.columns.tolist()}")
        return

    df_plot = df_pivot.dropna(subset=[metric_x, metric_y]).copy()
    if df_plot.empty:
        logger.warning(f"No data points with both '{metric_y}' and '{metric_x}' available for scatter plot.")
        return

    df_plot['strategy_type'] = df_plot['Strategy'].apply(_get_strategy_type)
    df_plot['base_model_id'] = df_plot['Model']

    metric_y_title = metric_y.replace('_', ' ').title()
    metric_x_title = metric_x.replace('_', ' ').replace(' Ms', ' (ms)').replace(' S', ' (s)').title()
    if metric_x == 'total_cost':
        metric_x_title += " ($)"
    title = f'{metric_y_title} vs. {metric_x_title} Trade-off'

    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(
        data=df_plot,
        x=metric_x,
        y=metric_y,
        hue='base_model_id',
        style='strategy_type',
        s=100, 
        alpha=0.8,
        edgecolor='k', 
        linewidth=0.5
    )

    ax.set_xlabel(metric_x_title)
    ax.set_ylabel(metric_y_title)
    ax.set_title(title, loc='left', fontweight='bold')

    if metric_y in ['accuracy', 'f1_score', 'precision', 'recall']:
        min_y = df_plot[metric_y].min()
        max_y = df_plot[metric_y].max()
        ax.set_ylim(bottom=min_y * 0.95 if min_y > 0 else -0.05,
                    top=max(1.0, max_y * 1.05) if max_y < 1 else max_y * 1.05)
    
    ax.legend(title='Legend (Model / Strategy)', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    filename_base = f'tradeoff_{metric_y}_vs_{metric_x}'
    output_path = Path(output_dir) / f'{filename_base}.png'
    try:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {output_path}: {e}")
        plt.close()

def _plot_total_time_comparison(df: pd.DataFrame, output_dir: Union[str, Path]):
    """Plots a comparison of average latency across models and strategies using Seaborn."""
    time_df = df[df['Metric'] == 'average_latency_ms'].copy()
    if time_df.empty:
        logger.warning("No average latency data (average_latency_ms) found to plot time comparison.")
        return

    plt.figure(figsize=(12, 7))
    num_models = time_df['Model'].nunique()
    models = sorted(time_df['Model'].unique()) 
    
    ax = sns.barplot(data=time_df, x='Strategy', y='Value', hue='Model', hue_order=models,
                     dodge=True, errorbar=None) 

    plt.title('Average Latency Comparison Across Strategies and Models', loc='left', fontweight='bold') 
    plt.ylabel('Average Latency (ms)') 
    plt.xlabel('Strategy') 
    plt.xticks(rotation=30, ha='right') 
    plt.yticks() 

    legend = ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    for container in ax.containers:
        labels = [f'{v:.0f}' if v >= 1 else f'{v:.3f}' for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='edge', padding=3,
                     fontsize=plt.rcParams['xtick.labelsize'] - 1) 

    ax.set_ylim(bottom=0)
    sns.despine() 
    plt.tight_layout(rect=[0, 0, 0.88, 1]) 

    output_path = Path(output_dir) / "average_latency_comparison.png"
    try:
        plt.savefig(output_path, bbox_inches='tight') 
        plt.close()
        logger.info(f"Saved plot: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {output_path}: {e}")
        plt.close()

def _plot_metric_comparison_for_strategy(df: pd.DataFrame, strategy_name: str, metrics_to_plot: list[str], output_dir: Union[str, Path]):
    """
    Generates separate bar charts comparing models for a specific strategy using Seaborn.
    Args:
        df: DataFrame containing the prepared plot data.
        strategy_name: The specific strategy name to plot comparisons for.
        metrics_to_plot: A list of metric names (strings) to generate plots for.
        output_dir: The directory to save the generated plots.
    """
    strategy_df_full = df[df['Strategy'] == strategy_name].copy()
    if strategy_df_full.empty:
        logger.warning(f"No data found for strategy '{strategy_name}'. Skipping metric comparison plots.")
        return

    num_models = strategy_df_full['Model'].nunique()
    if num_models == 0:
        logger.info(f"No models found for strategy '{strategy_name}'. Skipping model comparison plots.")
        return

    models = sorted(strategy_df_full['Model'].unique()) 

    for metric in metrics_to_plot:
        metric_df = strategy_df_full[strategy_df_full['Metric'] == metric]
        if metric_df.empty:
            logger.warning(f"No data found for metric '{metric}' in strategy '{strategy_name}'. Skipping plot.")
            continue

        plt.figure(figsize=(max(6, num_models * 1.5), 5)) 

        ax = sns.barplot(data=metric_df, x='Model', y='Value', order=models,
                         hue='Model', legend=False, 
                         errorbar=None) 

        metric_title = metric.replace('_', ' ').title()
        
        if metric == 'total_cost':
            ylabel = f'{metric_title} ($)'
            label_fmt = '${:,.3f}'
        elif metric == 'average_latency_ms':
            ylabel = f'{metric_title} (ms)'
            label_fmt = '{:.0f}'
        elif metric == 'total_run_time_s':
            ylabel = f'{metric_title} (s)'
            label_fmt = '{:.1f}s'
        elif metric == 'total_tokens':
            ylabel = f'{metric_title} (tokens)'
            label_fmt = '{:,.0f}'
        else:
            ylabel = metric_title
            label_fmt = '{:.3f}'

        plot_title = f"{metric_title} for {strategy_name}"
        
        plt.title(plot_title, loc='left', fontweight='bold', wrap=True) 
        plt.ylabel(ylabel) 
        plt.xlabel('Model') 
        plt.xticks(rotation=45, ha="right", fontsize=plt.rcParams["xtick.labelsize"] * 0.9) 
        plt.yticks() 
        
        for container in ax.containers:
            labels = [label_fmt.format(v) for v in container.datavalues]
            ax.bar_label(container, labels=labels,
                         fontsize=plt.rcParams['xtick.labelsize'] -1, padding=3) 

        if metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            ax.set_ylim(bottom=0, top=max(1.05, metric_df['Value'].max() * 1.1))
        elif metric_df['Value'].min() >= 0:
             ax.set_ylim(bottom=0, top=metric_df['Value'].max() * 1.15 if metric_df['Value'].max() > 0 else 0.1)

        if metric_df['Value'].max() == 0:
             ax.set_ylim(bottom=-0.001, top=0.01)
             ax.set_yticks([0])
        
        sns.despine()
        plt.tight_layout()
        safe_strategy_name = strategy_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('/', '')
        output_path = Path(output_dir) / f"{safe_strategy_name}_strategy_{metric}_comparison.png"
        try:
            plt.savefig(output_path, bbox_inches='tight') 
            plt.close()
            logger.info(f"Saved plot: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save plot {output_path}: {e}")
            plt.close()

def _plot_confusion_matrix(matrix: Union[np.ndarray, List[List[int]]], labels: List[str], model_id: str, strategy_name: str, output_dir: Union[str, Path]):
    """Generates and saves a confusion matrix heatmap using Seaborn."""
    output_path = Path(output_dir) / f"confusion_matrix_{model_id}_{strategy_name.replace(' ', '_')}.png"

    if isinstance(matrix, list):
        matrix = np.array(matrix)

    if matrix.size == 0 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(labels):
        logger.error(f"Invalid matrix or labels for confusion matrix: {model_id}/{strategy_name}. Matrix shape: {matrix.shape}, Labels: {len(labels)}. Skipping plot.")
        return

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 10})
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    ax.set_title(f'Confusion Matrix: {model_id} ({strategy_name})', fontsize=12, loc='left', fontweight='bold') 
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    try:
        plt.savefig(output_path)
        plt.close() 
        logger.info(f"Saved confusion matrix: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save confusion matrix {output_path}: {e}")
        plt.close()

def _prepare_detailed_data_for_plots(all_model_runs_summary: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Prepares detailed per-item evaluation data from all model runs into a long-form pandas DataFrame.

    Args:
        all_model_runs_summary: A dictionary containing summaries of model runs.
                                 Expected structure: 
                                 {model_id: {strategy_name: {"detailed_results": [{metric_item_1}, ...]}}}

    Returns:
        A pandas DataFrame with columns like 'Model', 'Strategy', 'question_id', 
        'cosine_similarity', 'self_grade_score', etc., for each item.
        Returns None if no detailed data can be extracted.
    """
    detailed_plot_data = []
    if not all_model_runs_summary:
        logger.warning("No model run summaries provided for detailed plotting.")
        return None

    for model_id, strategies in all_model_runs_summary.items():
        if not isinstance(strategies, dict):
            logger.warning(f"Expected dictionary of strategies for model '{model_id}', got {type(strategies)}. Skipping for detailed plot.")
            continue
        for strategy_name, metrics_data in strategies.items():
            if not isinstance(metrics_data, dict):
                logger.warning(f"Expected dictionary of metrics data for model '{model_id}', strategy '{strategy_name}', got {type(metrics_data)}. Skipping for detailed plot.")
                continue
            
            detailed_results_list = metrics_data.get("detailed_results")
            if not isinstance(detailed_results_list, list):
                logger.debug(f"No 'detailed_results' list found for {model_id}/{strategy_name}. Skipping.")
                continue

            for item_result in detailed_results_list:
                if not isinstance(item_result, dict):
                    logger.warning(f"Item in 'detailed_results' for {model_id}/{strategy_name} is not a dict: {item_result}. Skipping.")
                    continue
                
                
                
                item_data_with_ids = item_result.copy()
                item_data_with_ids['Model'] = model_id
                item_data_with_ids['Strategy'] = strategy_name
                
                
                
                item_data_with_ids['question_identifier'] = item_result.get('question_hash', item_result.get('question', 'unknown_question'))

                detailed_plot_data.append(item_data_with_ids)

    if not detailed_plot_data:
        logger.warning("No valid detailed data points extracted for plotting.")
        return None

    df_detailed = pd.DataFrame(detailed_plot_data)
    
    numeric_cols = ['cosine_similarity', 'self_grade_score', 
                    'rouge_l_precision', 'rouge_l_recall', 'rouge_l_f1measure']
    for col in numeric_cols:
        if col in df_detailed.columns:
            df_detailed[col] = pd.to_numeric(df_detailed[col], errors='coerce')
    
    if df_detailed.empty:
        logger.warning("Detailed DataFrame is empty after processing. No detailed plots will be generated.")
        return None

    logger.info(f"Prepared detailed DataFrame for plotting with {len(df_detailed)} rows.")
    return df_detailed


def plot_cosine_similarity_distribution(detailed_df: pd.DataFrame, output_dir: Union[str, Path]):
    """
    Generates a KDE plot for the distribution of Cosine Similarity scores.
    Uses hue for Model and separate plots (facets) for Strategy if multiple.
    """
    if detailed_df is None or detailed_df.empty or 'cosine_similarity' not in detailed_df.columns:
        logger.warning("Skipping Cosine Similarity distribution plot: No valid data.")
        return

    plot_df = detailed_df.dropna(subset=['cosine_similarity']).copy()
    if plot_df.empty:
        logger.warning("Skipping Cosine Similarity distribution plot: No non-NaN cosine similarity data.")
        return
    
    num_models = plot_df['Model'].nunique()
    num_strategies = plot_df['Strategy'].nunique()
        
    if num_strategies > 1:
        g = sns.FacetGrid(plot_df, col="Strategy", hue="Model", col_wrap=min(3, num_strategies), sharey=False, sharex=True, height=4, aspect=1.2)
        g.map(sns.kdeplot, "cosine_similarity", fill=True, alpha=.5, warn_singular=False)
        g.add_legend(title='Model')
        g.set_axis_labels("Cosine Similarity Score", "Density")
        g.set_titles("{col_name}") 
        main_title = "Distribution of Cosine Similarity Scores by Strategy"
    elif num_models > 0 : 
        plt.figure(figsize=(8, 5))
        ax = sns.kdeplot(data=plot_df, x="cosine_similarity", hue="Model", fill=True, alpha=.5, warn_singular=False)
        ax.set_xlabel("Cosine Similarity Score")
        ax.set_ylabel("Density")
        main_title = f"Distribution of Cosine Similarity Scores ({plot_df['Strategy'].iloc[0] if num_strategies == 1 else ''})" 
        plt.legend(title='Model')
    else: 
        logger.warning("Not enough diversity in data to plot cosine similarity distribution effectively.")
        return
        
    plt.suptitle(main_title, y=1.03, fontsize=plt.rcParams["axes.titlesize"] + 2) 
    plt.tight_layout(rect=[0, 0, 1, 0.98]) 

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_filename = output_path / "dist_cosine_similarity.png"
    plt.savefig(chart_filename, dpi=plt.rcParams["savefig.dpi"])
    plt.close()
    logger.info(f"Saved chart: {chart_filename}")

def plot_correlation_cosine_self_eval(detailed_df: pd.DataFrame, output_dir: Union[str, Path]):
    """
    Generates a scatter plot correlating Cosine Similarity with Self-Evaluation Scores.
    Includes a regression line and Pearson correlation coefficient.
    Uses hue for Model and facets for Strategy if multiple.
    """
    if detailed_df is None or detailed_df.empty or \
       'cosine_similarity' not in detailed_df.columns or \
       'self_grade_score' not in detailed_df.columns:
        logger.warning("Skipping Cosine vs. Self-Grade correlation plot: Missing required columns (cosine_similarity, self_grade_score) or empty DataFrame.")
        return

    plot_df = detailed_df[['Model', 'Strategy', 'cosine_similarity', 'self_grade_score']].copy()
    plot_df.dropna(subset=['cosine_similarity', 'self_grade_score'], inplace=True)

    if plot_df.empty:
        logger.warning("Skipping Cosine vs. Self-Grade correlation plot: DataFrame is empty after dropping NaNs.")
        return
    
    if 'Model' not in plot_df.columns:
        plot_df['Model'] = "Unknown Model"
    if 'Strategy' not in plot_df.columns:
        plot_df['Strategy'] = "Unknown Strategy"
        
    def calculate_corr(data, label_x, label_y, color, **kwargs):
        ax = plt.gca()
        
        valid_data = data.dropna(subset=[label_x, label_y])
        if not valid_data.empty and len(valid_data) > 1:
            try:
                r, p_value = pearsonr(valid_data[label_x], valid_data[label_y])
                ax.text(0.05, 0.95, f'r = {r:.2f}\\np = {p_value:.2g}',
                        transform=ax.transAxes, ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
                        fontsize=9) 
            except ValueError as e: 
                 ax.text(0.05, 0.95, f'r = N/A\\n(Error: {e})',
                        transform=ax.transAxes, ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
                        fontsize=9)
        else:
            ax.text(0.05, 0.95, 'r = N/A\\n(Insufficient data)',
                    transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
                    fontsize=9)
            
    unique_models = plot_df['Model'].nunique()
    unique_strategies = plot_df['Strategy'].nunique()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    base_filename = "correlation_cosine_vs_self_eval"
    chart_filename = ""

    try:
        if unique_models > 1 or unique_strategies > 1:
            g = sns.FacetGrid(plot_df, col="Strategy", hue="Model", col_wrap=min(2, unique_strategies), sharex=True, sharey=True, height=5, aspect=1.2)
            g.map_dataframe(sns.regplot, x="cosine_similarity", y="self_grade_score", scatter_kws={'alpha':0.5}, ci=95)
            g.map_dataframe(calculate_corr, label_x="cosine_similarity", label_y="self_grade_score") 
            
            g.set_axis_labels("Cosine Similarity", "Self-Evaluation Score (1-10)")
            g.set_titles(col_template="{col_name} Strategy")
            g.add_legend(title='Model', loc='upper right', bbox_to_anchor=(1, 1.05)) 
            plt.subplots_adjust(top=0.9) 
            g.fig.suptitle('Cosine Similarity vs. LLM Self-Evaluation', fontsize=16, y=0.98) 
            chart_filename = output_path / f"{base_filename}_faceted.png"

        else: 
            plt.figure(figsize=(8, 6)) 
            ax = sns.regplot(data=plot_df, x="cosine_similarity", y="self_grade_score", scatter_kws={'alpha':0.5}, ci=95)
            
            
            if not plot_df.empty and len(plot_df) > 1:
                try:
                    r, p_value = pearsonr(plot_df["cosine_similarity"].dropna(), plot_df["self_grade_score"].dropna())
                    ax.text(0.05, 0.95, f'r = {r:.2f}\\np = {p_value:.2g}',
                            transform=ax.transAxes, ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
                            fontsize=10) 
                except ValueError as e:
                     ax.text(0.05, 0.95, f'r = N/A\\n(Error: {e})',
                            transform=ax.transAxes, ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
                            fontsize=10)
            else:
                ax.text(0.05, 0.95, 'r = N/A\\n(Insufficient data)',
                        transform=ax.transAxes, ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
                        fontsize=10)

            model_name = plot_df['Model'].iloc[0]
            strategy_name = plot_df['Strategy'].iloc[0]
            plt.title(f'Cosine Similarity vs. Self-Evaluation\\n({model_name} - {strategy_name})', fontsize=14) 
            plt.xlabel("Cosine Similarity", fontsize=12) 
            plt.ylabel("Self-Evaluation Score (1-10)", fontsize=12) 
            plt.tight_layout()
            chart_filename = output_path / f"{base_filename}_{model_name.replace(' ', '_')}_{strategy_name.replace(' ', '_')}.png"

        plt.savefig(chart_filename, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved chart: {chart_filename}")

    except Exception as e:
        logger.error(f"Error generating cosine vs. self-grade correlation plot: {e}", exc_info=True)

def plot_error_heatmap_cosine_self_eval(detailed_df: pd.DataFrame, output_dir: Union[str, Path]):
    """
    Generates a heatmap showing the joint distribution of binned Cosine Similarity and Self-Grade Scores.
    Facets by Strategy if multiple strategies are present.
    """
    if detailed_df is None or detailed_df.empty or \
       'cosine_similarity' not in detailed_df.columns or \
       'self_grade_score' not in detailed_df.columns:
        logger.warning("Skipping Error Heatmap (Cosine vs. Self-Grade): Missing required columns.")
        return

    plot_df = detailed_df.dropna(subset=['cosine_similarity', 'self_grade_score']).copy()
    if plot_df.empty:
        logger.warning("Skipping Error Heatmap: No non-NaN data for both cosine similarity and self-grade scores.")
        return
    cosine_bins = np.linspace(0, 1, 6) 
    self_grade_bins = np.arange(0.5, 11, 2) 
    plot_df['cosine_bin'] = pd.cut(plot_df['cosine_similarity'], bins=cosine_bins, include_lowest=True, right=True)
    plot_df['self_grade_bin'] = pd.cut(plot_df['self_grade_score'], bins=self_grade_bins, include_lowest=True, right=True)
    num_strategies = plot_df['Strategy'].unique().shape[0]
    max_count_overall = 0
    if num_strategies > 1:
        for strategy_name, group_df in plot_df.groupby('Strategy'):
            heatmap_data = group_df.groupby(['self_grade_bin', 'cosine_bin'], observed=False).size().unstack(fill_value=0)
            if not heatmap_data.empty:
                max_count_overall = max(max_count_overall, heatmap_data.max().max())
    else:
        heatmap_data_single = plot_df.groupby(['self_grade_bin', 'cosine_bin'], observed=False).size().unstack(fill_value=0)
        if not heatmap_data_single.empty:
            max_count_overall = heatmap_data_single.max().max()
    
    vmax = max_count_overall if max_count_overall > 0 else None 

    if num_strategies > 1:
        unique_strategies = sorted(plot_df['Strategy'].unique())
        num_cols_facet = min(2, len(unique_strategies)) 
        num_rows_facet = (len(unique_strategies) + num_cols_facet - 1) // num_cols_facet
        fig_height = 5 * num_rows_facet
        fig_width = 6 * num_cols_facet
        fig, axes = plt.subplots(num_rows_facet, num_cols_facet, figsize=(fig_width, fig_height), squeeze=False)
        axes = axes.flatten() 

        for i, strategy_name in enumerate(unique_strategies):
            ax = axes[i]
            strategy_df = plot_df[plot_df['Strategy'] == strategy_name]
            if strategy_df.empty:
                ax.set_title(f"{strategy_name}\n(No data)", fontsize=10)
                ax.axis('off')
                continue

            heatmap_data = strategy_df.groupby(['self_grade_bin', 'cosine_bin'], observed=False).size().unstack(fill_value=0)
            heatmap_data = heatmap_data.sort_index(ascending=False)
            
            sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="viridis", ax=ax, linewidths=.5, cbar=True, vmax=vmax)
            ax.set_title(strategy_name, fontsize=10)
            ax.set_xlabel("Cosine Similarity Bin")
            ax.set_ylabel("Self-Grade Score Bin")
        
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        main_title = "Error Heatmap: Cosine Similarity vs. Self-Grade Score"
        plt.suptitle(main_title, y=1.0, fontsize=plt.rcParams["axes.titlesize"] + 2)
        plt.tight_layout(rect=[0, 0, 1, 0.97]) 

    else: 
        plt.figure(figsize=(7, 6))
        ax = plt.gca()
        heatmap_data = plot_df.groupby(['self_grade_bin', 'cosine_bin'], observed=False).size().unstack(fill_value=0)
        heatmap_data = heatmap_data.sort_index(ascending=False)

        sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="viridis", ax=ax, linewidths=.5, cbar=True, vmax=vmax)
        strategy_name_title = plot_df['Strategy'].iloc[0] if not plot_df['Strategy'].empty else ""
        main_title = f"Error Heatmap: Cosine vs. Self-Grade ({strategy_name_title})"
        plt.title(main_title, fontsize=plt.rcParams["axes.titlesize"] + 2)
        ax.set_xlabel("Cosine Similarity Bin")
        ax.set_ylabel("Self-Grade Score Bin")
        plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_filename = output_path / "heatmap_cosine_vs_self_grade.png"
    plt.savefig(chart_filename, dpi=plt.rcParams["savefig.dpi"])
    plt.close()
    logger.info(f"Saved chart: {chart_filename}")


def plot_self_grading_calibration_curve(detailed_df: pd.DataFrame, output_dir: Union[str, Path], performance_metric: str = 'cosine_similarity'):
    """
    Generates a calibration curve for LLM self-grading scores against a specified performance metric.

    Args:
        detailed_df: DataFrame with detailed per-item results, including 'self_grade_score'
                     and the `performance_metric` column (e.g., 'cosine_similarity').
        output_dir: Directory to save the plot.
        performance_metric: The column name to use as the measure of actual performance.
    """
    if detailed_df is None or detailed_df.empty or \
       'self_grade_score' not in detailed_df.columns or \
       performance_metric not in detailed_df.columns:
        logger.warning(f"Skipping Self-Grading Calibration Curve: Missing 'self_grade_score' or '{performance_metric}'.")
        return

    plot_df = detailed_df.dropna(subset=['self_grade_score', performance_metric]).copy()
    if plot_df.empty:
        logger.warning(f"Skipping Self-Grading Calibration Curve: No non-NaN data for scores and '{performance_metric}'.")
        return
    
    bins = np.arange(0.5, 11.5, 1) 
    labels = [str(int(i)) for i in np.arange(1, 11, 1)] 
    plot_df['self_grade_bin'] = pd.cut(plot_df['self_grade_score'], bins=bins, labels=labels, right=True, include_lowest=True)
    
    calibration_data = plot_df.groupby(['Strategy', 'Model', 'self_grade_bin'], observed=False)[performance_metric].agg(
        mean_performance='mean',
        std_performance='std',
        count='count'
    ).reset_index()
    
    calibration_data['ci_lower'] = calibration_data['mean_performance'] - 1.96 * (calibration_data['std_performance'] / np.sqrt(calibration_data['count']))
    calibration_data['ci_upper'] = calibration_data['mean_performance'] + 1.96 * (calibration_data['std_performance'] / np.sqrt(calibration_data['count']))
    
    calibration_data.loc[calibration_data['count'] <= 1, ['ci_lower', 'ci_upper']] = calibration_data.loc[calibration_data['count'] <= 1, 'mean_performance']
    calibration_data.dropna(subset=['mean_performance'], inplace=True) 

    if calibration_data.empty:
        logger.warning(f"Skipping Self-Grading Calibration Curve: No data after binning for '{performance_metric}'.")
        return

    num_strategies = calibration_data['Strategy'].nunique()
    performance_metric_title = performance_metric.replace('_',' ').title()

    if num_strategies > 1:
        g = sns.FacetGrid(calibration_data, col="Strategy", hue="Model", col_wrap=min(3, num_strategies), sharey=True, height=4, aspect=1.3)
        g.map_dataframe(lambda data, color: plt.plot(data['self_grade_bin'].astype(str), data['mean_performance'], marker='o', label=data['Model'].iloc[0]))
        def plot_with_ci(data, **kwargs):
            ax = plt.gca()
            model_name = data['Model'].iloc[0] 
            ax.plot(data['self_grade_bin'].astype(str), data['mean_performance'], marker='o', label=model_name, color=kwargs.get('color'))
            ax.fill_between(data['self_grade_bin'].astype(str), data['ci_lower'], data['ci_upper'], alpha=0.2, color=kwargs.get('color'))

        g.map_dataframe(plot_with_ci) 
        g.add_legend(title='Model')
        g.set_axis_labels("Self-Grade Score Bin (1-10)", f"Mean {performance_metric_title}")
        g.set_titles("{col_name}")
        main_title = f"LLM Self-Grading Calibration ({performance_metric_title})"

    else: 
        plt.figure(figsize=(9, 6))
        ax = plt.gca()
        for model_name, model_df in calibration_data.groupby('Model'):
            ax.plot(model_df['self_grade_bin'].astype(str), model_df['mean_performance'], marker='o', label=model_name)
            ax.fill_between(model_df['self_grade_bin'].astype(str), model_df['ci_lower'], model_df['ci_upper'], alpha=0.2)
        
        ax.set_xlabel("Self-Grade Score Bin (1-10)")
        ax.set_ylabel(f"Mean {performance_metric_title}")
        strategy_name_title = calibration_data['Strategy'].iloc[0] if not calibration_data['Strategy'].empty else ""
        main_title = f"LLM Self-Grading Calibration ({performance_metric_title}) - {strategy_name_title}"
        plt.title(main_title)
        plt.legend(title='Model')
        plt.xticks(rotation=45, ha='right')
        sns.despine()

    plt.suptitle(main_title, y=1.02, fontsize=plt.rcParams["axes.titlesize"] + 1) 
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_filename = output_path / f"calibration_self_grade_vs_{performance_metric}.png"
    plt.savefig(chart_filename, dpi=plt.rcParams["savefig.dpi"])
    plt.close()
    logger.info(f"Saved chart: {chart_filename}")

def plot_radar_evaluation_metrics(df_summary: pd.DataFrame, output_dir: Union[str, Path], 
                                  metrics_for_radar: Optional[List[str]] = None, 
                                  top_n_models: Optional[int] = 5):
    """
    Generates a radar chart comparing models (or model-strategy pairs) across multiple evaluation dimensions.
    Normalizes metrics to a [0-1] scale for comparability.

    Args:
        df_summary: DataFrame with aggregated summary metrics (from _prepare_plot_data).
        output_dir: Directory to save the plot.
        metrics_for_radar: List of metric names (from df_summary columns) to use as radar axes.
                           If None, a default set is used.
        top_n_models: If there are many models/strategies, limit to top N based on a primary metric (e.g., ROUGE-L F1).
                      Set to None to include all. For simplicity, this example might plot all available.
    """
    if df_summary is None or df_summary.empty:
        logger.warning("Skipping Radar Chart: No summary data provided.")
        return

    if metrics_for_radar is None:
        metrics_for_radar = [
            'avg_cosine_similarity', 'avg_self_grade_score', 
            'avg_rouge_l_f1measure', 'avg_rouge_l_precision', 'avg_rouge_l_recall'
        ]

    try:
        pivot_df = df_summary.pivot_table(index=['Model', 'Strategy'], columns='Metric', values='Value')
    except Exception as e:
        logger.error(f"Failed to pivot summary DataFrame for radar chart: {e}", exc_info=True)
        return
    
    relevant_metrics = [m for m in metrics_for_radar if m in pivot_df.columns]
    if not relevant_metrics:
        logger.warning(f"Skipping Radar Chart: None of the specified metrics for radar are available in the summary data. Tried: {metrics_for_radar}")
        return
    
    radar_df = pivot_df[relevant_metrics].dropna()
    if radar_df.empty:
        logger.warning("Skipping Radar Chart: No data remains after selecting metrics and dropping NaNs.")
        return
    
    normalized_df = radar_df.copy()
    for metric in relevant_metrics:
        min_val = radar_df[metric].min()
        max_val = radar_df[metric].max()
        if max_val == min_val: 
            normalized_df[metric] = 0.5 
        else:
            normalized_df[metric] = (radar_df[metric] - min_val) / (max_val - min_val)
    

    if 'avg_self_grade_score' in relevant_metrics and 'avg_self_grade_score' in radar_df.columns:
        min_val_sg = 1.0
        max_val_sg = 10.0
        actual_min_sg = radar_df['avg_self_grade_score'].min() if radar_df['avg_self_grade_score'].notna().any() else min_val_sg
        actual_max_sg = radar_df['avg_self_grade_score'].max() if radar_df['avg_self_grade_score'].notna().any() else max_val_sg
        
        if actual_max_sg == actual_min_sg:
            normalized_df['avg_self_grade_score'] = 0.5
        else:
            normalized_df['avg_self_grade_score'] = (radar_df['avg_self_grade_score'] - actual_min_sg) / (actual_max_sg - actual_min_sg)

    categories = list(normalized_df.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] 
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))    
    category_labels_pretty = [cat.replace('avg_', '').replace('_', ' ').title() for cat in categories]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(category_labels_pretty)
    ax.set_yticks(np.linspace(0, 1, 5)) 
    ax.set_yticklabels([f"{val:.1f}" for val in np.linspace(0, 1, 5)])
    ax.set_ylim(0, 1)
    color_cycle = plt.cm.get_cmap('tab10', len(normalized_df.index)) 

    for i, (idx, row) in enumerate(normalized_df.iterrows()):
        data = row.values.flatten().tolist()
        data += data[:1] 
        label = f"{idx[0]} ({idx[1]})" 
        ax.plot(angles, data, linewidth=1.5, linestyle='solid', label=label, color=color_cycle(i))
        ax.fill(angles, data, color=color_cycle(i), alpha=0.2)

    plt.title("Radar Chart: Multi-Metric Performance Comparison", size=16, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True, fontsize=8) 
    
    handles, labels = ax.get_legend_handles_labels()
    
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=min(3, len(labels)), fontsize=8)

    plt.tight_layout(pad=2.0) 

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_filename = output_path / "radar_metrics_comparison.png"
    plt.savefig(chart_filename, dpi=plt.rcParams["savefig.dpi"], bbox_inches='tight')
    plt.close()
    logger.info(f"Saved chart: {chart_filename}")


def generate_all_charts(all_model_run_summaries: dict, charts_output_dir: Union[str, Path]):
    """
    Generates all comparison charts based on the summary data using Matplotlib/Seaborn.

    Args:
        all_model_run_summaries: Dictionary containing metric summaries for different models and strategies.
                                 This now also includes "detailed_results" for per-item scores.
        charts_output_dir: The directory path to save the generated charts.
    """
    output_dir = Path(charts_output_dir) 
    output_dir.mkdir(parents=True, exist_ok=True)
    df_summary = _prepare_plot_data(all_model_run_summaries)
    df_detailed = _prepare_detailed_data_for_plots(all_model_run_summaries)

    if (df_summary is None or df_summary.empty) and (df_detailed is None or df_detailed.empty):
        logger.warning("Plotting skipped: No valid summary or detailed data prepared.")
        ui_utils.print_warning("Plotting skipped: No valid summary or detailed data prepared.")
        return
    
    logger.info("Generating plots using Matplotlib/Seaborn...")

    if df_detailed is not None and not df_detailed.empty:
        logger.info("Generating detailed distribution plots...")
        try:
            plot_cosine_similarity_distribution(df_detailed, output_dir)
        except Exception as e:
            logger.error(f"Error generating cosine similarity distribution plot: {e}", exc_info=True)
        
        logger.info("Generating detailed correlation plots...")
        try:
            plot_correlation_cosine_self_eval(df_detailed, output_dir)
        except Exception as e:
            logger.error(f"Error generating cosine vs. self-grade correlation plot: {e}", exc_info=True)

        logger.info("Generating detailed heatmaps...")
        try:
            plot_error_heatmap_cosine_self_eval(df_detailed, output_dir)
        except Exception as e:
            logger.error(f"Error generating cosine vs. self-grade heatmap: {e}", exc_info=True)

        logger.info("Generating self-grading calibration curves...")
        try:
            plot_self_grading_calibration_curve(df_detailed, output_dir, performance_metric='cosine_similarity')
            
            if 'rouge_l_f1measure' in df_detailed.columns:
                 plot_self_grading_calibration_curve(df_detailed, output_dir, performance_metric='rouge_l_f1measure')
        except Exception as e:
            logger.error(f"Error generating self-grading calibration curve: {e}", exc_info=True)

        
        if 'question_type' in df_detailed.columns:
            logger.info("'question_type' column found, will attempt to generate quality by type plot.")
            
        else:
            logger.info("'question_type' column not found in detailed data. Skipping 'Answer Quality vs. Question Type' plot.")

    else:
        logger.info("Skipping detailed plots as no detailed data was prepared.")
    
    if df_summary is None or df_summary.empty:
        logger.info("Skipping aggregated plots as no summary data was prepared.")
    else:
        logger.info("Generating aggregated summary plots...") 
        primary_metric = 'avg_rouge_l_f1measure'
        latency_metric = 'average_latency_ms'
        cost_metric = 'total_cost'
        logger.info("Generating strategy comparison plots...")
        key_metrics_for_strategy_comparison = [primary_metric, 'avg_rouge_l_precision', 'avg_rouge_l_recall', latency_metric, cost_metric, 'avg_cosine_similarity', 'avg_self_grade_score']
        available_metrics_for_plot = df_summary['Metric'].unique()

        for metric in key_metrics_for_strategy_comparison:
            if metric in available_metrics_for_plot:
                logger.info(f"Generating strategy comparison plot for: {metric}")
                _plot_metric_by_strategy_comparison(df_summary, output_dir, metric)
            else:
                logger.info(f"Skipping strategy comparison plot for '{metric}': Metric not found in prepared data.")

        logger.info("Generating SC-CoT N sample comparison plots...")
        sc_metrics = [primary_metric, latency_metric, cost_metric, 'avg_cosine_similarity']
        for metric in sc_metrics:
            if metric in available_metrics_for_plot:
                _plot_sc_comparison(df_summary, output_dir, metric=metric)
            else:
                logger.info(f"Skipping SC comparison plot for '{metric}': Metric not found in prepared data.")

        logger.info("Generating trade-off scatter plots...")
        if primary_metric in available_metrics_for_plot and latency_metric in available_metrics_for_plot:
            _plot_scatter_tradeoff(df_summary, output_dir, metric_y=primary_metric, metric_x=latency_metric)
        else:
            logger.warning(f"Skipping {primary_metric} vs {latency_metric} scatter plot: One or both metrics not found.")
        
        if primary_metric in available_metrics_for_plot and cost_metric in available_metrics_for_plot:
            _plot_scatter_tradeoff(df_summary, output_dir, metric_y=primary_metric, metric_x=cost_metric)    
        else:
            logger.warning(f"Skipping {primary_metric} vs {cost_metric} scatter plot: One or both metrics not found.")

        if latency_metric in available_metrics_for_plot and cost_metric in available_metrics_for_plot:
            _plot_scatter_tradeoff(df_summary, output_dir, metric_y=latency_metric, metric_x=cost_metric)   
        else:
            logger.warning(f"Skipping {latency_metric} vs {cost_metric} scatter plot: One or both metrics not found.")

        logger.info("Generating average latency comparison plot...")
        if 'total_run_time_s' in available_metrics_for_plot:
            _plot_total_time_comparison(df_summary, output_dir)
        else:
            logger.info("Skipping total run time comparison plot: 'total_run_time_s' not found in prepared data.")

        logger.info("Generating per-strategy metric comparison plots...")
        all_strategies = df_summary['Strategy'].unique()
        metrics_per_strategy = []
        if primary_metric in available_metrics_for_plot: 
            metrics_per_strategy.append(primary_metric)    
        if primary_metric == 'accuracy' and 'f1_score' in available_metrics_for_plot:
            metrics_per_strategy.append('f1_score')
        
        if latency_metric in available_metrics_for_plot:
            metrics_per_strategy.append(latency_metric)
        if cost_metric in available_metrics_for_plot:
            metrics_per_strategy.append(cost_metric)
        if 'total_tokens' in available_metrics_for_plot:
            metrics_per_strategy.append('total_tokens')
        
        seen = set()
        metrics_per_strategy = [x for x in metrics_per_strategy if not (x in seen or seen.add(x))]

        for strategy in all_strategies:
             logger.debug(f"Generating plots for strategy: {strategy}")
             
             available_metrics_for_strat = df_summary[(df_summary['Strategy'] == strategy) & (df_summary['Metric'].isin(metrics_per_strategy))]['Metric'].unique()
             if available_metrics_for_strat.size > 0:
                 _plot_metric_comparison_for_strategy(df_summary, strategy, list(available_metrics_for_strat), output_dir)
             else:
                 logger.debug(f"No relevant metrics found for strategy '{strategy}' to plot per-strategy comparison.")        
        logger.info("Generating radar chart for multi-metric comparison...")
        try:
            plot_radar_evaluation_metrics(df_summary, output_dir)
        except Exception as e:
            logger.error(f"Error generating radar chart: {e}", exc_info=True)

        logger.info("Generating confusion matrices...")
        for model_id, strategies in all_model_run_summaries.items():
            for strategy_name, results in strategies.items():
                if isinstance(results, dict) and 'confusion_matrix' in results and 'labels' in results:
                    matrix = results['confusion_matrix']
                    labels = results['labels']
                    if matrix and labels: 
                         _plot_confusion_matrix(matrix, labels, model_id, strategy_name, output_dir)
                    else:
                         logger.warning(f"Skipping confusion matrix for {model_id}/{strategy_name}: Empty matrix or labels.")
                         
    logger.info("Finished generating Matplotlib/Seaborn plots.")

    
    
    

    
    
    
    
    
    












