#!/usr/bin/env python3
"""
JSON to CSV Processor for CFA Essay Results

This script processes JSON files from the CFA essay reproducer results
and converts them into a consolidated CSV file.

Path structure: CFA_ESSAY_REPRODUCER/results/{strategy}/evaluated_results_{model}__{strategy}.json
Output: all_results.csv
"""

import json
import csv
import os
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def get_results_directory() -> Path:
    """
    Get the results directory path relative to the script location.
    
    Returns:
        Path: Path to the results directory
    """
    # Script is in CFA_ESSAY_REPRODUCER/src/utils
    # Results are in CFA_ESSAY_REPRODUCER/results
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent.parent / "results"
    return results_dir


def find_json_files(results_dir: Path) -> List[Path]:
    """
    Find all JSON files matching the expected pattern.
    
    Args:
        results_dir (Path): Path to the results directory
        
    Returns:
        List[Path]: List of JSON file paths
    """
    pattern = "*/evaluated_results_*__*.json"
    json_files = list(results_dir.glob(pattern))
    
    if not json_files:
        print(f"Warning: No JSON files found in {results_dir} matching pattern {pattern}")
    else:
        print(f"Found {len(json_files)} JSON files to process")
    
    return json_files


def extract_model_strategy_from_path(file_path: Path) -> tuple[str, str]:
    """
    Extract model and strategy from the file path.
    
    Args:
        file_path (Path): Path to the JSON file
        
    Returns:
        tuple: (model, strategy)
    """
    # Known strategies
    strategies = ["default_essay", "self_consistency_essay_n3", "self_consistency_essay_n5", "self_discover_essay"]
    
    # Known models from the results table
    models = [
        "claude-3.5-haiku", "claude-3.5-sonnet", "claude-3.7-sonnet", "claude-opus-4", "claude-sonnet-4",
        "codestral-latest-official", "deepseek-r1", "gemini-2.5-flash", "gemini-2.5-pro", 
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "grok-3", "grok-3-mini-beta-high-effort",
        "grok-3-mini-beta-low-effort", "groq-llama-4-maverick", "groq-llama-4-scout", "groq-llama-guard-4",
        "groq-llama3.1-8b-instant", "groq-llama3.3-70b", "mistral-large-official", "o3-mini", "o4-mini",
        "palmyra-fin-default"
    ]
    
    # Extract strategy from parent directory name
    strategy = file_path.parent.name
    
    # Validate strategy
    if strategy not in strategies:
        print(f"Warning: Unknown strategy '{strategy}' in {file_path}")
    
    # Extract model from filename: evaluated_results_{model}__{strategy}.json
    filename = file_path.stem  # Remove .json extension
    # Remove "evaluated_results_" prefix and "__{strategy}" suffix
    prefix = "evaluated_results_"
    suffix = f"__{strategy}"
    
    if filename.startswith(prefix) and filename.endswith(suffix):
        model = filename[len(prefix):-len(suffix)]
    else:
        # This shouldn't happen with the known file pattern, but just in case
        print(f"Warning: Unexpected filename pattern in {file_path}")
        parts = filename.split("__")
        if len(parts) >= 2:
            model = parts[0].replace("evaluated_results_", "")
        else:
            model = filename.replace("evaluated_results_", "")
    
    # Validate model
    if model not in models:
        print(f"Warning: Unknown model '{model}' in {file_path}")
    
    return model, strategy


def process_json_file(file_path: Path) -> Dict[str, Any]:
    """
    Process a single JSON file and extract relevant data.
    
    Args:
        file_path (Path): Path to the JSON file
        
    Returns:
        Dict[str, Any]: Processed data row
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and dict structures
        if isinstance(data, list):
            if len(data) > 0:
                data = data[0]  # Take the first item if it's a list
            else:
                print(f"Warning: Empty list in {file_path}")
                return None
        
        # Extract model and strategy from file path
        model, strategy = extract_model_strategy_from_path(file_path)
        
        # Get run_timestamp from regrade_info.regrade_timestamp
        run_timestamp = ""
        if isinstance(data, dict) and 'regrade_info' in data and isinstance(data['regrade_info'], dict) and 'regrade_timestamp' in data['regrade_info']:
            run_timestamp = data['regrade_info']['regrade_timestamp']
        else:
            print(f"Warning: No regrade_timestamp found in {file_path}")
            # Fallback to file modification time if regrade_timestamp not available
            file_stat = file_path.stat()
            run_timestamp = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
        
        # Safely get values with proper checks
        def safe_get(obj, key, default=''):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return default
        
        # Convert response_time from seconds to milliseconds
        response_time = safe_get(data, 'response_time', 0)
        latency_ms = response_time * 1000 if response_time else 0
        
        # Create the row data
        row_data = {
            'model': model,
            'strategy': strategy,
            'run_timestamp': run_timestamp,
            'cosine_similarity': safe_get(data, 'cosine_similarity'),
            'self_grade_score': safe_get(data, 'self_grade_score'),
            'rouge_l_precision': safe_get(data, 'rouge_l_precision'),
            'rouge_l_recall': safe_get(data, 'rouge_l_recall'),
            'rouge_l_f1measure': safe_get(data, 'rouge_l_f1measure'),
            'latency_ms': latency_ms,
            'input_tokens': safe_get(data, 'input_tokens'),
            'output_tokens': safe_get(data, 'output_tokens'),
            'answer_length': safe_get(data, 'answer_length')
        }
        
        return row_data
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_all_files(json_files: List[Path]) -> List[Dict[str, Any]]:
    """
    Process all JSON files and extract data.
    
    Args:
        json_files (List[Path]): List of JSON file paths
        
    Returns:
        List[Dict[str, Any]]: List of processed data rows
    """
    all_data = []
    
    for file_path in json_files:
        print(f"Processing: {file_path}")
        row_data = process_json_file(file_path)
        if row_data:
            all_data.append(row_data)
    
    return all_data


def write_csv(data: List[Dict[str, Any]], output_file: Path) -> None:
    """
    Write the processed data to a CSV file.
    
    Args:
        data (List[Dict[str, Any]]): Processed data rows
        output_file (Path): Output CSV file path
    """
    if not data:
        print("No data to write to CSV")
        return
    
    headers = [
        'model', 'strategy', 'run_timestamp', 'cosine_similarity', 
        'self_grade_score', 'rouge_l_precision', 'rouge_l_recall', 
        'rouge_l_f1measure', 'latency_ms', 'input_tokens', 
        'output_tokens', 'answer_length'
    ]
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Successfully wrote {len(data)} rows to {output_file}")
        
    except Exception as e:
        print(f"Error writing CSV file: {e}")


def main():
    """
    Main function to orchestrate the JSON to CSV conversion process.
    """
    print("Starting JSON to CSV conversion process...")
    
    # Get the results directory
    results_dir = get_results_directory()
    print(f"Results directory: {results_dir}")
    
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return
    
    # Find all JSON files
    json_files = find_json_files(results_dir)
    
    if not json_files:
        print("No JSON files found to process")
        return
    
    # Process all files
    all_data = process_all_files(json_files)
    
    # Write to CSV
    output_file = Path("all_results.csv")
    write_csv(all_data, output_file)
    
    print("Process completed!")


if __name__ == "__main__":
    main()