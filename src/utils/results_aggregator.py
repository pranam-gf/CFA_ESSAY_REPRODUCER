import os
import json
import csv
import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)
CSV_HEADERS = [
    "original_question_id",  
    "folder",
    "vignette",
    "question",
    "explanation",    
    "prompt",
    "llm_answer", 
    "raw_llm_answer", 
    "cleaned_llm_answer", 
    "answer_length",
    "model_id",
    "config_id",
    "strategy",
    "error",
    "selected_sample_index",
    "selector_llm_response",
    "all_cot_samples", 
    "self_discover_reasoning", 
    "self_discover_final_essay", 
    "input_tokens",
    "output_tokens",
    "response_time", 
    "cosine_similarity",
    "rouge_l_precision",
    "rouge_l_recall",
    "rouge_l_f1measure",
    "self_grade_score",
    "self_grade_justification",
    "self_grade_error",
    "raw_self_grade_api_response" 
    
]

def extract_data_from_record(record: Dict[str, Any], strategy_folder: str) -> Dict[str, Any]:
    """
    Extracts relevant fields from a single record for the CSV.
    Handles missing keys gracefully.
    """
    extracted = {}
    
    extracted["self_discover_reasoning"] = None
    extracted["self_discover_final_essay"] = None

    current_strategy = record.get("strategy")
    if not current_strategy: 
        if "default_essay" in strategy_folder:
            current_strategy = "default_essay"
        elif "self_consistency_essay_n3" in strategy_folder:
            current_strategy = "self_consistency_essay_n3"
        elif "self_consistency_essay_n5" in strategy_folder:
            current_strategy = "self_consistency_essay_n5"
        elif "self_discover_essay" in strategy_folder:
            current_strategy = "self_discover_essay"
    
    if current_strategy == "self_discover_essay":
        raw_answer_for_sd = record.get("raw_llm_answer")
        if raw_answer_for_sd and isinstance(raw_answer_for_sd, str):
            marker_regex = r"^(?:#+\s*|\*\*\s*|[\d.]+\s*)*Full Essay Answer(?:[:*]*\s*)?$"
            match = re.search(marker_regex, raw_answer_for_sd, re.MULTILINE | re.IGNORECASE)
            if match:
                reasoning_part = raw_answer_for_sd[:match.start()].strip()
                essay_part = raw_answer_for_sd[match.end():].strip()
                if essay_part: 
                    extracted["self_discover_reasoning"] = reasoning_part
                    extracted["self_discover_final_essay"] = essay_part
                else:
                    extracted["self_discover_reasoning"] = raw_answer_for_sd 
                    question_id_for_log = record.get("original_question_id", record.get("question_id", record.get("question_hash", "N/A")))
                    logger.debug(f"Self-Discover marker found for ID {question_id_for_log}, but no content followed for the essay. Storing full raw answer in reasoning.")
            else:
                extracted["self_discover_reasoning"] = raw_answer_for_sd 
                question_id_for_log = record.get("original_question_id", record.get("question_id", record.get("question_hash", "N/A")))
                logger.debug(f"Self-Discover marker 'Full Essay Answer' variations not found in raw_llm_answer for ID {question_id_for_log}. Storing full raw answer in reasoning.")


    for header in CSV_HEADERS:
        if header == "self_discover_reasoning" or header == "self_discover_final_essay":   
            if header not in extracted: 
                 extracted[header] = None
            continue 
        elif header == "all_cot_samples" and header in record:
            
            samples = record.get(header)
            if isinstance(samples, list):
                extracted[header] = f"Count: {len(samples)}" 
            else:
                extracted[header] = samples
        elif header == "raw_self_grade_api_response" and header in record:
            
            raw_response = record.get(header)
            if isinstance(raw_response, str) and len(raw_response) > 500: 
                extracted[header] = raw_response[:497] + "..."
            else:
                extracted[header] = raw_response
        elif header == "original_question_id":
            
            extracted[header] = record.get("question_id", record.get("question_hash", record.get("id")))
        else:
            extracted[header] = record.get(header)
            
    
    if not extracted.get("strategy") and current_strategy:
        extracted["strategy"] = current_strategy
    elif not extracted.get("strategy") and "strategy" in record: 
         extracted["strategy"] = record["strategy"]
    return extracted

def aggregate_results_to_csv(results_base_dir: str, output_csv_path: str) -> None:
    """
    Aggregates results from JSON files in specified subdirectories of results_base_dir
    into a single CSV file.
    Args:
        results_base_dir: The base directory where strategy subdirectories are located (e.g., "results/").
        output_csv_path: The path to save the aggregated CSV file (e.g., "results/all_results.csv").
    """
    strategy_subfolders = [
        "default_essay",
        "self_consistency_essay_n3",
        "self_consistency_essay_n5",
        "self_discover_essay"
    ]

    all_records_for_csv: List[Dict[str, Any]] = []
    found_files_count = 0

    for strategy_folder_name in strategy_subfolders:
        current_strategy_path = os.path.join(results_base_dir, strategy_folder_name)
        if not os.path.isdir(current_strategy_path):
            logger.warning(f"Strategy folder not found: {current_strategy_path}")
            continue

        for filename in os.listdir(current_strategy_path):
            if filename.startswith("evaluated_results_") and filename.endswith(".json"):
                filepath = os.path.join(current_strategy_path, filename)
                logger.info(f"Processing file: {filepath}")
                found_files_count += 1
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        if isinstance(data, list):
                            for record in data:
                                if isinstance(record, dict):
                                    all_records_for_csv.append(extract_data_from_record(record, strategy_folder_name))
                                else:
                                    logger.warning(f"Skipping non-dict item in list in file {filepath}: {type(record)}")
                        
                        elif isinstance(data, dict) and 'results' in data and isinstance(data['results'], list):
                             for record in data['results']:
                                if isinstance(record, dict):
                                    all_records_for_csv.append(extract_data_from_record(record, strategy_folder_name))
                                else:
                                    logger.warning(f"Skipping non-dict item in 'results' list in file {filepath}: {type(record)}")
                        else:
                            logger.warning(f"Skipping file with unexpected structure: {filepath}. Expected list or dict with 'results' list.")
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from file {filepath}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error processing file {filepath}: {e}", exc_info=True)
    
    if not all_records_for_csv:
        logger.warning(f"No records were extracted. CSV file will be empty or not created. Found {found_files_count} 'evaluated_results_*.json' files.")
        if found_files_count == 0:
             logger.error("No 'evaluated_results_*.json' files found in any strategy directory. Ensure they exist and paths are correct.")
        return

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS, extrasaction='ignore') 
            writer.writeheader()
            writer.writerows(all_records_for_csv)
        logger.info(f"Successfully aggregated {len(all_records_for_csv)} records into {output_csv_path}")
    except IOError as e:
        logger.error(f"Error writing CSV to {output_csv_path}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during CSV writing: {e}", exc_info=True)

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s') 
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(project_root, "results")
    output_csv = os.path.join(results_dir, "all_aggregated_results.csv") 
    logger.info(f"Project root determined as: {project_root}")
    logger.info(f"Attempting to read from: {results_dir}")
    logger.info(f"Attempting to write to: {output_csv}")

    aggregate_results_to_csv(results_dir, output_csv)

    
    
    