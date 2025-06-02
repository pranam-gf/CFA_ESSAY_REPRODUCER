#!/usr/bin/env python3
"""
CFA Essay Re-grading Script for Unbiased Evaluation

This script addresses the critical bias issue where each LLM was grading its own generated essays.
Instead, it uses GPT-4.1 as a consistent grader across all models to ensure fair and unbiased evaluation,
reproducing the evaluation methodology from the JP Morgan CFA paper (Mahfouz et al., 2024).

Key Features:
- Uses proper CFA Level III grading functions with correct answer context
- Automatically matches updated_data.json and answer_grading_details.json by folder + position
- Gets question context, correct answer (explanation), and grading criteria from data files
- Consistent GPT-4.1 grader for all results with access to correct answers
- Efficient score-only grading for faster processing

Usage:
    # Preferred method (run as module):
    python -m src.utils.regrade_with_gpt4_1 --strategy default_essay --dry-run
    python -m src.utils.regrade_with_gpt4_1 --strategy all --execute
    
    # Alternative method (run script directly):
    python src/utils/regrade_with_gpt4_1.py --strategy default_essay --dry-run
    python src/utils/regrade_with_gpt4_1.py --strategy all --execute
"""

import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
import re
import inspect

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import PROJECT_ROOT, OPENAI_API_KEY
from src.llm_clients import get_llm_response

from src.prompts.grading_prompts import get_full_cfa_level_iii_efficient_grading_prompt


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('regrade_log.txt')
    ]
)
logger = logging.getLogger(__name__)

# Verify prompt function import
try:
    from src.prompts.grading_prompts import get_full_cfa_level_iii_efficient_grading_prompt
    
    # Check function signature
    sig = inspect.signature(get_full_cfa_level_iii_efficient_grading_prompt)
    expected_params = ['question', 'vignette', 'answer_grading_details', 'correct_answer', 'student_answer', 'min_score', 'max_score']
    actual_params = list(sig.parameters.keys())
    
    if actual_params != expected_params:
        logger.warning(f"Prompt function signature mismatch!")
        logger.warning(f"Expected: {expected_params}")
        logger.warning(f"Actual: {actual_params}")
    else:
        logger.info("✅ Prompt function signature verified")
        
except ImportError as e:
    logger.error(f"Failed to import grading prompt function: {e}")
    raise

GPT4_1_CONFIG = {
    "config_id": "gpt-4.1",
    "type": "openai", 
    "model_id": "gpt-4.1-2025-04-14",
    "parameters": {
        "temperature": 0.0,  # Set to 0.0 for maximum consistency and strict grading
        "max_tokens": 10
    }
}

class ImprovedEssayRegrader:
    """Re-grades essay results using GPT-4.1 with proper CFA Level III grading system including correct answer context."""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.results_dir = PROJECT_ROOT / "results"
        self.data_dir = PROJECT_ROOT / "data"
        self.stats = {
            "files_processed": 0,
            "essays_regraded": 0,
            "errors": 0,
            "score_changes": [],
            "processing_time": 0
        }
        
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found. Cannot use GPT-4.1 for re-grading.")
    
    def load_data_files(self) -> Tuple[List[Dict], List[Dict]]:
        """Load and validate updated_data.json and answer_grading_details.json files."""
        try:
            updated_data_path = self.data_dir / "updated_data.json"
            with open(updated_data_path, 'r', encoding='utf-8') as f:
                updated_data = json.load(f)
            
            grading_details_path = self.data_dir / "answer_grading_details.json"
            with open(grading_details_path, 'r', encoding='utf-8') as f:
                grading_details = json.load(f)
            
            logger.info(f"Loaded {len(updated_data)} entries from updated_data.json")
            logger.info(f"Loaded {len(grading_details)} entries from answer_grading_details.json")
            
            return updated_data, grading_details
        
        except Exception as e:
            logger.error(f"Error loading data files: {e}")
            raise
    
    def create_question_lookup(self, updated_data: List[Dict], grading_details: List[Dict]) -> Dict[str, Dict]:
        """Create a lookup dictionary matching entries by folder + position."""
        question_lookup = {}
        
        folder_positions = {}
        for entry in updated_data:
            folder = entry['folder']
            if folder not in folder_positions:
                folder_positions[folder] = []
            folder_positions[folder].append(entry)
        
        for folder, entries in folder_positions.items():
            for position, entry in enumerate(entries):
                key = f"{folder}:{position}"
                question_lookup[key] = {
                    'folder': folder,
                    'position': position,
                    'question': entry.get('question', ''),
                    'vignette': entry.get('vignette', ''),
                    'explanation': entry.get('explanation', ''),
                    'grading_details': None,
                    'max_score': None
                }
        
        grading_folder_positions = {}
        for entry in grading_details:
            folder = entry['folder']
            if folder not in grading_folder_positions:
                grading_folder_positions[folder] = []
            grading_folder_positions[folder].append(entry)
        
        matched_count = 0
        for folder, entries in grading_folder_positions.items():
            for position, entry in enumerate(entries):
                key = f"{folder}:{position}"
                if key in question_lookup:
                    question_lookup[key]['grading_details'] = entry.get('grading_details', '')
                    question_lookup[key]['max_score'] = entry.get('max_score')
                    matched_count += 1
                else:
                    logger.warning(f"No matching question found for grading details at {key}")
        
        logger.info(f"Successfully matched {matched_count} question-grading pairs")
        
        complete_lookup = {k: v for k, v in question_lookup.items() if v['grading_details']}
        logger.info(f"Final lookup contains {len(complete_lookup)} complete entries")
        
        return complete_lookup
    
    def debug_question_matching(self, question_lookup: Dict[str, Dict], sample_file: Path = None) -> None:
        """Debug function to verify question matching is working correctly."""
        logger.info("=== DEBUGGING QUESTION MATCHING ===")
        
        # Show sample of question_lookup
        sample_keys = list(question_lookup.keys())[:3]
        for key in sample_keys:
            context = question_lookup[key]
            logger.info(f"Lookup key: {key}")
            logger.info(f"  Question: {context['question'][:100]}...")
            logger.info(f"  Max score: {context['max_score']}")
            logger.info(f"  Has grading details: {bool(context['grading_details'])}")
        
        # If sample file provided, test matching
        if sample_file and sample_file.exists():
            logger.info(f"\nTesting matching with sample file: {sample_file}")
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list) and data:
                    entry = data[0]
                    matched_context = self.find_matching_question_context(entry, question_lookup)
                    
                    logger.info(f"Sample entry folder: {entry.get('folder', 'N/A')}")
                    logger.info(f"Sample entry question: {entry.get('question', 'N/A')[:100]}...")
                    logger.info(f"Matched: {'Yes' if matched_context else 'No'}")
                    
                    if matched_context:
                        logger.info(f"Matched folder: {matched_context['folder']}")
                        logger.info(f"Matched position: {matched_context['position']}")
                        logger.info(f"Max score: {matched_context['max_score']}")
            
            except Exception as e:
                logger.error(f"Error in debug matching: {e}")
    
    def extract_score_from_response(self, response_text: str, max_score: int) -> Optional[int]:
        """Extract numerical score from GPT-4.1 response with strict validation."""
        if not response_text:
            return None
            
        response_text = response_text.strip()
        
        # Look for integers only (no decimals for strict scoring)
        numbers = re.findall(r'\b(\d+)\b', response_text)
        if numbers:
            try:
                score = int(numbers[0])
                if 0 <= score <= max_score:
                    return score
                else:
                    logger.warning(f"Score {score} outside valid range [0, {max_score}]")
            except ValueError:
                pass
        
        logger.warning(f"Could not extract valid integer score from response: {response_text[:200]}")
        return None
    
    def regrade_essay_with_context(self, question_context: Dict, generated_answer: str) -> Tuple[Optional[int], Optional[str]]:
        """Re-grade a single essay using GPT-4.1 with strict CFA Level III grading."""
        try:
            grading_details_text = question_context.get('grading_details', '')
            correct_answer = question_context.get('explanation', '')
            max_score = question_context.get('max_score')
            question = question_context.get('question', '')
            vignette = question_context.get('vignette', '')
            folder = question_context.get('folder', 'Unknown')
            position = question_context.get('position', 'Unknown')
            
            # Log grading start with context
            question_preview = question[:80] + "..." if len(question) > 80 else question
            logger.info(f"🔍 GRADING: {folder}:{position} (max:{max_score}) - {question_preview}")
            
            if not grading_details_text:
                logger.warning("No grading details available for this question")
                return None, None
            
            if not question:
                logger.warning("No question text available for grading context")
                return None, None
            
            if max_score is None:
                logger.warning(f"Max score not found for question: {question[:100]}...")
                score_match = re.search(r"(\d+)\s*points", grading_details_text, re.IGNORECASE)
                if score_match:
                    max_score = int(score_match.group(1))
                    logger.info(f"Extracted max_score {max_score} from grading_details text.")
                else:
                    logger.warning("Could not extract max_score from grading_details text. Defaulting to 10.")
                    max_score = 10

            min_score = 0
            
            # Get the correct answer for reference
            correct_answer = (question_context.get('explanation', '') or 
                            "See grading details for scoring criteria")
            
            prompt = get_full_cfa_level_iii_efficient_grading_prompt(
                question=question,
                vignette=vignette,
                answer_grading_details=grading_details_text,
                correct_answer=correct_answer,
                student_answer=generated_answer,
                min_score=min_score,
                max_score=max_score,
                correct_answer=correct_answer
            )
            
            response = get_llm_response(
                prompt=prompt,
                model_config=GPT4_1_CONFIG,
                is_json_response_expected=False
            )
            
            raw_response_content = response.get('response_content')

            if raw_response_content:
                score = self.extract_score_from_response(raw_response_content, max_score)
                if score is not None: 
                    logger.info(f"✅ GRADED: {folder}:{position} → Score: {score}/{max_score} | Response: {raw_response_content.strip()[:50]}...")
                    return score, raw_response_content
                else:
                    logger.warning(f"❌ GRADE FAILED: {folder}:{position} - Could not extract score from: {raw_response_content.strip()[:100]}...")
            else:
                logger.error(f"❌ GRADE FAILED: {folder}:{position} - Invalid or empty response from GPT-4.1: {response}")
                
        except Exception as e:
            logger.error(f"Error re-grading essay: {e}", exc_info=True)
            self.stats["errors"] += 1
            
        return None, None
    
    def find_matching_question_context(self, entry: Dict, question_lookup: Dict[str, Dict]) -> Optional[Dict]:
        """Find matching question context for a result entry."""
        folder = entry.get('folder', '')
        question = entry.get('question', '')
        
        # Try exact match first
        for key, context in question_lookup.items():
            if context['folder'] == folder and context['question'] == question:
                logger.debug(f"Found exact match for {folder}:{question[:50]}...")
                return context
        
        # Try position-based match
        entry_position = entry.get('position_in_file')
        if folder and entry_position is not None:
             key_guess = f"{folder}:{entry_position}"
             if key_guess in question_lookup:
                 logger.debug(f"Found match by folder and entry position for {key_guess}")
                 return question_lookup[key_guess]

        # Try partial question match within same folder
        folder_matches = [v for k, v in question_lookup.items() if v['folder'] == folder]
        if folder_matches:
            for context in folder_matches:
                if question and context.get('question') and question[:100] in context['question']:
                    logger.debug(f"Found partial question match for {folder}:{question[:50]}...")
                    return context
        
        logger.warning(f"No matching context found for folder: {folder}, question: {question[:50]}...")
        return None
    
    def process_evaluated_result_file(self, file_path: Path, question_lookup: Dict[str, Dict]) -> bool:
        """Process a single evaluated results JSON file."""
        logger.info(f"📂 Processing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                if isinstance(data, dict) and 'results' in data and isinstance(data['results'], list):
                    data = data['results']
                    logger.info(f"Data in {file_path} was a dict, using 'results' list.")
                else:
                    logger.error(f"Expected list format or dict with 'results' list in {file_path}, got {type(data)}")
                    return False
                
            logger.info(f"📊 Found {len(data)} entries to process in {file_path.name}")
            updated_count = 0
            
            for i, entry in enumerate(data):
                logger.info(f"📝 Processing entry {i+1}/{len(data)} in {file_path.name}")
                
                question_context = self.find_matching_question_context(entry, question_lookup)
                if not question_context:
                    logger.warning(f"No matching context found for entry {i} ('{entry.get('question_id', 'N/A')}') in {file_path}")
                    continue
                
                if 'final_answer' in entry:
                    generated_answer = entry.get('final_answer', '')
                    logger.debug(f"Using final_answer for self-consistency strategy entry {i}")
                else:
                    generated_answer = (entry.get('llm_answer', '') or 
                                       entry.get('cleaned_llm_answer', '') or 
                                       entry.get('raw_llm_answer', ''))
                
                if not generated_answer:
                    logger.warning(f"No generated answer found for entry {i} ('{entry.get('question_id', 'N/A')}') in {file_path}")
                    continue
                
                original_score = entry.get('self_grade_score')
                
                if self.dry_run:
                    question_preview = question_context.get('question', '')[:60] + "..." if len(question_context.get('question', '')) > 60 else question_context.get('question', '')
                    logger.info(f"🔍 [DRY RUN] Would re-grade entry {i+1}: {question_context.get('folder', 'Unknown')}:{question_context.get('position', 'Unknown')} (max:{question_context.get('max_score', 'Unknown')}) - {question_preview}")
                    logger.info(f"   Original score: {original_score}")
                    time.sleep(0.05) 
                    updated_count += 1
                else:
                    new_score, raw_llm_response_text = self.regrade_essay_with_context(question_context, generated_answer)
                    
                    if new_score is not None:
                        entry['self_grade_score'] = new_score
                        entry['self_grade_justification'] = "Re-graded by GPT-4.1 using CFA Level III efficient (score-only) grading with correct answer context."
                        entry['raw_self_grade_api_response'] = raw_llm_response_text if raw_llm_response_text else f"Score: {new_score}"
                        
                        entry['regrade_info'] = {
                            'regraded_by': GPT4_1_CONFIG['model_id'],
                            'original_score': original_score,
                            'regrade_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'grading_system': 'cfa_level_iii_efficient_with_correct_answer',
                            'matched_folder': question_context['folder'],
                            'matched_position': question_context['position'],
                            'includes_correct_answer': True
                        }
                        
                        if original_score is not None:
                            try:
                                score_change = new_score - int(original_score)
                                self.stats["score_changes"].append({
                                    'file': file_path.name,
                                    'entry_index': i,
                                    'question_id': entry.get('question_id', 'N/A'),
                                    'original': original_score,
                                    'new': new_score,
                                    'change': score_change,
                                    'folder': question_context['folder']
                                })
                                if score_change != 0:
                                    logger.info(f"Score change in {file_path.name}[{i}] ('{entry.get('question_id', 'N/A')}'): {original_score} → {new_score} ({score_change:+d})")
                            except (ValueError, TypeError):
                                 logger.warning(f"Could not calculate score change for entry {i} due to original score type: {original_score}")
                        
                        updated_count += 1
                        self.stats["essays_regraded"] += 1
                        
                        time.sleep(0.1) 
                    else:
                        logger.warning(f"Failed to re-grade entry {i} ('{entry.get('question_id', 'N/A')}') in {file_path}")
            
            if not self.dry_run and updated_count > 0:
                backup_path = file_path.with_suffix('.json.backup')
                if not backup_path.exists():
                    try:
                        file_path.rename(backup_path)
                        logger.info(f"Created backup: {backup_path}")
                    except OSError as e:
                        logger.error(f"Could not create backup for {file_path}: {e}. Skipping save.")
                        return False
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Updated {updated_count} entries in {file_path}")
            elif not self.dry_run and updated_count == 0:
                logger.info(f"ℹ️  No entries were updated in {file_path} during execute mode.")
            
            logger.info(f"✅ Completed processing {file_path.name}: {updated_count} entries processed")
            return True
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            self.stats["errors"] += 1
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}")
            self.stats["errors"] += 1
            return False
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            self.stats["errors"] += 1
            return False
    
    def process_strategy_folder(self, strategy_name: str, question_lookup: Dict[str, Dict]) -> bool:
        """Process all evaluated result files in a strategy folder."""
        strategy_folder = self.results_dir / strategy_name
        
        if not strategy_folder.exists():
            logger.error(f"Strategy folder not found: {strategy_folder}")
            return False
        
        result_files = [f for f in strategy_folder.glob("evaluated_results_*.json") 
                       if not f.name.endswith('.json.backup')]
        
        if not result_files:
            logger.warning(f"No evaluated result files found in {strategy_folder} (excluding .backup files)")
            return False
        
        logger.info(f"Found {len(result_files)} files to process in {strategy_name}")
        
        success_count = 0
        for file_path in result_files:
            if self.process_evaluated_result_file(file_path, question_lookup):
                success_count += 1
        
        self.stats["files_processed"] += len(result_files)

        if success_count == len(result_files) and len(result_files) > 0:
             logger.info(f"Successfully processed all {success_count}/{len(result_files)} files in {strategy_name}")
        elif success_count > 0:
             logger.warning(f"Processed {success_count}/{len(result_files)} files in {strategy_name} with some errors.")
        elif len(result_files) > 0:
             logger.error(f"Failed to process any files in {strategy_name}.")
        
        return success_count > 0 or not result_files
    
    def run(self, strategies: List[str]) -> Dict[str, Any]:
        """Run the re-grading process for specified strategies."""
        start_time = time.time()
        
        logger.info("Loading data files and creating question-grading lookup...")
        try:
            updated_data, grading_details = self.load_data_files()
            question_lookup = self.create_question_lookup(updated_data, grading_details)
        except Exception as e:
            logger.error(f"Failed to load data or create lookup: {e}")
            self.stats["processing_time"] = time.time() - start_time
            return {"success": False, "error": f"Data loading/lookup creation failed: {e}", "stats": self.stats}

        if not question_lookup:
            logger.error("No question-grading matches found. Cannot proceed.")
            self.stats["processing_time"] = time.time() - start_time
            return {"success": False, "error": "No matching data found", "stats": self.stats}
        
        # Add debug call in dry-run mode
        if self.dry_run:
            # Find a sample file for debugging
            sample_file = None
            for strategy_dir in self.results_dir.iterdir():
                if strategy_dir.is_dir():
                    sample_files = list(strategy_dir.glob("evaluated_results_*.json"))
                    if sample_files:
                        sample_file = sample_files[0]
                        break
            self.debug_question_matching(question_lookup, sample_file)
        
        processed_strategies = []
        
        if "all" in strategies:
            all_dirs = [d.name for d in self.results_dir.iterdir() if d.is_dir()]
            strategies_to_process = [
                d_name for d_name in all_dirs 
                if "essay" in d_name or "default" in d_name or "consistency" in d_name or "discover" in d_name
            ]
            if not strategies_to_process:
                strategies_to_process = all_dirs

            logger.info(f"Processing all identified strategy folders: {strategies_to_process}")
        else:
            strategies_to_process = strategies
        
        for strategy in strategies_to_process:
            logger.info(f"--- Starting strategy: {strategy} ---")
            if self.process_strategy_folder(strategy, question_lookup):
                processed_strategies.append(strategy)
                logger.info(f"--- Completed strategy: {strategy} ---")
            else:
                logger.error(f"--- Failed to fully process strategy: {strategy} ---")
        
        self.stats["processing_time"] = time.time() - start_time
        
        self.print_summary(processed_strategies)
        
        return {
            "success": len(processed_strategies) > 0 or not strategies_to_process,
            "processed_strategies": processed_strategies,
            "stats": self.stats
        }
    
    def print_summary(self, processed_strategies: List[str]) -> None:
        """Print summary of re-grading results."""
        logger.info("=" * 60)
        logger.info("RE-GRADING SUMMARY")
        logger.info("=" * 60)
        
        if self.dry_run:
            logger.info("DRY RUN MODE - No files were actually modified")
        
        logger.info(f"Attempted strategies: {', '.join(processed_strategies) if processed_strategies else 'None'}")
        logger.info(f"Files attempted: {self.stats['files_processed']}")
        logger.info(f"Essays re-graded: {self.stats['essays_regraded']}")
        logger.info(f"Errors encountered during file processing/regrading: {self.stats['errors']}")
        logger.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        
        if self.stats["score_changes"]:
            changes = self.stats["score_changes"]
            improvements = [c for c in changes if c['change'] > 0]
            declines = [c for c in changes if c['change'] < 0]
            no_change = [c for c in changes if c['change'] == 0]
            
            logger.info(f"\nScore Changes (based on {len(changes)} comparable scores):")
            logger.info(f"  Improvements: {len(improvements)}")
            logger.info(f"  Declines: {len(declines)}")
            logger.info(f"  No change: {len(no_change)}")
            
            if improvements:
                avg_improvement = sum(c['change'] for c in improvements) / len(improvements)
                logger.info(f"  Average improvement: +{avg_improvement:.2f}")
            
            if declines:
                avg_decline = sum(c['change'] for c in declines) / len(declines)
                logger.info(f"  Average decline: {avg_decline:.2f}")
        else:
            logger.info("\nNo score changes recorded (or original scores were not comparable).")

def main():
    """Main function to run the improved re-grading script."""
    parser = argparse.ArgumentParser(description="CFA Essay Re-grading with GPT-4.1 for Unbiased Evaluation")
    parser.add_argument(
        "--strategy", 
        required=True,
        help="Strategy to process (e.g., 'default_essay', 'self_consistency_essay_n3', or 'all')"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Run in dry-run mode (show what would be done without making changes)"
    )
    parser.add_argument(
        "--execute", 
        action="store_true",
        help="Execute the re-grading (required for actual changes)"
    )
    
    args = parser.parse_args()
    
    if args.execute and args.dry_run:
        logger.error("Cannot specify both --execute and --dry-run. Defaulting to dry-run.")
        dry_run = True
    elif args.execute:
        dry_run = False
        logger.info("EXECUTE MODE: Files will be modified.")
    else:
        dry_run = True
        logger.info("DRY RUN MODE: No files will be modified. Use --execute to make changes.")

    try:
        regrader = ImprovedEssayRegrader(dry_run=dry_run)
        strategies = [s.strip() for s in args.strategy.split(',')]
        
        result = regrader.run(strategies)
        
        if result["success"]:
            logger.info("Re-grading process completed.")
        else:
            logger.error(f"Re-grading process encountered issues: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Fatal error during script execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()