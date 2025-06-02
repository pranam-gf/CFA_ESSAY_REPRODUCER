#!/usr/bin/env python3
"""
CFA Essay Similarity Score Regrading Script

This script recalculates and updates the cosine similarity and ROUGE-L scores
for all evaluated essay results, ensuring consistency across all result files.

Usage:
    python -m src.utils.regrade_similarity_scores --strategy default_essay --dry-run
    python -m src.utils.regrade_similarity_scores --strategy all --execute
"""

import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any
import sys
import shutil

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import PROJECT_ROOT
from src.evaluations.essay_evaluation import calculate_cosine_similarity, calculate_rouge_l_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('similarity_regrade_log.txt')
    ]
)
logger = logging.getLogger(__name__)

class SimilarityScoreRegrader:
    """Recalculates and updates similarity scores (cosine and ROUGE-L) for essay results."""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.results_dir = PROJECT_ROOT / "results"
        self.stats = {
            "files_processed": 0,
            "entries_processed": 0,
            "entries_updated": 0,
            "errors": 0,
            "processing_time": 0
        }
    
    def process_evaluated_result_file(self, file_path: Path) -> bool:
        """Process a single evaluated results JSON file."""
        logger.info(f"📂 Processing file: {file_path}")
        
        try:
            # Extract model name and method from filename
            filename = file_path.name
            # Format: evaluated_results_MODEL__METHOD.json
            if "__" in filename:
                parts = filename.replace("evaluated_results_", "").replace(".json", "").split("__")
                model_name = parts[0]
                method_name = parts[1] if len(parts) > 1 else "unknown"
            else:
                model_name = "unknown"
                method_name = "unknown"
            
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
                # Get the necessary texts for recalculation
                cleaned_generated_answer = entry.get("cleaned_llm_answer")
                reference_answer = entry.get("explanation")
                
                # Skip if required data is missing
                if not cleaned_generated_answer or not reference_answer or entry.get("error"):
                    logger.info(f"⚠️ Q{i+1} - {model_name} - {method_name}: Skipped (missing data)")
                    continue
                
                self.stats["entries_processed"] += 1
                
                # Recalculate scores (but don't update entry in dry run mode)
                if self.dry_run:
                    logger.info(f"✓ Q{i+1} - {model_name} - {method_name}: Would update (dry run)")
                    updated_count += 1
                else:
                    # Actually update the scores in execute mode
                    # Recalculate and update cosine similarity
                    new_cosine = calculate_cosine_similarity(cleaned_generated_answer, reference_answer)
                    if new_cosine is not None:
                        entry["cosine_similarity"] = new_cosine
                    
                    # Recalculate and update ROUGE scores
                    new_rouge_scores = calculate_rouge_l_score(cleaned_generated_answer, reference_answer)
                    if new_rouge_scores:
                        entry["rouge_l_precision"] = new_rouge_scores["precision"]
                        entry["rouge_l_recall"] = new_rouge_scores["recall"]
                        entry["rouge_l_f1measure"] = new_rouge_scores["f1measure"]
                    
                    logger.info(f"✓ Q{i+1} - {model_name} - {method_name}: Updated")
                    updated_count += 1
                    self.stats["entries_updated"] += 1
            
            if not self.dry_run and updated_count > 0:
                # Create backup of original file
                backup_path = file_path.with_suffix('.json.backup')
                if not backup_path.exists():
                    try:
                        shutil.copy2(file_path, backup_path)
                        logger.info(f"Created backup: {backup_path}")
                    except OSError as e:
                        logger.error(f"Could not create backup for {file_path}: {e}. Skipping save.")
                        return False
                
                # Save updated data
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Updated {updated_count} entries in {file_path}")
            elif self.dry_run:
                logger.info(f"[DRY RUN] Would update {updated_count} entries in {file_path}")
            
            logger.info(f"✅ Completed processing {file_path.name}")
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
    
    def process_strategy_folder(self, strategy_name: str) -> bool:
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
            if self.process_evaluated_result_file(file_path):
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
        """Run the similarity score regrading process for specified strategies."""
        start_time = time.time()
        
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
            if self.process_strategy_folder(strategy):
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
        """Print summary of similarity score regrading results."""
        logger.info("=" * 60)
        logger.info("SIMILARITY SCORE REGRADING SUMMARY")
        logger.info("=" * 60)
        
        if self.dry_run:
            logger.info("DRY RUN MODE - No files were actually modified")
        
        logger.info(f"Attempted strategies: {', '.join(processed_strategies) if processed_strategies else 'None'}")
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Entries processed: {self.stats['entries_processed']}")
        logger.info(f"Entries updated: {self.stats['entries_updated']}")
        logger.info(f"Errors encountered: {self.stats['errors']}")
        logger.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")

def main():
    """Main function to run the similarity score regrading script."""
    parser = argparse.ArgumentParser(description="Recalculate and update cosine similarity and ROUGE-L scores for CFA essay results")
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
        help="Execute the regrading (required for actual changes)"
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
        regrader = SimilarityScoreRegrader(dry_run=dry_run)
        strategies = [s.strip() for s in args.strategy.split(',')]
        
        result = regrader.run(strategies)
        
        if result["success"]:
            logger.info("Similarity score regrading process completed successfully.")
        else:
            logger.error(f"Similarity score regrading process encountered issues: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Fatal error during script execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()
