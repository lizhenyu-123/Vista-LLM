import json
import argparse
import os
import re
import glob
import pandas as pd
from typing import List, Dict, Optional
from collections import defaultdict

# ==============================================================================
# 1. Common Utility Functions
# ==============================================================================

def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and stripping extra spaces."""
    if not isinstance(text, str): 
        return ""
    return ' '.join(text.lower().strip().split())

def print_row(name: str, stats: Dict, is_sub: bool = False):
    """Format and print a single row of evaluation statistics."""
    acc = stats.get('accuracy', 0.0)
    corr = stats.get('correct', 0)
    tot = stats.get('total', 0)
    err = stats.get('errors', 0)
    
    prefix = "   - " if is_sub else ""
    name_fmt = f"{prefix}{name}"
    
    col1 = f"{name_fmt:<20}"
    col2 = f"{acc:>6.2f}%"
    col3 = f"({corr}/{tot})"
    
    err_str = f"[Errors: {err}]" if err > 0 else "[OK]"
        
    print(f"{col1} | {col2} | {col3:<15} | {err_str}")


# ==============================================================================
# 2. Multi-Task MCQ Evaluation (Used for MLVU & MVBench)
# ==============================================================================

def _parse_mcq_answer(item: Dict) -> Optional[str]:
    """Parse the generated choice letter and map it to the candidate text."""
    generated_answer = item.get("generated_answer", "")
    candidates = item.get("candidates", [])
    
    if not candidates or not isinstance(candidates, list) or not isinstance(generated_answer, str):
        return None
        
    # Match A-Z ignoring case and surrounding punctuation
    match = re.search(r'^\s*([A-Z])\b', generated_answer.strip(), re.IGNORECASE)
    if not match:
        return None 

    # Convert letter to index (A=0, B=1, ...)
    option_letter = match.group(1).upper()
    option_index = ord(option_letter) - ord('A')

    if 0 <= option_index < len(candidates):
        return candidates[option_index]
    
    return None 

def _calculate_multitask_file_stats(results: List[Dict]) -> Dict:
    """Calculate correct, total, and error counts for a single task JSON file."""
    if not results:
        return {"correct": 0, "total": 0, "errors": 0}

    correct_count = 0
    error_count = 0
    valid_count = 0
    
    for item in results:
        generated_answer = item.get("generated_answer", "")
        correct_answer = item.get("correct_answer", "N/A")
        
        # Error checking
        if generated_answer is None:
            continue
        if isinstance(generated_answer, str) and generated_answer.strip().startswith("[ERROR]"):
            error_count += 1
            continue
        if "N/A" in correct_answer:
            continue
            
        valid_count += 1
        
        # Matching logic
        norm_corr_ans = normalize_text(correct_answer)
        resolved_mcq_answer = _parse_mcq_answer(item)
        
        is_correct = False
        
        # Strict Match
        if resolved_mcq_answer and normalize_text(resolved_mcq_answer) == norm_corr_ans:
            is_correct = True
        elif normalize_text(generated_answer) == norm_corr_ans:
            is_correct = True

        # Loose Match
        if not is_correct:
            if resolved_mcq_answer and norm_corr_ans in normalize_text(resolved_mcq_answer):
                is_correct = True
            elif norm_corr_ans in normalize_text(generated_answer):
                is_correct = True
        
        if is_correct:
            correct_count += 1
            
    return {"correct": correct_count, "total": valid_count, "errors": error_count}

def evaluate_multitask_benchmark(base_path: str, suffix: str, dataset_name: str) -> Dict:
    """Aggregate accuracy across multiple sub-task folders for MLVU/MVBench."""
    search_pattern = os.path.join(base_path, f"*_{suffix}")
    matching_dirs = sorted(glob.glob(search_pattern))
    
    if not matching_dirs:
        print(f"WARNING: [{dataset_name}] No directories found matching: {search_pattern}")
        return {}

    target_files = []
    for dir_path in matching_dirs:
        if not os.path.isdir(dir_path): continue
        json_files = glob.glob(os.path.join(dir_path, '*.json'))
        if json_files:
            target_files.append(json_files[0])
    
    print(f"INFO: [{dataset_name}] Found {len(target_files)} result files.")

    agg_stats = {"correct": 0, "total": 0, "errors": 0}
    
    for file_path in target_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_stats = _calculate_multitask_file_stats(data)
            agg_stats["correct"] += file_stats["correct"]
            agg_stats["total"] += file_stats["total"]
            agg_stats["errors"] += file_stats["errors"]
            
        except Exception as e:
            print(f"ERROR: Failed to read {file_path}: {e}")

    agg_stats["accuracy"] = (agg_stats["correct"] / agg_stats["total"] * 100) if agg_stats["total"] > 0 else 0.0
    return agg_stats


# ==============================================================================
# 3. VideoMME Evaluation Logic
# ==============================================================================

def evaluate_videomme(results_files: List[str], ann_file: str) -> Dict:
    """Evaluate VideoMME including overall, short, medium, and long categories."""
    print(f"INFO: [VideoMME] Loading annotations from {ann_file}...")
    try:
        ground_truth_df = pd.read_parquet(ann_file)
        ground_truth_map = {
            row['question_id']: {'answer': row['answer'], 'duration': row['duration']} 
            for _, row in ground_truth_df.iterrows()
        }
    except Exception as e:
        print(f"ERROR: [VideoMME] Failed to load annotations: {e}")
        return {}

    stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': 0})
    
    for res_file in results_files:
        if not os.path.exists(res_file): 
            continue
        try:
            with open(res_file, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
        except Exception as e: 
            print(f"ERROR: [VideoMME] Failed to read {res_file}: {e}")
            continue

        for pred in predictions:
            q_id = pred.get('question_id')
            gen_ans = pred.get('generated_answer')
            
            # Check for generation errors
            if isinstance(gen_ans, str) and gen_ans.strip().startswith('[ERROR]'):
                gt = ground_truth_map.get(q_id)
                if gt:
                    stats['overall']['errors'] += 1
                    stats[gt['duration']]['errors'] += 1
                continue
            
            if not q_id or not gen_ans: continue

            gt = ground_truth_map.get(q_id)
            if not gt: continue

            is_correct = str(gen_ans).strip().upper().startswith(gt['answer'].upper())

            stats['overall']['total'] += 1
            stats[gt['duration']]['total'] += 1
            
            if is_correct:
                stats['overall']['correct'] += 1
                stats[gt['duration']]['correct'] += 1

    final_output = {}
    for key, val in stats.items():
        acc = (val['correct'] / val['total'] * 100) if val['total'] > 0 else 0.0
        final_output[key] = {
            'accuracy': acc,
            'correct': val['correct'],
            'total': val['total'],
            'errors': val['errors']
        }
    return final_output


# ==============================================================================
# 4. LongVideoBench Evaluation Logic
# ==============================================================================

def evaluate_longvideobench(results_file: str, ann_file: str) -> Dict:
    """Evaluate LongVideoBench by extracting choice letters and comparing with GT."""
    print(f"INFO: [LongVideoBench] Processing results...")
    try:
        with open(results_file, 'r', encoding='utf-8') as f: results_data = json.load(f)
        with open(ann_file, 'r', encoding='utf-8') as f: annotation_data = json.load(f)
    except Exception as e:
        print(f"ERROR: [LongVideoBench] Failed to load files: {e}")
        return {}

    annotation_map = {item['id']: item for item in annotation_data}
    correct = total = errors = 0
    
    for result in results_data:
        q_id = result.get("question_id")
        gen_ans = result.get("generated_answer", "")
        ann = annotation_map.get(q_id)
        
        if not ann: continue

        if not gen_ans or (isinstance(gen_ans, str) and "[ERROR]" in gen_ans):
            errors += 1
            continue
        
        correct_idx = ann.get("correct_choice")
        if correct_idx is None: continue
        
        # Convert numeric correct_choice to letter (0 -> A, 1 -> B, etc.)
        correct_letter = chr(ord('A') + correct_idx)
        total += 1
        
        # Extract the first valid A-Z character from prediction
        match = re.search(r'[A-Z]', gen_ans.upper())
        if match and match.group(0) == correct_letter:
            correct += 1
            
    return {
        "accuracy": (correct / total * 100) if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "errors": errors
    }


# ==============================================================================
# 5. Main Execution Script
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Integrated Evaluation Script for Video Benchmarks")
    
    parser.add_argument("--suffix", type=str, default="domain0.85_agg0.5_tau0.80", 
                        help="Suffix used to filter sub-task folders for MLVU/MVBench")
    
    # LongVideoBench Arguments
    parser.add_argument("--lvb_file", type=str, default="./results/longvideobench/results.json",
                        help="Path to LongVideoBench results.json")
    parser.add_argument("--lvb_ann", type=str, default="./annotations/lvb_val.json", 
                        help="Path to LVB annotation json")
    
    # VideoMME Arguments
    parser.add_argument("--videomme_files", nargs='+', type=str, 
                        default=[
                            "./results/videomme/short/results.json",
                            "./results/videomme/medium/results.json",
                            "./results/videomme/long/results.json",
                        ],
                        help="List of paths to VideoMME results (short/medium/long)")
    parser.add_argument("--videomme_ann", type=str, default="./annotations/VideoMME/test.parquet", 
                        help="Path to VideoMME parquet")

    # MLVU Arguments
    parser.add_argument("--mlvu_base", type=str, default="./results/mlvu",
                        help="Base path to MLVU results")
    
    # MVBench Arguments
    parser.add_argument("--mvbench_base", type=str, default="./results/mvbench",
                        help="Base path to MVBench results")

    args = parser.parse_args()
    final_results = {} 
    
    print(f"\n{'='*80}")
    print(f"EVALUATION REPORT (Suffix: {args.suffix})")
    print(f"{'='*80}")

    # Process VideoMME
    if args.videomme_files:
        vmme_stats = evaluate_videomme(args.videomme_files, args.videomme_ann)
        if vmme_stats:
            final_results['VideoMME'] = vmme_stats 

    # Process MLVU
    if args.mlvu_base:
        stats = evaluate_multitask_benchmark(args.mlvu_base, args.suffix, "MLVU")
        if stats: final_results['MLVU'] = {'overall': stats}

    # Process MVBench
    if args.mvbench_base:
        stats = evaluate_multitask_benchmark(args.mvbench_base, args.suffix, "MVBench")
        if stats: final_results['MVBench'] = {'overall': stats}

    # Process LongVideoBench
    if args.lvb_file and os.path.exists(args.lvb_file):
        stats = evaluate_longvideobench(args.lvb_file, args.lvb_ann)
        if stats: final_results['LongVideoBench'] = {'overall': stats}

    # --- Print Formatting ---
    print(f"\n{'='*80}")
    print(f"{'DATASET':<20} | {'ACC':>7} | {'(CORRECT/TOTAL)':<15} | {'ERRORS'}")
    print(f"{'-'*80}")

    total_acc_sum = 0
    count = 0
    order = ['VideoMME', 'MLVU', 'MVBench', 'LongVideoBench']
    
    for name in order:
        if name in final_results:
            dataset_data = final_results[name]
            
            if name == 'VideoMME':
                if 'overall' in dataset_data:
                    stats = dataset_data['overall']
                    print_row(name, stats)
                    total_acc_sum += stats['accuracy']
                    count += 1
                    
                    for sub in ['short', 'medium', 'long']:
                        if sub in dataset_data:
                            print_row(sub.capitalize(), dataset_data[sub], is_sub=True)
            
            elif 'overall' in dataset_data:
                stats = dataset_data['overall']
                print_row(name, stats)
                total_acc_sum += stats['accuracy']
                count += 1

    print(f"{'-'*80}")
    if count > 0:
        avg = total_acc_sum / count 
        print(f"{'AVERAGE':<20} | {avg:>6.2f}% | {'(Macro Avg)':<15} |")
    else:
        print("No valid results found. Check your file paths.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()