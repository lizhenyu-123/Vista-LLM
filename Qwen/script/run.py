# --- START OF FILE run_qwen2_5_vl.py ---

import os
import json
import argparse
import time
import gc
import logging
import copy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor

from model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from model.blip2_preprocessor import Blip2RegionExtractor

from dataset.videomme import VideoMMEDataset, extract_subtitles_for_keyframes, load_video_videomme
from dataset.mlvu import MLVUDataset, load_video_mlvu
from dataset.longvideobench import LongVideoBenchDataset, load_video_longvideobench
from dataset.mvbench import MVBenchDataset, data_list, load_video_mvbench
from utils.utils import load_preextracted_frames_and_meta

def prepare_qwen_prompt(dataset_name, question, candidates, video_id=None, srt_path=None, frame_indices=None, original_fps=None):
    """
    根据不同的数据集构建对应的 Prompt。
    """
    if dataset_name == 'videomme':
        subtitles_text = extract_subtitles_for_keyframes(frame_indices, original_fps, srt_path) if srt_path else None
        options_str = "\n".join([f"{opt}" for opt in candidates])
        base_prompt = (
            f"Select the best answer to the following multiple-choice question based on the video. "
            f"Respond with only the letter (A, B, C, or D) of the correct option.\n"
            f"{question}\n{options_str}\nThe best answer is:"
        )
        return f"This video's subtitles are listed below:\n{subtitles_text}\n\n{base_prompt}" if subtitles_text else base_prompt

    elif dataset_name == 'mlvu':
        options_str = "\n".join([f"{chr(65+j)}. {cand}" for j, cand in enumerate(candidates)])
        return f"Based on the video, answer the following question. Only provide the text of the correct option.\n\nQuestion: {question}\n\nOptions:\n{options_str}\n\nAnswer:"

    elif dataset_name == 'longvideobench':
        prompt = f"Question: {question}\nOptions:\n"
        prompt += "\n".join([f"{chr(ord('A')+j)}. {cand}" for j, cand in enumerate(candidates)])
        prompt += "\nAnswer with the option's letter from the given choices directly."
        return prompt

    elif dataset_name == 'mvbench':
        options_str = "\n".join([f"{chr(65+j)}. {cand}" for j, cand in enumerate(candidates)])
        return (
            f"Carefully watch the video and pay attention to the cause and sequence of events, "
            f"the detail and movement of objects, and the action and pose of persons. "
            f"Based on your observations, select the best option that accurately addresses the question.\n"
            f"Question: {question}\n"
            f"Options:\n{options_str}\n"
            f"Only give the best option."
        )
    return ""


def get_question_embedding(question, processor, model, device):
    """
    获取 Qwen2.5-VL 的问题句子级别的 Embedding
    """
    question_ids = processor.tokenizer(question, return_tensors="pt").input_ids.to(device)
    # Qwen2.5 语言模型的 embed_tokens 路径
    question_word_embeddings = model.model.language_model.embed_tokens(question_ids) 
    question_sentence_embedding = question_word_embeddings[0].mean(dim=0)
    return question_sentence_embedding


def generate(args):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logging.info("--- Phase 1: Loading Qwen2.5-VL Models ---")
    
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2", 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()

    feature_model = None
    if args.use_way: 
        logging.info("Dynamic token way is enabled. Loading BLIP2 model.")
        feature_model = Blip2RegionExtractor(
            train_path=args.train_path,
            device_map="auto",
        )
        feature_model.model.eval()

    logging.info(f"--- Phase 2: Loading Dataset '{args.dataset}' ---")
    if args.dataset == 'videomme':
        hf_full_dataset = load_dataset('parquet', data_files={'test': args.ann_file})['test']
        hf_filtered_dataset = hf_full_dataset if args.task_type == "all" else hf_full_dataset.filter(lambda ex: ex['duration'] == args.task_type)
        dataset = VideoMMEDataset(hf_dataset=hf_filtered_dataset, video_root=args.video_root)

    elif args.dataset == 'mlvu':
        with open(args.ann_file, 'r', encoding='utf-8') as f:
            dataset_json = json.load(f)
        dataset = MLVUDataset(dataset_json, args.video_root, args.task_type)

    elif args.dataset == 'longvideobench':
        dataset = LongVideoBenchDataset(data_path=args.video_root, annotation_file=args.ann_file)

    elif args.dataset == 'mvbench':
        prefix = data_list[args.task_type][1]
        json_path = data_list[args.task_type][0]
        with open(os.path.join(args.ann_file, json_path), 'r', encoding='utf-8') as f:
            dataset_json = json.load(f)
        data_dir = os.path.join(args.frames_root, prefix) if args.use_preextracted_frames else args.video_root
        dataset = MVBenchDataset(data_dir=data_dir, dataset_json=dataset_json, task_type=args.task_type)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}.")

    def custom_collate_fn(batch):
        keys = batch[0].keys()
        return {key: [d[key] for d in batch] for key in keys}

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=4, pin_memory=True
    )

    logging.info(f"\n--- Phase 3: Generating Answers for '{args.dataset}' Task ---")
    output_file = os.path.join(args.output_dir, f"{args.dataset}_results.json")
    
    results, processed_ids = [], set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                processed_ids = {res.get("question_id") for res in results}
            logging.info(f"Loaded {len(results)} existing results. Resuming...")
        except (json.JSONDecodeError, IOError):
            pass

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Answering {args.dataset} questions")
        for batch in progress_bar:
            batch_start_time = time.time()
            valid_batch_samples = []

            batch_input_text = []
            batch_video_inputs = []   
            batch_questions = []     
            batch_question_embeddings = []  

            for i in range(len(batch["question_id"])):
                video_id = batch["video_id"][i]
                video_path = batch["video_path"][i]
                question_id = batch["question_id"][i]
                question = batch["question"][i]
                candidates = batch["candidates"][i]
                
                if question_id in processed_ids: 
                    continue

                try:
                    # --- 1. 视频/帧提取 ---
                    if args.use_preextracted_frames:
                        frames_dir = os.path.join(args.frames_root, args.task_type)
                        if args.dataset == 'mlvu':
                            preextracted_path = os.path.join(frames_dir, os.path.splitext(video_id)[0])
                        elif args.dataset == 'longvideobench':
                            preextracted_path = os.path.join(args.frames_root, video_id)
                        elif args.dataset == 'mvbench':
                            preextracted_path = os.path.splitext(video_path)[0]
                        else:
                            preextracted_path = os.path.join(frames_dir, video_id)
                        
                        raw_frames, frame_indices, original_fps, frame_time, video_time = load_preextracted_frames_and_meta(preextracted_path)
                    else:
                        if args.dataset == 'videomme':
                            raw_frames, frame_indices, original_fps = load_video_videomme(video_path, args.n_frames)
                        elif args.dataset == 'mlvu':
                            raw_frames, frame_time, video_time = load_video_mlvu(video_path, args.n_frames, 1, force_sample=True)
                        elif args.dataset == 'longvideobench':
                            raw_frames, frame_time = load_video_longvideobench(video_path, batch["duration"][i], 64)
                        elif args.dataset == 'mvbench':
                            raw_frames, frame_time, video_time = load_video_mvbench(video_path, args.n_frames, 1, force_sample=True)

                    srt_path = os.path.join(args.srt_root, f"{video_id}.srt") if args.dataset == 'videomme' and args.srt_root else None
                    prompt_text = prepare_qwen_prompt(
                        dataset_name=args.dataset,
                        question=question,
                        candidates=candidates,
                        video_id=video_id,
                        srt_path=srt_path,
                        frame_indices=frame_indices if args.dataset == 'videomme' else None,
                        original_fps=original_fps if args.dataset == 'videomme' else None
                    )

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "video", "video": raw_frames},
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]
                    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    batch_input_text.append(text_input)
                    batch_video_inputs.append(raw_frames) 

                    if args.use_way: 
                        question_embed = get_question_embedding(question, processor, model, device)
                        batch_question_embeddings.append(question_embed)
                        batch_questions.append(question)

                    valid_batch_samples.append({key: batch[key][i] for key in batch})

                except Exception as e:
                    logging.error(f"Error pre-processing {question_id}: {e}", exc_info=True)
                    results.append({"question_id": question_id, "generated_answer": f"[ERROR] Pre-processing failed: {e}"})
                    processed_ids.add(question_id)

            if not valid_batch_samples:
                continue

            try:
                inputs = processor(
                    text=batch_input_text,
                    videos=batch_video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                final_question_embeddings = torch.stack(batch_question_embeddings, dim=0) if batch_question_embeddings else None

                t1_gen = time.time()
                
                generate_kwargs = {
                    **inputs,
                    "max_new_tokens": 32,
                    "do_sample": False,
                }
                
                if args.use_way:
                    generate_kwargs.update({
                        "feature_model": feature_model,
                        "raw_images": batch_video_inputs,
                        "doimain_rate": args.domain_rate,  
                        "question_embedding": final_question_embeddings,
                        "questions": batch_questions,
                    })

                generated_ids = model.generate(**generate_kwargs)
                
                t2_gen = time.time()
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                text_outputs = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                
                batch_end_time = time.time()
                logging.info(f"Processed batch of size {len(valid_batch_samples)} in {batch_end_time - batch_start_time:.2f}s.")

                # --- 6. 保存结果 ---
                for i, output_text in enumerate(text_outputs):
                    sample = valid_batch_samples[i]
                    results.append({
                        "question_id": sample["question_id"],
                        "question": sample["question"],
                        "candidates": sample["candidates"],
                        "correct_answer": sample.get("answer", "N/A"),
                        "generated_answer": output_text.strip(),
                        "generate_time": t2_gen - t1_gen,
                        "all_time": batch_end_time - batch_start_time,
                    })
                    processed_ids.add(sample["question_id"])

            except Exception as e:
                logging.error(f"Error during batch inference: {e}", exc_info=True)
                for sample in valid_batch_samples:
                    results.append({"question_id": sample["question_id"], "generated_answer": f"[ERROR] Batch inference failed: {e}"})
                    processed_ids.add(sample["question_id"])
            finally:
                if 'inputs' in locals(): del inputs
                if 'generated_ids' in locals(): del generated_ids
                gc.collect()
                if torch.cuda.is_available(): 
                    torch.cuda.empty_cache()
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)

    logging.info(f"Processing finished. All results saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on Video QA datasets with Qwen2.5-VL.")

    parser.add_argument('--dataset', type=str, required=True, choices=['videomme', 'mlvu', "longvideobench", "mvbench"], help="The dataset to run inference on.")
    
    parser.add_argument('--model-path', type=str, required=True, help="Path to the Qwen2.5-VL model.")
    parser.add_argument('--feature-checkpoint', type=str, default=None, help="Path to the BLIP2 checkpoint for dynamic tokens.")
    parser.add_argument('--train-path', type=str, default=None, help="Path to the trained checkpoint for BLIP2.")
    parser.add_argument('--output-dir', type=str, default=None, help="Directory to save results and logs.")
    
    parser.add_argument('--ann-file', type=str, default=None, help="Path to the annotation file (JSON/Parquet).")
    parser.add_argument('--video-root', type=str, default=None, help="Base root directory for videos.")
    parser.add_argument('--srt-root', type=str, default=None, help="Root directory of the subtitle files (VideoMME).")
    parser.add_argument('--task-type', type=str, default="short", help="Task type or duration filter.")
    parser.add_argument('--frames-root', type=str, default=None, help="Root directory of the pre-extracted frames.")

    parser.add_argument('--videomme-ann-file', type=str, default=None)
    parser.add_argument('--videomme-video-root', type=str, default=None)
    parser.add_argument('--mlvu-ann-file', type=str, default=None)
    parser.add_argument('--mlvu-video-root', type=str, default=None)
    parser.add_argument('--longvideobench-ann-file', type=str, default=None)
    parser.add_argument('--longvideobench-video-root', type=str, default=None)
    parser.add_argument('--mvbench-ann-file', type=str, default=None)
    parser.add_argument('--mvbench-video-root', type=str, default=None)

    parser.add_argument('--batch-size', type=int, default=1, help="Batch size for model inference.")
    parser.add_argument('--n-frames', type=int, default=64, help="Number of frames to uniformly sample from the video.")
    parser.add_argument('--use-way', action='store_true', help="Enable dynamic token selection way (requires BLIP2).")
    parser.add_argument('--use-preextracted-frames', action='store_true', help="If set, load frames from a pre-extracted directory instead of video files.")
    
    parser.add_argument('--domain-rate', type=float, default=0.0, help="Rate of BLIP2 tokens and MMGVID tokens.")

    args = parser.parse_args()

    if args.dataset == 'videomme':
        args.ann_file = args.ann_file or args.videomme_ann_file
        args.video_root = os.path.join((args.video_root or args.videomme_video_root), args.task_type)
    elif args.dataset == 'mlvu':
        args.ann_file = os.path.join((args.ann_file or args.mlvu_ann_file), f"{args.task_type}.json")
        args.video_root = os.path.join((args.video_root or args.mlvu_video_root), args.task_type)
    elif args.dataset == 'longvideobench':
        args.ann_file = args.ann_file or args.longvideobench_ann_file
        args.video_root = args.video_root or args.longvideobench_video_root
    elif args.dataset == 'mvbench':
        args.ann_file = args.ann_file or args.mvbench_ann_file
        args.video_root = args.video_root or args.mvbench_video_root

    folder_name = f"{args.task_type}_domain{args.domain_rate}_tau{os.getenv('TAU_SIMILARITY_THRESHOLD', 0.95)}"
    base_out = args.output_dir or f"./results_{args.dataset}"
    args.output_dir = os.path.join(base_out, folder_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 设置日志 ---
    log_file_path = os.path.join(args.output_dir, f"run_log_{args.dataset}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[logging.FileHandler(log_file_path, mode='a'), logging.StreamHandler()]
    )

    logging.info(f"--- Starting new run on '{args.dataset}' ---")
    logging.info("Configuration:")
    for key, value in vars(args).items(): 
        logging.info(f"{key:<30}: {value}")
    
    logging.info("-" * 50)
    interested_env_vars =[
        "USE_AGGREGATION",
        "SELECTION_MODE",
        "RETENTION_RATIO", 
        "R_MIN_RATIO", 
        "TAU_SIMILARITY_THRESHOLD"
    ]
    logging.info("Environment Variables Configuration:")
    for key in interested_env_vars:
        logging.info(f"{key:<30}: {os.getenv(key, 'Not Set (Use Code Default)')}")
    logging.info("-" * 50 + "\n")
    
    try:
        generate(args)
        logging.info("Script finished successfully.")
    except Exception as e:
        logging.error("An unhandled exception occurred in the main script.", exc_info=True)