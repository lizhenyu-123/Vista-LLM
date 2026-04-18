"""
Video Frame Extractor

A multiprocessing script to extract a fixed number of uniformly sampled frames 
from video files. It also generates a metadata.json for each video containing 
details like FPS, total frames, and the specific indices sampled.
"""

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import signal

import cv2
import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm

# --- Global Configurations & Signals ---
SUPPORTED_EXTENSIONS = {'.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv', '.wmv'}
terminate_flag = mp.Event()

def handle_sigint(signum, frame):
    """Catch Ctrl+C (SIGINT/SIGTERM) and trigger the global termination flag."""
    print("\n[!] Ctrl+C detected, stopping all workers...")
    terminate_flag.set()

signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)


# --- Core Logic ---
def extract_and_save_frames(video_path, num_frames_to_sample, output_dir):
    """
    Extracts uniformly sampled frames from a video and saves them as images.
    Also creates a metadata.json file in the output directory.

    Args:
        video_path (str): Path to the input video.
        num_frames_to_sample (int): Number of frames to extract.
        output_dir (str): Directory to save extracted frames and metadata.

    Returns:
        tuple: (video_path, success_boolean, status_message)
    """
    try:
        # Check termination flag before starting a new task
        if terminate_flag.is_set():
            return video_path, False, "Terminated manually"

        if not os.path.exists(video_path):
            return video_path, False, "File not found"

        # Skip if already processed (checked via the existence of metadata.json)
        meta_filepath = os.path.join(output_dir, "metadata.json")
        if os.path.exists(meta_filepath):
            return video_path, True, "Already processed"

        os.makedirs(output_dir, exist_ok=True)

        # decord supports most ffmpeg-compatible formats
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        original_fps = vr.get_avg_fps()

        if total_frames == 0:
            return video_path, False, "Video has zero frames"

        # Uniformly sample frame indices
        indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
        frames = vr.get_batch(indices).asnumpy()

        for i, frame_index in enumerate(indices):
            # Check termination flag during the extraction loop
            if terminate_flag.is_set():
                return video_path, False, "Terminated manually"

            frame_image = frames[i]
            
            # decord outputs RGB, but OpenCV requires BGR for saving
            frame_image_bgr = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
            output_filename = f"frame_{frame_index:06d}.jpg"
            output_filepath = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_filepath, frame_image_bgr)

        # Save metadata
        metadata = {
            'original_fps': original_fps,
            'total_frames': total_frames,
            'frame_indices': indices.tolist(),
            'num_sampled_frames': len(indices),
            'video_time': total_frames / original_fps if original_fps > 0 else 0
        }
        
        with open(meta_filepath, 'w') as f:
            json.dump(metadata, f, indent=4)

        return video_path, True, "Success"

    except Exception as e:
        return video_path, False, str(e)


def worker_function(params):
    """Unpack parameters for the executor map function."""
    video_path, n_frames, output_dir = params
    return extract_and_save_frames(video_path, n_frames, output_dir)


# --- Main Entry Point ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-extract frames and metadata from videos using multiple processes.")
    parser.add_argument('--video-root', type=str, default="./videos", help="Root directory containing input videos.")
    parser.add_argument('--output-root', type=str, default="./video_frames", help="Root directory to save extracted frames.")
    parser.add_argument('--n-frames', type=int, default=64, help="Number of frames to sample per video.")
    parser.add_argument('--num-workers', type=int, default=None, help="Number of concurrent workers. Defaults to CPU count.")
    args = parser.parse_args()

    if args.num_workers is None:
        args.num_workers = os.cpu_count()
    
    print(f"Using {args.num_workers} CPU cores.")
    print(f"Scanning for videos in '{args.video_root}' (Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)})...")
    
    video_files = []
    
    # Traverse directory to find all supported video files
    for root, dirs, files in os.walk(args.video_root):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                video_files.append(os.path.join(root, file))

    print(f"Found {len(video_files)} videos to process.")

    # Prepare task arguments
    tasks = []
    for video_path in video_files:
        relative_path = os.path.relpath(os.path.dirname(video_path), args.video_root)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(args.output_root, relative_path, video_name)
        tasks.append((video_path, args.n_frames, video_output_dir))

    success_count = 0
    fail_count = 0
    already_processed_count = 0

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # chunksize=1 ensures a smooth progress bar, but can be increased for massive numbers of small tasks.
            results_iterator = executor.map(worker_function, tasks, chunksize=1)

            for video_path, success, message in tqdm(results_iterator, total=len(tasks), desc="Extracting Frames"):
                if terminate_flag.is_set():
                    raise KeyboardInterrupt

                if success:
                    if message == "Already processed":
                        already_processed_count += 1
                    else:
                        success_count += 1
                else:
                    fail_count += 1
                    print(f"\n[!] Failed to process {os.path.basename(video_path)}: {message}")

    except KeyboardInterrupt:
        print("\n[!] KeyboardInterrupt caught, terminating workers...")
        terminate_flag.set()
        executor.shutdown(wait=False, cancel_futures=True)
        os._exit(0)  # Force exit to prevent hanging processes

    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")
        terminate_flag.set()
        executor.shutdown(wait=False, cancel_futures=True)
        os._exit(1)

    # Print summary
    print("\n" + "="*32)
    print("   Frame Extraction Summary   ")
    print("="*32)
    print(f" Successfully processed : {success_count}")
    print(f" Already processed      : {already_processed_count} (skipped)")
    print(f" Failed                 : {fail_count}")
    print("="*32)