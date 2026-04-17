import os
import json
import glob
import logging 
import numpy as np
import pysubs2
from PIL import Image
from torch.utils.data import Dataset 
from decord import VideoReader, cpu

class VideoMMEDataset(Dataset):
    def __init__(self, hf_dataset, video_root):
        self.hf_dataset = hf_dataset
        self.video_root = video_root

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        video_id = sample.get('videoID')
        question = sample.get('question')
        answer = sample.get('answer')
        candidates = sample.get('options',[])
        video_filename = f"{video_id}.mp4" 
        video_path = os.path.join(self.video_root, video_filename)

        if not os.path.exists(video_path):
            logging.warning(f"Video file not found at {video_path}")

        return {
            "video_path": video_path,
            "video_id": video_id,
            "question_id": sample.get('question_id'),
            "question": question,
            "candidates": candidates,
            "answer": answer,
        }

def extract_subtitles_for_keyframes(keyframe_indices, video_fps, srt_path):
    if not srt_path or not os.path.exists(srt_path) or video_fps == 0:
        return ""
    try:
        subs = pysubs2.load(srt_path, encoding="utf-8")
        if not subs: return ""
    except Exception as e:
        logging.warning(f"Could not load or parse subtitle file {srt_path}: {e}")
        return ""

    found_subtitles, seen_subtitles = [], set()
    for frame_index in keyframe_indices:
        current_time = pysubs2.make_time(fps=video_fps, frames=frame_index)
        for sub in subs:
            if sub.start <= current_time < sub.end:
                sub_text = sub.text.replace("\\N", " ").strip()
                if sub_text and sub_text not in seen_subtitles:
                    found_subtitles.append(sub_text)
                    seen_subtitles.add(sub_text)
                break
    return "\n".join(found_subtitles)

def load_video_videomme(video_path, num_frames):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    original_fps = vr.get_avg_fps()
    
    return frames, indices, original_fps