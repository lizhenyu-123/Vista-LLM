import os
import logging 
import hashlib
import numpy as np
from torch.utils.data import Dataset
from decord import VideoReader, cpu

class MLVUDataset(Dataset):
    def __init__(self, dataset_json, video_root, task_type):
        self.dataset = dataset_json
        self.video_root = video_root
        logging.info(f"Loaded {len(self.dataset)} samples for the '{task_type}' task.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        answer = sample.get('answer')
        video_id = sample.get('video')
        question = sample.get('question')
        candidates = sample.get('candidates')

        if not isinstance(question, str):
            question = str(question)

        question_bytes = question.encode('utf-8')
        question_hash = hashlib.sha256(question_bytes).hexdigest()
        unique_task_id = f"{video_id}-{question_hash}"
        video_path = os.path.join(self.video_root, video_id)

        if not os.path.exists(video_path):
            logging.warning(f"Video file not found at {video_path}")

        return {
            "video_path": video_path,
            "video_id": video_id,
            "question_id": unique_task_id, 
            "question": question,
            "candidates": candidates,
            "answer": answer,
        }


def load_video_mlvu(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time =[i / fps for i in frame_idx]
    
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time =[i / vr.get_avg_fps() for i in frame_idx]
        
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    
    return spare_frames, frame_time, video_time