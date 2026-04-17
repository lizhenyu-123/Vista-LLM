import os
import json
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from PIL import Image

class LongVideoBenchDataset(Dataset):
    def __init__(self, data_path: str, annotation_file: str):
        super().__init__()
        self.data_path = data_path
        with open(os.path.join(data_path, annotation_file)) as f:
            self.data = json.load(f)
        
    def __getitem__(self, index: int) -> dict:
        di = self.data[index]
        video_id = di["video_path"].replace(".mp4", "")

        return {
            "question_id": di["id"],
            "video_id": video_id,
            "video_path": os.path.join(self.data_path, "videos", di["video_path"]),
            "question": di["question"],
            "candidates": di["candidates"],
            "answer": chr(ord("A") + di.get("correct_choice", -1)),
            "duration": di["duration"],
            "starting_timestamp_for_subtitles": di["starting_timestamp_for_subtitles"],
        }
    
    def __len__(self) -> int:
        return len(self.data)

def load_video_longvideobench(video_file, duration, max_num_frames=16):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices =[int(total_valid_frames / num_frames) * i for i in range(num_frames)]
    
    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]
    
    return [Image.fromarray(fr).convert("RGB") for fr in frames], frame_timestamps