import os
import numpy as np
from torch.utils.data import Dataset
from decord import VideoReader, cpu

data_list = {
    "Action Sequence": ("action_sequence.json", f"star/Charades_v1_480/", "video", True), 
    "Action Prediction": ("action_prediction.json", f"star/Charades_v1_480/", "video", True), 
    "Action Antonym": ("action_antonym.json", f"ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", f"Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", f"FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", f"clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", f"star/Charades_v1_480/", "video", True), 
    "Object Shuffle": ("object_shuffle.json", f"perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", f"clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", f"sta/sta_video/", "video", True),  
    "Scene Transition": ("scene_transition.json", f"scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", f"perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", f"clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", f"clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", f"perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", f"nturgbd/", "video", False),
    "Character Order": ("character_order.json", f"perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", f"vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", f"tvqa/frames_fps3_hq/", "frame", True),  
    "Counterfactual Inference": ("counterfactual_inference.json", f"clevrer/video_validation/", "video", False),
}

class MVBenchDataset(Dataset):
    def __init__(self, data_dir, dataset_json, task_type):
        self.data_dir = data_dir
        self.dataset = dataset_json
        self.task_type = task_type
        
    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        video_path = os.path.join(self.data_dir, sample['video'])
            
        return {
            'video_path' : video_path,
            'question_id': sample['video'],
            'video_id': sample['video'],
            'question': sample['question'], 
            'answer': sample['answer'],
            'task_type': self.task_type,
            'candidates': sample['candidates'],
        }

def load_video_mvbench(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx =[i for i in range(0, len(vr), fps)]
    frame_time =[i / fps for i in frame_idx]
    
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time =[i / vr.get_avg_fps() for i in frame_idx]
        
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time