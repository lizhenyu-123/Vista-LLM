import os
import json
import glob
import logging

import numpy as np
from PIL import Image

def load_preextracted_frames_and_meta(frames_dir):
    meta_path = os.path.join(frames_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found in {frames_dir}")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    frame_indices = np.array(metadata['frame_indices'])
    original_fps = metadata['original_fps']
    video_time = metadata.get('video_time', 0.0) 
    frame_time_list = [i / original_fps for i in frame_indices]
    frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time_list])
    image_files = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
    
    if not image_files:
        raise FileNotFoundError(f"No image files (.jpg) found in {frames_dir}")
        
    frames_list =[]
    if len(image_files) != len(frame_indices):
        logging.warning(f"Mismatch between image files ({len(image_files)}) and indices ({len(frame_indices)}) in {frames_dir}. Using indices in metadata.")
        for fp in frame_indices:
            img = Image.open(image_files[fp-1])
            frames_list.append(np.array(img, dtype=np.uint8))
    else:
        for fp in image_files:
            img = Image.open(fp)
            frames_list.append(np.array(img, dtype=np.uint8))
            
    raw_frames = np.stack(frames_list, axis=0)
    return raw_frames, frame_indices, original_fps, frame_time_str, video_time