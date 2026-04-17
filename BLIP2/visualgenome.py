
import json
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class VisualGenomeDataset(Dataset):
    def __init__(self, vg_root_dir, processor=None):
        self.vg_root = vg_root_dir
        self.image_dir = os.path.join(vg_root_dir, 'VG_100K') 
        
        print("Loading image metadata...")
        with open(os.path.join(vg_root_dir, 'image_data.json'), 'r') as f:
            imgs_meta = json.load(f)
            self.id_to_filename = {img['image_id']: img['url'].split('/')[-1] for img in imgs_meta}

        print("Loading region descriptions (this might take a while)...")
        with open(os.path.join(vg_root_dir, 'region_descriptions.json'), 'r') as f:
            regions_data = json.load(f)
        
        self.samples = []
        for entry in regions_data:
            image_id = entry['id']
            if image_id not in self.id_to_filename:
                continue
                
            filename = self.id_to_filename[image_id]
            full_path = os.path.join(self.image_dir, filename)
            
            for region in entry['regions']:
                phrase = region['phrase']
                if len(phrase.split()) >= 3: 
                    self.samples.append({
                        'image_path': full_path,
                        'text': phrase
                    })
        
        print(f"Loaded {len(self.samples)} region-text pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            image = Image.open(item['image_path']).convert('RGB')
            return image, item['text']
        except Exception as e:
            print(f"Error loading image {item['image_path']}: {e}")
            return None