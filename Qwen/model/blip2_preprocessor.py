import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, Blip2Processor
from utils.inference_loader import load_model_for_inference

class Blip2RegionExtractor:
    def __init__(self, train_path=None, device="cuda", device_map="cuda"):
        print(f"INFO: Loading BLIP-2 model from {train_path}")
        self.model = load_model_for_inference(
            model_path=train_path, 
            device_map=device_map, 
            attn_implementation='eager', 
            torch_dtype=torch.bfloat16
        )
        
        self.device = self.model.device
        self.processor = Blip2Processor.from_pretrained(train_path)
        self.qformer_tokenizer = BertTokenizer.from_pretrained(train_path)

        self.model.eval()
        print("INFO: BLIP-2 model loaded successfully.")
    
    def _get_qformer_features(self, pixel_values: torch.FloatTensor, input_ids: torch.LongTensor, interpolate_pos_encoding=False):
        """Helper method to extract Q-Former cross-attention features."""
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=True,
        )
        image_embeds = vision_outputs[0]

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        
        question_embeds = self.model.embeddings(
            input_ids=input_ids,
            query_embeds=query_tokens,
        )
        
        target_dtype = self.model.qformer.layernorm.weight.dtype
        question_embeds = question_embeds.to(dtype=target_dtype)
        
        query_outputs = self.model.qformer(
            query_embeds=question_embeds,
            query_length=query_tokens.shape[1],
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
            output_attentions=True
        )
        return query_outputs
    
    def extract_relevant_patch_indices_dynamic_no_crop(self, 
                                                       video_frames_list, 
                                                       question_texts, 
                                                       thw_grid):
        """
        Extract attention map from Q-Former based on dynamic video inputs without center cropping.
        
        Args:
            video_frames_list: List of PIL Images, Numpy arrays, or Tensors representing a batch of videos.
            question_texts: List of strings representing questions for each video.
            thw_grid: Grid tensor [T, H, W] containing target dimensions from the primary LLM vision encoder.
        """
        target_device = self.model.device

        # ==============================================================================
        # 0. Data Preprocessing: Convert to Tensor (Batch * Time, C, H, W)
        # ==============================================================================
        if isinstance(video_frames_list, list):
            if len(video_frames_list) == 0:
                return []
            if isinstance(video_frames_list[0], np.ndarray):
                video_tensor = torch.from_numpy(np.stack(video_frames_list)).to(target_device)
            elif isinstance(video_frames_list[0], torch.Tensor):
                video_tensor = torch.stack(video_frames_list).to(target_device)
            else:
                raise ValueError("Unsupported input type. Please ensure elements are Tensors or Numpy arrays.")
        elif isinstance(video_frames_list, torch.Tensor):
            video_tensor = video_frames_list.to(target_device)
        else:
            raise ValueError(f"Unexpected input type: {type(video_frames_list)}")

        # Handle 5D input (Batch, Time, C, H, W) -> Flatten to 4D (Batch*Time, C, H, W)
        if video_tensor.dim() == 5:
            b, t, d1, d2, d3 = video_tensor.shape
            video_tensor = video_tensor.view(b * t, d1, d2, d3)

        # Adjust channel order to (N, C, H, W) if it is currently (N, H, W, C)
        if video_tensor.shape[-1] == 3 or video_tensor.shape[-1] < video_tensor.shape[1]: 
            video_tensor = video_tensor.permute(0, 3, 1, 2)

        # Normalize uint8 to [0, 1] float
        if video_tensor.dtype == torch.uint8:
            video_tensor = video_tensor.float() / 255.0

        # ==============================================================================
        # 1. Uniform Resize (Squash/Stretch) without Center Crop
        # ==============================================================================
        # Use Qwen's target grid size directly to ensure BLIP's resolution matches LLM's vision encoder
        h_grid_qwen = int(thw_grid[-2] // 2) 
        w_grid_qwen = int(thw_grid[-1] // 2)
        patch_size = 14
    
        blip_target_h = h_grid_qwen * patch_size
        blip_target_w = w_grid_qwen * patch_size
        
        # Manual interpolation (bicubic) prevents automatic Center Cropping
        resized_tensor = F.interpolate(
            video_tensor, 
            size=(blip_target_h, blip_target_w), 
            mode='bicubic', 
            align_corners=False
        )

        # Construct textual queries (repeat question for each frame in the video)
        all_questions_repeated = []
        batch_size = len(video_frames_list)
        num_frames_per_video = [t.shape[0] for t in video_frames_list]
        for i in range(batch_size):
            all_questions_repeated.extend([question_texts[i]] * num_frames_per_video[i])

        # ==============================================================================
        # 2. Tokenize Inputs (Explicitly disable Processor geometric transforms)
        # ==============================================================================
        text_inputs = self.qformer_tokenizer(
            all_questions_repeated, 
            padding='longest', 
            truncation=True,
            return_tensors="pt"
        ).to(target_device)

        # Apply processor strictly for normalization, bypassing any auto-resize or crop
        image_inputs = self.processor(
            images=resized_tensor, 
            return_tensors="pt", 
            do_resize=False,        
            do_center_crop=False,   
            do_rescale=False,       # assuming input is already scaled by 255.0 above
            input_data_format="channels_first" 
        ).to(target_device)
        
        # ==============================================================================
        # 3. Model Inference
        # ==============================================================================
        with torch.no_grad():
            query_outputs = self._get_qformer_features(
                pixel_values=image_inputs.pixel_values, 
                input_ids=text_inputs.input_ids,
                interpolate_pos_encoding=True
            )
        
        # ==============================================================================
        # 4. Extract Attention & Temporal Fusion (T -> T/2)
        # ==============================================================================
        qformer_cross_attentions = query_outputs.cross_attentions[-1]
        attention_map, _ = qformer_cross_attentions.max(dim=1) 
        
        # Remove CLS token (Index 0)
        raw_attention = attention_map.sum(dim=1)[:, 1:] 
        
        T, N_features = raw_attention.shape
        
        # Pad odd sequence lengths by repeating the last frame's attention
        if T % 2 != 0:
            raw_attention = torch.cat([raw_attention, raw_attention[-1:]], dim=0)
            T += 1 
        
        # Merge adjacent frames using Max Pooling
        grouped_attention = raw_attention.view(T // 2, 2, N_features)
        fused_attention, _ = grouped_attention.max(dim=1)
        
        return fused_attention