import torch
from transformers import BertTokenizer, Blip2Processor
from utils.inference_loader import load_model_for_inference

class Blip2RegionExtractor:
    def __init__(self, train_path=None, device_map="auto"):
        print(f"Loading BLIP-2 model: {train_path}")
        self.model = load_model_for_inference(model_path=train_path, device_map=device_map, attn_implementation='eager', torch_dtype=torch.bfloat16)
        
        self.device = self.model.device
        self.processor = Blip2Processor.from_pretrained(train_path)
        self.qformer_tokenizer = BertTokenizer.from_pretrained(train_path)

        self.model.eval()
        print("BLIP-2 model loaded.")

    def _get_qformer_features(self, pixel_values: torch.FloatTensor, input_ids: torch.LongTensor):
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
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

    def extract_relevant_patch_indices_dynamic(self, 
                                               video_frames_list, 
                                               question_texts, 
                                               ):
        if len(video_frames_list) == 0:
            return torch.empty(0), torch.tensor([0])

        target_device = self.model.device
        batch_size = len(video_frames_list)

        flat_video_frames =[]  
        for i in range(batch_size):
            video_frames = video_frames_list[i] 
            flat_video_frames.extend(video_frames)
        num_frames_per_video = [t.shape[0] for t in video_frames_list]
        
        all_questions_repeated =[]
        for i in range(batch_size):
            all_questions_repeated.extend([question_texts[i]] * num_frames_per_video[i])

        image_inputs = self.processor(images=flat_video_frames, return_tensors="pt").to(target_device)
        text_inputs = self.qformer_tokenizer(
            all_questions_repeated, 
            padding='longest', 
            truncation=True,
            return_tensors="pt"
        ).to(target_device)
        input_ids = text_inputs.input_ids
        
        with torch.no_grad():  
            query_outputs = self._get_qformer_features(
                pixel_values=image_inputs.pixel_values, 
                input_ids=input_ids
            )
        
        qformer_cross_attentions = query_outputs.cross_attentions[-1]
        
        attention_map, _ = qformer_cross_attentions.max(dim=1)  
        image_patch_attention = attention_map.sum(dim=1)     
        image_patch_attention = image_patch_attention[:, 1:]     

        attentions_per_video = list(torch.split(image_patch_attention, num_frames_per_video, dim=0))

        return attentions_per_video[0]