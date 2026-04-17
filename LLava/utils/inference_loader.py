import torch
import os
from transformers import Blip2Config
from transformers import Blip2ForImageTextRetrieval

def load_model_for_inference(
    model_path="Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.bfloat16, 
    device_map="cuda", 
    attn_implementation="eager"
):
    print(f"Loading base configuration: {model_path}")
    config = Blip2Config.from_pretrained(model_path)
    
    config.qformer_config.use_qformer_text_input = True

    model = Blip2ForImageTextRetrieval.from_pretrained(
        model_path,
        config=config,   
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation
    )

    print("Applying weight sharing...")
    for layer in model.qformer.encoder.layer:
        layer.intermediate = layer.intermediate_query
        layer.output = layer.output_query

    model.eval()
    return model