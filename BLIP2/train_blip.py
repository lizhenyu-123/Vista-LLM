import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2ForImageTextRetrieval, BertTokenizer, Blip2Processor
from datetime import datetime
from functools import partial
from BLIP2.visualgenome import VisualGenomeDataset

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def setup_logging(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_file = os.path.join(output_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class KeyTokenProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.GELU(),                
            nn.Dropout(dropout),      
            nn.Linear(hidden_dim, output_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class Blip2KeyTokenTrainer(nn.Module):
    def __init__(self, model_name="Salesforce/blip2-itm-vit-g", freeze_vision=True):
        super().__init__()
        
        logging.info(f"Loading model: {model_name}...")
        self.model = Blip2ForImageTextRetrieval.from_pretrained(model_name, torch_dtype=torch.float32)
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # put your local path here

        for layer in self.model.qformer.encoder.layer:
            if not hasattr(layer, 'intermediate') or layer.intermediate is None:
                layer.intermediate = layer.intermediate_query
                layer.output = layer.output_query

        if freeze_vision:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            logging.info("Vision Encoder frozen.")
        else:
            logging.info("Vision Encoder is trainable.")
        
        self.vision_dim = 1408 
        self.itc_dim = 256     
        self.patch_proj = KeyTokenProjector(
            input_dim=self.vision_dim, 
            output_dim=self.itc_dim,
            hidden_dim=768,
            dropout=0.1
        )
            
        self.itm_loss_fct = nn.CrossEntropyLoss()

    def load_weights_from_checkpoint(self, checkpoint_path):
        logging.info(f"Overwriting weights from checkpoint: {checkpoint_path}")
        
        weight_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu")
            msg = self.model.load_state_dict(state_dict, strict=False)
            logging.info(f"Main model weights loaded. Missing keys: {msg.missing_keys}")
        else:
            logging.warning(f"Main model weights not found at {weight_path}!")

        patch_path = os.path.join(checkpoint_path, "patch_proj.bin")
        if os.path.exists(patch_path):
            patch_state = torch.load(patch_path, map_location="cpu")
            self.patch_proj.load_state_dict(patch_state)
            logging.info("Patch projector weights loaded.")
        else:
            logging.warning(f"Patch projector weights not found at {patch_path}!")

    def forward(self, pixel_values, text_input_ids, text_attention_mask, 
                alpha=1.0, K=64, temperature=0.1):
        
        batch_size = pixel_values.shape[0]
        device = pixel_values.device

        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            return_dict=True
        )
        image_embeds = vision_outputs.last_hidden_state 
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

        query_embeds = self.model.embeddings(input_ids=text_input_ids)
        text_output = self.model.qformer(
            query_embeds=query_embeds,
            query_length=0,
            attention_mask=text_attention_mask,
            return_dict=True
        )
        text_feat = text_output.last_hidden_state[:, 0, :]
        text_feat_norm = F.normalize(self.model.text_projection(text_feat), dim=-1)

        with torch.no_grad():
            query_tokens = self.model.query_tokens.expand(batch_size, -1, -1)
            query_output = self.model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True
            )
            image_feat = F.normalize(self.model.vision_projection(query_output.last_hidden_state), dim=-1)
            image_feat_norm = image_feat.mean(dim=1) 

            sims = image_feat_norm @ text_feat_norm.t()
            sims.fill_diagonal_(-100.0)
            weights_t2i = F.softmax(sims, dim=1)
            weights_i2t = F.softmax(sims, dim=0)
            neg_text_idx = torch.multinomial(weights_i2t, 1).squeeze()
            neg_image_idx = torch.multinomial(weights_t2i, 1).squeeze()

        image_embeds_world = torch.cat([
            image_embeds,               
            image_embeds,               
            image_embeds[neg_image_idx] 
        ], dim=0)
        
        text_ids_world = torch.cat([
            text_input_ids,             
            text_input_ids[neg_text_idx], 
            text_input_ids              
        ], dim=0)
        
        text_atts_world = torch.cat([
            text_attention_mask,
            text_attention_mask[neg_text_idx],
            text_attention_mask
        ], dim=0)

        itm_labels = torch.cat([
            torch.ones(batch_size, dtype=torch.long, device=device),  
            torch.zeros(2 * batch_size, dtype=torch.long, device=device) 
        ])

        pos_mask = torch.cat([
            torch.ones(batch_size, dtype=torch.bool, device=device),
            torch.zeros(2 * batch_size, dtype=torch.bool, device=device)
        ])

        bs_world = 3 * batch_size
        shuffle_idx = torch.randperm(bs_world, device=device)

        image_embeds_world = image_embeds_world[shuffle_idx]
        text_ids_world = text_ids_world[shuffle_idx]
        text_atts_world = text_atts_world[shuffle_idx]
        itm_labels = itm_labels[shuffle_idx]
        pos_mask = pos_mask[shuffle_idx] 

        query_tokens_world = self.model.query_tokens.expand(bs_world, -1, -1)
        query_atts_world = torch.ones(query_tokens_world.size()[:-1], dtype=torch.long, device=device)
        image_atts_world = torch.ones(image_embeds_world.size()[:-1], dtype=torch.long, device=device)
        attention_mask_world = torch.cat([query_atts_world, text_atts_world], dim=1)

        query_embeds_world = self.model.embeddings(
            input_ids=text_ids_world,
            query_embeds=query_tokens_world,
        )

        itm_outputs = self.model.qformer(
            query_embeds=query_embeds_world,
            query_length=query_tokens_world.shape[1],
            attention_mask=attention_mask_world,
            encoder_hidden_states=image_embeds_world,
            encoder_attention_mask=image_atts_world,
            output_attentions=True,
            return_dict=True
        )
        
        itm_logits = self.model.itm_head(itm_outputs.last_hidden_state[:, :query_tokens_world.size(1), :]).mean(dim=1)
        loss_itm = self.itm_loss_fct(itm_logits, itm_labels)

        if pos_mask.sum() > 0: 
            cross_attentions_pos = itm_outputs.cross_attentions[-1][pos_mask] 
            patch_score = cross_attentions_pos.mean(dim=1).sum(dim=1)[:, 1:] 
            valid_image_embeds = image_embeds_world[pos_mask][:, 1:, :]
            
            topk_values, _ = torch.topk(patch_score, k=K, dim=-1, sorted=True)
            threshold = topk_values[:, -1].unsqueeze(1)
            soft_topk_mask = torch.sigmoid((patch_score - threshold) / temperature)
            base_weights = F.softmax(patch_score / temperature, dim=-1)
            masked_weights = base_weights * soft_topk_mask
            final_weights = masked_weights / (masked_weights.sum(dim=-1, keepdim=True) + 1e-9)

            weighted_visual_feat = torch.bmm(
                final_weights.unsqueeze(1), 
                valid_image_embeds
            ).squeeze(1)
            
            indices_world = torch.arange(bs_world, device=device)
            indices_shuffled = indices_world[shuffle_idx] 
            
            original_ids = indices_shuffled[pos_mask] 
            target_text_feat = text_feat[original_ids] 
            
            proj_visual = self.patch_proj(weighted_visual_feat) 
            proj_visual = F.normalize(proj_visual, dim=-1)
            target_text_feat = F.normalize(self.model.text_projection(target_text_feat), dim=-1)
            
            similarity = (proj_visual * target_text_feat).sum(dim=-1)
            loss_alignment = 1 - similarity.mean()
        else:
            loss_alignment = torch.tensor(0.0, device=device, requires_grad=True)

        query_out_grad = self.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True
        )
        image_feat_grad = F.normalize(self.model.vision_projection(query_out_grad.last_hidden_state), dim=-1)
        image_feat_vec = image_feat_grad.mean(dim=1) 
        
        sim_i2t = image_feat_vec @ text_feat_norm.t() / 0.07
        sim_t2i = text_feat_norm @ image_feat_vec.t() / 0.07
        
        targets = torch.arange(batch_size, device=device)
        
        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) + 
            F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        total_loss = loss_itm + (alpha * loss_alignment) +  loss_itc

        return {
            "loss": total_loss,
            "loss_itm": loss_itm,
            "loss_alignment": loss_alignment,
            "loss_itc": loss_itc
        }

def collate_fn(batch, processor):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None
    
    images, texts = zip(*batch)
    pixel_values = processor(images=list(images), return_tensors="pt")["pixel_values"]
    return pixel_values, texts

def save_checkpoint(trainer, optimizer, output_dir, step):
    save_path = os.path.join(output_dir, f"checkpoint-{step}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    trainer.model.config.qformer_config.use_qformer_text_input = True
    logging.info(f"Saving checkpoint to {save_path}...")
    trainer.model.save_pretrained(save_path, safe_serialization=False)
    trainer.qformer_tokenizer.save_pretrained(save_path)
    torch.save(trainer.patch_proj.state_dict(), os.path.join(save_path, "patch_proj.bin"))
    training_state = {
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': step
    }
    torch.save(training_state, os.path.join(save_path, "training_state.pt"))
    
    logging.info(f"Save complete (Model + Optimizer state at step {step}).")

def main():
    parser = argparse.ArgumentParser(description="Train BLIP-2 for Key Token Extraction")
    
    parser.add_argument("--output_dir", type=str, default="./output_blip2_keytoken", help="Output directory for checkpoints and logs")
    parser.add_argument("--root_dir", type=str, default="./dataset/VG", help="Root directory of the dataset")
    parser.add_argument("--model_name", type=str, default="Salesforce/blip2-itm-vit-g", help="Pretrained model path or name")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every X steps")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of training steps")
    parser.add_argument("--log_steps", type=int, default=2, help="Log metrics every X steps")
    parser.add_argument("--accum_steps", type=int, default=20, help="Gradient accumulation steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume training from")
    
    parser.add_argument("--alpha", type=float, default=2.0, help="Weight for Alignment Loss")
    parser.add_argument("--K", type=int, default=64, help="K value for Soft Top-K")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature coefficient for Soft Top-K")
    
    args = parser.parse_args()
    
    logger = setup_logging(args.output_dir)
    logger.info(f"Parameters: {args}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    trainer = Blip2KeyTokenTrainer(model_name=args.model_name, freeze_vision=True)
    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        trainer.load_weights_from_checkpoint(args.resume_from_checkpoint)
    trainer.to(device)
    trainer.train()

    logger.info("Preparing dataset...")
    
    dataset = VisualGenomeDataset(args.root_dir)
    collate_fn_with_proc = partial(collate_fn, processor=trainer.processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_with_proc, num_workers=4)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, trainer.parameters()), lr=args.lr)
    
    global_step = 0

    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        state_path = os.path.join(args.resume_from_checkpoint, "training_state.pt")
        if os.path.exists(state_path):
            logger.info("Loading optimizer state and global step...")
            checkpoint_state = torch.load(state_path, map_location=device)
            optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
        else:
            logger.warning("training_state.pt not found in checkpoint directory. Optimizer starts from scratch.")   
            
    for epoch in range(args.epochs):
        logger.info(f"Starting Epoch {epoch+1}/{args.epochs}")
        
        for step, (pixel_values, texts) in enumerate(dataloader):
            
            if global_step >= args.max_steps:
                logger.info(f"Reached max steps {args.max_steps}. Ending training.")
                break
            
            if pixel_values is None:
                continue
                
            try:
                pixel_values = pixel_values.to(device)
                
                text_inputs = trainer.qformer_tokenizer(
                    list(texts),
                    padding=True,
                    truncation=True,
                    max_length=32,
                    return_tensors="pt"
                ).to(device)
                
                outputs = trainer(
                    pixel_values=pixel_values,
                    text_input_ids=text_inputs.input_ids,
                    text_attention_mask=text_inputs.attention_mask,
                    alpha=args.alpha,
                    K=args.K,
                    temperature=args.temperature
                )
                
                loss = outputs['loss']
                
                loss = loss / args.accum_steps 
                loss.backward()

                if (step + 1) % args.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    global_step += 1 
                    
                    if global_step % args.log_steps == 0:
                        current_real_loss = loss.item() * args.accum_steps
                        logger.info(f"Ep {epoch+1} | Step {global_step} | Loss: {current_real_loss:.4f} "
                                    f"| ITM: {outputs['loss_itm']:.4f} | Align: {outputs['loss_alignment']:.4f} | ITC: {outputs['loss_itc']:.4f}")
                    
                    if global_step % args.save_steps == 0:
                        save_checkpoint(trainer, optimizer, args.output_dir, global_step)
            
                if global_step >= args.max_steps:
                    break
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"CUDA Out of Memory at step {global_step}. Clearing cache.")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                else:
                    logger.error(f"RuntimeError at step {global_step}: {e}")
                    optimizer.zero_grad()
        
        if global_step >= args.max_steps:
            break

    logger.info("Training finished. Saving final model...")
    save_checkpoint(trainer, optimizer, args.output_dir, "final_model")

if __name__ == "__main__":
    main()