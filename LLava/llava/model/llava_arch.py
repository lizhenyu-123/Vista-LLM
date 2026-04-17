#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
import os
import math
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random
import concurrent.futures

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor

def standard_dpc(candidate_tokens: torch.Tensor, budget: int, n_neighbors: int = 7):
    if candidate_tokens.numel() == 0 or budget <= 0:
        return torch.empty(0, dtype=torch.long, device=candidate_tokens.device)

    x = candidate_tokens.float()
    N, D = x.shape
    budget = min(budget, N)

    dist_matrix = torch.cdist(x, x) / (D ** 0.5)
    k_final = min(n_neighbors + 1, N)
    dist_knn, _ = torch.topk(dist_matrix, k=k_final, dim=1, largest=False)

    if k_final > 1:
        rho = torch.exp(-(dist_knn[:, 1:] ** 2).mean(dim=1))
    else:
        rho = torch.ones(N, device=x.device)

    rho = rho + torch.rand_like(rho) * 1e-6
    higher_density_mask = rho[None, :] > rho[:, None]
    max_dist = dist_matrix.max()
    dist_masked = torch.where(higher_density_mask, dist_matrix, max_dist)

    delta, _ = dist_masked.min(dim=1)
    is_max_rho = rho == rho.max()
    delta[is_max_rho] = dist_matrix[is_max_rho].max(dim=1).values

    score = rho * delta
    _, top_indices = torch.topk(score, k=budget)

    return top_indices

def select_supplementary_tokens(
    patch_features_per_frame: torch.Tensor,
    dynamic_token_budgets: list[int],
    all_pruning_indices: list[torch.Tensor],
    segments: list[list[int]],
    use_aggregation: bool = False,
    aggregation_alpha: float = 0.5
):
    device = patch_features_per_frame.device
    num_frames, tokens_per_frame, feature_dim = patch_features_per_frame.shape
    feature_dtype = patch_features_per_frame.dtype

    final_frame_features = {}

    for frame_idx in range(num_frames):
        current_frame_all_features = patch_features_per_frame[frame_idx]
        must_select_indices = all_pruning_indices[frame_idx].to(device)
        
        all_indices = torch.arange(tokens_per_frame, device=device)
        mask = torch.ones(tokens_per_frame, dtype=torch.bool, device=device)
        if must_select_indices.numel() > 0:
            mask[must_select_indices] = False
        candidate_indices = all_indices[mask]
        candidate_features = current_frame_all_features[candidate_indices]
        
        total_budget = dynamic_token_budgets[frame_idx]
        supplementary_budget = total_budget - len(must_select_indices)
        supplementary_budget = max(0, min(supplementary_budget, len(candidate_indices)))

        supplementary_indices = torch.tensor([], dtype=torch.long, device=device)
        if supplementary_budget > 0 and candidate_features.numel() > 0:
            relative_indices = standard_dpc(candidate_features, supplementary_budget)
            supplementary_indices = candidate_indices[relative_indices]

        key_indices = torch.cat([must_select_indices, supplementary_indices])
        key_indices = torch.sort(key_indices).values

        if use_aggregation and key_indices.numel() > 0:
            discarded_mask = torch.ones(tokens_per_frame, dtype=torch.bool, device=device)
            discarded_mask[key_indices] = False
            discarded_indices = all_indices[discarded_mask]
            
            if discarded_indices.numel() == 0:
                final_frame_features[frame_idx] = (current_frame_all_features[key_indices], key_indices)
                continue

            key_features = current_frame_all_features[key_indices]
            discarded_features = current_frame_all_features[discarded_indices]
            
            dists = torch.cdist(discarded_features, key_features, p=2)
            closest_key_indices_map = torch.argmin(dists, dim=1)
            
            num_key_tokens = key_features.shape[0]
            aggregated_discarded_sum = torch.zeros_like(key_features, device=device)
            index_map_expanded = closest_key_indices_map.unsqueeze(1).expand(-1, feature_dim)
            aggregated_discarded_sum.scatter_add_(0, index_map_expanded, discarded_features)

            counts = torch.zeros(num_key_tokens, device=device, dtype=feature_dtype)
            counts.scatter_add_(0, closest_key_indices_map, torch.ones_like(closest_key_indices_map, dtype=feature_dtype))

            counts_safe = counts.unsqueeze(1).clamp(min=1.0)
            mean_aggregated_features = aggregated_discarded_sum / counts_safe
            
            final_key_features = key_features.clone()
            update_mask = (counts > 0)
            
            final_key_features[update_mask] = \
                (1 - aggregation_alpha) * key_features[update_mask] + \
                aggregation_alpha * mean_aggregated_features[update_mask]
            
            final_frame_features[frame_idx] = (final_key_features, key_indices)
        else:
            if key_indices.numel() > 0:
                final_frame_features[frame_idx] = (current_frame_all_features[key_indices], key_indices)
            else:
                final_frame_features[frame_idx] = (torch.empty(0, feature_dim, device=device), torch.empty(0, dtype=torch.long, device=device))

    return final_frame_features

def get_my_dynamic_token_budgets(
    patch_features_per_frame: torch.Tensor,
    question_embedding: torch.Tensor,
    total_retention_ratio: float = 0.25,
    grid_size: int = 14,
    R_MIN_RATIO: float = 0.10
):
    TAU_SIMILARITY_THRESHOLD = float(os.environ.get("TAU_SIMILARITY_THRESHOLD", 0.95))
    
    device = patch_features_per_frame.device
    num_frames, tokens_per_frame, feature_dim = patch_features_per_frame.shape

    frame_features = torch.mean(patch_features_per_frame, dim=1)
    frame_features = F.normalize(frame_features, p=2, dim=1)

    similarities = torch.sum(frame_features[:-1] * frame_features[1:], dim=1)
    boundaries = (similarities < TAU_SIMILARITY_THRESHOLD).nonzero().squeeze(-1).tolist()
    
    segments =[]
    start_idx = 0
    for boundary in boundaries:
        segments.append(list(range(start_idx, boundary + 1)))
        start_idx = boundary + 1
    segments.append(list(range(start_idx, num_frames)))
    
    i = 0
    while i < len(segments):
        if len(segments[i]) == 1:
            current_segment_idx = segments[i][0]
            current_segment_feature = frame_features[current_segment_idx]
            
            sim_left = -1
            if i > 0:
                left_neighbor_feature = torch.mean(frame_features[segments[i-1]], dim=0)
                sim_left = torch.sum(current_segment_feature * F.normalize(left_neighbor_feature, dim=0))

            sim_right = -1
            if i < len(segments) - 1:
                right_neighbor_feature = torch.mean(frame_features[segments[i+1]], dim=0)
                sim_right = torch.sum(current_segment_feature * F.normalize(right_neighbor_feature, dim=0))

            if sim_left > sim_right:
                segments[i-1].extend(segments.pop(i))
            else:
                segments[i][:0] = segments.pop(i)
        else:
            i += 1

    segment_features_list =[torch.mean(frame_features[seg], dim=0) for seg in segments]
    segment_features = torch.stack(segment_features_list)
    
    segment_features_norm = F.normalize(segment_features, p=2, dim=1)
    question_embedding_norm = F.normalize(question_embedding.squeeze(), p=2, dim=0)
    relevance_scores = torch.matmul(segment_features_norm, question_embedding_norm)

    weights = F.softmax(relevance_scores, dim=0)

    total_tokens = num_frames * tokens_per_frame
    target_total_tokens = total_tokens * total_retention_ratio
    
    min_tokens_per_frame = int(round(tokens_per_frame * R_MIN_RATIO))
    final_frame_budgets = [min_tokens_per_frame] * num_frames
    
    current_allocated_tokens = sum(final_frame_budgets)
    tokens_to_distribute = int(target_total_tokens - current_allocated_tokens)
    tokens_to_distribute = max(0, tokens_to_distribute)

    segment_capacities =[]
    for seg in segments:
        num_frames_in_seg = len(seg)
        max_tokens_for_seg = num_frames_in_seg * tokens_per_frame
        min_tokens_for_seg = num_frames_in_seg * min_tokens_per_frame
        capacity = max_tokens_for_seg - min_tokens_for_seg
        segment_capacities.append(capacity)
    
    segment_capacities = torch.tensor(segment_capacities, device=device, dtype=torch.int)
    allocated_extra_tokens = torch.zeros(len(segments), dtype=torch.int, device=device)
    
    while tokens_to_distribute > 0:
        eligible_mask = (segment_capacities - allocated_extra_tokens) > 0
        
        if not torch.any(eligible_mask):
            break 
            
        eligible_weights = weights * eligible_mask.float()
        total_eligible_weight = torch.sum(eligible_weights)
        
        if total_eligible_weight <= 1e-8:
            num_eligible = torch.sum(eligible_mask).item()
            if num_eligible == 0: break
            normalized_eligible_weights = (1.0 / num_eligible) * eligible_mask.float()
        else:
            normalized_eligible_weights = eligible_weights / total_eligible_weight

        tentative_allocations = (tokens_to_distribute * normalized_eligible_weights).round().int()
        
        current_capacity = segment_capacities - allocated_extra_tokens
        actual_allocations = torch.min(tentative_allocations, current_capacity)
        
        allocated_extra_tokens += actual_allocations
        distributed_this_round = torch.sum(actual_allocations).item()
        tokens_to_distribute -= distributed_this_round

        if distributed_this_round == 0 and tokens_to_distribute > 0:
            eligible_mask = (segment_capacities - allocated_extra_tokens) > 0
            if torch.any(eligible_mask):
                eligible_weights = weights * eligible_mask.float()
                winner_idx = torch.argmax(eligible_weights).item()
                can_take = (segment_capacities[winner_idx] - allocated_extra_tokens[winner_idx]).item()
                add_amount = int(min(tokens_to_distribute, can_take))
                allocated_extra_tokens[winner_idx] += add_amount
            tokens_to_distribute = 0
            
    for seg_idx, extra_tokens in enumerate(allocated_extra_tokens):
        extra_tokens_for_segment = extra_tokens.item()
        num_frames_in_segment = len(segments[seg_idx])
        if num_frames_in_segment > 0:
            extra_tokens_per_frame = extra_tokens_for_segment / num_frames_in_segment
            for frame_idx in segments[seg_idx]:
                final_frame_budgets[frame_idx] += int(round(extra_tokens_per_frame))
                final_frame_budgets[frame_idx] = min(tokens_per_frame, final_frame_budgets[frame_idx])

    return final_frame_budgets, segments


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat is not 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_newline_token(self, feat, pos, grid_size, newline_token):
        row_pos = pos // grid_size
        expanded_feat =[]
        
        for row in range(grid_size):
            mask = (row_pos == row)
            find_row_feat = feat[mask]
            if find_row_feat.shape[0] > 0:
                expanded_feat.append(torch.cat((find_row_feat, newline_token), dim=0))
        
        if len(expanded_feat) > 0:
            return torch.cat(expanded_feat, dim=0)
        else:
            return torch.empty(0, feat.shape[-1], device=feat.device)

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def _apply_dynamic_token_compression(
        self,
        image_feature,
        grid_size,
        feature_model,
        raw_images,
        questions,
        question_embedding,
        domain_rate,
        blip_future,
        blip_executor,
        mm_newline_position,
        question_id
    ):
        target_device = image_feature.device
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]
        newline_token_embedding = self.model.image_newline[None].to(target_device)

        dynamic_budget_tokens, segments = get_my_dynamic_token_budgets(
            patch_features_per_frame=image_feature,
            question_embedding=question_embedding,
            grid_size=grid_size,
            total_retention_ratio=float(os.environ.get("RETENTION_RATIO", 0.25)),
            R_MIN_RATIO=float(os.environ.get("R_MIN_RATIO", 0.10)),
        )

        attention_scores_all = None
        if blip_future is not None:
            attention_scores_all = blip_future.result()
            blip_executor.shutdown(wait=False)

        all_pruning_indices =[]

        if attention_scores_all is not None and len(attention_scores_all) > 0:
            attention_scores_all = attention_scores_all.to(target_device)
            for frame_idx in range(len(attention_scores_all)):
                k_budget = int(dynamic_budget_tokens[frame_idx] * domain_rate) 
                scores = attention_scores_all[frame_idx]
                
                if k_budget <= 0:
                    all_pruning_indices.append(torch.tensor([], device=target_device, dtype=torch.long))
                    continue

                k_safe = min(k_budget, scores.shape[0])
                _, top_indices_blip = torch.topk(scores, k=k_safe, dim=-1)
                
                blip_grid_size = int(scores.shape[0] ** 0.5)
                llava_grid_size = grid_size 

                if blip_grid_size == llava_grid_size:
                    final_indices = top_indices_blip
                else:
                    h_blip = top_indices_blip // blip_grid_size
                    w_blip = top_indices_blip % blip_grid_size
                    h_norm = (h_blip.float() + 0.5) / blip_grid_size
                    w_norm = (w_blip.float() + 0.5) / blip_grid_size
                    h_llava = torch.clamp(torch.floor(h_norm * llava_grid_size).long(), 0, llava_grid_size - 1)
                    w_llava = torch.clamp(torch.floor(w_norm * llava_grid_size).long(), 0, llava_grid_size - 1)
                    final_indices = h_llava * llava_grid_size + w_llava

                all_pruning_indices.append(final_indices)
        else:
            for _ in range(num_frames):
                all_pruning_indices.append(torch.tensor([], device=target_device, dtype=torch.long))
        
        final_frame_features_map = select_supplementary_tokens(
            patch_features_per_frame=image_feature,
            dynamic_token_budgets=dynamic_budget_tokens,
            all_pruning_indices=all_pruning_indices,
            segments=segments,
            use_aggregation=os.environ.get("USE_AGGREGATION", "False") == "True",
            aggregation_alpha=float(os.environ.get("AGGREGATION_ALPHA", 0.5)),
        )

        final_sequence_parts =[]
        for i in range(num_frames):
            frame_data = final_frame_features_map.get(i)
            if frame_data is None: 
                continue
            
            pruned_feats, pruned_indices = frame_data
            if pruned_feats.numel() == 0:
                continue
            
            if mm_newline_position == "grid":
                reconstructed_frame_feat = self.add_newline_token(
                    pruned_feats, pruned_indices, grid_size, newline_token_embedding
                )
                final_sequence_parts.append(reconstructed_frame_feat)
            elif mm_newline_position == "frame":
                frame_with_newline = torch.cat((pruned_feats, newline_token_embedding), dim=0)
                final_sequence_parts.append(frame_with_newline)
            else:
                final_sequence_parts.append(pruned_feats)

        if not final_sequence_parts:
            return torch.empty(0, feature_dim, device=target_device)
        
        if mm_newline_position == "one_token":
            final_sequence_parts.append(newline_token_embedding)
            
        return torch.cat(final_sequence_parts, dim=0)

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None,
            feature_model=None,
            raw_images=None,
            domain_rate=None,
            question_embedding=None,
            questions = None,
            question_id = None,
            ):
        
        # BLIP2 token selection
        blip_executor = None
        blip_future = None
        if feature_model is not None and raw_images is not None:
            def run_blip_background():
                return feature_model.extract_relevant_patch_indices_dynamic(
                    video_frames_list=raw_images, 
                    question_texts=questions, 
                )
            blip_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            blip_future = blip_executor.submit(run_blip_background)

        grid_size = None
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            t1 = time.time()
            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]


            encoded_image_features = self.encode_images(concat_images)
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                    grid_size = int(image_features[0].shape[1] ** 0.5)
                else:
                    image_features.append(image_feat)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            # image_feature = self.add_token_per_grid(image_feature)
                            # if getattr(self.config, "add_faster_video", False):
                            #     faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                            #     # Add a token for each frame
                            #     concat_slow_fater_token = []
                            #     # import pdb; pdb.set_trace()
                            #     for _ in range(image_feature.shape[0]):
                            #         if _ % self.config.faster_token_stride == 0:
                            #             concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                            #         else:
                            #             concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                            #     # import pdb; pdb.set_trace()
                            #     image_feature = torch.cat(concat_slow_fater_token)

                            #     # print("!!!!!!!!!!!!")
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            # image_feature = self.add_token_per_frame(image_feature)

                            # new_image_features.append(image_feature.flatten(0, 1))
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "one_token":
                            # one-token
                            # image_feature = image_feature.flatten(0, 1)
                            # if 'unpad' in mm_patch_merge_type:
                            #     image_feature = torch.cat((
                            #         image_feature,
                            #         self.model.image_newline[None].to(image_feature.device)
                            #     ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")
        #Compression Stage
        if feature_model is not None:
            new_image_features =[]
            final_pruned_features = self._apply_dynamic_token_compression(
                image_feature=image_features[0],    
                grid_size=grid_size,
                feature_model=feature_model,
                raw_images=raw_images,
                questions=questions,
                question_embedding=question_embedding,
                domain_rate=domain_rate,
                blip_future=blip_future,
                blip_executor=blip_executor,
                mm_newline_position=mm_newline_position,
                question_id=question_id
            )
            new_image_features.append(final_pruned_features)
            image_features = new_image_features

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
