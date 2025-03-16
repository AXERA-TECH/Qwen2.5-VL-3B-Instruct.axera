import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionSdpaAttention,\
                                                                apply_rotary_pos_emb_vision,\
                                                                Qwen2_5_VLVisionBlock,\
                                                                Qwen2_5_VisionTransformerPretrainedModel,\
                                                                Qwen2_5_VLForConditionalGeneration
import onnxruntime as ort
import numpy as np

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(1, L, S, dtype=query.dtype).to(attn_mask.device)
    
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
    return attn_weight @ value
class Qwen2_5_VLVisionSdpaAttentionExport(Qwen2_5_VLVisionSdpaAttention):

    # def forward(
    #     self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    # ) -> torch.Tensor:
    #     seq_length = hidden_states.shape[0]
    #     q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    #     q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    #     k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

    #     q = q.transpose(0, 1)
    #     k = k.transpose(0, 1)
    #     v = v.transpose(0, 1)
    #     attn_output = scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
    #     attn_output = attn_output.transpose(0, 1)
    #     attn_output = attn_output.reshape(seq_length, -1)
    #     attn_output = self.proj(attn_output)
    #     return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            print(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        # for i in range(1, len(cu_seqlens)):
        #     attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output

class Qwen2_5_VLVisionBlockExport(Qwen2_5_VLVisionBlock):
    def __init__(self, config, attn_implementation: str = "sdpa"):
        super().__init__(config, attn_implementation)
        self.attn = Qwen2_5_VLVisionSdpaAttentionExport(
            config.hidden_size, num_heads=config.num_heads
        )

    def forward(self, hidden_states, attention_mask, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

class Qwen2_5_VisionTransformerPretrainedModelInfer(Qwen2_5_VisionTransformerPretrainedModel):
    

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        torch.save(hidden_states, "hidden_states.pth")
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        torch.save(rotary_pos_emb, "rotary_pos_emb.pth")
        torch.save(cu_seqlens, "cu_seqlens.pth")
        torch.save(cu_window_seqlens, "cu_window_seqlens.pth")
        torch.save(window_index, "window_index.pth")

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def forward_by_second(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """

        assert grid_thw.shape[0]==1, f"not support shape:{grid_thw.shape}"

        t,grid_h,grid_w = grid_thw[0]

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        
        print("hidden_states.shape",hidden_states.shape)    # grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size

        
        llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

        thw, dim = hidden_states.shape
        hidden_states = hidden_states.view(t, -1, dim)
        
        torch.save(window_index[0:llm_grid_h*llm_grid_w], "window_index.pth")
        
        torch.save(cu_seqlens[0:2], "cu_seqlens.pth")
        

        win_idx_t = window_index[0:llm_grid_h*llm_grid_w]

        cu_win_seqlens_t = cu_window_seqlens[0 : 1+ num_windows_h*num_windows_w]
        cu_seqlens_t = cu_seqlens[0:2]

        cu_win_seqlens_t = torch.tensor(
                                        cu_win_seqlens_t,
                                        device=hidden_states.device,
                                        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
                                        )
        cu_win_seqlens_t = torch.unique_consecutive(cu_win_seqlens_t)
        torch.save(cu_win_seqlens_t, "cu_window_seqlens.pth")
        
        rope_t = rotary_pos_emb[0: grid_h*grid_w]
    
        seq_len = thw//t
        

        rope_t = rope_t.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rope_t = rope_t[win_idx_t, :, :]
        rope_t = rope_t.reshape(seq_len, -1)

        torch.save(rope_t, "rotary_pos_emb.pth")

        emb = torch.cat((rope_t, rope_t), dim=-1)
        pos_embs = (emb.cos(), emb.sin())


        out = []
        for ti in range(t):
            ht = hidden_states[ti]
            print("ht.shape",ht.shape)
            torch.save(ht, "hidden_states.pth")
            ht = self.patch_embed(ht)
            ht = ht.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
            ht = ht[win_idx_t, :, :]
            ht = ht.reshape(seq_len, -1)
            
            for layer_num, blk in enumerate(self.blocks):
                if layer_num in self.fullatt_block_indexes:
                    cu_seqlens_now = cu_seqlens_t
                else:
                    cu_seqlens_now = cu_win_seqlens_t
                
                ht = blk(ht, cu_seqlens=cu_seqlens_now, position_embeddings=pos_embs)

            ht = self.merger(ht)
            reverse_indices = torch.argsort(win_idx_t)
            ht = ht[reverse_indices, :]

            out.append(ht)
        out = torch.cat(out, 0)
        return out
            
    def forward_by_second_1(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        # hidden_states = hidden_states.permute(0,2,3,1)
        t, channel, grid_hw,  tpp  = hidden_states.shape
        # t, grid_hw,  tpp, channel  = hidden_states.shape
        # hidden_states = hidden_states.permute(0,1,3,2).reshape(t*grid_hw, channel*tpp)

        assert grid_thw.shape[0]==1, f"not support shape:{grid_thw.shape}"

        t,grid_h,grid_w = grid_thw[0]

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        
        print("hidden_states.shape",hidden_states.shape)    # grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size

        
        llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

        # thw, dim = hidden_states.shape
        # hidden_states = hidden_states.view(t, -1, dim)
        
        torch.save(window_index[0:llm_grid_h*llm_grid_w], "window_index.pth")
        
        torch.save(cu_seqlens[0:2], "cu_seqlens.pth")
        

        win_idx_t = window_index[0:llm_grid_h*llm_grid_w]

        cu_win_seqlens_t = cu_window_seqlens[0 : 1+ num_windows_h*num_windows_w]
        cu_seqlens_t = cu_seqlens[0:2]

        cu_win_seqlens_t = torch.tensor(
                                        cu_win_seqlens_t,
                                        device=hidden_states.device,
                                        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
                                        )
        cu_win_seqlens_t = torch.unique_consecutive(cu_win_seqlens_t)
        torch.save(cu_win_seqlens_t, "cu_window_seqlens.pth")
        
        rope_t = rotary_pos_emb[0: grid_h*grid_w]
    
        seq_len = grid_hw
        

        rope_t = rope_t.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rope_t = rope_t[win_idx_t, :, :]
        rope_t = rope_t.reshape(seq_len, -1)

        torch.save(rope_t, "rotary_pos_emb.pth")

        emb = torch.cat((rope_t, rope_t), dim=-1)
        pos_embs = (emb.cos(), emb.sin())


        out = []
        for ti in range(t):
            ht = hidden_states[ti:ti+1]
            print("ht.shape",ht.shape)
            torch.save(ht, "hidden_states.pth")
            ht = ht.permute(0,2,3,1)
            ht = ht.permute(0,1,3,2).reshape(grid_hw, channel*tpp)
            ht = self.patch_embed(ht)
            ht = ht.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
            ht = ht[win_idx_t, :, :]
            ht = ht.reshape(seq_len, -1)
            
            for layer_num, blk in enumerate(self.blocks):
                if layer_num in self.fullatt_block_indexes:
                    cu_seqlens_now = cu_seqlens_t
                else:
                    cu_seqlens_now = cu_win_seqlens_t
                
                ht = blk(ht, cu_seqlens=cu_seqlens_now, position_embeddings=pos_embs)

            ht = self.merger(ht)
            reverse_indices = torch.argsort(win_idx_t)
            ht = ht[reverse_indices, :]

            out.append(ht)
        out = torch.cat(out, 0)
        np.save("vit_out.npy", out.cpu().numpy())
        return out

def generate_attnmask(seq_length, cu_seqlens):
    attention_mask = torch.zeros([1, seq_length, seq_length],  dtype=torch.bool)
    for i in range(1, cu_seqlens.shape[0]):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

    return attention_mask

class Qwen2_5_VisionTransformerPretrainedModelExport(Qwen2_5_VisionTransformerPretrainedModel):

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        
        h = torch.load("hidden_states.pth","cpu", weights_only=True)
        cu_seqlens = torch.load("cu_seqlens.pth","cpu", weights_only=True)
        cu_window_seqlens = torch.load("cu_window_seqlens.pth","cpu", weights_only=True)

        seq_length = h.shape[0] if h.shape[0]!=1 else h.shape[2]
        # seq_length = h.shape[0] if h.shape[0]!=1 else h.shape[1]
        self.attention_mask = generate_attnmask(seq_length, cu_seqlens)
        self.attention_mask_window = generate_attnmask(seq_length, cu_window_seqlens)

        self.rotary_pos_emb_ = torch.load("rotary_pos_emb.pth","cpu", weights_only=True)

        self.window_index = torch.load("window_index.pth","cpu", weights_only=True)
        self.reverse_indices = torch.argsort(self.window_index)

        self.blocks = nn.ModuleList(
            [Qwen2_5_VLVisionBlockExport(config, config._attn_implementation) for _ in range(config.depth)]
        )
    
    def forward_export(self, hidden_states):

        device = hidden_states.device
        self.attention_mask = self.attention_mask.to(device)
        self.attention_mask_window = self.attention_mask_window.to(device)
        self.rotary_pos_emb_ = self.rotary_pos_emb_.to(device)
        self.reverse_indices = self.reverse_indices.to(device)

        hidden_states = self.patch_embed(hidden_states)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[self.window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                attention_mask_now = self.attention_mask
            else:
                attention_mask_now = self.attention_mask_window

            hidden_states = blk(
                hidden_states,
                attention_mask=attention_mask_now,
                rotary_pos_emb=self.rotary_pos_emb_,
            )

        hidden_states = self.merger(hidden_states)
        # reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[self.reverse_indices, :]

        return hidden_states

    def forward_export_by_second_1(self, hidden_states):
        hidden_states = hidden_states.permute(0,2,3,1)
        t, grid_hw,  tpp, channel = hidden_states.shape
        print("hidden_states.shape",hidden_states.shape)
        device = hidden_states.device

        hidden_states = hidden_states.permute(0,1,3,2).reshape(grid_hw, channel*tpp)
        
        self.attention_mask = self.attention_mask.to(device)
        self.attention_mask_window = self.attention_mask_window.to(device)

        self.rotary_pos_emb_ = self.rotary_pos_emb_.to(device)
        self.reverse_indices = self.reverse_indices.to(device)

        hidden_states = self.patch_embed(hidden_states)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[self.window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                attention_mask_now = self.attention_mask
            else:
                attention_mask_now = self.attention_mask_window

            hidden_states = blk(
                hidden_states,
                attention_mask=attention_mask_now,
                rotary_pos_emb=self.rotary_pos_emb_,
            )

        hidden_states = self.merger(hidden_states)
        # reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[self.reverse_indices, :]

        return hidden_states

    def forward_export_part1(self, hidden_states):

        hidden_states = self.patch_embed(hidden_states)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[self.window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        device = hidden_states.device
        self.attention_mask = self.attention_mask.to(device)
        self.attention_mask_window = self.attention_mask_window.to(device)
        self.rotary_pos_emb_ = self.rotary_pos_emb_.to(device)

        blocks_num = len(self.blocks)
        for layer_num in range(blocks_num//2):
            blk = self.blocks[layer_num]
            if layer_num in self.fullatt_block_indexes:
                attention_mask_now = self.attention_mask
            else:
                attention_mask_now = self.attention_mask_window

            hidden_states = blk(
                hidden_states,
                attention_mask=attention_mask_now,
                rotary_pos_emb=self.rotary_pos_emb_,
            )
        return hidden_states

    def forward_export_part2(self, hidden_states, ):

        device = hidden_states.device
        self.attention_mask = self.attention_mask.to(device)
        self.attention_mask_window = self.attention_mask_window.to(device)
        self.rotary_pos_emb_ = self.rotary_pos_emb_.to(device)
        self.reverse_indices = self.reverse_indices.to(device)

        blocks_num = len(self.blocks)
        for layer_num in range(blocks_num//2, blocks_num):
            blk = self.blocks[layer_num]
            if layer_num in self.fullatt_block_indexes:
                attention_mask_now = self.attention_mask
            else:
                attention_mask_now = self.attention_mask_window

            hidden_states = blk(
                hidden_states,
                attention_mask=attention_mask_now,
                rotary_pos_emb=self.rotary_pos_emb_,
            )

        hidden_states = self.merger(hidden_states)
        
        hidden_states = hidden_states[self.reverse_indices, :]

        return hidden_states

    def forward_onnx(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        def generate_attnmask(seq_length, cu_seqlens):
            attention_mask = torch.zeros([1, seq_length, seq_length], device=cu_seqlens.device, dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

            return attention_mask
        print("Qwen2_5_VisionTransformerPretrainedModel grid_thw",grid_thw)
        print("Qwen2_5_VisionTransformerPretrainedModel hidden_states",hidden_states.shape)              # [14308, 1176]
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        print("rotary_pos_emb.shape",rotary_pos_emb.shape)      # [14308, 40]
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        print("hidden_states",hidden_states.shape)              # [14308, 1280]
        print("window_index.shape",window_index.shape)
        print("window_index[0:33]",window_index[0:33])
        # window_index[0:33] tensor([  0,   1,   2,   3,  73,  74,  75,  76, 146, 147, 148, 149, 219, 220,
        # 221, 222,   4,   5,   6,   7,  77,  78,  79,  80, 150, 151, 152, 153,
        # 223, 224, 225, 226,   8])
        seq_len, _ = hidden_states.size()
        
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)    
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # return hidden_states
        print("test Vision Encoder Onnx -------------------")
        session1 = ort.InferenceSession("Qwen2.5-VL-3B-Instruct_vision.onnx", providers=["CPUExecutionProvider"])
        
        inputs = {"hidden_states": hidden_states.cpu().numpy().astype(np.float32),}
        hidden_states = session1.run(["hidden_states_out"], inputs)[0]
        hidden_states = torch.from_numpy(hidden_states).to(grid_thw.device)
        return hidden_states

    def forward_onnx_by_second(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
       
        t, grid_h, grid_w = grid_thw[0]

        print("test Vision Encoder Onnx -------------------")
        session = ort.InferenceSession("Qwen2.5-VL-3B-Instruct_vision.onnx", providers=["CPUExecutionProvider"])
        
        
        thw, dim = hidden_states.shape
        hidden_states = hidden_states.view(t, -1, dim)

        outputs = []
        for ti in range(t):
            ht = hidden_states[ti]

            inputs = {"hidden_states": ht.cpu().numpy().astype(np.float32),}
            out = session.run(["hidden_states_out"], inputs)[0]
            out = torch.from_numpy(out).to(grid_thw.device)
            outputs.append(out)
        outputs = torch.cat(outputs, 0)
        return outputs

    def forward_onnx_by_second_1(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        
        print("test Vision Encoder Onnx -------------------")
        session = ort.InferenceSession("Qwen2.5-VL-3B-Instruct_vision.onnx", providers=["CPUExecutionProvider"])
        
        
        t = hidden_states.shape[0]
        print("h shape",hidden_states.shape)
        outputs = []
        for ti in range(t):
            ht = hidden_states[ti:ti+1]

            inputs = {"hidden_states": ht.cpu().numpy().astype(np.float32),}
            out = session.run(["hidden_states_out"], inputs)[0]
            out = torch.from_numpy(out).to(grid_thw.device)
            outputs.append(out)
        outputs = torch.cat(outputs, 0)
        return outputs

    def forward_onnx_two_parts(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        def generate_attnmask(seq_length, cu_seqlens):
            attention_mask = torch.zeros([1, seq_length, seq_length], device=cu_seqlens.device, dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

            return attention_mask
        print("Qwen2_5_VisionTransformerPretrainedModel grid_thw",grid_thw)
        print("Qwen2_5_VisionTransformerPretrainedModel hidden_states",hidden_states.shape)              # [14308, 1176]
        
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        print("rotary_pos_emb.shape",rotary_pos_emb.shape)      # [14308, 40]
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        print("hidden_states",hidden_states.shape)              # [14308, 1280]
        print("window_index.shape",window_index.shape)
        print("window_index[0:33]",window_index[0:33])
        # window_index[0:33] tensor([  0,   1,   2,   3,  73,  74,  75,  76, 146, 147, 148, 149, 219, 220,
        # 221, 222,   4,   5,   6,   7,  77,  78,  79,  80, 150, 151, 152, 153,
        # 223, 224, 225, 226,   8])
        seq_len, _ = hidden_states.size()
        
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)    
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # return hidden_states
        print("test Vision Encoder Onnx two parts-------------------")
        session1 = ort.InferenceSession("Qwen2.5-VL-3B-Instruct_vision_part1.onnx", providers=["CPUExecutionProvider"])
        session2 = ort.InferenceSession("Qwen2.5-VL-3B-Instruct_vision_part2.onnx", providers=["CPUExecutionProvider"])
        # attention_mask = generate_attnmask(hidden_states.shape[0], cu_seqlens).to(torch.uint8)
        # attention_mask_window = generate_attnmask(hidden_states.shape[0], cu_window_seqlens).to(torch.uint8)
       
        inputs = {"hidden_states": hidden_states.cpu().numpy().astype(np.float32),
                    # "rotary_pos_emb":rotary_pos_emb.cpu().numpy().astype(np.float32),
                    # "attention_mask":attention_mask.cpu().numpy().astype(np.uint8),
                    # "attention_mask_window":attention_mask_window.cpu().numpy().astype(np.uint8)
                    }
        hidden_states = session1.run(["hidden_states_out"], inputs)[0]

        inputs = {"hidden_states": hidden_states,
                    # "rotary_pos_emb":rotary_pos_emb.cpu().numpy().astype(np.float32),
                    # "attention_mask":attention_mask.cpu().numpy().astype(np.uint8),
                    # "attention_mask_window":attention_mask_window.cpu().numpy().astype(np.uint8),
                    # "window_index":window_index.cpu().numpy().astype(np.int32)
                    }
        hidden_states = session2.run(["hidden_states_out"], inputs)[0]

        hidden_states = torch.from_numpy(hidden_states).to(grid_thw.device)
        return hidden_states

class Qwen2_5_VLForConditionalGenerationExport(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModelExport._from_config(config.vision_config)

class Qwen2_5_VLForConditionalGenerationInfer(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModelInfer._from_config(config.vision_config)