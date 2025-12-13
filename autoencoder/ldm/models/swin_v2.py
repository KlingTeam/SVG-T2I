from typing import Tuple, Optional, List, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import timm
try:
    # timm <= 0.9 often exposes DropPath here
    from timm.models.layers import DropPath  # type: ignore
except Exception:
    try:
        # newer timm exposes layers at top-level timm.layers
        from timm.layers import DropPath  # type: ignore
    except Exception:
        # Fallback: no-op DropPath for environments without timm layers (debug-only)
        class DropPath(nn.Identity):
            def __init__(self, drop_prob: float = 0.0):
                super().__init__()

__all__: List[str] = [
    "SwinTransformerStage",
    "SwinTransformerBlock",
    "DeformableSwinTransformerBlock",
    "SwinTransformerV2",
    "SwinV2MLPHead",
    "SwinV2FPNHead",
    "SwinV2Encoder",
]


class FeedForward(nn.Sequential):
    """
    Feed forward module used in the transformer encoder.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 dropout: float = 0.) -> None:
        """
        Constructor method
        :param in_features: (int) Number of input features
        :param hidden_features: (int) Number of hidden features
        :param out_features: (int) Number of output features
        :param dropout: (float) Dropout factor
        """
        # Call super constructor and init modules
        super().__init__(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_features, out_features=out_features),
            nn.Dropout(p=dropout)
        )


def bchw_to_bhwc(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, height, width, channels]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, height, width, channels]
    :return: (torch.Tensor) Output tensor of the shape [batch size, height, width, channels]
    """
    return input.permute(0, 2, 3, 1)


def bhwc_to_bchw(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, channels, height, width]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, height, width, channels]
    :return: (torch.Tensor) Output tensor of the shape [batch size, channels, height, width]
    """
    return input.permute(0, 3, 1, 2)


def unfold(input: torch.Tensor,
           window_size: int) -> torch.Tensor:
    """
    Unfolds (non-overlapping) a given feature map by the given window size (stride = window size)
    :param input: (torch.Tensor) Input feature map of the shape [batch size, channels, height, width]
    :param window_size: (int) Window size to be applied
    :return: (torch.Tensor) Unfolded tensor of the shape [batch size * windows, channels, window size, window size]
    """
    # Get original shape
    _, channels, height, width = input.shape
    # Unfold input
    output: torch.Tensor = input.unfold(dimension=3, size=window_size, step=window_size) \
        .unfold(dimension=2, size=window_size, step=window_size)
    # Reshape to [batch size * windows, channels, window size, window size]
    output: torch.Tensor = output.permute(0, 2, 3, 1, 5, 4).reshape(-1, channels, window_size, window_size)
    return output


def fold(input: torch.Tensor,
         window_size: int,
         height: int,
         width: int) -> torch.Tensor:
    """
    Fold a tensor of windows again to a 4D feature map
    :param input: (torch.Tensor) Input tensor of windows [batch size * windows, channels, window size, window size]
    :param window_size: (int) Window size to be reversed
    :param height: (int) Height of the feature map
    :param width: (int) Width of the feature map
    :return: (torch.Tensor) Folded output tensor of the shape [batch size, channels, height, width]
    """
    # Get channels of windows
    channels: int = input.shape[1]
    # Get original batch size
    batch_size: int = int(input.shape[0] // (height * width // window_size // window_size))
    # Reshape input to
    output: torch.Tensor = input.view(batch_size, height // window_size, width // window_size, channels,
                                      window_size, window_size)
    output: torch.Tensor = output.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, height, width)
    return output


class WindowMultiHeadAttention(nn.Module):
    """
    This class implements window-based Multi-Head-Attention.
    """

    def __init__(self,
                 in_features: int,
                 window_size: int,
                 number_of_heads: int,
                 dropout_attention: float = 0.,
                 dropout_projection: float = 0.,
                 meta_network_hidden_features: int = 256,
                 sequential_self_attention: bool = False) -> None:
        """
        Constructor method
        :param in_features: (int) Number of input features
        :param window_size: (int) Window size
        :param number_of_heads: (int) Number of attention heads
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_projection: (float) Dropout rate after projection
        :param meta_network_hidden_features: (int) Number of hidden features in the two layer MLP meta network
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        """
        # Call super constructor
        super(WindowMultiHeadAttention, self).__init__()
        # Check parameter
        assert (in_features % number_of_heads) == 0, \
            "The number of input features (in_features) are not divisible by the number of heads (number_of_heads)."
        # Save parameters
        self.in_features: int = in_features
        self.window_size: int = window_size
        self.number_of_heads: int = number_of_heads
        self.sequential_self_attention: bool = sequential_self_attention
        # Init query, key and value mapping as a single layer
        self.mapping_qkv: nn.Module = nn.Linear(in_features=in_features, out_features=in_features * 3, bias=True)
        # Init attention dropout
        self.attention_dropout: nn.Module = nn.Dropout(dropout_attention)
        # Init projection mapping
        self.projection: nn.Module = nn.Linear(in_features=in_features, out_features=in_features, bias=True)
        # Init projection dropout
        self.projection_dropout: nn.Module = nn.Dropout(dropout_projection)
        # Init meta network for positional encodings
        self.meta_network: nn.Module = nn.Sequential(
            nn.Linear(in_features=2, out_features=meta_network_hidden_features, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=meta_network_hidden_features, out_features=number_of_heads, bias=True))
        # Init tau
        self.register_parameter("tau", torch.nn.Parameter(torch.ones(1, number_of_heads, 1, 1)))
        # Init pair-wise relative positions (log-spaced)
        self.__make_pair_wise_relative_positions()

    def __make_pair_wise_relative_positions(self) -> None:
        """
        Method initializes the pair-wise relative positions to compute the positional biases
        """
        indexes: torch.Tensor = torch.arange(self.window_size, device=self.tau.device)
        # Explicitly set indexing to avoid future warnings and keep old behavior
        coordinates: torch.Tensor = torch.stack(torch.meshgrid([indexes, indexes], indexing="ij"), dim=0)
        coordinates: torch.Tensor = torch.flatten(coordinates, start_dim=1)
        relative_coordinates: torch.Tensor = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates: torch.Tensor = relative_coordinates.permute(1, 2, 0).reshape(-1, 2).float()
        relative_coordinates_log: torch.Tensor = torch.sign(relative_coordinates) \
                                                 * torch.log(1. + relative_coordinates.abs())
        self.register_buffer("relative_coordinates_log", relative_coordinates_log)

    def update_resolution(self,
                          new_window_size: int,
                          **kwargs: Any) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param kwargs: (Any) Unused
        """
        # Set new window size
        self.window_size: int = new_window_size
        # Make new pair-wise relative positions
        self.__make_pair_wise_relative_positions()

    def __get_relative_positional_encodings(self) -> torch.Tensor:
        """
        Method computes the relative positional encodings
        :return: (torch.Tensor) Relative positional encodings [1, number of heads, window size ** 2, window size ** 2]
        """
        relative_position_bias: torch.Tensor = self.meta_network(self.relative_coordinates_log)
        relative_position_bias: torch.Tensor = relative_position_bias.permute(1, 0)
        relative_position_bias: torch.Tensor = relative_position_bias.reshape(self.number_of_heads,
                                                                              self.window_size * self.window_size,
                                                                              self.window_size * self.window_size)
        return relative_position_bias.unsqueeze(0)

    def __self_attention(self,
                         query: torch.Tensor,
                         key: torch.Tensor,
                         value: torch.Tensor,
                         batch_size_windows: int,
                         tokens: int,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function performs standard (non-sequential) scaled cosine self-attention
        :param query: (torch.Tensor) Query tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param key: (torch.Tensor) Key tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param value: (torch.Tensor) Value tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param batch_size_windows: (int) Size of the first dimension of the input tensor (batch size * windows)
        :param tokens: (int) Number of tokens in the input
        :param mask: (Optional[torch.Tensor]) Attention mask for the shift case
        :return: (torch.Tensor) Output feature map of the shape [batch size * windows, tokens, channels]
        """
        # Compute attention map with scaled cosine attention
        attention_map: torch.Tensor = torch.einsum("bhqd, bhkd -> bhqk", query, key) \
                                      / torch.maximum(torch.norm(query, dim=-1, keepdim=True)
                                                      * torch.norm(key, dim=-1, keepdim=True).transpose(-2, -1),
                                                      torch.tensor(1e-06, device=query.device, dtype=query.dtype))
        attention_map: torch.Tensor = attention_map / self.tau.clamp(min=0.01)
        # Apply relative positional encodings
        attention_map: torch.Tensor = attention_map + self.__get_relative_positional_encodings()
        # Apply mask if utilized
        if mask is not None:
            number_of_windows: int = mask.shape[0]
            attention_map: torch.Tensor = attention_map.view(batch_size_windows // number_of_windows, number_of_windows,
                                                             self.number_of_heads, tokens, tokens)
            attention_map: torch.Tensor = attention_map + mask.unsqueeze(1).unsqueeze(0)
            attention_map: torch.Tensor = attention_map.view(-1, self.number_of_heads, tokens, tokens)
        attention_map: torch.Tensor = attention_map.softmax(dim=-1)
        # Perform attention dropout
        attention_map: torch.Tensor = self.attention_dropout(attention_map)
        # Apply attention map and reshape
        output: torch.Tensor = torch.einsum("bhal, bhlv -> bhav", attention_map, value)
        output: torch.Tensor = output.permute(0, 2, 1, 3).reshape(batch_size_windows, tokens, -1)
        return output

    def __sequential_self_attention(self,
                                    query: torch.Tensor,
                                    key: torch.Tensor,
                                    value: torch.Tensor,
                                    batch_size_windows: int,
                                    tokens: int,
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function performs sequential scaled cosine self-attention
        :param query: (torch.Tensor) Query tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param key: (torch.Tensor) Key tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param value: (torch.Tensor) Value tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param batch_size_windows: (int) Size of the first dimension of the input tensor (batch size * windows)
        :param tokens: (int) Number of tokens in the input
        :param mask: (Optional[torch.Tensor]) Attention mask for the shift case
        :return: (torch.Tensor) Output feature map of the shape [batch size * windows, tokens, channels]
        """
        # Init output tensor
        output: torch.Tensor = torch.ones_like(query)
        # Compute relative positional encodings fist
        relative_position_bias: torch.Tensor = self.__get_relative_positional_encodings()
        # Iterate over query and key tokens
        for token_index_query in range(tokens):
            # Compute attention map with scaled cosine attention
            attention_map: torch.Tensor = \
                torch.einsum("bhd, bhkd -> bhk", query[:, :, token_index_query], key) \
                / torch.maximum(torch.norm(query[:, :, token_index_query], dim=-1, keepdim=True)
                                * torch.norm(key, dim=-1, keepdim=False),
                                torch.tensor(1e-06, device=query.device, dtype=query.dtype))
            attention_map: torch.Tensor = attention_map / self.tau.clamp(min=0.01)[..., 0]
            # Apply positional encodings
            attention_map: torch.Tensor = attention_map + relative_position_bias[..., token_index_query, :]
            # Apply mask if utilized
            if mask is not None:
                number_of_windows: int = mask.shape[0]
                attention_map: torch.Tensor = attention_map.view(batch_size_windows // number_of_windows,
                                                                 number_of_windows, self.number_of_heads, 1,
                                                                 tokens)
                attention_map: torch.Tensor = attention_map \
                                              + mask.unsqueeze(1).unsqueeze(0)[..., token_index_query, :].unsqueeze(3)
                attention_map: torch.Tensor = attention_map.view(-1, self.number_of_heads, tokens)
            attention_map: torch.Tensor = attention_map.softmax(dim=-1)
            # Perform attention dropout
            attention_map: torch.Tensor = self.attention_dropout(attention_map)
            # Apply attention map and reshape
            output[:, :, token_index_query] = torch.einsum("bhl, bhlv -> bhv", attention_map, value)
        output: torch.Tensor = output.permute(0, 2, 1, 3).reshape(batch_size_windows, tokens, -1)
        return output

    def forward(self,
                input: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size * windows, channels, height, width]
        :param mask: (Optional[torch.Tensor]) Attention mask for the shift case
        :return: (torch.Tensor) Output tensor of the shape [batch size * windows, channels, height, width]
        """
        # Save original shape
        batch_size_windows, channels, height, width = input.shape
        tokens: int = height * width
        # Reshape input to [batch size * windows, tokens (height * width), channels]
        input: torch.Tensor = input.reshape(batch_size_windows, channels, tokens).permute(0, 2, 1)
        # Perform query, key, and value mapping
        query_key_value: torch.Tensor = self.mapping_qkv(input)
        query_key_value: torch.Tensor = query_key_value.view(batch_size_windows, tokens, 3, self.number_of_heads,
                                                             channels // self.number_of_heads).permute(2, 0, 3, 1, 4)
        query, key, value = query_key_value[0], query_key_value[1], query_key_value[2]
        # Perform attention
        if self.sequential_self_attention:
            output: torch.Tensor = self.__sequential_self_attention(query=query, key=key, value=value,
                                                                    batch_size_windows=batch_size_windows,
                                                                    tokens=tokens,
                                                                    mask=mask)
        else:
            output: torch.Tensor = self.__self_attention(query=query, key=key, value=value,
                                                         batch_size_windows=batch_size_windows, tokens=tokens,
                                                         mask=mask)
        # Perform linear mapping and dropout
        output: torch.Tensor = self.projection_dropout(self.projection(output))
        # Reshape output to original shape [batch size * windows, channels, height, width]
        output: torch.Tensor = output.permute(0, 2, 1).view(batch_size_windows, channels, height, width)
        return output


class SwinTransformerBlock(nn.Module):
    """
    This class implements the Swin transformer block.
    """

    def __init__(self,
                 in_channels: int,
                 number_of_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: float = 0.0,
                 sequential_self_attention: bool = False,
                 input_resolution: Optional[Tuple[int, int]] = None) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param input_resolution: (Optional[Tuple[int, int]]) Input resolution; if None, lazily inferred at first forward
        :param number_of_heads: (int) Number of attention heads to be utilized
        :param window_size: (int) Window size to be utilized
        :param shift_size: (int) Shifting size to be used
        :param ff_feature_ratio: (int) Ratio of the hidden dimension in the FFN to the input channels
        :param dropout: (float) Dropout in input mapping
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_path: (float) Dropout in main path
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        """
        # Call super constructor
        super(SwinTransformerBlock, self).__init__()
        # Save parameters
        self.in_channels: int = in_channels
        # If not provided, will be set on first forward/update
        self.input_resolution: Tuple[int, int] = input_resolution if input_resolution is not None else (0, 0)
        # Fixed window size policy: keep the initialization window_size throughout training/inference.
        self.fixed_window_size: int = window_size
        self.window_size: int = self.fixed_window_size
        # Preserve original (parity-based) shift size as base; actual shift decided per resolution on update
        self._base_shift_size: int = shift_size
        # Start with no shift until resolution is known
        self.shift_size: int = 0
        self.make_windows: bool = True
        # Init normalization layers
        self.normalization_1: nn.Module = nn.LayerNorm(normalized_shape=in_channels)
        self.normalization_2: nn.Module = nn.LayerNorm(normalized_shape=in_channels)
        # Init window attention module
        self.window_attention: WindowMultiHeadAttention = WindowMultiHeadAttention(
            in_features=in_channels,
            window_size=self.window_size,
            number_of_heads=number_of_heads,
            dropout_attention=dropout_attention,
            dropout_projection=dropout,
            sequential_self_attention=sequential_self_attention)
        # Init dropout layer
        self.dropout: nn.Module = DropPath(drop_prob=dropout_path) if dropout_path > 0. else nn.Identity()
        # Init feed-forward network
        self.feed_forward_network: nn.Module = FeedForward(in_features=in_channels,
                                                           hidden_features=int(in_channels * ff_feature_ratio),
                                                           dropout=dropout,
                                                           out_features=in_channels)
        # Make attention mask if resolution is known at init; otherwise, it will be created on first update
        if input_resolution is not None and all(v > 0 for v in input_resolution):
            self.__make_attention_mask()

    def __make_attention_mask(self) -> None:
        """
        Method generates the attention mask used in shift case
        """
        # Make masks for shift case
        if self.shift_size > 0:
            height, width = self.input_resolution
            mask: torch.Tensor = torch.zeros(height, width, device=self.window_attention.tau.device)
            height_slices: Tuple = (slice(0, -self.window_size),
                                    slice(-self.window_size, -self.shift_size),
                                    slice(-self.shift_size, None))
            width_slices: Tuple = (slice(0, -self.window_size),
                                   slice(-self.window_size, -self.shift_size),
                                   slice(-self.shift_size, None))
            counter: int = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    mask[height_slice, width_slice] = counter
                    counter += 1
            mask_windows: torch.Tensor = unfold(mask[None, None], self.window_size)
            mask_windows: torch.Tensor = mask_windows.reshape(-1, self.window_size * self.window_size)
            attention_mask: Optional[torch.Tensor] = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attention_mask: Optional[torch.Tensor] = attention_mask.masked_fill(attention_mask != 0, float(-100.0))
            attention_mask: Optional[torch.Tensor] = attention_mask.masked_fill(attention_mask == 0, float(0.0))
        else:
            attention_mask: Optional[torch.Tensor] = None
        # Save mask
        self.register_buffer("attention_mask", attention_mask)

    def update_resolution(self,
                          new_window_size: int,
                          new_input_resolution: Tuple[int, int]) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param new_input_resolution: (Tuple[int, int]) New input resolution
        """
        # Update input resolution
        self.input_resolution: Tuple[int, int] = new_input_resolution
        # Keep window size fixed to initialization value (ignore requested new_window_size)
        self.window_size: int = self.fixed_window_size
        # Enable/disable shift per resolution; when any side <= window size, disable shift
        self.shift_size: int = 0 if min(self.input_resolution) <= self.window_size else self._base_shift_size
        self.make_windows: bool = True
        # Update attention mask
        self.__make_attention_mask()
        # Update attention module
        self.window_attention.update_resolution(new_window_size=self.window_size)

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, in channels, height, width]
        """
        # Save shape
        batch_size, channels, height, width = input.shape
        # 1) Use fixed window size (as initialized)
        chosen_ws = self.window_size
        # Determine padding for this chosen window size
        pad_h: int = (chosen_ws - height % chosen_ws) % chosen_ws
        pad_w: int = (chosen_ws - width % chosen_ws) % chosen_ws
        use_shift: bool = (self.shift_size > 0)
        # Update internal resolution + window size (also updates masks and attention tables)
        # If padding is needed and we intend to use shift, compute masks for padded resolution to preserve behavior
        target_res: Tuple[int, int] = (height + pad_h, width + pad_w) if (pad_h or pad_w) and use_shift else (height, width)
        if (self.input_resolution != target_res) or (self.window_size != chosen_ws):
            self.update_resolution(new_window_size=chosen_ws, new_input_resolution=target_res)
        local_ws = self.window_size  # after update
        # Apply padding if needed
        if pad_h or pad_w:
            x = F.pad(input, (0, pad_w, 0, pad_h))  # pad (left,right,top,bottom) on W,H
        else:
            x = input
        H_pad, W_pad = x.shape[-2], x.shape[-1]
        # Apply shift (now safe even when padded since mask matches padded resolution)
        if use_shift:
            shifted: torch.Tensor = torch.roll(input=x, shifts=(-self.shift_size, -self.shift_size), dims=(-1, -2))
        else:
            shifted = x
        # Window partition
        patches: torch.Tensor = unfold(input=shifted, window_size=local_ws)
        # Attention mask only when shift is enabled
        attn_mask = (getattr(self, "attention_mask", None) if use_shift else None)
        attended: torch.Tensor = self.window_attention(patches, mask=attn_mask)
        # Merge windows
        merged: torch.Tensor = fold(input=attended, window_size=local_ws, height=H_pad, width=W_pad)
        # Reverse shift if applied
        if use_shift:
            restored: torch.Tensor = torch.roll(input=merged, shifts=(self.shift_size, self.shift_size), dims=(-1, -2))
        else:
            restored = merged
        # Crop back to original size if padded
        if pad_h or pad_w:
            output_shift: torch.Tensor = restored[:, :, :height, :width]
        else:
            output_shift = restored
        # Perform normalization
        output_normalize: torch.Tensor = self.normalization_1(output_shift.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # Skip connection
        output_skip: torch.Tensor = self.dropout(output_normalize) + input
        # Feed forward network, normalization and skip connection
        output_feed_forward: torch.Tensor = self.feed_forward_network(
            output_skip.view(batch_size, channels, -1).permute(0, 2, 1)).permute(0, 2, 1)
        output_feed_forward: torch.Tensor = output_feed_forward.view(batch_size, channels, height, width)
        output_normalize: torch.Tensor = bhwc_to_bchw(self.normalization_2(bchw_to_bhwc(output_feed_forward)))
        output: torch.Tensor = output_skip + self.dropout(output_normalize)
        return output


class DeformableSwinTransformerBlock(SwinTransformerBlock):
    """
    This class implements a deformable version of the Swin Transformer block.
    Inspired by: https://arxiv.org/pdf/2201.00520.pdf
    """

    def __init__(self,
                 in_channels: int,
                 number_of_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: float = 0.0,
                 sequential_self_attention: bool = False,
                 offset_downscale_factor: int = 2,
                 input_resolution: Optional[Tuple[int, int]] = None) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
    :param input_resolution: (Optional[Tuple[int, int]]) Input resolution; if None, lazily inferred
        :param number_of_heads: (int) Number of attention heads to be utilized
        :param window_size: (int) Window size to be utilized
        :param shift_size: (int) Shifting size to be used
        :param ff_feature_ratio: (int) Ratio of the hidden dimension in the FFN to the input channels
        :param dropout: (float) Dropout in input mapping
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_path: (float) Dropout in main path
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        :param offset_downscale_factor: (int) Downscale factor of offset network
        """
        # Call super constructor
        super(DeformableSwinTransformerBlock, self).__init__(
            in_channels=in_channels,
            number_of_heads=number_of_heads,
            window_size=window_size,
            shift_size=shift_size,
            ff_feature_ratio=ff_feature_ratio,
            dropout=dropout,
            dropout_attention=dropout_attention,
            dropout_path=dropout_path,
            sequential_self_attention=sequential_self_attention,
            input_resolution=input_resolution
        )
        # Save parameter
        self.offset_downscale_factor: int = offset_downscale_factor
        self.number_of_heads: int = number_of_heads
        # Make default offsets if resolution is known; otherwise create a placeholder and lazily build on update
        if all(v > 0 for v in self.input_resolution):
            self.__make_default_offsets()
        else:
            self.register_buffer("default_grid", torch.zeros(1, 1, 1, 2))
        # Init offset network
        self.offset_network: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=offset_downscale_factor,
                      padding=3, groups=in_channels, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels=in_channels, out_channels=2 * self.number_of_heads, kernel_size=1, stride=1,
                      padding=0, bias=True)
        )

    def __make_default_offsets(self) -> None:
        """
        Method generates the default sampling grid (inspired by kornia)
        """
        # Init x and y coordinates
        x: torch.Tensor = torch.linspace(0, self.input_resolution[1] - 1, self.input_resolution[1],
                                         device=self.window_attention.tau.device)
        y: torch.Tensor = torch.linspace(0, self.input_resolution[0] - 1, self.input_resolution[0],
                                         device=self.window_attention.tau.device)
        # Normalize coordinates to a range of [-1, 1]
        x: torch.Tensor = (x / (self.input_resolution[1] - 1) - 0.5) * 2
        y: torch.Tensor = (y / (self.input_resolution[0] - 1) - 0.5) * 2
        # Make grid [2, height, width]
        # Keep historical behavior by specifying indexing
        grid: torch.Tensor = torch.stack(torch.meshgrid([x, y], indexing="ij")).transpose(1, 2)
        # Reshape grid to [1, height, width, 2]
        grid: torch.Tensor = grid.unsqueeze(dim=0).permute(0, 2, 3, 1)
        # Register in module
        self.register_buffer("default_grid", grid)

    def update_resolution(self, new_window_size: int, new_input_resolution: Tuple[int, int]) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param new_input_resolution: (Tuple[int, int]) New input resolution
        """
        # Update resolution and window size
        super(DeformableSwinTransformerBlock, self).update_resolution(new_window_size=new_window_size,
                                                                      new_input_resolution=new_input_resolution)
        # Update default sampling grid
        self.__make_default_offsets()

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        # Get input shape
        batch_size, channels, height, width = input.shape
        # Compute offsets of the shape [batch size, 2, height / r, width / r]
        offsets: torch.Tensor = self.offset_network(input)
        # Upscale offsets to the shape [batch size, 2 * number of heads, height, width]
        offsets: torch.Tensor = F.interpolate(input=offsets,
                                              size=(height, width), mode="bilinear", align_corners=True)
        # Reshape offsets to [batch size, number of heads, height, width, 2]
        offsets: torch.Tensor = offsets.reshape(batch_size, -1, 2, height, width).permute(0, 1, 3, 4, 2)
        # Flatten batch size and number of heads and apply tanh
        offsets: torch.Tensor = offsets.view(-1, height, width, 2).tanh()
        # Cast offset grid to input data type
        if input.dtype != self.default_grid.dtype:
            self.default_grid = self.default_grid.type(input.dtype)
        # Construct offset grid
        offset_grid: torch.Tensor = self.default_grid.repeat_interleave(repeats=offsets.shape[0], dim=0) + offsets
        # Reshape input to [batch size * number of heads, channels / number of heads, height, width]
        input: torch.Tensor = input.view(batch_size, self.number_of_heads, channels // self.number_of_heads, height,
                                         width).flatten(start_dim=0, end_dim=1)
        # Apply sampling grid
        input_resampled: torch.Tensor = F.grid_sample(input=input, grid=offset_grid.clip(min=-1, max=1),
                                                      mode="bilinear", align_corners=True, padding_mode="reflection")
        # Reshape resampled tensor again to [batch size, channels, height, width]
        input_resampled: torch.Tensor = input_resampled.view(batch_size, channels, height, width)
        return super(DeformableSwinTransformerBlock, self).forward(input=input_resampled)


class PatchMerging(nn.Module):
    """
    This class implements the patch merging approach which is essential a strided convolution with normalization before
    """

    def __init__(self,
                 in_channels: int) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        """
        # Call super constructor
        super(PatchMerging, self).__init__()
        # Init normalization
        self.normalization: nn.Module = nn.LayerNorm(normalized_shape=4 * in_channels)
        # Init linear mapping
        self.linear_mapping: nn.Module = nn.Linear(in_features=4 * in_channels, out_features=2 * in_channels,
                                                   bias=False)

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, 2 * in channels, height // 2, width // 2]
        """
        # Get original shape
        batch_size, channels, height, width = input.shape
        # Reshape input to [batch size, in channels, height, width]
        input: torch.Tensor = bchw_to_bhwc(input)
        # Unfold input
        input: torch.Tensor = input.unfold(dimension=1, size=2, step=2).unfold(dimension=2, size=2, step=2)
        input: torch.Tensor = input.reshape(batch_size, input.shape[1], input.shape[2], -1)
        # Normalize input
        input: torch.Tensor = self.normalization(input)
        # Perform linear mapping
        output: torch.Tensor = bhwc_to_bchw(self.linear_mapping(input))
        return output


class PatchEmbedding(nn.Module):
    """
    Module embeds a given image into patch embeddings.
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 96,
                 patch_size: int = 4) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param patch_size: (int) Patch size to be utilized
        :param image_size: (int) Image size to be used
        """
        # Call super constructor
        super(PatchEmbedding, self).__init__()
        # Save parameters
        self.out_channels: int = out_channels
        # Init linear embedding as a convolution
        self.linear_embedding: nn.Module = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                     kernel_size=(patch_size, patch_size),
                                                     stride=(patch_size, patch_size))
        # Init layer normalization
        self.normalization: nn.Module = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass transforms a given batch of images into a patch embedding
        :param input: (torch.Tensor) Input images of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Patch embedding of the shape [batch size, patches + 1, out channels]
        """
        # Perform linear embedding
        embedding: torch.Tensor = self.linear_embedding(input)
        # Perform normalization
        embedding: torch.Tensor = bhwc_to_bchw(self.normalization(bchw_to_bhwc(embedding)))
        return embedding


class SwinTransformerStage(nn.Module):
    """
    This class implements a stage of the Swin transformer including multiple layers.
    """

    def __init__(self,
                 in_channels: int,
                 depth: int,
                 downscale: bool,
                 number_of_heads: int,
                 window_size: int = 7,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: Union[List[float], float] = 0.0,
                 use_checkpoint: bool = False,
                 sequential_self_attention: bool = False,
                 use_deformable_block: bool = False,
                 input_resolution: Optional[Tuple[int, int]] = None) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param depth: (int) Depth of the stage (number of layers)
        :param downscale: (bool) If true input is downsampled (see Fig. 3 or V1 paper)
        :param input_resolution: (Optional[Tuple[int, int]]) Input resolution; if None, lazily inferred
        :param number_of_heads: (int) Number of attention heads to be utilized
        :param window_size: (int) Window size to be utilized
        :param shift_size: (int) Shifting size to be used
        :param ff_feature_ratio: (int) Ratio of the hidden dimension in the FFN to the input channels
        :param dropout: (float) Dropout in input mapping
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_path: (float) Dropout in main path
        :param use_checkpoint: (bool) If true checkpointing is utilized
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        :param use_deformable_block: (bool) If true deformable block is used
        """
        # Call super constructor
        super(SwinTransformerStage, self).__init__()
        # Save parameters
        self.use_checkpoint: bool = use_checkpoint
        self.downscale: bool = downscale
        # Init downsampling
        self.downsample: nn.Module = PatchMerging(in_channels=in_channels) if downscale else nn.Identity()
        # Update resolution and channels
        if input_resolution is not None:
            self.input_resolution: Tuple[int, int] = (input_resolution[0] // 2, input_resolution[1] // 2) \
                if downscale else input_resolution
        else:
            self.input_resolution: Tuple[int, int] = (0, 0)
        in_channels = in_channels * 2 if downscale else in_channels
        # Get block
        block = DeformableSwinTransformerBlock if use_deformable_block else SwinTransformerBlock
        # Init blocks
        self.blocks: nn.ModuleList = nn.ModuleList([
        block(in_channels=in_channels,
            number_of_heads=number_of_heads,
            window_size=window_size,
            shift_size=0 if ((index % 2) == 0) else window_size // 2,
            ff_feature_ratio=ff_feature_ratio,
            dropout=dropout,
            dropout_attention=dropout_attention,
            dropout_path=dropout_path[index] if isinstance(dropout_path, list) else dropout_path,
            sequential_self_attention=sequential_self_attention,
            input_resolution=self.input_resolution)
            for index in range(depth)])

    def update_resolution(self, new_window_size: int, new_input_resolution: Tuple[int, int]) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param new_input_resolution: (Tuple[int, int]) New input resolution
        """
        # Update resolution
        self.input_resolution: Tuple[int, int] = (new_input_resolution[0] // 2, new_input_resolution[1] // 2) \
            if self.downscale else new_input_resolution
        # Update resolution of each block
        for block in self.blocks:
            block.update_resolution(new_window_size=new_window_size, new_input_resolution=self.input_resolution)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, 2 * channels, height // 2, width // 2]
        """
        # Downscale input tensor
        output: torch.Tensor = self.downsample(input)
        # Forward pass of each block
        for block in self.blocks:
            # Perform checkpointing if utilized
            if self.use_checkpoint:
                output: torch.Tensor = checkpoint.checkpoint(block, output)
            else:
                output: torch.Tensor = block(output)
        return output
    


class SwinTransformerV2(nn.Module):
    """
    This class implements the Swin Transformer without classification head.
    """

    def __init__(self,
                 in_channels: int,
                 embedding_channels: int,
                 depths: Tuple[int, ...],
                 number_of_heads: Tuple[int, ...],
                 window_size: int = 7,
                 patch_size: int = 4,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: float = 0.2,
                 use_checkpoint: bool = False,
                 sequential_self_attention: bool = False,
                 use_deformable_block: bool = False,
                 input_resolution: Optional[Tuple[int, int]] = None) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param depth: (int) Depth of the stage (number of layers)
        :param downscale: (bool) If true input is downsampled (see Fig. 3 or V1 paper)
    :param input_resolution: (Optional[Tuple[int, int]]) Input resolution; if None, lazily inferred
        :param number_of_heads: (int) Number of attention heads to be utilized
        :param window_size: (int) Window size to be utilized
        :param shift_size: (int) Shifting size to be used
        :param ff_feature_ratio: (int) Ratio of the hidden dimension in the FFN to the input channels
        :param dropout: (float) Dropout in input mapping
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_path: (float) Dropout in main path
        :param use_checkpoint: (bool) If true checkpointing is utilized
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        :param use_deformable_block: (bool) If true deformable block is used
        """
        # Call super constructor
        super(SwinTransformerV2, self).__init__()
        self.embedding_channels: int = embedding_channels
        # Save parameters
        self.patch_size: int = patch_size
        # Track window size for dynamic resolution updates
        self.window_size: int = window_size
        # Cache last seen input resolution to avoid redundant updates
        self._last_input_resolution: Optional[Tuple[int, int]] = None
        # Init patch embedding
        self.patch_embedding: nn.Module = PatchEmbedding(in_channels=in_channels, out_channels=embedding_channels,
                                                         patch_size=patch_size)
        # Compute patch resolution (or placeholder if unknown yet)
        patch_resolution: Tuple[int, int] = (
            (input_resolution[0] // patch_size, input_resolution[1] // patch_size)
            if input_resolution is not None else (0, 0)
        )
        # Path dropout dependent on depth
        dropout_path = torch.linspace(0., dropout_path, sum(depths)).tolist()
        # Init stages
        self.stages: nn.ModuleList = nn.ModuleList()
        for index, (depth, number_of_head) in enumerate(zip(depths, number_of_heads)):
            self.stages.append(
                SwinTransformerStage(
                    in_channels=embedding_channels * (2 ** max(index - 1, 0)),
                    depth=depth,
                    downscale=not (index == 0),
                    number_of_heads=number_of_head,
                    window_size=window_size,
                    ff_feature_ratio=ff_feature_ratio,
                    dropout=dropout,
                    dropout_attention=dropout_attention,
                    dropout_path=dropout_path[sum(depths[:index]):sum(depths[:index + 1])],
                    use_checkpoint=use_checkpoint,
                    sequential_self_attention=sequential_self_attention,
                    use_deformable_block=use_deformable_block and (index > 0),
                    input_resolution=(patch_resolution[0] // (2 ** max(index - 1, 0)),
                                      patch_resolution[1] // (2 ** max(index - 1, 0)))
                ))

    def update_resolution(self, new_window_size: int, new_input_resolution: Tuple[int, int]) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param new_input_resolution: (Tuple[int, int]) New input resolution
        """
        # Compute new patch resolution
        new_patch_resolution: Tuple[int, int] = (new_input_resolution[0] // self.patch_size,
                                                 new_input_resolution[1] // self.patch_size)
        # Update resolution of each stage
        for index, stage in enumerate(self.stages):
            stage.update_resolution(new_window_size=new_window_size,
                                    new_input_resolution=(new_patch_resolution[0] // (2 ** max(index - 1, 0)),
                                                          new_patch_resolution[1] // (2 ** max(index - 1, 0))))

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :return: (List[torch.Tensor]) List of features from each stage
        """
        # Dynamically update internal resolutions/masks when input HxW changes
        h, w = int(input.shape[-2]), int(input.shape[-1])
        cur_res: Tuple[int, int] = (h, w)
        if self._last_input_resolution != cur_res:
            # Ensure stages/blocks have correct masks for current size
            self.update_resolution(new_window_size=self.window_size, new_input_resolution=cur_res)
            self._last_input_resolution = cur_res
        # Perform patch embedding
        output: torch.Tensor = self.patch_embedding(input)
        # Init list to store feature
        features: List[torch.Tensor] = []
        # Forward pass of each stage
        for stage in self.stages:
            output: torch.Tensor = stage(output)
            features.append(output)
        return features


class SwinV2MLPHead(nn.Module):
    """A lightweight MLP head that maps a single-stage feature to desired channels C at stride 1x.

    Typical usage here: take stage-3 feature (stride 16) and project to out_channels via 1x1 Conv MLP.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Cin, Hs, Ws)  -> (B, Cout, Hs, Ws)
        return self.proj(x)


class SwinV2FPNHead(nn.Module):
    """Fuse 4 SwinV2 stage features to a single stride-16 feature map via simple FPN-style head.

    Steps per stage i (i=1..4):
    - 1x1 conv to unify channels to out_channels
    - resample to target size (by default size of stage3, i.e., stride-16)
      * downsample: adaptive avg pool to target size
      * upsample: bilinear interpolate to target size
    - sum all and apply a 3x3 conv for smoothing
    """
    def __init__(self, in_channels_per_stage: List[int], out_channels: int):
        super().__init__()
        assert len(in_channels_per_stage) == 4, "Expect 4 stage channels"
        self.proj = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0)
            for c in in_channels_per_stage
        ])
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        assert len(feats) == 4, "Expect 4 stage features"
        target_h, target_w = feats[2].shape[-2], feats[2].shape[-1]
        outs: List[torch.Tensor] = []
        for i, (f, proj) in enumerate(zip(feats, self.proj)):
            x = proj(f)
            h, w = x.shape[-2], x.shape[-1]
            if (h, w) != (target_h, target_w):
                if h > target_h or w > target_w:
                    x = F.adaptive_avg_pool2d(x, output_size=(target_h, target_w))
                else:
                    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            outs.append(x)
        fused = outs[0] + outs[1] + outs[2] + outs[3]
        return self.smooth(fused)



class EnhancedFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.GELU):
        super().__init__()
        # 初始融合卷积：输入通道数是所有特征拼接后的总和
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) # 添加 Batch Norm
        self.act = activation()
        # 第二个卷积用于进一步精炼特征
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

class SwinV2SkipHead(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()
        assert len(in_channels) == 4, "Expect 4 stage channels"
        C0, C1, C2, C3 = in_channels # 1/4, 1/8, 1/16, 1/32 对应的通道
        
        # 1. 降采样模块：将高分辨率特征降采样到 1/16
        # C0 (1/4) -> C2 (1/16)
        self.down_c0 = nn.Conv2d(C0, C2, kernel_size=4, stride=4, padding=0) 
        # C1 (1/8) -> C2 (1/16)
        self.down_c1 = nn.Conv2d(C1, C2, kernel_size=2, stride=2, padding=0) 
        
        # 2. 上采样模块：将低分辨率特征上采样到 1/16
        # C3 (1/32) -> C2 (1/16)
        self.up_c3 = nn.ConvTranspose2d(C3, C2, kernel_size=2, stride=2)
        
        # 3. 融合块：在 1/16 处融合所有特征
        # 输入通道总和：C2(from C0) + C2(from C1) + C2(skip C2) + C2(from C3) = 4 * C2
        self.fusion_block = EnhancedFusionBlock(
            in_channels=C2 * 4, 
            out_channels=out_channels 
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        assert len(feats) == 4, "Expect 4 stage features"
        x0, x1, x2, x3 = feats # 1/4, 1/8, 1/16, 1/32

        # 1. 对高分辨率特征降采样到 1/16
        x0_down = self.down_c0(x0) 
        x1_down = self.down_c1(x1)
        
        # 2. 对低分辨率特征上采样到 1/16
        x3_up = self.up_c3(x3)

        # 3. 在 1/16 处拼接所有特征
        fused = torch.cat([x0_down, x1_down, x2, x3_up], dim=1)
        
        # 4. 融合与精炼
        return self.fusion_block(fused)



class SwinV2Encoder(nn.Module):
    """Unified encoder that couples SwinV2 backbone with an output head.

    Goals:
    - Provide a single module that returns (B, C_out, H/16, W/16)
    - Head can be 'fpn' (fuses all 4 stages) or 'mlp' (use one stage, default stage3/stride-16)

    Args:
    - variant: one of {"t","s","b","l","h","g"}
    - out_channels: desired output feature channels C_out
    - head: 'fpn' or 'mlp'
    - mlp_stage_index: which stage to use for mlp head (0..3). Default 2 (stage3, stride-16)
    - All other kwargs are passed to the backbone builder (e.g., input_resolution, window_size, patch_size, etc.)
    """
    def __init__(
        self,
        variant: str = "t",
        *,
        out_channels: int = 16,
        head: str = "fpn",
        mlp_stage_index: int = 2,
    in_channels: int = 3,
    input_resolution: Optional[Tuple[int, int]] = None,
        window_size: int = 8,
        patch_size: int = 4,
        use_checkpoint: bool = False,
        sequential_self_attention: bool = False,
        ff_feature_ratio: int = 4,
        dropout: float = 0.0,
        dropout_attention: float = 0.0,
        dropout_path: float = 0.2,
        use_deformable_block: bool = False,
    ) -> None:
        super().__init__()
        self.head_type = head.lower()
        assert self.head_type in {"fpn", "mlp", "skip"}, f"Unsupported head={head}"
        self.mlp_stage_index = int(mlp_stage_index)
        assert 0 <= self.mlp_stage_index <= 3, "mlp_stage_index must be in [0,3]"
        self.out_channels = out_channels
        self.patch_size = patch_size

        # Build backbone by variant
        builders = {
            "t": swin_transformer_v2_t,
            "s": swin_transformer_v2_s,
            "b": swin_transformer_v2_b,
            "l": swin_transformer_v2_l,
            "h": swin_transformer_v2_h,
            "g": swin_transformer_v2_g,
        }
        assert variant in builders, f"Unknown SwinV2 variant '{variant}'"
        self.backbone: SwinTransformerV2 = builders[variant](
            input_resolution=input_resolution,
            window_size=window_size,
            in_channels=in_channels,
            use_checkpoint=use_checkpoint,
            sequential_self_attention=sequential_self_attention,
            patch_size=patch_size,
            ff_feature_ratio=ff_feature_ratio,
            dropout=dropout,
            dropout_attention=dropout_attention,
            dropout_path=dropout_path,
            use_deformable_block=use_deformable_block,
        )

        assert self.head_type == "skip"
        in_cs = [self.backbone.embedding_channels * i for i in [1, 2, 4, 8]]
        self._head = SwinV2SkipHead(in_channels=in_cs, out_channels=self.out_channels)
        

    @property
    def output_stride(self) -> int:
        # For FPN we target stage-3 size (index 2), which corresponds to stride = patch_size * 2^(2) = 4*patch_size
        # With default patch_size=4, output stride = 16. For MLP head we compute by stage index.
        if self.head_type == "fpn":
            return self.patch_size * (2 ** 2)
        # stage i has stride patch_size * 2^max(i-1,0) but our stage indexing here is 0..3 with stage2 being stride 16
        # Given our stage construction: stage0 stride=patch, stage1=2*patch, stage2=4*patch, stage3=8*patch
        strides = [self.patch_size, self.patch_size * 2, self.patch_size * 4, self.patch_size * 8]
        return strides[self.mlp_stage_index]

    # def _ensure_head(self, feats: List[torch.Tensor]) -> None:
    #     if self._head is not None:
    #         return
    #     if self.head_type == "fpn":
    #         in_cs = [f.shape[1] for f in feats]
    #         self._head = SwinV2FPNHead(in_channels_per_stage=in_cs, out_channels=self.out_channels)
    #     elif self.head_type == "skip":
    #         in_cs = [f.shape[1] for f in feats]
    #         print(f'Building SwinV2SkipHead with in_channels={in_cs}, out_channels={self.out_channels}')
    #         self._head = SwinV2SkipHead(in_channels=in_cs, out_channels=self.out_channels)
    #     else:  # mlp
    #         in_c = feats[self.mlp_stage_index].shape[1]
    #         self._head = SwinV2MLPHead(in_channels=in_c, out_channels=self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone produces 4 stage features
        feats = self.backbone(x)
        # Lazily build head with correct channel dims
        # for f in feats:
            # print(f.shape)
        # self._ensure_head(feats)
        assert self._head is not None
        self._head = self._head.to(feats[0].device)
        # Produce (B, C_out, H/16, W/16)-like map
        if self.head_type == "fpn":
            return self._head(feats)  # type: ignore[arg-type]
        elif self.head_type == "skip":
            tmp = self._head(feats)  # type: ignore[arg-type]
            return tmp.flatten(2)  # (B, C_out, H*W)
        return self._head(feats[self.mlp_stage_index])  # type: ignore[index]


def swin_transformer_v2_t(input_resolution: Optional[Tuple[int, int]] = None,
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformerV2:
    """
    Function returns a tiny Swin Transformer V2 (SwinV2-T: C = 96, layer numbers = {2, 2, 6, 2}) for feature extraction
    :param input_resolution: (Optional[Tuple[int, int]]) Input resolution; if None, lazily inferred
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformerV2) Tiny Swin Transformer V2
    """
    return SwinTransformerV2(input_resolution=input_resolution,
                             window_size=window_size,
                             in_channels=in_channels,
                             use_checkpoint=use_checkpoint,
                             sequential_self_attention=sequential_self_attention,
                             embedding_channels=96,
                             depths=(2, 2, 6, 2),
                             number_of_heads=(3, 6, 12, 24),
                             **kwargs)


def swin_transformer_v2_s(input_resolution: Optional[Tuple[int, int]] = None,
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformerV2:
    """
    Function returns a small Swin Transformer V2 (SwinV2-S: C = 96, layer numbers ={2, 2, 18, 2}) for feature extraction
    :param input_resolution: (Optional[Tuple[int, int]]) Input resolution; if None, lazily inferred
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformerV2) Small Swin Transformer V2
    """
    return SwinTransformerV2(input_resolution=input_resolution,
                             window_size=window_size,
                             in_channels=in_channels,
                             use_checkpoint=use_checkpoint,
                             sequential_self_attention=sequential_self_attention,
                             embedding_channels=96,
                             depths=(2, 2, 18, 2),
                             number_of_heads=(3, 6, 12, 24),
                             **kwargs)


def swin_transformer_v2_b(input_resolution: Optional[Tuple[int, int]] = None,
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformerV2:
    """
    Function returns a base Swin Transformer V2 (SwinV2-B: C = 128, layer numbers ={2, 2, 18, 2}) for feature extraction
    :param input_resolution: (Optional[Tuple[int, int]]) Input resolution; if None, lazily inferred
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformerV2) Base Swin Transformer V2
    """
    return SwinTransformerV2(input_resolution=input_resolution,
                             window_size=window_size,
                             in_channels=in_channels,
                             use_checkpoint=use_checkpoint,
                             sequential_self_attention=sequential_self_attention,
                             embedding_channels=128,
                             depths=(2, 2, 18, 2),
                             number_of_heads=(4, 8, 16, 32),
                             **kwargs)


def swin_transformer_v2_l(input_resolution: Optional[Tuple[int, int]] = None,
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformerV2:
    """
    Function returns a large Swin Transformer V2 (SwinV2-L: C = 192, layer numbers ={2, 2, 18, 2}) for feature extraction
    :param input_resolution: (Optional[Tuple[int, int]]) Input resolution; if None, lazily inferred
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformerV2) Large Swin Transformer V2
    """
    return SwinTransformerV2(input_resolution=input_resolution,
                             window_size=window_size,
                             in_channels=in_channels,
                             use_checkpoint=use_checkpoint,
                             sequential_self_attention=sequential_self_attention,
                             embedding_channels=192,
                             depths=(2, 2, 18, 2),
                             number_of_heads=(6, 12, 24, 48),
                             **kwargs)


def swin_transformer_v2_h(input_resolution: Optional[Tuple[int, int]] = None,
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformerV2:
    """
    Function returns a large Swin Transformer V2 (SwinV2-H: C = 352, layer numbers = {2, 2, 18, 2}) for feature extraction
    :param input_resolution: (Optional[Tuple[int, int]]) Input resolution; if None, lazily inferred
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformerV2) Large Swin Transformer V2
    """
    return SwinTransformerV2(input_resolution=input_resolution,
                             window_size=window_size,
                             in_channels=in_channels,
                             use_checkpoint=use_checkpoint,
                             sequential_self_attention=sequential_self_attention,
                             embedding_channels=352,
                             depths=(2, 2, 18, 2),
                             number_of_heads=(11, 22, 44, 88),
                             **kwargs)


def swin_transformer_v2_g(input_resolution: Optional[Tuple[int, int]] = None,
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformerV2:
    """
    Function returns a giant Swin Transformer V2 (SwinV2-G: C = 512, layer numbers = {2, 2, 42, 2}) for feature extraction
    :param input_resolution: (Optional[Tuple[int, int]]) Input resolution; if None, lazily inferred
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformerV2) Giant Swin Transformer V2
    """
    return SwinTransformerV2(input_resolution=input_resolution,
                             window_size=window_size,
                             in_channels=in_channels,
                             use_checkpoint=use_checkpoint,
                             sequential_self_attention=sequential_self_attention,
                             embedding_channels=512,
                             depths=(2, 2, 42, 2),
                             number_of_heads=(16, 32, 64, 128),
                             **kwargs)


# ------------------------
# Debug / quick benchmark
# ------------------------
def count_parameters(model: nn.Module) -> int:
    # Count all parameters (trainable + non-trainable) for a full parameter count
    return sum(p.numel() for p in model.parameters())


def format_count(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


if __name__ == "__main__":
    # Show param counts and output shapes for different SwinV2 variants and the unified encoder
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_resolution = None
    window_size = 8  # choose 8 so that patch grid (H/4,W/4) is divisible by 8 when possible
    x = torch.randn(1, 3, 1024, 1024, device=device)

    variants = {
        "swinv2_t": swin_transformer_v2_t,
        "swinv2_s": swin_transformer_v2_s,
        "swinv2_b": swin_transformer_v2_b,
        "swinv2_l": swin_transformer_v2_l,
        # Very large models; uncomment if you have enough memory
        # "swinv2_h": swin_transformer_v2_h,
        # "swinv2_g": swin_transformer_v2_g,
    }

    for name, builder in variants.items():
        model = builder(input_resolution=input_resolution, window_size=window_size, in_channels=3,
                        use_checkpoint=False, sequential_self_attention=False, patch_size=4)
        model = model.to(device)
        params = count_parameters(model)
        print(f"[{name}] params: {format_count(params)}")
        with torch.no_grad():
            feats = model(x)
        for i, f in enumerate(feats):
            print(f"  stage{i+1} output shape: {tuple(f.shape)}")
        # Build an FPN fusion head to fuse all 4 stages into (B, C, H/16, W/16)
        C_out = 8  # desired output channels; adjust as needed
        in_cs = [f.shape[1] for f in feats]
        fpn_head = SwinV2FPNHead(in_channels_per_stage=in_cs, out_channels=C_out).to(device)
        with torch.no_grad():
            out_map = fpn_head(feats)
        print(f"  fused stride-16 map: {tuple(out_map.shape)}  (expect (B, {C_out}, {1024//16}, {1024//16}))")
        print("-")

    # Demo unified encoder (variant 't')
    enc = SwinV2Encoder(
        variant="t",
        out_channels=8,
        head="skip",            # try 'mlp' with mlp_stage_index=2 as well
        mlp_stage_index=2,
        in_channels=3,
        input_resolution=input_resolution,
        window_size=window_size,
        patch_size=4,
        use_checkpoint=False,
        sequential_self_attention=False,
        use_deformable_block=False,
    ).to(device)
    params = count_parameters(enc)
    print(f"[encoder] params: {format_count(params)}")
    print(f"[encoder] output_stride: {enc.output_stride}")
    y = enc(x)
    print(f"[encoder] output map: {tuple(y.shape)}  ")
    # Test dynamic-resolution forward with a different size (non-square, non-divisible)
    x2 = torch.randn(1, 3, 768, 1344, device=device)
    y2 = enc(x2)
    print(f"[encoder] output map (768x1344): {tuple(y2.shape)}  (expect (B, 8, {768//16}, {1344//16}))")