import torch
import torch.nn as nn
import sys
import os

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

# Add ToConv path to sys.path
toconv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if toconv_path not in sys.path:
    sys.path.insert(0, toconv_path)


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        # ToConv compression settings
        self.use_token_compression = getattr(args, 'use_token_compression', False)
        self.compression_input_size = getattr(args, 'compression_input_size', 24)
        self.compression_output_size = getattr(args, 'compression_output_size', 12)
        self.token_compressor = None

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        # Initialize ToConv token compressor if enabled
        if self.use_token_compression:
            try:
                from vision_token_compression.models.token_compressor import TokenCompressor

                # Get hidden_dim from CLIP config
                hidden_dim = self.vision_tower.config.hidden_size

                self.token_compressor = TokenCompressor(
                    input_grid_size=self.compression_input_size,
                    output_grid_size=self.compression_output_size,
                    hidden_dim=hidden_dim
                )
                self.token_compressor.to(self.vision_tower.device)
                print(f"ToConv Token Compression enabled: {self.compression_input_size}x{self.compression_input_size} → {self.compression_output_size}x{self.compression_output_size}")
            except Exception as e:
                print(f"Warning: Failed to load TokenCompressor: {e}")
                print("Token compression will be disabled.")
                self.use_token_compression = False
                self.token_compressor = None

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)

                # Apply token compression if enabled
                if self.use_token_compression and self.token_compressor is not None:
                    image_feature = self.token_compressor(image_feature)

                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

            # Apply token compression if enabled
            if self.use_token_compression and self.token_compressor is not None:
                image_features = self.token_compressor(image_features)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        base_size = self.config.image_size // self.config.patch_size
        if self.use_token_compression and self.token_compressor is not None:
            return self.compression_output_size
        return base_size

    @property
    def num_patches(self):
        if self.use_token_compression and self.token_compressor is not None:
            return self.compression_output_size ** 2
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        # Initialize ToConv token compressor if enabled
        if self.use_token_compression:
            try:
                from vision_token_compression.models.token_compressor import TokenCompressor

                # Get hidden_dim from CLIP config
                hidden_dim = self.vision_tower.config.hidden_size

                self.token_compressor = TokenCompressor(
                    input_grid_size=self.compression_input_size,
                    output_grid_size=self.compression_output_size,
                    hidden_dim=hidden_dim
                )
                self.token_compressor.to(self.vision_tower.device)
                print(f"ToConv Token Compression enabled for S2: {self.compression_input_size}x{self.compression_input_size} → {self.compression_output_size}x{self.compression_output_size}")
            except Exception as e:
                print(f"Warning: Failed to load TokenCompressor: {e}")
                print("Token compression will be disabled.")
                self.use_token_compression = False
                self.token_compressor = None

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)

        # Apply token compression if enabled
        if self.use_token_compression and self.token_compressor is not None:
            image_features = self.token_compressor(image_features)

        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
