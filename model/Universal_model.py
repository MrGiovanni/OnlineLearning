import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Unet import UNet3D


class Universal_model(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        
        self.backbone = UNet3D()
        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 8, kernel_size=1)
        )
        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
        )

        weight_nums, bias_nums = [], []
        weight_nums.append(8*8)
        weight_nums.append(8*8)
        weight_nums.append(8*1)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.controller = nn.Conv3d(256+256, sum(weight_nums+bias_nums), kernel_size=1, stride=1, padding=0)
        self.register_buffer('organ_embedding', torch.randn(out_channels, 512))
        self.text_to_vision = nn.Linear(512, 256)
        self.class_num = out_channels

    def load_params(self, model_dict):
        store_dict = self.backbone.state_dict()
        for key in model_dict.keys():
            if 'out_tr' not in key:
                store_dict[key.replace("module.", "")] = model_dict[key]
        self.backbone.load_state_dict(store_dict)

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 7))
        for i in range(N):
            task_encoding[i, task_id[i]]=1
        return task_encoding.cuda()

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):

            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, x_in):
        dec4, out = self.backbone(x_in)

        task_encoding = F.relu(self.text_to_vision(self.organ_embedding))
        task_encoding = task_encoding.unsqueeze(2).unsqueeze(2).unsqueeze(2)    

        x_feat = self.GAP(dec4)
        b = x_feat.shape[0]
        logits_array = []
        for i in range(b):
            x_cond = torch.cat([x_feat[i].unsqueeze(0).repeat(self.class_num,1,1,1,1), task_encoding], 1)
            params = self.controller(x_cond)
            params.squeeze_(-1).squeeze_(-1).squeeze_(-1)
            
            head_inputs = self.precls_conv(out[i].unsqueeze(0))
            head_inputs = head_inputs.repeat(self.class_num,1,1,1,1)
            N, _, D, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, D, H, W)

            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)

            logits = self.heads_forward(head_inputs, weights, biases, N)
            logits_array.append(logits.reshape(1, -1, D, H, W))
        
        out = torch.cat(logits_array,dim=0)

        return out, x_feat