import torch
import torch.nn as nn
from loguru import logger

from .backbone import *
from .head import hmr_head, smpl_head, smil_head
from .backbone.utils import get_backbone_info
from ..utils.train_utils import add_smpl_params_to_dict, prepare_statedict


class HMR(nn.Module):
    def __init__(
            self,
            smpl_model_type='smpl',
            backbone='resnet50',
            img_res=224,
            pretrained=None,
            num_betas=10,
    ):
        super(HMR, self).__init__()
        self.backbone = eval(backbone)(pretrained=True)
        self.head = hmr_head(
            num_input_features=get_backbone_info(backbone)['n_output_channels'],
        )
        self.smpl_model_type = smpl_model_type
        if self.smpl_model_type == 'SMPL':
            self.smpl = smpl_head(img_res=img_res)
        elif self.smpl_model_type == 'SMIL':
            self.smpl = smil_head(img_res=img_res, num_betas=num_betas)
        if pretrained is not None:
            self.load_pretrained(pretrained)

    def forward(self, images):
        features = self.backbone(images)
        hmr_output = self.head(features)
        smpl_output = self.smpl(
            rotmat=hmr_output['pred_pose'],
            shape=hmr_output['pred_shape'],
            cam=hmr_output['pred_cam'],
            normalize_joints2d=True,
        )
        smpl_output.update(hmr_output)
        return smpl_output

    def load_pretrained(self, file):
        logger.info(f'Loading pretrained weights from {file}')
        try:
            state_dict = torch.load(file, weights_only=False)['model']
        except:
            try:
                state_dict = prepare_statedict(torch.load(file, weights_only=False)['state_dict'])
            except:
                state_dict = add_smpl_params_to_dict(torch.load(file, weights_only=False))
        self.backbone.load_state_dict(state_dict, strict=False)
        self.head.load_state_dict(state_dict, strict=False)
