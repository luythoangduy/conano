"""
Trainer for RDLGC with BYOL-style architecture.

Key differences from original trainer:
1. Calls update_momentum_encoder() after each optimization step
2. Sets total_steps for momentum scheduling
3. Uses BYOL loss (no negative samples)
"""

import os
import copy
import glob
import shutil
import datetime
import time

import tabulate
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from util.metric import get_evaluator
from timm.data import Mixup

import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
    from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad

from ._base_trainer import BaseTrainer
from . import TRAINER
from util.vis import vis_rgb_gt_amp, read_data, save_data
from util.bank import MemoryBank
import torch.nn.functional as F


@TRAINER.register_module
class RDLGCBYOLTrainer(BaseTrainer):
    """
    Trainer for RDLGC with BYOL-style architecture.
    
    Key features:
    1. Updates momentum encoder after each optimization step
    2. Uses BYOL loss without negative samples
    3. Supports momentum scheduling (constant, cosine, linear)
    """
    
    def __init__(self, cfg):
        super(RDLGCBYOLTrainer, self).__init__(cfg)
        
        # Handle both DDP and non-DDP cases
        net_module = self.net.module if hasattr(self.net, 'module') else self.net
        
        # === Setup optimizers ===
        # Optimizer for projection layer + predictor
        proj_params = list(net_module.proj_layer.parameters()) + \
                      list(net_module.predictor.parameters())
        self.optim.proj_opt = get_optim(cfg.optim.proj_opt.kwargs, proj_params, lr=cfg.optim.lr)
        
        # Temporarily remove proj_layer and predictor for distill_opt
        proj_layer = net_module.proj_layer
        predictor = net_module.predictor
        net_module.proj_layer = None
        net_module.predictor = None
        self.optim.distill_opt = get_optim(cfg.optim.distill_opt.kwargs, self.net, lr=cfg.optim.lr * 5)
        net_module.proj_layer = proj_layer
        net_module.predictor = predictor
        
        # === Set total steps for momentum scheduling ===
        total_steps = cfg.trainer.iter_full if hasattr(cfg.trainer, 'iter_full') else \
                     cfg.trainer.epoch_full * cfg.data.train_size
        net_module.set_total_steps(total_steps)
        
        # Log BYOL-specific info
        if self.master:
            log_msg(self.logger, f"[BYOL] Total steps: {total_steps}")
            log_msg(self.logger, f"[BYOL] Momentum schedule: {net_module.momentum_schedule}")
            if net_module.momentum_schedule != 'constant':
                log_msg(self.logger, f"[BYOL] Momentum range: {net_module.momentum_start} → {net_module.momentum_end}")

    def set_input(self, inputs):
        """Set input data"""
        self.imgs = inputs['img'].cuda()
        self.aug_imgs = inputs.get('aug_img', None)
        self.imgs_mask = inputs['img_mask'].cuda()
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly']
        self.img_path = inputs['img_path']
        self.labels = inputs['label'].cuda()
        self.bs = self.imgs.shape[0]
        
        if self.aug_imgs is not None:
            self.aug_imgs = self.aug_imgs.cuda()

    def forward(self):
        """Forward pass"""
        outputs = self.net(self.imgs, self.aug_imgs)
        (self.feats_t, self.feats_s, self.feats_t_k, 
         self.feats_t_q_grid, self.feats_t_k_grid, 
         self.glb_feats, self.glb_feats_k) = outputs

    def backward_term(self, loss_term, optim):
        """Backward pass with gradient clipping"""
        optim.proj_opt.zero_grad()
        optim.distill_opt.zero_grad()
        
        if self.loss_scaler:
            self.loss_scaler(
                loss_term, optim, 
                clip_grad=self.cfg.loss.clip_grad, 
                parameters=self.net.parameters(),
                create_graph=self.cfg.loss.create_graph
            )
        else:
            loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
            if self.cfg.loss.clip_grad is not None:
                dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
            
            optim.proj_opt.step()
            optim.distill_opt.step()

    def optimize_parameters(self):
        """Optimization step with BYOL-style momentum update"""
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        
        with self.amp_autocast():
            self.forward()
            
            # === Reconstruction loss (cosine similarity) ===
            loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s)
            
            # === Global contrastive loss (SCL) ===
            loss_glb = self.loss_terms['scl'](self.glb_feats, self.labels)
            
            # === BYOL Dense loss (no negatives!) ===
            # Note: q_grid has predictor output, k_grid doesn't
            loss_den = self.loss_terms['dense'](
                self.feats_t,           # q_b: online backbone for matching
                self.feats_t_k,         # k_b: target backbone for matching  
                self.feats_t_q_grid,    # q_grid: online output (with predictor)
                self.feats_t_k_grid,    # k_grid: target output (no predictor)
                self.labels
            )
            
            loss = loss_cos + loss_glb + loss_den

        self.backward_term(loss, self.optim)

        # === CRITICAL: Update momentum encoder after each step ===
        net_module = self.net.module if hasattr(self.net, 'module') else self.net
        if hasattr(net_module, 'update_momentum_encoder'):
            net_module.update_momentum_encoder()

        # === Logging ===
        update_log_term(
            self.log_terms.get('cos'), 
            reduce_tensor(loss_cos, self.world_size).clone().detach().item(), 
            1, self.master
        )
        update_log_term(
            self.log_terms.get('glb'), 
            reduce_tensor(loss_glb, self.world_size).clone().detach().item(), 
            1, self.master
        )
        update_log_term(
            self.log_terms.get('dense'), 
            reduce_tensor(loss_den, self.world_size).clone().detach().item(),
            1, self.master
        )
        
        # Log momentum value periodically
        if self.iter % 100 == 0 and self.master:
            current_momentum = net_module.get_current_momentum()
            if isinstance(current_momentum, torch.Tensor):
                current_momentum = current_momentum.item()
            log_msg(self.logger, f"[BYOL] Step {self.iter}, Momentum: {current_momentum:.4f}")

    @torch.no_grad()
    def test(self):
        """Test/evaluation loop"""
        if self.master:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
        
        self.reset(isTrain=False)
        imgs_masks, anomaly_maps, cls_names, anomalys = [], [], [], []
        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)
        glb_feats = []
        labels = []
        
        while batch_idx < test_length:
            t1 = get_timepc()
            batch_idx += 1
            test_data = next(test_loader)
            self.set_input(test_data)
            self.forward()
            
            # Compute anomaly maps
            feats_t = self.feats_t
            feats_s = self.feats_s
            
            anomaly_map_list = []
            for f_t, f_s in zip(feats_t, feats_s):
                # Compute feature difference
                diff = (f_t - f_s) ** 2
                # Average over channels and upsample
                diff = diff.mean(dim=1, keepdim=True)
                diff = F.interpolate(diff, size=self.imgs.shape[-2:], mode='bilinear', align_corners=False)
                anomaly_map_list.append(diff)
            
            # Combine multi-scale anomaly maps
            anomaly_map = sum(anomaly_map_list) / len(anomaly_map_list)
            anomaly_map = anomaly_map.squeeze(1)  # (B, H, W)
            
            # Collect results
            for i in range(self.bs):
                imgs_masks.append(self.imgs_mask[i].cpu().numpy())
                anomaly_maps.append(anomaly_map[i].cpu().numpy())
                cls_names.append(self.cls_name[i])
                anomalys.append(self.anomaly[i])
            
            t2 = get_timepc()
            
            if self.master:
                if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
                    msg = f"Test [{batch_idx}/{test_length}] Time: {t2-t1:.3f}s"
                    log_msg(self.logger, msg)
        
        # Compute metrics
        if self.master:
            # Use evaluator to compute AUROC, etc.
            results = self.evaluator.compute(
                anomaly_maps=anomaly_maps,
                gt_masks=imgs_masks,
                cls_names=cls_names,
                anomalys=anomalys
            )
            
            for metric_name, value in results.items():
                log_msg(self.logger, f"{metric_name}: {value:.4f}")
            
            return results
        
        return None


# ============================================================================
# Alternative: Minimal changes to existing trainer
# ============================================================================

def patch_existing_trainer(trainer_class):
    """
    Decorator to patch existing trainer with BYOL momentum updates.
    
    Usage:
        @patch_existing_trainer
        class RDLGCTrainer(BaseTrainer):
            ...
    """
    original_init = trainer_class.__init__
    original_optimize = trainer_class.optimize_parameters
    
    def new_init(self, cfg):
        original_init(self, cfg)
        
        # Add predictor to optimizer
        net_module = self.net.module if hasattr(self.net, 'module') else self.net
        if hasattr(net_module, 'predictor'):
            # Add predictor params to proj_opt
            predictor_params = list(net_module.predictor.parameters())
            for param in predictor_params:
                self.optim.proj_opt.add_param_group({'params': param})
            
            # Set total steps
            total_steps = cfg.trainer.iter_full if hasattr(cfg.trainer, 'iter_full') else \
                         cfg.trainer.epoch_full * cfg.data.train_size
            net_module.set_total_steps(total_steps)
    
    def new_optimize(self):
        original_optimize(self)
        
        # Update momentum encoder
        net_module = self.net.module if hasattr(self.net, 'module') else self.net
        if hasattr(net_module, 'update_momentum_encoder'):
            net_module.update_momentum_encoder()
    
    trainer_class.__init__ = new_init
    trainer_class.optimize_parameters = new_optimize
    
    return trainer_class