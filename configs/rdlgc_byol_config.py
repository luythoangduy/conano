"""
Example configuration for RDLGC with BYOL-style architecture.

Key differences from original config:
1. Model uses 'rd_lgc_byol' instead of 'rd_lgc'
2. Dense loss uses 'BYOLDenseLoss' instead of 'DenseCLLoss'
3. Added momentum scheduling options
"""

from argparse import Namespace

# ============================================================================
# Data Configuration
# ============================================================================
class cfg_data(Namespace):
    def __init__(self):
        Namespace.__init__(self)
        
        self.root = 'data/mvtec'
        self.dataset = 'mvtec'
        self.cls_names = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                          'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                          'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        
        self.train_bs = 16
        self.test_bs = 16
        self.num_workers = 8
        
        self.resize = 256
        self.cropsize = 256


# ============================================================================
# Model Configuration (BYOL-style)
# ============================================================================
class cfg_model(Namespace):
    def __init__(self):
        Namespace.__init__(self)
        
        # Teacher encoder (frozen)
        self.model_t = Namespace()
        self.model_t.name = 'timm_wide_resnet50_2'
        self.model_t.kwargs = dict(
            pretrained=True,
            checkpoint_path='model/pretrain/wide_resnet50_racm-8234f177.pth',
            strict=False,
            features_only=True,
            out_indices=[1, 2, 3]
        )
        
        # Student decoder
        self.model_s = Namespace()
        self.model_s.name = 'de_wide_resnet50_2'
        self.model_s.kwargs = dict(
            pretrained=False,
            checkpoint_path='',
            strict=False
        )
        
        # Main model - BYOL style
        self.model = Namespace()
        self.model.name = 'rd_lgc_byol'  # Use BYOL-style model
        self.model.kwargs = dict(
            pretrained=False,
            checkpoint_path='',
            strict=True,
            model_t=self.model_t,
            model_s=self.model_s,
            dp=False,  # Use sparse projection if True
            
            # === BYOL Momentum Settings ===
            momentum_schedule='cosine',  # 'constant', 'linear', 'cosine'
            momentum_start=0.9,          # Starting momentum (for scheduled)
            momentum_end=0.999,          # Ending momentum (for scheduled)
            # momentum=0.99,             # Use this if momentum_schedule='constant'
        )


# ============================================================================
# Loss Configuration (BYOL-style)
# ============================================================================
class cfg_loss(Namespace):
    def __init__(self):
        Namespace.__init__(self)
        
        # Reconstruction loss
        self.cos = dict(
            name='CosLoss',
            kwargs=dict(lam=1.0)
        )
        
        # Global contrastive loss (class-level)
        self.scl = dict(
            name='SupConLoss',
            kwargs=dict(
                temperature=0.1,
                lam=1.0
            )
        )
        
        # Dense local loss - BYOL style (NO negatives!)
        self.dense = dict(
            name='BYOLDenseLoss',  # Changed from DenseCLLoss
            kwargs=dict(
                lam=1.0,
                use_spatial_matching=True,  # Use spatial correspondence
            )
        )
        
        # Alternative: Class-aware BYOL loss
        # self.dense = dict(
        #     name='ClassAwareBYOLDenseLoss',
        #     kwargs=dict(
        #         lam=1.0,
        #         use_spatial_matching=True,
        #     )
        # )
        
        self.clip_grad = 1.0
        self.retain_graph = False
        self.create_graph = False


# ============================================================================
# Optimizer Configuration
# ============================================================================
class cfg_optim(Namespace):
    def __init__(self):
        Namespace.__init__(self)
        
        self.lr = 0.005
        
        # Optimizer for projection + predictor
        self.proj_opt = Namespace()
        self.proj_opt.kwargs = dict(
            opt='sgd',
            momentum=0.9,
            weight_decay=0.0001
        )
        
        # Optimizer for distillation (decoder)
        self.distill_opt = Namespace()
        self.distill_opt.kwargs = dict(
            opt='sgd',
            momentum=0.9,
            weight_decay=0.0001
        )


# ============================================================================
# Trainer Configuration
# ============================================================================
class cfg_trainer(Namespace):
    def __init__(self):
        Namespace.__init__(self)
        
        self.name = 'RDLGCBYOLTrainer'  # Use BYOL trainer
        
        self.epoch_full = 200
        self.iter_full = None  # Set automatically based on data
        
        self.resume_dir = ''
        self.save_per = 10
        self.test_per = 10


# ============================================================================
# Full Configuration
# ============================================================================
class Config:
    def __init__(self):
        self.data = cfg_data()
        self.model = cfg_model()
        self.loss = cfg_loss()
        self.optim = cfg_optim()
        self.trainer = cfg_trainer()
        
        # Logging
        self.logging = Namespace()
        self.logging.log_per = 10
        self.logging.test_log_per = 10


# Create config instance
cfg = Config()


# ============================================================================
# Alternative configs for ablation study
# ============================================================================

def get_config_constant_momentum():
    """Config with constant momentum (like MoCo)"""
    cfg = Config()
    cfg.model.model.kwargs['momentum_schedule'] = 'constant'
    cfg.model.model.kwargs['momentum'] = 0.99
    return cfg


def get_config_linear_momentum():
    """Config with linear momentum schedule"""
    cfg = Config()
    cfg.model.model.kwargs['momentum_schedule'] = 'linear'
    cfg.model.model.kwargs['momentum_start'] = 0.9
    cfg.model.model.kwargs['momentum_end'] = 0.999
    return cfg


def get_config_class_aware():
    """Config with class-aware BYOL loss"""
    cfg = Config()
    cfg.loss.dense = dict(
        name='ClassAwareBYOLDenseLoss',
        kwargs=dict(lam=1.0, use_spatial_matching=True)
    )
    return cfg


def get_config_no_spatial_matching():
    """Config without spatial matching (direct position correspondence)"""
    cfg = Config()
    cfg.loss.dense['kwargs']['use_spatial_matching'] = False
    return cfg