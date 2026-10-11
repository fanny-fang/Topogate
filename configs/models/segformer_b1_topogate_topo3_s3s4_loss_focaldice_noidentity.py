CONFIG = {
    'name': 'segformer_b1_topogate_topo3_s3s4_loss_focaldice_noidentity',
    'backbone': {'type': 'segformer_b1', 'pretrained': True, 'in_channels': 3, 'allow_fallback': False},
    'fusion': {'type': 'topogate', 'inject_layers': [2, 3], 'reduction': 4, 'residual_init': 0.15, 'modulation_init': 0.05, 'min_gate': 0.05, 'use_identity': False},
    'topo_channels': ['rel_dem', 'slope', 'curvature'],
    'data': {'topo_indices': [0, 1, 2], 'topo_channel_names': ['rel_dem', 'slope', 'curvature']},
    'decoder': {'channels': 256, 'num_classes': 1},
    'loss': {'bce': 0.0, 'dice': 0.5, 'focal': 0.5, 'focal_alpha': 0.25, 'focal_gamma': 2.0, 'gate_l2': 0.0, 'ignore_index': 255},
    'train': {'lr': 6e-05, 'weight_decay': 0.01, 'iters': 40000, 'warmup_iters': 1500, 'backbone_lr_mult': 0.5, 'topo_lr_mult': 3.0, 'fuser_lr_mult': 3.0},
}
