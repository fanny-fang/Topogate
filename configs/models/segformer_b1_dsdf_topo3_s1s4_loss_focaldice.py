CONFIG = {
    'name': 'segformer_b1_dsdf_topo3_s1s4_loss_focaldice',
    'backbone': {'type': 'segformer_b1', 'pretrained': True, 'in_channels': 3, 'allow_fallback': False},
    'fusion': {'type': 'dsdf', 'late_stage': 3, 'early_hidden': 32, 'early_gain_max': 1.0, 'num_heads': 4, 'late_residual': False},
    'topo_channels': ['rel_dem', 'slope', 'curvature'],
    'data': {'topo_indices': [0, 1, 2], 'topo_channel_names': ['rel_dem', 'slope', 'curvature']},
    'decoder': {'channels': 256, 'num_classes': 1},
    'loss': {'bce': 0.0, 'dice': 0.5, 'focal': 0.5, 'focal_alpha': 0.25, 'focal_gamma': 2.0, 'ignore_index': 255},
    'train': {'lr': 6e-05, 'weight_decay': 0.01, 'iters': 40000, 'warmup_iters': 1500},
}
