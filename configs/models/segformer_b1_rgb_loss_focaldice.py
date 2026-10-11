CONFIG = {
    'name': 'segformer_b1_rgb_loss_focaldice',
    'backbone': {'type': 'segformer_b1', 'pretrained': True, 'in_channels': 3, 'allow_fallback': False},
    'fusion': {'type': 'rgb'},
    'topo_channels': [],
    'decoder': {'channels': 256, 'num_classes': 1},
    'loss': {'bce': 0.0, 'dice': 0.5, 'focal': 0.5, 'focal_alpha': 0.25, 'focal_gamma': 2.0, 'ignore_index': 255},
    'train': {'lr': 6e-05, 'weight_decay': 0.01, 'iters': 40000, 'warmup_iters': 1500},
}
