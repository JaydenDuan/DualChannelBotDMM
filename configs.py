# configs/config.py
import yaml
import os


class Config:

    def __init__(self, config_path=None):
        self.config = self._default_config()

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                self.config.update(custom_config)

    def _default_config(self):
        return {
            'model': {
                'embedding_dim': 128,
                'feature_dim': 128,
                'num_temporal_steps': 5,
                'dropout': 0.3,
                'temperature': 0.1,
                'alpha': 0.5,
                'num_heads': 8
            },

            'training': {
                'batch_size': 32,
                'lr': 0.00005,
                'weight_decay': 5e-4,
                'epochs': 100,
                'patience': 20,
                'clip_grad': 1.0,
                'lambda_feature_contrast': 0.1,
                'lambda_class_contrast': 0.1
            },

            'data': {
                'num_folds': 10,
                'test_size': 0.2,
                'random_seed': 42,
                'num_workers': 4
            },

            'paths': {
                'data_dir': 'dual_format_cv/cross_validation',
                'output_dir': 'experiments/results',
                'checkpoint_dir': 'checkpoints'
            }
        }

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def set(self, key, value):
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)