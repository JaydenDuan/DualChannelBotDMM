# training/cross_validation.py
import torch
import numpy as np
import json
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import logging
from .trainer import Trainer
from models.botdmm import BotDMM
from data.data_loader import FoldDataLoader

logger = logging.getLogger(__name__)


class CrossValidationTrainer:
    def __init__(self, cv_dir, config, device='cuda'):
        self.cv_dir = cv_dir
        self.config = config
        self.device = device
        self.k_folds = 10

    def train_fold(self, fold_id):
        logger.info(f"Training Fold {fold_id}")
        data_loader = FoldDataLoader(self.cv_dir, fold_id, self.device)
        train_loader = data_loader.get_train_loader(batch_size=self.config['batch_size'])
        val_loader = data_loader.get_val_loader(batch_size=self.config['batch_size'])
        model = BotDMM(
            embedding_dimension=self.config['embedding_dim'],
            feature_dim=self.config['feature_dim'],
            num_temporal_steps=self.config['num_steps'],
            dropout=self.config['dropout'],
            temperature=self.config['temperature'],
            alpha=self.config['alpha']
        )
        trainer = Trainer(model, self.config, self.device)
        best_val_acc = trainer.train(train_loader, val_loader, self.config['epochs'])
        val_metrics = trainer.evaluate(val_loader)

        fold_results = {
            'fold_id': fold_id,
            'best_val_acc': best_val_acc,
            'val_metrics': val_metrics
        }
        model_path = os.path.join(self.cv_dir, f'fold_{fold_id}_model.pt')
        torch.save(model.state_dict(), model_path)

        return fold_results

    def run_cross_validation(self):
        all_fold_results = []
        for fold_id in range(self.k_folds):
            try:
                fold_results = self.train_fold(fold_id)
                all_fold_results.append(fold_results)
                results_file = os.path.join(self.cv_dir, f'fold_{fold_id}_results.json')
                with open(results_file, 'w') as f:
                    json.dump(fold_results, f, indent=2)

            except Exception as e:
                logger.error(f"Error in fold {fold_id}: {e}")
                continue

        cv_summary = self._compute_cv_statistics(all_fold_results)
        summary_file = os.path.join(self.cv_dir, 'cv_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(cv_summary, f, indent=2)

        self._print_cv_summary(cv_summary)

        return cv_summary

    def _compute_cv_statistics(self, all_fold_results):
        metrics = ['accuracy', 'f1', 'mcc', 'precision', 'recall']
        cv_summary = {}

        for metric in metrics:
            values = [result['val_metrics'][metric] for result in all_fold_results]
            cv_summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }

        cv_summary['num_folds'] = len(all_fold_results)
        return cv_summary

    def _print_cv_summary(self, cv_summary):
        logger.info("\n" + "=" * 60)
        logger.info("CROSS VALIDATION RESULTS")
        logger.info("=" * 60)

        for metric, stats in cv_summary.items():
            if metric != 'num_folds':
                logger.info(f"{metric.upper()}:")
                logger.info(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
                logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    def test_on_holdout(self, test_loader):
        best_fold_id = self._find_best_fold()
        model_path = os.path.join(self.cv_dir, f'fold_{best_fold_id}_model.pt')

        model = BotDMM(
            embedding_dimension=self.config['embedding_dim'],
            feature_dim=self.config['feature_dim'],
            num_temporal_steps=self.config['num_steps'],
            dropout=self.config['dropout'],
            temperature=self.config['temperature'],
            alpha=self.config['alpha']
        )
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)

        trainer = Trainer(model, self.config, self.device)
        test_metrics = trainer.evaluate(test_loader)

        logger.info("\n" + "=" * 60)
        logger.info("HOLDOUT TEST RESULTS")
        logger.info("=" * 60)
        for metric, value in test_metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")

        return test_metrics

    def _find_best_fold(self):
        best_acc = 0
        best_fold = 0

        for fold_id in range(self.k_folds):
            results_file = os.path.join(self.cv_dir, f'fold_{fold_id}_results.json')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    fold_results = json.load(f)
                    if fold_results['best_val_acc'] > best_acc:
                        best_acc = fold_results['best_val_acc']
                        best_fold = fold_id

        return best_fold