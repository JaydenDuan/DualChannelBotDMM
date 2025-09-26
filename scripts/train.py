# scripts/train.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import logging
from configs.config import Config
from training.cross_validation import CrossValidationTrainer
from data.data_loader import FoldDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train BotDMM model")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='cv',
                        choices=['cv', 'single', 'test'],
                        help='Training mode: cv for cross-validation, single for single fold, test for testing')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold ID for single fold training')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()
    config = Config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if args.mode == 'cv':
        # Run cross-validation
        logger.info("Starting 10-fold cross-validation...")
        cv_trainer = CrossValidationTrainer(
            cv_dir=config.get('paths.data_dir'),
            config=config.get('training'),
            device=device
        )
        cv_summary = cv_trainer.run_cross_validation()
        logger.info("Cross-validation completed!")

    elif args.mode == 'single':
        logger.info(f"Training single fold {args.fold}...")
        cv_trainer = CrossValidationTrainer(
            cv_dir=config.get('paths.data_dir'),
            config=config.get('training'),
            device=device
        )
        fold_results = cv_trainer.train_fold(args.fold)
        logger.info(f"Fold {args.fold} training completed!")

    elif args.mode == 'test':
        logger.info("Testing on holdout set...")
        cv_trainer = CrossValidationTrainer(
            cv_dir=config.get('paths.data_dir'),
            config=config.get('training'),
            device=device
        )
        test_loader = FoldDataLoader(
            cv_dir=config.get('paths.data_dir'),
            fold_id=-1,
            device=device
        ).get_val_loader(batch_size=config.get('training.batch_size'))

        test_results = cv_trainer.test_on_holdout(test_loader)
        logger.info("Testing completed!")


if __name__ == "__main__":
    main()