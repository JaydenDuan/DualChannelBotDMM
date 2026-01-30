import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader

from training.trainer import Trainer
from models.botdmm import BotDMM
from data.data_loader import Twibot20Dataset, BotSimDataset, collate_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train(args, config, device):
    logger.info("=" * 60)
    logger.info(f"Training BotDMM - Dataset: {args.dataset}")
    logger.info("=" * 60)

    logger.info("Loading datasets...")
    if args.dataset == 'botsim':
        train_dataset = BotSimDataset(args.data_dir, split='train', num_steps=config['num_steps'])
        val_dataset = BotSimDataset(args.data_dir, split='val', num_steps=config['num_steps'])
        test_dataset = BotSimDataset(args.data_dir, split='test', num_steps=config['num_steps'])
        num_classes = 2
    else:
        train_dataset = Twibot20Dataset(args.data_dir, split='train', num_steps=config['num_steps'])
        val_dataset = Twibot20Dataset(args.data_dir, split='val', num_steps=config['num_steps'])
        test_dataset = Twibot20Dataset(args.data_dir, split='test', num_steps=config['num_steps'])
        num_classes = 3

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    sample = train_dataset[0]
    num_prop_size = sample['num_prop'].shape[0]
    llm_features_size = sample['llm_features'].shape[0]
    logger.info(f"num_prop_size: {num_prop_size}, llm_features_size: {llm_features_size}")

    model = BotDMM(
        des_size=768,
        tweet_size=768,
        amr_size=768,
        num_prop_size=num_prop_size,
        llm_features_size=llm_features_size,
        embedding_dimension=config['embedding_dim'],
        feature_dim=config['feature_dim'],
        num_temporal_steps=config['num_steps'],
        dropout=config['dropout'],
        temperature=config['temperature'],
        alpha=config['alpha'],
        num_classes=num_classes
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    trainer = Trainer(model, config, device)
    best_val_acc, best_model_path = trainer.train(train_loader, val_loader, config['epochs'])

    logger.info("\n" + "=" * 60)
    logger.info("Testing on holdout test set...")
    logger.info("=" * 60)

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.to(device)
    test_metrics = trainer.evaluate(test_loader)

    f1_type = "binary" if num_classes == 2 else "macro"
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 ({f1_type}): {test_metrics['f1']:.4f}")
    logger.info(f"Test MCC: {test_metrics['mcc']:.4f}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train BotDMM model")

    parser.add_argument("--data_dir", type=str, default="Dataset/Twibot20", help="Data directory")
    parser.add_argument("--save_dir", type=str, default="./models_saved", help="Model save directory")
    parser.add_argument("--dataset", type=str, default="twibot20", choices=["twibot20", "botsim"],
                        help="Dataset to use: twibot20 (3-class) or botsim (binary)")

    parser.add_argument("--alpha", type=float, default=0.5, help="Orthogonal constraint coefficient")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--feature_dim", type=int, default=128, help="Feature dimension")
    parser.add_argument("--num_steps", type=int, default=5, help="Number of temporal steps")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    parser.add_argument("--lambda_feature_contrast", type=float, default=0.1, help="Feature contrastive loss weight")
    parser.add_argument("--lambda_class_contrast", type=float, default=0.1, help="Class contrastive loss weight")
    parser.add_argument("--temperature", type=float, default=0.1, help="Contrastive learning temperature")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f"Using device: {device}")

    num_classes = 2 if args.dataset == 'botsim' else 3
    config = {
        'embedding_dim': args.embedding_dim,
        'feature_dim': args.feature_dim,
        'num_steps': args.num_steps,
        'dropout': args.dropout,
        'temperature': args.temperature,
        'alpha': args.alpha,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'patience': args.patience,
        'clip_grad': args.clip_grad,
        'lambda_feature_contrast': args.lambda_feature_contrast,
        'lambda_class_contrast': args.lambda_class_contrast,
        'batch_size': args.batch_size,
        'save_dir': args.save_dir,
        'num_classes': num_classes,
        'dataset': args.dataset
    }

    os.makedirs(args.save_dir, exist_ok=True)
    train(args, config, device)
    logger.info("\nTraining completed!")


if __name__ == "__main__":
    main()
