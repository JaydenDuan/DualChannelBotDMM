import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import logging
from training.cross_validation import CrossValidationTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train BotDMM model")
    
    # Data paths
    parser.add_argument("--cross_validation_dir", type=str, 
                        default="dual_format_cv/cross_validation",
                        help="Cross validation data directory")
    parser.add_argument("--results_dir", type=str, 
                        default="./cv_results",
                        help="Results save directory")
    
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Orthogonal constraint coefficient α")
    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--feature_dim", type=int, default=128,
                        help="Feature dimension")
    parser.add_argument("--num_steps", type=int, default=5,
                        help="Number of temporal steps")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    
    parser.add_argument("--lr", type=float, default=0.00005,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs per fold")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--clip_grad", type=float, default=1.0,
                        help="Gradient clipping")
    
    parser.add_argument("--lambda_feature_contrast", type=float, default=0.1,
                        help="Feature contrastive loss weight")
    parser.add_argument("--lambda_class_contrast", type=float, default=0.1,
                        help="Class contrastive loss weight")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Contrastive learning temperature")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA")
    parser.add_argument("--test_on_holdout", action="store_true", default=True,
                        help="Test best model on holdout set")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create config from command line arguments
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
        'batch_size': 32  # Default value
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    
    cv_trainer = CrossValidationTrainer(
        cv_dir=args.cross_validation_dir,
        config=config,
        device=device
    )
    
    cv_trainer.results_dir = args.results_dir
    os.makedirs(args.results_dir, exist_ok=True)
    
    logger.info("Starting 10-fold cross-validation...")
    cv_summary = cv_trainer.run_cross_validation()
    
    # Test on holdout if requested
    if args.test_on_holdout:
        logger.info("Testing on holdout set...")
        # Load test data loader here
        # test_results = cv_trainer.test_on_holdout(test_loader)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
