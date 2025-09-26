# training/trainer.py
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('lr', 0.00005),
            weight_decay=config.get('weight_decay', 5e-4)
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(train_loader, desc="Training"):
            des, tweets, amrs, num_prop, llm_features, edge_indices, labels = self._unpack_batch(batch)

            self.optimizer.zero_grad()

            outputs = self.model(des, tweets, amrs, num_prop, llm_features, edge_indices, labels)

            cls_loss = self.criterion(outputs['logits'], labels)
            cont_loss = outputs['feature_contrastive_loss']
            class_cont_loss = outputs['class_contrastive_loss']

            total_loss_batch = (cls_loss +
                                self.config.get('lambda_feature_contrast', 0.1) * cont_loss +
                                self.config.get('lambda_class_contrast', 0.1) * class_cont_loss)

            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('clip_grad', 1.0))
            self.optimizer.step()

            total_loss += total_loss_batch.item()

            preds = outputs['logits'].argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)

        return avg_loss, train_acc

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                des, tweets, amrs, num_prop, llm_features, edge_indices, labels = self._unpack_batch(batch)

                outputs = self.model(des, tweets, amrs, num_prop, llm_features, edge_indices)

                loss = self.criterion(outputs['logits'], labels)
                total_loss += loss.item()

                preds = outputs['logits'].argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='binary'),
            'mcc': matthews_corrcoef(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='binary'),
            'recall': recall_score(all_labels, all_preds, average='binary')
        }

        return metrics

    def train(self, train_loader, val_loader, num_epochs):
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            train_loss, train_acc = self.train_epoch(train_loader)
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            val_metrics = self.evaluate(val_loader)
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val F1: {val_metrics['f1']:.4f}, Val MCC: {val_metrics['mcc']:.4f}")
            self.scheduler.step(val_metrics['accuracy'])
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                self.save_checkpoint(epoch, val_metrics)
            else:
                patience_counter += 1

            if patience_counter >= self.config.get('patience', 20):
                logger.info("Early stopping triggered")
                break

        return best_val_acc

    def _unpack_batch(self, batch):
        pass

    def save_checkpoint(self, epoch, metrics):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
        logger.info(f"Checkpoint saved for epoch {epoch}")