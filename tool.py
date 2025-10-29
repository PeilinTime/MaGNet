from torch import nn
from torcheval.metrics.functional import binary_accuracy, binary_auprc, binary_auroc, binary_f1_score, binary_precision, binary_recall
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast


def train_epoch(model, dataloader, criterion, optimizer, scaler, device="cuda"):
    model.train()
    total_loss = 0
    output_list = []
    y_list = []

    progress_bar = tqdm(dataloader, desc='Training')
    for X_batch, y_batch in progress_bar:  # X_batch: [B, N, T, F], y_batch:[B, N]
        batch_size = X_batch.size(0)

        optimizer.zero_grad()

        batch_outputs = []
        batch_labels = []
        total_batch_loss = 0

        with autocast():
            for b in range(batch_size):
                X = X_batch[b].to(device)  # X: [N, T, F]
                y = y_batch[b].to(device)  # y: [N]

                # Store labels for metrics calculation
                batch_labels.append(y)

                # Forward pass
                outputs, _, _, _, _, _ = model(X)  # model's output
                batch_outputs.append(outputs)

                loss = criterion(outputs, y)
                total_batch_loss += loss

            avg_batch_loss = total_batch_loss / batch_size
            scaler.scale(avg_batch_loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            batch_outputs_cat = torch.cat(batch_outputs, dim=0)
            batch_labels_cat = torch.cat(batch_labels, dim=0)

            output_list.append(batch_outputs_cat.detach())
            y_list.append(batch_labels_cat.detach())

            batch_metrics = Metrics.calculate_metrics(batch_outputs_cat, batch_labels_cat)

            progress_bar.set_postfix(
                {'loss': avg_batch_loss.item(), 'accuracy': batch_metrics['accuracy'],
                 'precision': batch_metrics['precision'],
                 'recall': batch_metrics['recall'], 'f1': batch_metrics['f1'],
                 'auroc': batch_metrics['auroc'], 'auprc': batch_metrics['auprc']})

        total_loss += avg_batch_loss.item()

    with torch.no_grad():
        all_outputs = torch.cat(output_list, dim=0)
        all_labels = torch.cat(y_list, dim=0)
        final_metrics = Metrics.calculate_metrics(all_outputs, all_labels)
        final_metrics['loss'] = total_loss / len(dataloader)

    return final_metrics


def validate(model, dataloader, criterion, device="cuda"):
    model.eval()
    total_loss = 0
    output_list = []
    y_list = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:  # X_batch: [B, N, T, F], y_batch:[B, N]
            batch_size = X_batch.size(0)
            batch_outputs = []
            batch_labels = []

            total_batch_loss = 0
            for b in range(batch_size):
                X = X_batch[b].to(device)  # X: [N, T, F]
                y = y_batch[b].to(device)  # y: [N]

                # Store labels for metrics calculation
                batch_labels.append(y)

                # Forward pass
                outputs, _, _, _, _, _ = model(X)  # model's output
                batch_outputs.append(outputs)

                # Compute loss
                loss = criterion(outputs, y)
                total_batch_loss += loss

            # Average batch loss
            avg_batch_loss = total_batch_loss / batch_size
            total_loss += avg_batch_loss.item()

            # Concatenate batch outputs and labels for overall metrics
            batch_outputs_cat = torch.cat(batch_outputs, dim=0)
            batch_labels_cat = torch.cat(batch_labels, dim=0)

            output_list.append(batch_outputs_cat)
            y_list.append(batch_labels_cat)

    # Calculate final metrics across all batches
    all_outputs = torch.cat(output_list, dim=0)
    all_labels = torch.cat(y_list, dim=0)
    final_metrics = Metrics.calculate_metrics(all_outputs, all_labels)
    final_metrics['loss'] = total_loss / len(dataloader)

    return final_metrics


class Metrics:
    @staticmethod
    def calculate_metrics(logits, targets) -> dict:
        """
        Calculate metrics using torchmetrics
        """
        preds = logits.argmax(dim=1)
        probabilities = nn.functional.softmax(logits, dim=1)[:, 1]

        # 2分类：
        accuracy = binary_accuracy(preds, targets)
        precision = binary_precision(preds, targets)
        recall = binary_recall(preds, targets)
        f1_score = binary_f1_score(preds, targets)
        auroc = binary_auroc(probabilities, targets)
        auprc = binary_auprc(probabilities, targets)

        return {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1_score.item(),
            'auroc': auroc.item(),
            'auprc': auprc.item()
        }


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0