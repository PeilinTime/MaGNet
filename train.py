import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler
from Dataset import StockDataset
from MaGNet import MaGNet
from tool import EarlyStopping, train_epoch, validate
from transformers import get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def main(epochs, dim, num_experts, num_heads_mha, num_channels, num_heads_CausalMHA,
         data_path, T, batch_size, num_MAGE, num_F2DAttn, num_TCH, TopK, M1,
         num_S2DAttn, num_GPH, M2):
    print("Loading and preprocessing data...")
    data= torch.load(data_path).to(device)


    total_date = data.shape[1]
    train_cutoff = int(total_date * 0.7)
    valid_cutoff = train_cutoff + int(total_date * 0.1)

    train_data = data[:, :train_cutoff]
    valid_data = data[:, train_cutoff:valid_cutoff]
    test_data = data[:, valid_cutoff:]

    epsilon = 1e-6
    train_data_mean = train_data.mean(dim=1, keepdim=True)  # Mean along F dimension
    train_data_std = train_data.std(dim=1, keepdim=True)  # Standard deviation along F dimension
    train_data = (train_data - train_data_mean) / (train_data_std + epsilon)

    valid_data_mean = valid_data.mean(dim=1, keepdim=True)  # Mean along F dimension
    valid_data_std = valid_data.std(dim=1, keepdim=True)  # Standard deviation along F dimension
    valid_data = (valid_data - valid_data_mean) / (valid_data_std + epsilon)

    test_data_mean = test_data.mean(dim=1, keepdim=True)  # Mean along F dimension
    test_data_std = test_data.std(dim=1, keepdim=True)  # Standard deviation along F dimension
    test_data = (test_data - test_data_mean) / (test_data_std + epsilon)

    # binary classification: given lookback T data for each stock on next day, rise = 1, otherwise 0
    train_dataset = StockDataset(train_data, T, device)
    valid_dataset = StockDataset(valid_data, T, device)
    test_dataset = StockDataset(test_data, T, device)




    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_train_batches = len(train_loader)
    num_valid_batches = len(valid_loader)
    num_test_batches = len(test_loader)

    N, F = data.shape[0], data.shape[2]

    model = MaGNet(N, T, F, dim, num_MAGE, num_experts,
                 num_heads_mha, num_F2DAttn, num_channels,
                 num_heads_CausalMHA, num_TCH, TopK, M1,
                 num_S2DAttn, num_GPH, M2, device=device, dropout=0.1).to(device)

    lr=1e-4
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=5)


    best_val_acc = 0.
    best_model_path = 'best_model.pth'

    history = {
        'train_loss': [], 'train_accuracy': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
        'train_auroc': [], 'train_auprc': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auroc': [],
        'val_auprc': [],
        'test_loss': [], 'test_accuracy': [], 'test_precision': [], 'test_recall': [], 'test_f1': [], 'test_auroc': [],
        'test_auprc': [],
    }

    print("Starting training...")

    steps_per_epoch = len(train_loader)
    num_training_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device = device)
        with torch.no_grad():
            val_metrics = validate(model, valid_loader, criterion, device = device)
            test_metrics = validate(model, test_loader, criterion, device = device)

        lr_scheduler.step()


        early_stopping(val_metrics['loss'])

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")

        for phase in ['train', 'val', 'test']:
            metrics = locals()[f'{phase}_metrics']
            history[f'{phase}_loss'].append(metrics['loss'])
            history[f'{phase}_accuracy'].append(metrics['accuracy'])
            history[f'{phase}_precision'].append(metrics['precision'])
            history[f'{phase}_recall'].append(metrics['recall'])
            history[f'{phase}_f1'].append(metrics['f1'])
            history[f'{phase}_auroc'].append(metrics['auroc'])
            history[f'{phase}_auprc'].append(metrics['auprc'])

        print(
            f"Train - Loss: {train_metrics['loss']:.4f}, accuracy: {train_metrics['accuracy']:.4f}, precision: {train_metrics['precision']:.4f}, recall: {train_metrics['recall']:.4f}, f1: {train_metrics['f1']:.4f}, auroc: {train_metrics['auroc']:.4f}, auprc: {train_metrics['auprc']:.4f}, ")
        print(
            f"Valid - Loss: {val_metrics['loss']:.4f}, accuracy: {val_metrics['accuracy']:.4f}, precision: {val_metrics['precision']:.4f}, recall: {val_metrics['recall']:.4f}, f1: {val_metrics['f1']:.4f}, auroc: {val_metrics['auroc']:.4f}, auprc: {val_metrics['auprc']:.4f}, ")
        print(
            f"Test - Loss: {test_metrics['loss']:.4f}, accuracy: {test_metrics['accuracy']:.4f}, precision: {test_metrics['precision']:.4f}, recall: {test_metrics['recall']:.4f}, f1: {test_metrics['f1']:.4f}, auroc: {test_metrics['auroc']:.4f}, auprc: {test_metrics['auprc']:.4f}, ")

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 4, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.plot(history['test_loss'], label='Test')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 4, 2)
    plt.plot(history['train_accuracy'], label='Train')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.plot(history['test_accuracy'], label='Test')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(2, 4, 3)
    plt.plot(history['train_precision'], label='Train')
    plt.plot(history['val_precision'], label='Validation')
    plt.plot(history['test_precision'], label='Test')
    plt.title('Precision History')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')

    plt.subplot(2, 4, 4)
    plt.plot(history['train_recall'], label='Train')
    plt.plot(history['val_recall'], label='Validation')
    plt.plot(history['test_recall'], label='Test')
    plt.title('Recall History')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')

    plt.subplot(2, 4, 5)
    plt.plot(history['train_f1'], label='Train')
    plt.plot(history['val_f1'], label='Validation')
    plt.plot(history['test_f1'], label='Test')
    plt.title('F1 History')
    plt.xlabel('Epoch')
    plt.ylabel('F1')

    plt.subplot(2, 4, 6)
    plt.plot(history['train_auroc'], label='Train')
    plt.plot(history['val_auroc'], label='Validation')
    plt.plot(history['test_auroc'], label='Test')
    plt.title('auroc History')
    plt.xlabel('Epoch')
    plt.ylabel('auroc')

    plt.subplot(2, 4, 7)
    plt.plot(history['train_auprc'], label='Train')
    plt.plot(history['val_auprc'], label='Validation')
    plt.plot(history['test_auprc'], label='Test')
    plt.title('auprc History')
    plt.xlabel('Epoch')
    plt.ylabel('auprc')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    # common parameters
    epochs = 30
    dim = 32 # size of Feature Embedding
    num_experts = 4 # number of experts in MoE in MAGE block
    num_heads_mha = 2 # number of heads in MHA in MAGE block
    num_channels = 4 # number of channels in Feature-wise/Stock-wise 2D Spatiotemporal Attention
    num_heads_CausalMHA = 2 # number of heads in CausalMHA in TCH




    ### For dataset DJIA:
    data_path = 'djia_alpha158_alpha360.pt'
    batch_size=32
    T = 20 # lookback window size
    num_MAGE = 2 # number of MAGE block
    num_F2DAttn = 1 # number of Feature-wise 2D Spatiotemporal Attention
    num_TCH = 1 # number of TCH
    M1 = 32 # number of hyperedges in TCH
    TopK = 32 # TopK sparsification in TCH
    num_S2DAttn = 1 # number of  Stock-wise 2D Spatiotemporal Attention
    num_GPH = 1 # number of GPH
    M2 = 16 # number of hyperedges in GPH
    main(epochs, dim, num_experts, num_heads_mha, num_channels, num_heads_CausalMHA,
         data_path, T, batch_size, num_MAGE, num_F2DAttn, num_TCH, TopK, M1,
         num_S2DAttn, num_GPH, M2)



    ### For dataset NASDAQ100:

    # data_path = 'nas100_alpha158_alpha360.pt'
    # T = 10  # lookback window size
    # batch_size = 24
    # num_MAGE = 1  # number of MAGE block
    # num_F2DAttn = 1  # number of Feature-wise 2D Spatiotemporal Attention
    # num_S2DAttn = 1  # number of  Stock-wise 2D Spatiotemporal Attention
    # num_TCH = 2  # number of TCH
    # TopK = 64  # TopK sparsification in TCH
    # M1 = 64  # number of hyperedges in TCH
    # num_GPH = 2  # number of GPH
    # M2 = 32  # number of hyperedges in GPH
    # main(epochs, dim, num_experts, num_heads_mha, num_channels, num_heads_CausalMHA,
    #      data_path, T, batch_size, num_MAGE, num_F2DAttn, num_TCH, TopK, M1,
    #      num_S2DAttn, num_GPH, M2)
    #




    # ### For dataset CSI300:

    # data_path = 'csi300_alpha158_alpha360.pt'
    # T = 10  # lookback window size
    # batch_size = 4
    # num_MAGE = 1  # number of MAGE block
    # num_F2DAttn = 1  # number of Feature-wise 2D Spatiotemporal Attention
    # num_S2DAttn = 1  # number of  Stock-wise 2D Spatiotemporal Attention
    # num_TCH = 2  # number of TCH
    # TopK = 64  # TopK sparsification in TCH
    # M1 = 64  # number of hyperedges in TCH
    # num_GPH = 1  # number of GPH
    # M2 = 32  # number of hyperedges in GPH
    # main(epochs, dim, num_experts, num_heads_mha, num_channels, num_heads_CausalMHA,
    #      data_path, T, batch_size, num_MAGE, num_F2DAttn, num_TCH, TopK, M1,
    #      num_S2DAttn, num_GPH, M2)
