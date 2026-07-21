import time
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import cfg
from src.dataset import get_dataloaders
from src.model import EffB0_QNN
from src.utils import check_anomalies, plot_training_curves

def train():

    print("Loading data...")
    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders()
    num_classes = len(class_names)
    
    print("Initializing model...")
    model = EffB0_QNN(num_classes=num_classes).to(cfg.DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW([
        {'params': model.cnn.parameters(), 'lr': cfg.LR_CNN},
        {'params': model.bridge.parameters(), 'lr': cfg.LR},
        {'params': model.quantum.parameters(), 'lr': cfg.LR},
        {'params': model.classifier.parameters(), 'lr': cfg.LR},
    ], weight_decay=cfg.WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCH, eta_min=1e-6)

    best_val_acc = 0.0
    best_epoch = 0
    patience_count = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    for epoch in range(1, cfg.EPOCH + 1):
        model.train()
        train_loss, train_correct, total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCH} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
            pbar.set_postfix(loss=loss.item())
            
        train_acc = train_correct / total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == targets).sum().item()
                val_total += targets.size(0)
                
        val_acc = val_correct / val_total
        scheduler.step()

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc * 100)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc * 100)
        history['lr'].append(optimizer.param_groups[1]['lr'])
        
        print(f"Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.2f}%")

        warnings = check_anomalies(history, epoch)
        for w in warnings:
            print(w)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_count = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }
            torch.save(checkpoint, cfg.RESULT_DIR / 'best_model.pt')
            print("--> Saved Best Model!")
        else:
            patience_count += 1
            if patience_count >= cfg.PATIENCE:
                print("Early stopping triggered!")
                break

    print("\nTraining Complete! Generating curves...")
    plot_training_curves(history, best_epoch, cfg, best_val_acc * 100)

if __name__ == "__main__":
    train()