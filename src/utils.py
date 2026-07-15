import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def get_grad_norm(model):

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def check_anomalies(history, current_epoch):

    warnings = []
    if np.isnan(history['train_loss'][-1]):
        warnings.append("CẢNH BÁO CRITICAL: Train Loss bị NaN (Exploding Gradients)!")
    
    if current_epoch >= 3:

        if history['train_loss'][-1] >= history['train_loss'][-2] >= history['train_loss'][-3]:
            warnings.append("CHÚ Ý: Train Loss đang có xu hướng tăng hoặc đi ngang!")

        if history['val_acc'][-1] == history['val_acc'][-2] == history['val_acc'][-3]:
            warnings.append("CHÚ Ý: Validation Accuracy đóng băng (Dấu hiệu Model Collapse)!")
    return warnings

def plot_training_curves(history, best_epoch, cfg, best_val_acc):

    epochs_ran = len(history['train_loss'])
    er = range(1, epochs_ran + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(er, history['train_loss'], 'b-', lw=2, label='Train')
    axes[0].plot(er, history['val_loss'],   'r-', lw=2, label='Val')
    axes[0].axvline(best_epoch, color='g', ls='--', alpha=0.7, label=f'Best (ep{best_epoch})')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(er, history['train_acc'], 'b-', lw=2, label='Train')
    axes[1].plot(er, history['val_acc'],   'r-', lw=2, label='Val')
    axes[1].axvline(best_epoch, color='g', ls='--', alpha=0.7)
    axes[1].set_title('Accuracy (%)'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(er, history['lr'], 'g-', lw=2)
    axes[2].set_title('Learning Rate'); axes[2].set_xlabel('Epoch')
    axes[2].set_yscale('log'); axes[2].grid(alpha=0.3)

    plt.suptitle(
        f'Eff-B0 + QNN Training — {cfg.num_Qubits}Q-{cfg.num.Layers}L | '
        f'Best Validation Acc={best_val_acc:.2f}%',
        fontsize=12, fontweight='bold')
    plt.tight_layout()

    curve_path = cfg.RESULTS_DIR / 'training_curves.png'
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved training curves to → {curve_path}')

def save_evaluation_results(all_labels, all_preds, class_names, test_acc, cfg):

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.3)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix — Test Acc: {test_acc:.2f}%', fontsize=13)
    plt.tight_layout()
    
    cm_path = cfg.RESULTS_DIR / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved Confusion Matrix to → {cm_path}')

    # 2. Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
    report_path = cfg.RESULTS_DIR / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f'Test Acc: {test_acc:.2f}%\n\n')
        f.write(report)
    print(f'Saved Classification Report to → {report_path}')