import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.config import cfg
from src.dataset import get_dataloaders
from src.model import EffB0_QNN

def evaluate():
    print("1. Loading test data...")
    _, _, test_loader, class_names, _ = get_dataloaders()
    num_classes = len(class_names)
    
    print("2. Loading best model...")
    # Khởi tạo lại kiến trúc model
    model = EffB0_QNN(num_classes=num_classes).to(cfg.DEVICE)
    
    # Load trọng số tốt nhất đã lưu khi train
    checkpoint = torch.load(cfg.RESULTS_DIR / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("3. Evaluating on Test Set...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(cfg.DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            
    print("4. Generating Reports...")
    # Đảm bảo thư mục kết quả tồn tại
    cfg.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    # 4.1. Tạo và lưu Báo cáo phân loại (Classification Report)
    report = classification_report(all_targets, all_preds, target_names=class_names)
    with open(cfg.RESULTS_DIR / 'classification_report.txt', 'w') as f:
        f.write(report)
        
    # 4.2. Tạo và lưu Ma trận nhầm lẫn (Confusion Matrix)
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(cfg.RESULTS_DIR / 'confusion_matrix.png')
    plt.close()
    
    print(f"✅ Đã đánh giá xong! Báo cáo được lưu tại: {cfg.RESULTS_DIR}")

if __name__ == "__main__":
    evaluate()