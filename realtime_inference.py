import cv2
import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# 1. Báo cho Python biết hãy tìm code trong thư mục yolov13 vừa tải về
sys.path.append(os.path.join(os.getcwd(), 'yolov13'))

# 2. Import module từ kho yolov13
# LƯU Ý: Dòng này phụ thuộc vào cách tác giả iMoonLab đặt tên file!
# Thông thường với các bản YOLO tùy biến, cú pháp sẽ giống như sau:
from models.experimental import attempt_load

from src.config import cfg
from src.model import EffB0_QNN

def run_realtime_pipeline():
    # 1. Tải mô hình YOLOv13 (Định vị & Bao khung)
    print("Loading YOLOv13 detector...")
    yolo_model = YOLO("weights/yolov13_blood.pt") # Đường dẫn tới trọng số YOLO của bạn

    # 2. Tải mô hình QNN (Phân loại chi tiết)
    print("Loading Hybrid QNN classifier...")
    # Thay num_classes bằng số lớp thực tế của bạn
    qnn_model = EffB0_QNN(num_classes=12).to(cfg.DEVICE) 
    qnn_model.load_state_dict(torch.load(cfg.RESULTS_DIR / 'best_model.pt')['model_state_dict'])
    qnn_model.eval()

    # Bộ biến đổi ảnh cho QNN
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Danh sách nhãn (cập nhật theo dataset của bạn)
    class_names = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil', ...] 

    # 3. Khởi động Camera/Webcam
    cap = cv2.VideoCapture(0) # Đổi thành đường dẫn video nếu không dùng webcam
    
    print("Mở luồng camera... Nhấn 'q' để thoát.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Bước 1: YOLO dự đoán Bounding Box
        results = yolo_model(frame, verbose=False)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Lấy tọa độ
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Cắt vùng tế bào
                cell_crop = frame[y1:y2, x1:x2]
                if cell_crop.size == 0:
                    continue
                    
                # Chuyển đổi định dạng cho QNN
                cell_pil = Image.fromarray(cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB))
                input_tensor = transform(cell_pil).unsqueeze(0).to(cfg.DEVICE)
                
                # Bước 2: QNN phân loại vùng đã cắt
                with torch.no_grad():
                    output = qnn_model(input_tensor)
                    pred_idx = output.argmax(1).item()
                    conf = torch.softmax(output, dim=1)[0, pred_idx].item()
                    label = class_names[pred_idx]
                
                # Bước 3: Vẽ khung và dán nhãn lên màn hình
                color = (0, 255, 0) # Màu xanh lá
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
        # Hiển thị
        cv2.imshow("Q-BloodVision Real-time", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_pipeline()