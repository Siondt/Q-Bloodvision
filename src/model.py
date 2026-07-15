import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from pennylane.qnn import TorchLayer
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from config import cfg

# Khởi tạo thiết bị lượng tử
dev = qml.device('lightning.qubit', wires=cfg.num_Qubits)

@qml.qnode(dev, interface='torch', diff_method='best')
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs * np.pi, wires=range(cfg.num_Qubits), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(cfg.num_Qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(cfg.num_Qubits)]

class FeatureBridge(nn.Module):
    def __init__(self, in_features, n_qubits, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_qubits),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

class EffB0_QNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Classical Backbone: EfficientNet-B0
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        for p in backbone.parameters():
            p.requires_grad = False
        for p in backbone.features[-3:].parameters():
            p.requires_grad = True
            
        self.cnn = nn.Sequential(
            backbone.features,
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 2. Dimensionality Reduction
        self.bridge = FeatureBridge(1280, cfg.num_Qubits, cfg.DROPOUT)
        
        # 3. Quantum Layer
        weight_shapes = {'weights': (cfg.num_Layers, cfg.num_Qubits, 3)}
        self.quantum = TorchLayer(quantum_circuit, weight_shapes)
        nn.init.uniform_(self.quantum.weights, 0, 2 * np.pi)
        
        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(cfg.num_Qubits, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.view(feat.size(0), -1)
        
        bridge_out = self.bridge(feat)
        q_out = self.quantum(bridge_out)
        
        # Residual connection
        combined = (bridge_out * 2 - 1) + q_out 
        return self.classifier(combined)