"""DeepConvLSTM (Ordonez & Roggen, Sensors 2016).

Four temporal convolutions applied to every input channel independently
(kernel (k, 1), so channels are not mixed until the LSTM), followed by two LSTM
layers and a linear classifier on the last time step.

forward(x) takes x of shape [B, C, T] and returns (embedding [B, hidden],
logits [B, num_classes]). The embedding is the representation the privileged
IMU alignment acts on. The same class serves WiFi (C=270) and IMU (C=30).
"""
from __future__ import annotations

import torch
from torch import nn


class DeepConvLSTM(nn.Module):
    def __init__(self, in_channels, num_classes, filters=64, kernel_size=5,
                 hidden=128, lstm_layers=2, dropout=0.5):
        super().__init__()
        self.config = dict(in_channels=in_channels, num_classes=num_classes,
                           filters=filters, kernel_size=kernel_size, hidden=hidden,
                           lstm_layers=lstm_layers, dropout=dropout)
        layers, c = [], 1
        for _ in range(4):
            layers += [nn.Conv2d(c, filters, (kernel_size, 1)), nn.ReLU(inplace=True)]
            c = filters
        self.convs = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(filters * in_channels, hidden, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2).unsqueeze(1)             # [B, 1, T, C]
        x = self.convs(x)                              # [B, F, T', C]
        b, f, t, c = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, f * c)  # [B, T', F*C]
        out, _ = self.lstm(self.dropout(x))
        emb = out[:, -1]                               # last time step
        return emb, self.classifier(self.dropout(emb))
