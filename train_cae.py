import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np

# 1. 아키텍처 정의 (논문 3.6절 기반)
class BasicCAE(nn.Module):
    def __init__(self):
        super(BasicCAE, self).__init__()
        # Encoder: 5개의 Conv 블록 (Downsampling)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(), # 112x112
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), # 56x56
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), # 28x28
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), # 14x14
            nn.Conv2d(128, 2, 3, stride=2, padding=1), nn.ReLU()   # 7x7 (잠재 텐서 2x7x7)
        )
        # Decoder: 5개의 Deconv 블록 (Upsampling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 128, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid() # 224x224, BCE를 위해 Sigmoid 사용
        )

    def forward(self, x):
        latent = self.encoder(x) # 2x7x7 텐서 생성
        flat_latent = torch.flatten(latent, start_dim=1) # 98차원 벡터
        reconstructed = self.decoder(latent)
        return reconstructed, flat_latent

# 2. 커스텀 데이터셋 (CSV 또는 이미지 로드용)
class BandStructureDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        # 실제 환경에서는 CSV 내 경로를 통해 이미지를 읽거나, 수치를 이미지화해야 함
        # 여기서는 예시로 랜덤 이미지를 생성하는 로직으로 구성함
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 예시: 224x224 단일 채널(흑백) 이미지 생성
        image = torch.randn(1, 224, 224) 
        if self.transform:
            image = self.transform(image)
        return image

# 3. 학습 함수
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicCAE().to(device)
    
    # 논문 설정: BCE Reconstruction Loss
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = BandStructureDataset(csv_file='test.csv')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    epochs = 50
    for epoch in range(epochs):
        train_loss = 0
        for data in loader:
            img = data.to(device)
            
            # Forward
            output, _ = model(img)
            loss = criterion(output, img) # 입력과 출력을 비교 (BCE Loss)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(loader):.4f}")

    # 모델 저장
    torch.save(model.state_dict(), "cae_model.pth")
    print("Model saved as cae_model.pth")

if __name__ == "__main__":
    train_model()
