import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from loader import ADE20KDataset
from loader import transform

train_img = 'ade/images/training'
train_msk = 'ade/annotations/training'

val_img = 'ade/images/validation'
val_msk = 'ade/annotations/validation'

train_dataset  = ADE20KDataset(train_img, train_msk, transform=transform)
validation_dataset = ADE20KDataset(val_img, val_msk, transform=transform) 

train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
valid_loader = DataLoader(validation_dataset, batch_size=6, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet50",         # Выбор энкодера
    encoder_weights="imagenet",        # Предобученные веса энкодера
    in_channels=3,                     # Число входных каналов (3 для RGB)
    classes=3,                         # Количество классов сегментации (1 для бинарной, больше — для мультиклассов)
).to(device)

criterion = nn.CrossEntropyLoss()  # Binary Cross Entropy для бинарной сегментации
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Пример обучающего цикла (упрощённый)
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"Эпоха {epoch+1}/{num_epochs}"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Эпоха [{epoch+1}/{num_epochs}], Потери: {avg_loss:.4f}")

print("Обучение завершено!")

torch.save(model.state_dict(), f"unet_epoch_{epoch+1}.pth")
print(f"Модель сохранена: unet_epoch_{epoch+1}.pth")
