import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from loader import ADE20KDataset
from loader import transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_mask(image):
    plt.figure(figsize=(12, 6))
    plt.imshow(image, cmap="gray")
    plt.title("Предсказанная маска")
    plt.show()

def extract_masked_region(image_path, mask):
    """ Вырезает область изображения, соответствующую предсказанной маске """
    img = Image.open(image_path).convert("RGB")
    imgnp = np.array(img)
    
    # Применяем маску: обнуляем фон, оставляя только объект
    masked_img = np.zeros_like(imgnp)
    masked_img[mask > 0] = imgnp[mask > 0]
    
    return Image.fromarray(masked_img)

def get_predicted(image_path):
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=3)
    model.load_state_dict(torch.load("unet_epoch_100.pth", map_location=device))
    model.to(device)
    model.eval()
    
    img = Image.open(image_path).convert("RGB")
    imgnp = np.array(img)
    input_image = transform(image=imgnp)['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_image)
    pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    pred_mask_resized = Image.fromarray(pred_mask.astype(np.uint8))
    pred_mask_resized = pred_mask_resized.resize((imgnp.shape[1], imgnp.shape[0]), Image.NEAREST)
    pred_mask_resized = np.array(pred_mask_resized)
    
    return pred_mask_resized

def main():
    image_path = 'test_img/img_2.jpg'
    predicted_mask = get_predicted(image_path)
    show_mask(predicted_mask)
    
    # Вырезаем объект по маске
    masked_region = extract_masked_region(image_path, predicted_mask)
    masked_region.show()  # Показываем результат
    masked_region.save("masked_image.png")  # Сохраняем вырезанный объект

if __name__ == '__main__':
    main()
