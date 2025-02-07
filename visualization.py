import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import os
from torchvision import transforms
from dataset import CustomDataset
from segnet import Segnet
import json

def visualize_predictions(model, dataloader, device, output_dir, num_samples=5):

    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    samples = random.sample(list(dataloader), num_samples)

    mean = np.array([0.485, 0.456, 0.406])  
    std = np.array([0.229, 0.224, 0.225])  

    for i, sample in enumerate(samples):
        inputs = sample['image'].to(device)
        masks = sample['mask'].to(device)
        masks = masks.squeeze(1)

        outputs = model(inputs)
        y_pred = torch.argmax(outputs, dim=1).data.cpu().numpy()
        y_true = masks.data.cpu().numpy()

        img = inputs.squeeze(0).permute(1, 2, 0).cpu().numpy()  
        img = (img * std) + mean  
        img = np.clip(img, 0, 1)  


        y_pred_colored = colorize_mask(y_pred[0])
        y_true_colored = colorize_mask(y_true[0])

        overlay_pred = (img * 0.5 + y_pred_colored * 0.5)  
        overlay_true = (img * 0.5 + y_true_colored * 0.5)  

        fig, axes = plt.subplots(1, 5, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(y_true[0])
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(y_pred[0])
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

        axes[3].imshow(overlay_true)
        axes[3].set_title("Overlay Ground Truth")
        axes[3].axis("off")

        axes[4].imshow(overlay_pred)
        axes[4].set_title("Overlay Prediction")
        axes[4].axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"sample_{i}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

def colorize_mask(mask):

    num_classes = np.max(mask) + 1
    colors = plt.cm.get_cmap("jet", num_classes)
    mask_colored = colors(mask / num_classes)[:, :, :3]  
    return mask_colored

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model predictions and save images.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model.")
    parser.add_argument('--version', type=str, default='n',choices=['n', 's', 'm', 'l', 'x'], help='Version of model')
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save images.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize.")
    parser.add_argument("--device", type=str, default="cuda", choices=['cpu', 'cuda'], help="Device to use")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument('--num_classes', type=int, default=80, help='Number of output classes')
    args = parser.parse_args()
    
    with open(os.path.join(args.root_dir, "test", "_annotations.coco.json"), "r") as f:
        coco_data = json.load(f)

    class_names = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    device = torch.device(args.device)
    model = Segnet(num_classes=args.num_classes, class_names=class_names, version=args.version)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    data_transforms = {
        'Image': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.imgsz, args.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'Mask': transforms.Compose([
            transforms.Resize((args.imgsz, args.imgsz)),
        ])
    }
    
    test_dataset = CustomDataset(args.root_dir, mode='test', transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    visualize_predictions(model, test_loader, device, args.output_dir, args.num_samples)
