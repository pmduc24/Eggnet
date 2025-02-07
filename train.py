import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from segnet import *
from loss import DiceCrossEntropyLoss
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from dataset import CustomDataset  
from segnet import Segnet
import os
import json
from val import validate_model, print_validation_results

def train(model, train_loader, valid_loader, criterion, optimizer, device, class_names, num_epochs, save_path):
    os.makedirs(save_path, exist_ok=True) 
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        
        for sample in train_loader:
            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)
            masks = masks.squeeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        
        metrics = validate_model(model, valid_loader, criterion, device, class_names)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {metrics['loss']:.4f}")
        print_validation_results(metrics, class_names)

        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            print(f"Best model saved at {os.path.join(save_path, 'best_model.pth')}")

    torch.save(model.state_dict(), os.path.join(save_path, "last_model.pth"))
    print(f"Last model saved at {os.path.join(save_path, 'last_model.pth')}")
    print("Training completed!")

def main():

    parser = argparse.ArgumentParser(description='Train a segmentation model')
    parser.add_argument('--version', type=str, default='n',choices=['n', 's', 'm', 'l', 'x'], help='Version of model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--num_classes', type=int, default=80, help='Number of output classes')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--root_dir', type=str, required=True, help='Path to training data')
    parser.add_argument('--save_path', type=str, default=".", help='Saving path')

    args = parser.parse_args()

    with open(os.path.join(args.root_dir, "train", "_annotations.coco.json"), "r") as f:
        coco_data = json.load(f)

    class_names = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    device = torch.device(args.device)
    
    model = Segnet(num_classes=args.num_classes, class_names=class_names, version=args.version)
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
            # transforms.ToPILImage(),                   
            transforms.Resize((args.imgsz, args.imgsz)),          
            # transforms.ToTensor()
        ])
    }

    train_dataset = CustomDataset(args.root_dir, mode = "train", transform=data_transforms)
    val_dataset = CustomDataset(args.root_dir, mode = "valid", transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = DiceCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model, train_loader, val_loader, criterion, optimizer, device, class_names, args.num_epochs, args.save_path)

if __name__ == '__main__':
    main()
