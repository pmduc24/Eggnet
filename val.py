import numpy as np
from sklearn.metrics import precision_score, recall_score, jaccard_score
import torch
from segnet import Segnet
import argparse
from torchvision import transforms
import os
import json
from dataset import CustomDataset  
from loss import DiceCrossEntropyLoss
from torch.utils.data import DataLoader

def acc_per_class(y_true, y_pred):
    """
    Tính độ chính xác cho từng lớp.
    
    Parameters:
        y_true (np.array): Nhãn thực tế.
        y_pred (np.array): Nhãn dự đoán.
    
    Returns:
        dict: Độ chính xác cho từng lớp.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred) 
    
    num_classes = len(np.unique(y_true))
    
    accuracies = {}
    for class_id in range(1, num_classes):

        class_mask = (y_true == class_id).astype(int)  
        
        correct = (y_true[class_mask == 1] == y_pred[class_mask == 1]).sum()
        total = class_mask.sum()
        
        accuracy = correct / total if total > 0 else 0
        accuracies[class_id] = accuracy
    
    return accuracies

def validate_model(model, dataloader, criterion, device, class_names):
    
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for sample in iter(dataloader):
            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)
            masks = masks.squeeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            y_pred = torch.argmax(outputs, dim=1).data.cpu().numpy().ravel()
            y_true = masks.data.cpu().numpy().ravel()

            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

    # Calculate metrics
    precision = precision_score(y_true_all, y_pred_all, average=None, labels=range(1, len(class_names)), zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average=None, labels=range(1, len(class_names)), zero_division=0)
    accuracy = acc_per_class(y_true_all, y_pred_all)
    iou = jaccard_score(y_true_all, y_pred_all, average=None, labels=range(1, len(class_names)), zero_division=0)


    # Calculate mean metrics for 'all'
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_iou = np.mean(iou)
    OA = np.mean(list(accuracy.values()))

    return {
        'loss': total_loss / len(dataloader),
        'precision': precision,
        'recall': recall,
        'accuracy': OA,
        'iou': iou,
        'class_accuracy': accuracy,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_iou': mean_iou
    }

def print_validation_results(metrics, class_names):
    print("Class\t\tP\t\tR\t\tAcc\t\tmIOU")
    print("--------------------------------------------------------------------------")
    
    # Print results for 'all'
    print(f"all\t\t{metrics['mean_precision']:.4f}\t\t{metrics['mean_recall']:.4f}\t\t{metrics['accuracy']:.4f}\t\t{metrics['mean_iou']:.4f}")
    
    # Print results for each class
    for i in range(1, len(class_names)):
        print(f"{class_names[i]}\t\t{metrics['precision'][i-1]:.4f}\t\t{metrics['recall'][i-1]:.4f}\t\t{metrics['class_accuracy'][i]:.4f}\t\t{metrics['iou'][i-1]:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Validate a segmentation model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--version', type=str, required=True, choices=['n', 's', 'm', 'l', 'x'], help='Version of model')
    parser.add_argument('--root_dir', type=str, required=True, help='Path to validation data')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--num_classes', type=int, default=80, help='Number of output classes')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    
    args = parser.parse_args()
    
    with open(os.path.join(args.root_dir, "valid", "_annotations.coco.json"), "r") as f:
        coco_data = json.load(f)
    
    class_names = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    
    device = torch.device(args.device)
    model = Segnet(num_classes=args.num_classes, class_names=class_names)
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
    
    val_dataset = CustomDataset(args.root_dir, mode='valid', transform=data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    criterion = DiceCrossEntropyLoss()
    metrics = validate_model(model, val_loader, criterion, device, class_names)
    
    print_validation_results(metrics, class_names)

if __name__ == '__main__':
    main()
