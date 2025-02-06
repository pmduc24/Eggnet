import numpy as np
from sklearn.metrics import precision_score, recall_score, jaccard_score
import torch

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
