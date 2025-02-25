import torch
from torch import nn
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import norm, sem
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import seaborn as sns

def compute_auc_ci(labels, probabilities, confidence=0.95):
    auc_score = roc_auc_score(labels, probabilities)
    n1 = sum(labels)
    n2 = len(labels) - n1

    q1 = auc_score / (2 - auc_score)
    q2 = 2 * auc_score**2 / (1 + auc_score)

    se_auc = np.sqrt((auc_score * (1 - auc_score) + (n1 - 1) * (q1 - auc_score**2) + (n2 - 1) * (q2 - auc_score**2)) / (n1 * n2))
    lower = auc_score - norm.ppf(confidence + (1 - confidence) / 2) * se_auc
    upper = auc_score + norm.ppf(confidence + (1 - confidence) / 2) * se_auc

    return lower, upper

def compute_p_value(labels, probabilities):
    auc_score = roc_auc_score(labels, probabilities)
    std_error = sem(probabilities)
    z_score = (auc_score - 0.5) / std_error
    p_value = norm.sf(abs(z_score)) * 2 
    return p_value

def save_results(results, save_dir, pr_curve_fig, roc_curve_figs, cm_fig):
    os.makedirs(save_dir, exist_ok=True)

    # Save results as JSON
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # Save Precision-Recall Curve
    pr_curve_path = os.path.join(save_dir, "pr_curve.png")
    pr_curve_fig.savefig(pr_curve_path)

    # Save ROC Curves
    for i, roc_fig in enumerate(roc_curve_figs):
        roc_curve_path = os.path.join(save_dir, f"roc_curve_class_{i}.png")
        roc_fig.savefig(roc_curve_path)

    # Save Confusion Matrix
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    cm_fig.savefig(cm_path)

    print(f"Results, PR Curve, ROC Curves, and Confusion Matrix saved to {save_dir}")

def test(model, test_loader, device, save_dir="test_results"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images, labels=labels)
            loss = outputs[0]
            logits = outputs[1]

            running_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(torch.softmax(logits, dim=1).cpu().numpy())

    # del images, labels, outputs, logits, loss
    torch.cuda.empty_cache()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    all_labels_one_hot = np.eye(len(set(all_labels)))[all_labels]
    auc_scores = []
    ci_bounds = []
    p_values = []
    roc_curve_figs = []

    for i in range(all_labels_one_hot.shape[1]):
        auc_score = roc_auc_score(all_labels_one_hot[:, i], np.array(all_logits)[:, i])
        auc_scores.append(auc_score)
        ci_low, ci_high = compute_auc_ci(all_labels_one_hot[:, i], np.array(all_logits)[:, i])
        ci_bounds.append((ci_low, ci_high))
        p_value = compute_p_value(all_labels_one_hot[:, i], np.array(all_logits)[:, i])
        p_values.append(p_value)

        # ROC Curve for each class
        fpr, tpr, _ = roc_curve(all_labels_one_hot[:, i], np.array(all_logits)[:, i])
        plt.figure()
        plt.plot(fpr, tpr, label=f'Class {i} ROC Curve (AUC = {auc_score:.4f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Class {i}')
        plt.legend(loc='lower right')
        plt.grid(True)
        roc_curve_figs.append(plt.gcf())
        plt.close()

    mean_auc = np.mean(auc_scores)

    # Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels_one_hot.ravel(), np.array(all_logits).ravel())
    average_precision = average_precision_score(all_labels_one_hot, np.array(all_logits), average="weighted")

    plt.figure()
    plt.plot(recall_vals, precision_vals, label=f'PR Curve (AP = {average_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    pr_curve_fig = plt.gcf()
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(all_labels), yticklabels=set(all_labels))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    cm_fig = plt.gcf()
    plt.close()

    # Print results
    print(f'Test Loss: {avg_loss:.4f}')
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Mean AUC: {mean_auc:.4f}')
    for i, (ci, p) in enumerate(zip(ci_bounds, p_values)):
        print(f'Class {i} AUC: {auc_scores[i]:.4f} (95% CI: {ci[0]:.4f} - {ci[1]:.4f}), p-value: {p:.4e}')
    print(f'Average Precision (AP): {average_precision:.4f}')

    # Save results
    results = {
        "Test Loss": avg_loss,
        "Test Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Mean AUC": mean_auc,
        "Average Precision": average_precision,
        "Class AUCs": [
            {"Class": i, "AUC": auc_scores[i], "CI": ci_bounds[i], "p-value": p_values[i]}
            for i in range(len(auc_scores))
        ]
    }

    save_results(results, save_dir, pr_curve_fig, roc_curve_figs, cm_fig)
