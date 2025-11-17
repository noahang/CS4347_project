import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from Model2.src.train import Classifier, Metrics
from Model2.src.dataset import move_data_to_device, get_data_loader
from Model2.src.hparams import Hparams


def evaluate_test_set(model_path, args):
    device = args['device']
    classifier = Classifier(device=device, model_path=model_path)
    model = classifier.model
    model.eval()

    test_loader = get_data_loader(split='test', args=args)

    print(f"Test Dataset size: {len(test_loader)}")
    print("Starting evaluation...")

    outs = []
    tgts = []
    metric = Metrics(nn.CrossEntropyLoss)

    with torch.no_grad():
        for batch in test_loader:
            x, tgt_center, tgt_mode = move_data_to_device(batch, args['device'])
            out = model(x)
            metric.update(out, (tgt_center, tgt_mode))

            outs.extend(out.cpu().numpy().tolist())
            tgts.extend((tgt_center.cpu().numpy()))

    outs = np.array(outs)
    tgts = np.array(tgts)

    accuracy = accuracy_score(tgts, outs)

    print("\n" + "=" * 50)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Test set size: {len(tgts)}")

    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(tgts, outs))

    # Confusion matrix
    cm = confusion_matrix(tgts, outs)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix if matplotlib is available
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('test_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'test_confusion_matrix.png'")
        plt.show()
    except Exception as e:
        print(f"Could not plot confusion matrix: {e}")

    # Calculate per-class accuracy
    print("\nPer-class Accuracy:")
    class_accuracies = {}
    unique_classes = np.unique(tgts)
    for class_id in unique_classes:
        class_mask = tgts == class_id
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(outs[class_mask] == class_id)
            class_accuracies[class_id] = class_accuracy
            print(f"Class {class_id}: {class_accuracy:.4f} ({np.sum(class_mask)} samples)")

    return {
        'accuracy': accuracy,
        'predictions': outs,
        'targets': tgts,
        'class_accuracies': class_accuracies,
        'confusion_matrix': cm
    }


def main():
    results = evaluate_test_set(
        model_path='.Model2/src/results/6s/best_model3.pth',
        args=Hparams.args_6s
    )

    output_file = 'test_evaluation_results.npy'
    np.save(output_file, results, allow_pickle=True)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()