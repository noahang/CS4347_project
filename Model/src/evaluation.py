import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing modules
from Model.src.train import Classifier
from Model.src.dataset import MyDataset, move_data_to_device, collate_fn
from Model.src.hparams import Hparams


def evaluate_test_set(model_path, args, test_data_dir=Hparams.args_2s['data_dir'], device='cuda'):
    """
    Evaluate the trained model on the test set

    Args:
        model_path: Path to the saved model weights
        test_data_dir: Directory containing test data
        args: Hyperparameters dictionary
        device: Device to run evaluation on
    """

    # Initialize model
    classifier = Classifier(device=device, model_path=model_path)
    model = classifier.model
    model.eval()

    # Create test dataset and dataloader
    test_dataset = MyDataset(ds_root=test_data_dir, split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args['num_workers'],
        collate_fn=collate_fn
    )

    print(f"Test set size: {len(test_dataset)}")
    print("Starting evaluation...")

    # Storage for predictions and targets
    all_predictions = []
    all_targets = []
    all_probabilities = []

    # Evaluation loop
    with torch.no_grad():
        for batch in test_loader:
            x, tgt = move_data_to_device(batch, device)

            # Forward pass
            out = model(x)
            print(out)
            print(out.shape)
            # out = torch.mean(out, dim=-1)
            # print(out.shape)
            probs = torch.softmax(out, dim=1)
            print(probs)
            print(probs.shape)
            preds = torch.argmax(probs, dim=1)
            print(preds)
            print(preds.shape)

            # Store results
            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(tgt.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)

    # Calculate metrics
    print(all_targets)
    print(all_predictions)
    accuracy = accuracy_score(all_targets, all_predictions)

    print("\n" + "=" * 50)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Test set size: {len(all_targets)}")

    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_predictions))

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
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
    unique_classes = np.unique(all_targets)
    for class_id in unique_classes:
        class_mask = all_targets == class_id
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(all_predictions[class_mask] == class_id)
            class_accuracies[class_id] = class_accuracy
            print(f"Class {class_id}: {class_accuracy:.4f} ({np.sum(class_mask)} samples)")

    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'class_accuracies': class_accuracies,
        'confusion_matrix': cm
    }


def main():
    # parser = argparse.ArgumentParser(description='Evaluate model on test set')
    # parser.add_argument('--model_path', type=str, required=True,
    #                     help='Path to the trained model weights')
    # parser.add_argument('--test_data_dir', type=str,
    #                     default='./data/2s-0.5s/splits/test',
    #                     help='Directory containing test data')
    # parser.add_argument('--device', type=str, default='cuda',
    #                     choices=['cuda', 'cpu'],
    #                     help='Device to run evaluation on')
    #
    # args_cmd = parser.parse_args()
    #
    # # Set device
    # device = args_cmd.device
    # if device == 'cuda' and not torch.cuda.is_available():
    #     print("CUDA not available, using CPU")
    #     device = 'cpu'
    #
    # # Update data directory in hyperparameters
    # eval_args = Hparams.args_2s.copy()
    # eval_args['data_dir'] = os.path.dirname(args_cmd.test_data_dir)  # Parent directory of test
    #
    # print(f"Using device: {device}")
    # print(f"Model path: {args_cmd.model_path}")
    # print(f"Test data directory: {args_cmd.test_data_dir}")

    os.chdir("../..")

    # Run evaluation
    results = evaluate_test_set(
        model_path='./src/results/best_model.pth',
        args=Hparams.args_2s,
        device='cpu'
    )

    # Save results
    output_file = 'test_evaluation_results.npy'
    np.save(output_file, results, allow_pickle=True)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()