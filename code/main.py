import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score, flat_precision_score, flat_recall_score


def read_conll_file(file_path):
    sentences = []
    labels = []

    current_sentence = []
    current_labels = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line == "":
                if len(current_sentence) > 0:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
                continue

            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) >= 5:
                word = parts[1]
                chunk_outer = parts[2]

                current_sentence.append(word)
                current_labels.append(chunk_outer)

    if len(current_sentence) > 0:
        sentences.append(current_sentence)
        labels.append(current_labels)

    return sentences, labels


def word_to_features(sentence, i):
    word = sentence[i]

    features = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "suffix3": word[-3:],
        "suffix2": word[-2:],
        "prefix2": word[:2],
        "prefix3": word[:3],
    }

    if i > 0:
        prev_word = sentence[i - 1]
        features.update({
            "prev_word.lower": prev_word.lower(),
            "prev_word.istitle": prev_word.istitle(),
            "prev_word.isupper": prev_word.isupper(),
        })
    else:
        features["BOS"] = True

    if i < len(sentence) - 1:
        next_word = sentence[i + 1]
        features.update({
            "next_word.lower": next_word.lower(),
            "next_word.istitle": next_word.istitle(),
            "next_word.isupper": next_word.isupper(),
        })
    else:
        features["EOS"] = True

    return features


def sentence_to_features(sentence):
    return [word_to_features(sentence, i) for i in range(len(sentence))]


def flatten(list_of_lists):
    result = []

    for item in list_of_lists:
        for value in item:
            result.append(value)

    return result


def save_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def calculate_dataset_statistics(sentences, labels, output_path):
    label_counts = {}

    for sentence_labels in labels:
        for label in sentence_labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

    total_tokens = sum(label_counts.values())

    stats = {
        "sentence_count": len(sentences),
        "token_count": total_tokens,
        "label_distribution": {}
    }

    for label, count in label_counts.items():
        stats["label_distribution"][label] = {
            "count": count,
            "ratio": count / total_tokens
        }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=4)

    return stats


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset", "data.conll")
    results_dir = os.path.join(base_dir, "results")

    os.makedirs(results_dir, exist_ok=True)

    sentences, labels = read_conll_file(dataset_path)

    print("Sentence count:", len(sentences))

    X = [sentence_to_features(sentence) for sentence in sentences]
    y = labels

    dataset_stats_path = os.path.join(results_dir, "dataset_statistics.json")
    calculate_dataset_statistics(sentences, labels, dataset_stats_path)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    all_true = []
    all_pred = []

    fold_results = []

    for fold_no, (train_index, test_index) in enumerate(kfold.split(X), start=1):
        X_train = [X[i] for i in train_index]
        X_test = [X[i] for i in test_index]

        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index]

        model = CRF(
            algorithm="lbfgs",
            max_iterations=100,
            all_possible_transitions=True
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        true_flat = flatten(y_test)
        pred_flat = flatten(y_pred)

        accuracy = accuracy_score(true_flat, pred_flat)
        precision = flat_precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = flat_recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = flat_f1_score(y_test, y_pred, average="weighted", zero_division=0)

        fold_result = {
            "fold": fold_no,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        fold_results.append(fold_result)

        all_true.extend(true_flat)
        all_pred.extend(pred_flat)

        print("Fold", fold_no)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("----------------------")

    avg_accuracy = np.mean([item["accuracy"] for item in fold_results])
    avg_precision = np.mean([item["precision"] for item in fold_results])
    avg_recall = np.mean([item["recall"] for item in fold_results])
    avg_f1 = np.mean([item["f1_score"] for item in fold_results])

    final_metrics = {
        "fold_results": fold_results,
        "average_accuracy": avg_accuracy,
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1_score": avg_f1
    }

    metrics_path = os.path.join(results_dir, "metrics.json")

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(final_metrics, file, ensure_ascii=False, indent=4)

    unique_labels = sorted(list(set(all_true)))

    report_path = os.path.join(results_dir, "classification_report.txt")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(classification_report(all_true, all_pred, labels=unique_labels, zero_division=0))

    confusion_matrix_path = os.path.join(results_dir, "confusion_matrix.png")
    save_confusion_matrix(all_true, all_pred, unique_labels, confusion_matrix_path)

    print("Average Accuracy:", avg_accuracy)
    print("Average Precision:", avg_precision)
    print("Average Recall:", avg_recall)
    print("Average F1-score:", avg_f1)

    print("Results saved to results folder.")


if __name__ == "__main__":
    main()