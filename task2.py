import os
import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import time


class CBIRSystemColorHistogram:
    def __init__(self, dataset_path, num_pins):
        self.image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        self.num_pins = num_pins
        self.features = [self.extract_features(cv2.imread(img_path)) for img_path in self.image_paths]
        self.labels = [int(os.path.splitext(os.path.basename(img_path))[0]) for img_path in self.image_paths]

    def extract_features(self, image):
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to reduce memory requirements
        resized_image = cv2.resize(image_rgb, (25, 25))

        # Calculate color histogram
        hist = self.calculate_color_histogram(resized_image)

        return hist

    def calculate_color_histogram(self, image):
        # Convert the image to float32
        image_float32 = image.astype(np.float32)

        # Calculate the histogram
        hist = cv2.calcHist([image_float32], [0, 1, 2], None,
                            [self.num_pins // 2, self.num_pins // 2, self.num_pins // 2],
                            [0, 256, 0, 256, 0, 256])

        # Normalize the histogram
        hist /= hist.sum()

        return hist.flatten()

    def compute_distance(self, query_features):
        distances = [cv2.compareHist(query_features, feat, cv2.HISTCMP_INTERSECT) for feat in self.features]
        return distances

    def rank_results(self, distances, query_index):
        # Exclude the query image from the ranked results
        sorted_indices = np.argsort(distances)[::-1]  # Reverse order to get descending similarity
        ranked_results = [idx for idx in sorted_indices if idx != query_index]
        return ranked_results

    def display_results(self, ranked_results, query_index):
        plt.figure(figsize=(15, 8))

        # Plot Query Image
        query_image = cv2.imread(self.image_paths[query_index])
        query_name = os.path.splitext(os.path.basename(self.image_paths[query_index]))[0]
        plt.subplot(3, 1, 1)
        plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
        plt.title(f"{query_name}")
        plt.axis('off')

        # Plot Retrieval Results - Second Row
        for i, idx in enumerate(ranked_results[:5]):
            img = cv2.imread(self.image_paths[idx])
            retrieved_name = os.path.splitext(os.path.basename(self.image_paths[idx]))[0]
            plt.subplot(3, 5, i + 6)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"{retrieved_name}")
            plt.axis('off')

        # Plot Retrieval Results - Third Row
        for i, idx in enumerate(ranked_results[5:10]):
            img = cv2.imread(self.image_paths[idx])
            retrieved_name = os.path.splitext(os.path.basename(self.image_paths[idx]))[0]
            plt.subplot(3, 5, i + 11)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"{retrieved_name}")
            plt.axis('off')

        plt.show()


def calculate_metrics(labels, ranked_results, true_category):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 100

    for i, idx in enumerate(ranked_results[:20]):
        retrieved_category = labels[idx]

        # Determine the category intervals
        true_category_start = true_category // 100 * 100
        true_category_end = true_category_start + 99

        retrieved_category_start = retrieved_category // 100 * 100
        retrieved_category_end = retrieved_category_start + 99

        # Check if the retrieved image is in the same category
        if true_category_start <= retrieved_category <= true_category_end:
            true_positives += 1
        else:
            false_positives += 1

    for i, idx in enumerate(ranked_results[20:]):
        retrieved_category = labels[idx]

        # Determine the category intervals
        true_category_start = true_category // 100 * 100
        true_category_end = true_category_start + 99

        retrieved_category_start = retrieved_category // 100 * 100
        retrieved_category_end = retrieved_category_start + 99

        # Check if the retrieved image is from another category
        if retrieved_category_start > true_category_end or retrieved_category_end < true_category_start:
            true_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # return precision, recall, f1, true_positives, false_positives, false_negatives
    return precision, recall, f1


def run_experiment_color_histogram(dataset_path, num_pins):
    cbir_system = CBIRSystemColorHistogram(dataset_path, num_pins)
    num_queries = 10
    # tp, fp, fn = 0, 0, 0

    all_labels, all_scores = [], []

    precision_avg, recall_avg, f1_avg, time_avg = 0, 0, 0, 0

    for query_index in range(num_queries):
        query_image_index = np.random.randint(len(cbir_system.image_paths))
        query_image = cv2.imread(cbir_system.image_paths[query_image_index])
        query_features = cbir_system.extract_features(query_image)

        print(f"Query Image: {cbir_system.image_paths[query_image_index]}")

        start_time = time.time()

        distances = cbir_system.compute_distance(query_features)
        ranked_results = cbir_system.rank_results(distances, query_image_index)

        true_category = cbir_system.labels[query_image_index]
        labels = cbir_system.labels

        # precision, recall, f1, tp_query, fp_query, fn_query = calculate_metrics(labels, ranked_results, true_category)

        precision, recall, f1 = calculate_metrics(labels, ranked_results, true_category)

        # tp += tp_query
        # fp += fp_query
        # fn += fn_query

        print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

        precision_avg += precision
        recall_avg += recall
        f1_avg += f1
        time_avg += (time.time() - start_time)

        # query_labels = [1 if i == query_index else 0 for i in range(len(cbir_system.image_paths))]
        query_labels = [1 if (i // 100) == (query_index % num_queries)
                        else 0 for i in range(len(cbir_system.image_paths))]
        all_labels.extend(query_labels)
        all_scores.extend([-dist for dist in distances])

    precision_avg /= num_queries
    recall_avg /= num_queries
    f1_avg /= num_queries
    time_avg /= num_queries

    print("\nPin:", num_pins)
    print("Average Metrics:")
    print(f"Average Precision: {precision_avg:.4f}")
    print(f"Average Recall: {recall_avg:.4f}")
    print(f"Average F1 Score: {f1_avg:.4f}")
    print(f"Average Time: {time_avg:.4f} seconds")

    # Construct ROC curve for each threshold
    plt.figure(figsize=(8, 8))

    fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')
    plt.show()

    # plt.plot(fpr, tpr, lw=2, label=f'Threshold = {threshold} (AUC = {roc_auc:.2f})')
    #
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic Curve')
    # plt.legend(loc='lower right')
    # plt.show()


dataset_path = ".\\dataset"
num_pins = [120]
for num_pins in num_pins:
    print(f"\nExperimenting with {num_pins} pins:")
    run_experiment_color_histogram(dataset_path, num_pins)
