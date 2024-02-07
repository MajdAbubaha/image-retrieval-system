import os
import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import time
from skimage import feature


class CBIRSystemBoVW_LBP:
    def __init__(self, dataset_path, cluster_size=10):
        self.image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        self.cluster_size = cluster_size
        self.features, self.labels = self.extract_bovw_lbp_features()

    def extract_lbp_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10 + 1), range=(0, 10))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist

    def extract_bovw_lbp_features(self):
        features = []
        labels = []

        for i, img_path in enumerate(self.image_paths):
            img = cv2.imread(img_path)
            lbp_hist = self.extract_lbp_features(img)
            features.append(lbp_hist)
            labels.append(i // self.cluster_size)  # Assign label based on the category (0-9)

        return np.array(features), np.array(labels)

    def compute_distance(self, query_hist):
        distances = [np.linalg.norm(query_hist - hist) for hist in self.features]
        return distances

    def rank_results(self, distances, query_index):
        # Exclude the query image from the ranked results
        sorted_indices = np.argsort(distances)
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

    def evaluate_performance(self, query_index):
        # Use a specific query image
        query_features = self.features[query_index]

        # Evaluate performance metrics
        true_category = int(os.path.splitext(os.path.basename(self.image_paths[query_index]))[0])
        labels = [int(os.path.splitext(os.path.basename(img_path))[0]) for img_path in self.image_paths]
        distances = self.compute_distance(query_features)

        # Precision, Recall, F1 Score
        ranked_results = self.rank_results(distances, query_index)
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 100

        for i, idx in enumerate(ranked_results[:30]):
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

        for i, idx in enumerate(ranked_results[30:]):
            retrieved_category = labels[idx]

            # Determine the category intervals
            true_category_start = true_category // 100 * 100
            true_category_end = true_category_start + 99

            retrieved_category_start = retrieved_category // 100 * 100
            retrieved_category_end = retrieved_category_start + 99

            # Check if the retrieved image is from another category
            if retrieved_category_start > true_category_end or retrieved_category_end < true_category_start:
                true_negatives += 1


        precision = true_positives / (true_positives + false_positives) if (
                                                                                   true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        return precision, recall, f1, distances


def run_experiment_bovw_lbp(dataset_path, cluster_size):
    # Create BoVW System with LBP
    cbir_system_bovw_lbp = CBIRSystemBoVW_LBP(dataset_path, cluster_size)

    # Perform experiments
    num_queries = len(cbir_system_bovw_lbp.image_paths) // cluster_size
    precision_avg, recall_avg, f1_avg, time_avg = 0, 0, 0, 0
    all_labels, all_scores = [], []

    for query_index in range(num_queries):
        # Use a random query image within the category
        category_start_index = query_index * cluster_size
        query_image_index = np.random.choice(range(category_start_index, category_start_index + cluster_size))
        query_hist = cbir_system_bovw_lbp.features[query_image_index]

        print(f"Query Path: {cbir_system_bovw_lbp.image_paths[query_image_index]}")

        # Measure time taken for retrieval
        start_time = time.time()

        # Perform retrieval and get ranked results
        distances = cbir_system_bovw_lbp.compute_distance(query_hist)
        ranked_results = cbir_system_bovw_lbp.rank_results(distances, query_image_index)

        # Evaluate performance metrics
        precision, recall, f1, distances = cbir_system_bovw_lbp.evaluate_performance(query_image_index)

        print(f"precision: {precision}, recall: {recall}, f1 score: {f1}")

        # Update average metrics
        precision_avg += precision
        recall_avg += recall
        f1_avg += f1
        time_avg += (time.time() - start_time)

        # Store labels and scores for ROC curve
        labels = [1 if (i // cluster_size) == (query_index % num_queries) else 0 for i in range(len(cbir_system_bovw_lbp.image_paths))]
        all_labels.extend(labels)
        all_scores.extend([-dist for dist in distances])  # Negative distances for ascending order

        # Display results for each query
        # cbir_system_bovw_lbp.display_results(ranked_results, query_image_index)

    # Calculate average metrics
    precision_avg /= num_queries
    recall_avg /= num_queries
    f1_avg /= num_queries
    time_avg /= num_queries

    print("Average Metrics:")
    print(f"Average Precision: {precision_avg:.2f}")
    print(f"Average Recall: {recall_avg:.2f}")
    print(f"Average F1 Score: {f1_avg:.2f}")
    print(f"Average Time: {time_avg:.2f} seconds")

    print(f"Labels: {all_labels}")
    print(f"Scores: {all_scores}")

    # Construct ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
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


dataset_path = ".\\dataset"
cluster_size = 100  # Set the number of clusters equal to the total number of categories
run_experiment_bovw_lbp(dataset_path, cluster_size)