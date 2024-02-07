############################################## Task 3.1 ###########################################################
'''import os
import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import time
from scipy.stats import skew

class CBIRSystemColorMoments:
    def __init__(self, dataset_path):
        self.image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        self.features, self.labels = self.extract_color_moments()

    def extract_color_moments(self):
        features = []
        labels = []

        for i, img_path in enumerate(self.image_paths):
            img = cv2.imread(img_path)
            color_moments = self.extract_features(img)
            features.append(color_moments)
            labels.append(i)  # Assign label based on the index

        return np.array(features), np.array(labels)

    def extract_features(self, image):
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calculate color moments: mean, standard deviation, and skewness for each channel (R, G, B)
        mean_values = np.mean(image_rgb, axis=(0, 1))
        std_dev_values = np.std(image_rgb, axis=(0, 1))
        skewness_values = [skew(image_rgb[:, :, i].ravel()) for i in range(3)]

        # Concatenate features
        color_moments = np.concatenate([mean_values, std_dev_values, skewness_values])

        return color_moments

    def compute_distance(self, query_features):
        distances = [np.linalg.norm(query_features - feat) for feat in self.features]
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
        true_positives = 0  # Initialize true positives counter

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

        false_positives = 10 - true_positives
        false_negatives = 100 - true_positives

        precision = true_positives / (true_positives + false_positives) \
            if (true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        return precision, recall, f1, distances


def run_experiment_color_moments(dataset_path, category_size):
    # Create CBIR System with Color Moments
    cbir_system_color_moments = CBIRSystemColorMoments(dataset_path)

    # Perform experiments
    num_queries = len(cbir_system_color_moments.image_paths) // category_size
    precision_avg, recall_avg, f1_avg, time_avg = 0, 0, 0, 0
    all_labels, all_scores = [], []

    for query_index in range(num_queries):
        # Use a random query image within the category
        category_start_index = query_index * category_size
        query_image_index = np.random.choice(range(category_start_index, category_start_index + category_size))
        query_image = cv2.imread(cbir_system_color_moments.image_paths[query_image_index])
        query_features = cbir_system_color_moments.extract_features(query_image)

        print(f"Query Path: {cbir_system_color_moments.image_paths[query_image_index]}")

        # Measure time taken for retrieval
        start_time = time.time()

        # Perform retrieval and get ranked results
        distances = cbir_system_color_moments.compute_distance(query_features)
        ranked_results = cbir_system_color_moments.rank_results(distances, query_image_index)

        # Evaluate performance metrics
        precision, recall, f1, distances = cbir_system_color_moments.evaluate_performance(query_image_index)

        print(f"precision: {precision}, recall: {recall}, f1 score: {f1}")

        # Update average metrics
        precision_avg += precision
        recall_avg += recall
        f1_avg += f1
        time_avg += (time.time() - start_time)

        # Store labels and scores for ROC curve
        labels = [1 if (i // category_size) == (query_index % num_queries)
                  else 0 for i in range(len(cbir_system_color_moments.image_paths))]

        # labels = [1 if i == query_index else 0 for i in range(len(cbir_system_color_moments.image_paths))]
        all_labels.extend(labels)
        all_scores.extend([-dist for dist in distances])  # Negative distances for ascending order

        # Display results for each query
        # cbir_system_color_moments.display_results(ranked_results, query_image_index)

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
category_size = 100  # Set the size of each category
run_experiment_color_moments(dataset_path, category_size)'''

############################################## Task 3.2 ###########################################################
'''import os
import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import time
from scipy.stats import skew

class CBIRSystemColorMoments:
    def __init__(self, dataset_path):
        self.image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        self.features, self.labels = self.extract_color_moments()

    def extract_color_moments(self):
        features = []
        labels = []

        for i, img_path in enumerate(self.image_paths):
            img = cv2.imread(img_path)
            color_moments = self.extract_features(img)
            features.append(color_moments)
            labels.append(i)  # Assign label based on the index

        return np.array(features), np.array(labels)

    def extract_features(self, image):
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calculate color moments: mean, standard deviation, and skewness for each channel (R, G, B)
        mean_values = np.mean(image_rgb, axis=(0, 1))
        std_dev_values = np.std(image_rgb, axis=(0, 1))
        skewness_values = [skew(image_rgb[:, :, i].ravel()) for i in range(3)]

        # Assign weights to the color moments
        weights = [1.0, 0.5, 10.0]  # Adjust the weights based on importance
        weighted_mean = np.multiply(mean_values, weights[0])
        weighted_std_dev = np.multiply(std_dev_values, weights[1])
        weighted_skewness = np.multiply(skewness_values, weights[2])

        # Concatenate features
        color_moments = np.concatenate([weighted_mean, weighted_std_dev, weighted_skewness])

        return color_moments

    def compute_distance(self, query_features):
        distances = [np.linalg.norm(query_features - feat) for feat in self.features]
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
        true_positives = 0  # Initialize true positives counter

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

        false_positives = 10 - true_positives
        false_negatives = 100 - true_positives

        precision = true_positives / (true_positives + false_positives) \
            if (true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        return precision, recall, f1, distances


def run_experiment_color_moments(dataset_path, category_size):
    # Create CBIR System with Color Moments
    cbir_system_color_moments = CBIRSystemColorMoments(dataset_path)

    # Perform experiments
    num_queries = len(cbir_system_color_moments.image_paths) // category_size
    precision_avg, recall_avg, f1_avg, time_avg = 0, 0, 0, 0
    all_labels, all_scores = [], []

    for query_index in range(num_queries):
        # Use a random query image within the category
        category_start_index = query_index * category_size
        query_image_index = np.random.choice(range(category_start_index, category_start_index + category_size))
        query_image = cv2.imread(cbir_system_color_moments.image_paths[query_image_index])
        query_features = cbir_system_color_moments.extract_features(query_image)

        print(f"Query Path: {cbir_system_color_moments.image_paths[query_image_index]}")

        # Measure time taken for retrieval
        start_time = time.time()

        # Perform retrieval and get ranked results
        distances = cbir_system_color_moments.compute_distance(query_features)
        ranked_results = cbir_system_color_moments.rank_results(distances, query_image_index)

        # Evaluate performance metrics
        precision, recall, f1, distances = cbir_system_color_moments.evaluate_performance(query_image_index)

        print(f"precision: {precision}, recall: {recall}, f1 score: {f1}")

        # Update average metrics
        precision_avg += precision
        recall_avg += recall
        f1_avg += f1
        time_avg += (time.time() - start_time)

        # Store labels and scores for ROC curve
        labels = [1 if (i // category_size) == (query_index % num_queries)
                  else 0 for i in range(len(cbir_system_color_moments.image_paths))]
        all_labels.extend(labels)
        all_scores.extend([-dist for dist in distances])  # Negative distances for ascending order

        # Display results for each query
        # cbir_system_color_moments.display_results(ranked_results, query_image_index)

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
category_size = 100  # Set the size of each category
run_experiment_color_moments(dataset_path, category_size)'''
############################################## Task 3.3 ###########################################################

import os
import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import time
from scipy.stats import skew, kurtosis

class CBIRSystemExtendedMoments:
    def __init__(self, dataset_path):
        self.image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        self.features, self.labels = self.extract_extended_moments()

    def extract_extended_moments(self):
        features = []
        labels = []

        for i, img_path in enumerate(self.image_paths):
            img = cv2.imread(img_path)
            moments = self.extract_features(img)
            features.append(moments)
            labels.append(i)  # Assign label based on the index

        return np.array(features), np.array(labels)

    def extract_features(self, image):
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calculate color moments: mean, standard deviation, and skewness for each channel (R, G, B)
        mean_values = np.mean(image_rgb, axis=(0, 1))
        std_dev_values = np.std(image_rgb, axis=(0, 1))
        skewness_values = [skew(image_rgb[:, :, i].ravel()) for i in range(3)]

        # Additional moments: Median, Mode, Kurtosis
        median_values = np.median(image_rgb, axis=(0, 1))
        mode_values = np.zeros(3)  # Placeholder for mode; update based on your preferred method
        kurtosis_values = [kurtosis(image_rgb[:, :, i].ravel()) for i in range(3)]

        # Assign weights to the moments
        weights = [10.0, 0.5, 1.0, 2.0, 3.0, 5.0]  # Adjust the weights based on importance
        weighted_mean = np.multiply(mean_values, weights[0])
        weighted_std_dev = np.multiply(std_dev_values, weights[1])
        weighted_skewness = np.multiply(skewness_values, weights[2])
        weighted_median = np.multiply(median_values, weights[3])
        weighted_kurtosis = np.multiply(kurtosis_values, weights[4])

        # Concatenate features
        moments = np.concatenate([weighted_mean, weighted_std_dev, weighted_skewness, weighted_median, weighted_kurtosis])

        return moments

    def compute_distance(self, query_features):
        distances = [np.linalg.norm(query_features - feat) for feat in self.features]
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
        true_positives = 0  # Initialize true positives counter

        for i, idx in enumerate(ranked_results[:10]):
            retrieved_category = labels[idx]

            # Determine the category intervals
            true_category_start = true_category // 100 * 100
            true_category_end = true_category_start + 99

            retrieved_category_start = retrieved_category // 100 * 100
            retrieved_category_end = retrieved_category_start + 99

            # Check if the retrieved image is in the same category
            if true_category_start <= retrieved_category <= true_category_end:
                true_positives += 1

        false_positives = 10 - true_positives
        false_negatives = 100 - true_positives

        precision = true_positives / (true_positives + false_positives) \
            if (true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        return precision, recall, f1, distances


def run_experiment_extended_moments(dataset_path, category_size):
    # Create CBIR System with Extended Moments
    cbir_system_extended_moments = CBIRSystemExtendedMoments(dataset_path)

    # Perform experiments
    num_queries = len(cbir_system_extended_moments.image_paths) // category_size
    precision_avg, recall_avg, f1_avg, time_avg = 0, 0, 0, 0
    all_labels, all_scores = [], []

    for query_index in range(num_queries):
        # Use a random query image within the category
        category_start_index = query_index * category_size
        query_image_index = np.random.choice(range(category_start_index, category_start_index + category_size))
        query_image = cv2.imread(cbir_system_extended_moments.image_paths[query_image_index])
        query_features = cbir_system_extended_moments.extract_features(query_image)

        print(f"Query Path: {cbir_system_extended_moments.image_paths[query_image_index]}")

        # Measure time taken for retrieval
        start_time = time.time()

        # Perform retrieval and get ranked results
        distances = cbir_system_extended_moments.compute_distance(query_features)
        ranked_results = cbir_system_extended_moments.rank_results(distances, query_image_index)

        # Evaluate performance metrics
        precision, recall, f1, distances = cbir_system_extended_moments.evaluate_performance(query_image_index)

        print(f"precision: {precision}, recall: {recall}, f1 score: {f1}")

        # Update average metrics
        precision_avg += precision
        recall_avg += recall
        f1_avg += f1
        time_avg += (time.time() - start_time)

        # Store labels and scores for ROC curve
        labels = [1 if (i // category_size) == (query_index % num_queries)
                  else 0 for i in range(len(cbir_system_extended_moments.image_paths))]
        all_labels.extend(labels)
        all_scores.extend([-dist for dist in distances])  # Negative distances for ascending order

        # Display results for each query
        cbir_system_extended_moments.display_results(ranked_results, query_image_index)

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
category_size = 100
run_experiment_extended_moments(dataset_path, category_size)
