import cv2
import numpy as np
import matplotlib.pyplot as plt


class CBIRSystemColorFeatures:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.images = [cv2.imread(img_path) for img_path in image_paths]
        self.features = [self.extract_features(img) for img in self.images]

    def extract_features(self, image):
        # Implement color feature extraction
        mean_std_features = []
        for channel in cv2.split(image):
            mean_std_features.extend([np.mean(channel), np.std(channel)])
        return mean_std_features

    def compute_distance(self, query_features):
        query_features = np.array(query_features)

        # Implement Euclidean distance computation
        distances = [np.linalg.norm(np.array(query_features) - np.array(feat)) for feat in self.features]
        return distances

    def rank_results(self, distances):
        # Rank results based on distances
        sorted_indices = np.argsort(distances)
        return sorted_indices

    def display_results(self, ranked_results):
        plt.figure(figsize=(15, 8))

        # Plot Query Image - First Row
        plt.subplot(3, 1, 1)
        plt.imshow(cv2.cvtColor(cv2.imread(self.image_paths[ranked_results[0]]), cv2.COLOR_BGR2RGB))
        plt.title("Query Image")
        plt.axis('off')

        # Plot Retrieval Results - Second Row
        for i, idx in enumerate(ranked_results[1:7], start=1):
            img = cv2.imread(self.image_paths[idx])
            plt.subplot(3, 6, i + 6)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"#{i}")
            plt.axis('off')

        # Plot Retrieval Results - Third Row
        for i, idx in enumerate(ranked_results[7:13], start=1):
            img = cv2.imread(self.image_paths[idx])
            plt.subplot(3, 6, i + 12)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"#{i + 6}")
            plt.axis('off')

        plt.show()


def run_experiment_once(query_image_path):
    # Load dataset
    dataset_path = ".\\dataset"
    image_paths = [f"{dataset_path}/{i}.jpg" for i in range(1, 1000)]

    # Create CBIR System
    cbir_system = CBIRSystemColorFeatures(image_paths)

    query_image = cv2.imread(query_image_path)
    query_features = cbir_system.extract_features(query_image)

    # Perform retrieval and get ranked results
    distances = cbir_system.compute_distance(query_features)
    ranked_results = cbir_system.rank_results(distances)

    cbir_system.display_results(ranked_results)


query_image_path = ".\\dataset\\133.jpg"
run_experiment_once(query_image_path)
