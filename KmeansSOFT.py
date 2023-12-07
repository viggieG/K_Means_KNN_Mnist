import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from starter import kmeans
def euclidean_distance(x1, x2):
    return np.linalg.norm(np.array(x1) - np.array(x2))

def read_data(file_name):
    data_set = []
    with open(file_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(',')
            label = tokens[0]
            attribs_list = []
            for x in tokens[1:]:
                try:
                    attribs_list.append(float(x))
                except ValueError:
                    pass
                    #print(f"Failed to convert '{x}' to float.")
            attribs = np.array(attribs_list, dtype=float)
            data_set.append([label, attribs])
    return data_set




def show(file_name, mode):
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ', end='')
                else:
                    print('*', end='')
            else:
                print('%4s ' % data_set[obs][1][idx], end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0], end='')
        print(' ')


def assign_labels_to_centroids(data, labels, centroids):
    """为每个聚类中心分配最常见的标签。"""
    cluster_labels = []
    distances = np.array([[euclidean_distance(x, c) for c in centroids] for x in data])
    clusters = np.argmin(distances, axis=1)
    for i in range(len(centroids)):
        counter = Counter(labels[clusters == i])
        if counter:
            most_common_label = counter.most_common(1)[0][0]
        else:
            # 如果聚类为空，随机分配一个标签
            most_common_label = np.random.choice(labels)
        cluster_labels.append(most_common_label)
    return cluster_labels


def kmeans_accuracy(train, query, k=1, max_iterations=100):
    train_data_list = read_data(train)
    train_data = np.array([x[1] for x in train_data_list])
    train_labels = np.array([x[0] for x in train_data_list])

    query_data_list = read_data(query)
    query_data = np.array([x[1] for x in query_data_list])
    query_labels = np.array([x[0] for x in query_data_list])

    m = 45
    centroids, _ = soft_kmeans(train_data, k, m, max_iterations)
    centroid_labels = assign_labels_to_centroids(train_data, train_labels, centroids)
    predictions = np.array(
        [centroid_labels[np.argmin([euclidean_distance(x, centroid) for centroid in centroids])] for x in query_data])

    correct_predictions = sum(p == q for p, q in zip(predictions, query_labels))
    accuracy = correct_predictions / len(query_labels)
    print(f'Kmeans Accuracy {accuracy:.2f} with {m} value')


def evaluate_kmeans_with_validation(train_file, valid_file, k_values,metric):
    k_accuracies = {}

    for k in k_values:
        # predictions = kmeans_modified(train_file, valid_file, metric, k)
        predictions = soft_kmeans(train_file,k)

        # read the true labels from the validation file
        validation_data = read_data(valid_file)
        true_labels = [row[0] for row in validation_data]

        accuracy = sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(predictions)
        k_accuracies[k] = accuracy

        # print the current k and its accuracy
        print(f"Accuracy for k = {k}: {accuracy:.2f}")

    best_k = max(k_accuracies, key=k_accuracies.get)
    best_accuracy = k_accuracies[best_k]
    return best_k, best_accuracy

def initialize_centroids(data, k):
    centroids = [data[np.random.choice(len(data))]]
    for _ in range(1, k):
        dist_sq = np.array([min([euclidean_distance(c, x) for c in centroids]) for x in data])
        probs = dist_sq / dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        centroids.append(data[i])
    return centroids



def compute_memberships(data, centroids, m):
    n_clusters = len(centroids)
    memberships = np.zeros((len(data), n_clusters))

    for i, x in enumerate(data):
        distances = [euclidean_distance(x, c)+ 1e-10 for c in centroids]
        for j in range(n_clusters):
            memberships[i][j] = 1.0 / np.sum([(distances[j] / d) ** (2.0 / (m - 1)) for d in distances])

    return memberships


def update_centroids(data, memberships, m):
    n_clusters = memberships.shape[1]
    centroids = np.zeros((n_clusters, data.shape[1]))

    for j in range(n_clusters):
        weights = memberships[:, j] ** m
        centroids[j] = np.sum(data * weights.reshape(-1, 1), axis=0) / np.sum(weights)

    return centroids


def soft_kmeans(data, k, m=2, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        old_centroids = centroids.copy()
        memberships = compute_memberships(data, centroids, m)
        centroids = update_centroids(data, memberships, m)
        if np.allclose(old_centroids, centroids):
            break
    return centroids, memberships


def main():
    # Assuming `train_data` is a numpy array of your training data
    # and `k` is the number of clusters
    accuracy2 = kmeans_accuracy('train.csv', 'valid.csv', 41, max_iterations=100)
    # print(f"soft KMeans accuracy: {accuracy2}")
    # k_values = list(range(1, 51,5))
    # evaluate_kmeans_with_validation('train.csv', 'valid.csv',k_values,'euclidean')



if __name__ == "__main__":
    main()
# Assuming `train_data` is a numpy array of your training data
# and `k` is the number of clusters
#k = 10
#m = 2

#centroids, memberships = soft_kmeans(train_data, k, m)