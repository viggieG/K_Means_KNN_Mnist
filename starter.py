import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE

# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    euclid_dist = round(np.sqrt(sum((a - b)**2)),2)
    return euclid_dist
        
# returns Cosine Similarity between vectors a dn b
def cosim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # prevent division by zero
    if norm_a == 0 or norm_b == 0:
        return 0 
    return dot_product / (norm_a * norm_b)

def distance(a, b, metric):
    if metric == "euclidean":
        return euclidean(a, b)
    elif metric == "cosim":
        return cosim(a, b)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def normalize_data(data):
    if len(data.shape) == 1:  # Check if data is 1D
        norm = np.linalg.norm(data, keepdims=True)
    else:
        norm = np.linalg.norm(data, axis=1, keepdims=True)
    return data/norm

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric):
    k = 1
    #read in training data
    train_list = read_data(train)
    #read in query data
    query_list = read_data(query)
    
    # normalize the data
    for i in range(len(train_list)):
        train_list[i][1] = normalize_data(np.array(train_list[i][1]).astype(float))
    for i in range(len(query_list)):
        query_list[i][1] = normalize_data(np.array(query_list[i][1]).astype(float))
        
    labels = []
    for q_item in query_list:
        _, q_vector = q_item
        distances = []
        
        for t_item in train_list:
            t_label, t_vector = t_item
            if metric == "cosim":
                dist = 1 - distance(np.array(q_vector).astype(float), np.array(t_vector).astype(float), metric)
            else:
                dist = distance(np.array(q_vector).astype(float), np.array(t_vector).astype(float), metric)
            distances.append((dist, t_label))

        # sort based on distances and take top-k labels
        sorted_distances = sorted(distances, key=lambda x: x[0])
        top_k_labels = [item[1] for item in sorted_distances[:k]]
        
        # determine the majority label among top-k labels
        predicted_label = max(set(top_k_labels), key=top_k_labels.count)
        labels.append(predicted_label)
    
    return labels

def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')

def knn_accuracy(train, test, metric):
    # Use your knn function to get predictions on the test set
    predictions = knn(train, test, metric)

    # Read in the true labels from the test set
    true_labels = [item[0] for item in read_data(test)]

    # Compare predictions to true labels to compute accuracy
    correct_predictions = sum([1 for predicted, true in zip(predictions, true_labels) if predicted == true])
    accuracy = correct_predictions / len(true_labels)
    print(f'KNN Accuracy {accuracy:.2f} with {metric}')
    
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f'Confusion Matrix ({metric} with {test})')
    plt.savefig(f'confusion_matrix_{metric}_test.png')
    
    return accuracy


def evaluate_knn_with_validation(train_file, valid_file, metric, k_values):
    k_accuracies = {}
    
    for k in k_values:
        predictions = knn_modified(train_file, valid_file, metric,k)
        
        # read the true labels from the validation file
        validation_data = read_data(valid_file)
        true_labels = [row[0] for row in validation_data] 
        
        accuracy = sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(predictions)
        k_accuracies[k] = accuracy

        # print the current k and its accuracy
        print(f"Accuracy for k = {k}: {accuracy:.2f} with {metric}")
        
    best_k = max(k_accuracies, key=k_accuracies.get)
    best_accuracy = k_accuracies[best_k]
    return best_k, best_accuracy

def knn_modified(train,query,metric,k):
    #read in training data
    train_list = read_data(train)
    #read in query data
    query_list = read_data(query)
    
    # Normalize the data
    for i in range(len(train_list)):
        train_list[i][1] = normalize_data(np.array(train_list[i][1]).astype(float))
    for i in range(len(query_list)):
        query_list[i][1] = normalize_data(np.array(query_list[i][1]).astype(float))
        
    labels = []
    for q_item in query_list:
        _, q_vector = q_item
        distances = []
        
        for t_item in train_list:
            t_label, t_vector = t_item
            if metric == "cosim":
                dist = 1 - distance(np.array(q_vector).astype(float), np.array(t_vector).astype(float), metric)
            else:
                dist = distance(np.array(q_vector).astype(float), np.array(t_vector).astype(float), metric)
            distances.append((dist, t_label))

        # Sort based on distances and take top-k labels
        sorted_distances = sorted(distances, key=lambda x: x[0])
        top_k_labels = [item[1] for item in sorted_distances[:k]]
        
        # Determine the majority label among top-k labels
        predicted_label = max(set(top_k_labels), key=top_k_labels.count)
        labels.append(predicted_label)
    
    return labels

def reduce_dimensionality(data, n_components=40):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train, query, metric, k, max_iterations=100, tolerance=1e-4,tsne_components=2):
    def normalize_data(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def euclidean_distance(p1, p2):
        return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

    def initialize_centroids_plusplus(data, k):
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

    def closest_centroid(sample, centroids):
        distances = [euclidean_distance(sample, centroid) for centroid in centroids]
        return np.argmin(distances)

    # Read in training data and query data
    train_list = read_data(train)
    query_list = read_data(query)
    train_labels = [item[0] for item in train_list]
    train_data = [item[1] for item in train_list]
    query_labels = [item[0] for item in query_list]
    query_data = [item[1] for item in query_list]

    # Normalize data
    train_data = [normalize_data(np.array(item).astype(float)) for item in train_data]
    query_data = [normalize_data(np.array(item).astype(float)) for item in query_data]

    def apply_tsne(train_data, query_data, tsne_components=2):
        # Combine train and query data
        combined_data = np.vstack([train_data, query_data])

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate=200.0)
        combined_data_tsne = tsne.fit_transform(combined_data)

        # Separate the transformed train and query data
        train_data_tsne = combined_data_tsne[:len(train_data)]
        query_data_tsne = combined_data_tsne[len(train_data):]

        return train_data_tsne, query_data_tsne

    # Then in your kmeans function:
    train_data_tsne, query_data_tsne = apply_tsne(train_data, query_data)


    # Initialize centroids using K-means++
    centroids = initialize_centroids_plusplus(train_data_tsne, k)  # 注意使用 t-SNE 转换后的数据
    old_centroids = centroids.copy()


    for _ in range(max_iterations):
        # Assign samples to closest centroids (create clusters)
        clusters = [[] for _ in range(k)]
        for idx, sample in enumerate(train_data_tsne):
            centroid_idx = closest_centroid(sample, centroids)
            clusters[centroid_idx].append((sample, train_labels[idx]))

        # Calculate new centroids from clusters
        for idx, cluster in enumerate(clusters):
            if cluster:  # check if cluster is not empty
                average = np.mean([item[0] for item in cluster], axis=0)
                centroids[idx] = average.tolist()

        # Check convergence
        diff = sum([np.linalg.norm(np.array(centroids[i]) - np.array(old_centroids[i])) for i in range(k)])
        if diff <= tolerance:
            break
        old_centroids = centroids.copy()

    # For each cluster, determine the dominant label
    dominant_labels = []
    for cluster in clusters:
        labels = [item[1] for item in cluster]
        dominant_label = Counter(labels).most_common(1)[0][0]
        dominant_labels.append(dominant_label)

    # Predict the labels for query data
    predictions = []
    for sample in query_data_tsne:
        centroid_idx = closest_centroid(sample, centroids)
        predictions.append(dominant_labels[centroid_idx])

    return predictions

def kmeans_accuracy(train, query, metric, k, max_iterations=100,tsne_components=2):
    predictions = kmeans(train, query, metric, k, max_iterations,tsne_components=2)

    query_list = read_data(query)
    query_labels = [item[0] for item in query_list]

    correct_predictions = sum(p == q for p, q in zip(predictions, query_labels))
    accuracy = correct_predictions / len(query_labels)
    print(f'Kmeans Accuracy {accuracy:.2f} with {metric}')
    cm = confusion_matrix(query_labels, predictions)
    disp = ConfusionMatrixDisplay(cm)
    # Display the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.savefig(f'confusion_matrix_kmeans_{metric}_test.png')

    return accuracy

def evaluate_kmeans_with_validation(train_file, valid_file, metric, k_values):
    k_accuracies = {}

    for k in k_values:
        #predictions = kmeans_modified(train_file, valid_file, metric, k)
        predictions = kmeans(train_file, valid_file, metric, k, max_iterations=100,tsne_components=2)

        # read the true labels from the validation file
        validation_data = read_data(valid_file)
        true_labels = [row[0] for row in validation_data]

        accuracy = sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(predictions)
        k_accuracies[k] = accuracy

        # print the current k and its accuracy
        print(f"Accuracy for k = {k}: {accuracy:.2f} with {metric}")

    best_k = max(k_accuracies, key=k_accuracies.get)
    best_accuracy = k_accuracies[best_k]
    return best_k, best_accuracy

def main():
    # show('valid.csv','pixels')
    # knn('train.csv','valid.csv','euclidean')
    knn_accuracy('train.csv','test.csv','euclidean')
    knn_accuracy('train.csv','test.csv','cosim')
    
    # k_values = list(range(1, 51,5))
    # best_k, best_accuracy = evaluate_knn_with_validation('train.csv', 'valid.csv', 'euclidean', k_values)
    # print(f"Best k for Euclidean: {best_k} with accuracy: {best_accuracy:.2f}")

    # best_k, best_accuracy = evaluate_knn_with_validation('train.csv', 'valid.csv', 'cosim', k_values)
    # print(f"Best k for Cosine Similarity: {best_k} with accuracy: {best_accuracy:.2f}")
    kmeans_cosim = kmeans_accuracy('train.csv', 'test.csv', 'cosim', k=31, max_iterations=100)
    print(f"KMeans accuracy: {kmeans_cosim:.2f}")
    kmeans_euclidean = kmeans_accuracy('train.csv', 'test.csv', 'euclidean', k=41, max_iterations=100)
    print(f"KMeans accuracy: {kmeans_euclidean:.2f}")
    
if __name__ == "__main__":
    main()
    

