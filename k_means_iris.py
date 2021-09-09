import pandas as pd
import numpy as np

ALGORITHM_NAME = "K-means"
DATA_PATH = "iris.csv"
SEPARATOR = ","

pd_full_data_set = pd.read_csv(DATA_PATH, sep=SEPARATOR)

pd_data_set = pd_full_data_set.copy()

no_of_instances = len(pd_data_set.index)
no_of_columns = len(pd_data_set.columns)
no_of_attributes = no_of_columns - 2
actual_class_column = no_of_columns - 1

unique_class_list_df = pd_data_set.iloc[:, actual_class_column]
unique_class_list_np = unique_class_list_df.unique()
unique_class_list_df = unique_class_list_df.drop_duplicates()

num_unique_classes = len(unique_class_list_df)

K = num_unique_classes

instance_id_colname = pd_data_set.columns[0]
class_column_colname = pd_data_set.columns[actual_class_column]
pd_data_set = pd_data_set.drop(columns=[instance_id_colname, class_column_colname])

np_data_set = pd_data_set.to_numpy(copy=True)

centroids = np_data_set[np.random.choice(np_data_set.shape[0], size=K, replace=False), :]

cluster_assignments = np.empty(no_of_instances)

centroids_the_same = False

max_iterations = 300

while max_iterations > 0 and not (centroids_the_same):
    for row in range(0, no_of_instances):

        this_instance = np_data_set[row]

        min_distance = float("inf")

        for row_centroid in range(0, K):
            this_centroid = centroids[row_centroid]

            distance = np.linalg.norm(this_instance - this_centroid)

            if distance < min_distance:
                cluster_assignments[row] = row_centroid
                min_distance = distance

    print("Cluster assignments completed for all " + str(no_of_instances) + " instances. Here they are:")
    print(cluster_assignments)
    print()
    print("Now calculating the new centroids...")
    print()
    old_centroids = centroids.copy()

    for row_centroid in range(0, K):

        for col_centroid in range(0, no_of_attributes):
            running_sum = 0.0
            count = 0.0
            average = None

            for row in range(0, no_of_instances):
                if (row_centroid == cluster_assignments[row]):
                    running_sum += np_data_set[row, col_centroid]
                    count += 1

                    if (count > 0):
                        average = running_sum / count
            centroids[row_centroid, col_centroid] = average

    print("New centroids have been created. Here they are:")
    print(centroids)
    print()

    centroids_the_same = np.array_equal(old_centroids, centroids)

    if centroids_the_same:
        print("Cluster membership is unchanged. Stopping criteria has been met.")

    max_iterations -= 1

actual_class_col_name = pd_full_data_set.columns[len(
    pd_full_data_set.columns) - 1]

pd_full_data_set = pd_full_data_set.reindex(
    columns=[*pd_full_data_set.columns.tolist( ), 'Cluster', 'Silhouette Coefficient', 'Predicted Class', ('Prediction Correct?')])

pd_full_data_set['Cluster'] = cluster_assignments

class_mappings = pd.DataFrame(index=range(K), columns=range(1))

for clstr in range(0, K):
    temp_df = pd_full_data_set.loc[pd_full_data_set['Cluster'] == clstr]
    class_mappings.iloc[clstr, 0] = temp_df.mode()[actual_class_col_name][0]

cluster_column = actual_class_column + 1
pred_class_column = actual_class_column + 3
pred_correct_column = actual_class_column + 4

for row in range(0, no_of_instances):

    for clstr in range(0, K):
        if clstr == pd_full_data_set.iloc[row, cluster_column]:
            pd_full_data_set.iloc[
                row, pred_class_column] = class_mappings.iloc[clstr, 0]

    if pd_full_data_set.iloc[row, pred_class_column] == pd_full_data_set.iloc[
        row, actual_class_column]:
        pd_full_data_set.iloc[row, pred_correct_column] = 1
    else:
        pd_full_data_set.iloc[row, pred_correct_column] = 0

print()
print()
print("Data Set")
print(pd_full_data_set)
print()
print()

# accuracy = (total correct predictions)/(total number of predictions)
accuracy = (pd_full_data_set.iloc[:, pred_correct_column].sum()) / no_of_instances
accuracy *= 100

print("Number of Instances : " + str(no_of_instances))
print("Value for k : " + str(K))
print("Accuracy : " + str(accuracy) + "%")