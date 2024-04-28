import os
import pickle
import numpy as np

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
        (dtype=np.int32)
    """
    ### YOUR CODE HERE

    meta_dir = os.path.join(data_dir, 'batches.meta')
    with open(meta_dir, 'rb') as fo:
        meta_data_dict = pickle.load(fo, encoding='latin1')
    label_names = meta_data_dict['label_names']

    # Load training data from data_batch_1 to data_batch_5
    x_train_batches = []
    y_train = np.array([], dtype=np.int32)
    for i in range(1, 6):
        data_batch_file = os.path.join(data_dir, f'data_batch_{i}')
        with open(data_batch_file, 'rb') as fo:
            batch_data = pickle.load(fo, encoding='latin1')
        x_train_batches.append(batch_data['data'])
        y_train = np.concatenate((y_train, np.array(batch_data['labels'], dtype=np.int32)))

    x_train = np.concatenate(x_train_batches).astype(np.float32)

    # Load test data from test_batch
    test_dataset = os.path.join(data_dir, 'test_batch')
    with open(test_dataset, 'rb') as fo:
        test_data_dict = pickle.load(fo, encoding='latin1')
    x_test = test_data_dict['data'].astype(np.float32)
    y_test = np.array(test_data_dict['labels'], dtype=np.int32)

    return x_train, y_train, x_test, y_test






    ### YOUR CODE HERE

    # return x_train, y_train, x_test, y_test

def train_valid_split_balanced(x_train, y_train, valid_ratio=0):
    """ Split the original training data into a new training dataset
        and a validation dataset using a random 80-20 split with
        similar class ratios.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        valid_ratio: A float specifying the validation ratio. Default is 0.2 (20%).

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    num_samples = x_train.shape[0]
    num_valid_samples = int(num_samples * valid_ratio)

    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(y_train, return_counts=True)

    # Initialize empty arrays for train and validation data
    x_train_new, y_train_new = np.array([]), np.array([])
    x_valid, y_valid = np.array([]), np.array([])

    # For each class, split the data and maintain similar class ratios
    for class_label in unique_classes:
        # Get indices of samples belonging to the current class
        class_indices = np.where(y_train == class_label)[0]
        np.random.shuffle(class_indices)  # Shuffle indices

        # Calculate the number of samples for the current class in the validation set
        num_valid_samples_class = int(class_counts[class_label] * valid_ratio)

        # Split indices into training and validation indices
        valid_indices = class_indices[:num_valid_samples_class]
        train_indices = class_indices[num_valid_samples_class:]

        # Concatenate the data for the current class to the respective arrays
        x_train_new = np.concatenate([x_train_new, x_train[train_indices]], axis=0) if x_train_new.size else x_train[train_indices]
        y_train_new = np.concatenate([y_train_new, y_train[train_indices]], axis=0) if y_train_new.size else y_train[train_indices]
        x_valid = np.concatenate([x_valid, x_train[valid_indices]], axis=0) if x_valid.size else x_train[valid_indices]
        y_valid = np.concatenate([y_valid, y_train[valid_indices]], axis=0) if y_valid.size else y_train[valid_indices]

    return x_train_new, y_train_new, x_valid, y_valid


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """

    ### YOUR CODE HERE
    split_index = int(len(x_train) * 0.8)

    # Split the data
    x_train_new, x_valid = x_train[:split_index], x_train[split_index:]
    y_train_new, y_valid = y_train[:split_index], y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid


def train_valid_split_v2(x_train, y_train, train_ratio=0.8):
    """Return the entire training dataset without splitting.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1 (unused).

    Returns:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        x_valid: An array of shape [0, 3072].
        y_valid: An array of shape [0,].
    """

    return x_train, y_train, np.array([]), np.array([])




def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 3072].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    x_test = np.load(data_dir).astype(np.float64)


    ### END CODE HERE

    return x_test
