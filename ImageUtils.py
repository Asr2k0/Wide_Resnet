import numpy as np
import matplotlib.pyplot as plt

""" This script implements the functions for data augmentation and preprocessing.
"""


def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])
    #visualize(image,'test.png')
    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image


def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        # Add 4 pixels along each side using reflection padding
        image = np.pad(image, ((4, 4), (4, 4), (0, 0)), mode='reflect')

        # Randomly crop a [32, 32] section of the image
        crop_x = np.random.randint(0, 9)
        crop_y = np.random.randint(0, 9)
        image = image[crop_x:crop_x + 32, crop_y:crop_y + 32, :]

        # Randomly flip the image horizontally
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1)

        # Randomly adjust brightness and contrast
        alpha = np.random.uniform(0.8, 1.2)  # Brightness factor
        beta = np.random.uniform(0.8, 1.2)  # Contrast factor
        image = np.clip(alpha * image + beta, 0, 255)

        # Add jittery pixels to simulate distortion
        jitter = np.random.randint(-5, 6, size=(32, 32, 3))  # Random jitter between -5 and 5
        image = np.clip(image + jitter, 0, 255)

        # Add random noise to the image
        if np.random.rand() < 0.5:
            noise = np.random.normal(loc=0, scale=20,
                                     size=image.shape)  # Gaussian noise with mean 0 and standard deviation 20
            image = np.clip(image + noise, 0, 255)

    # Subtract off the mean and divide by the standard deviation of the pixels
    image = (image - np.mean(image, axis=(0, 1), keepdims=True)) / np.std(image, axis=(0, 1), keepdims=True)

    return image


def plot_image(image):
    image = np.transpose(image.reshape(3, 32, 32), [1, 2, 0])
    plt.imshow(image.astype('uint8'))
    plt.show()


def plot_actual_vs_predicted_image(image, actual_label, predicted_label):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image.astype('uint8'))
    plt.title(f'Actual: {actual_label}')
    plt.subplot(1, 2, 2)
    plt.imshow(image.astype('uint8'))
    plt.title(f'Predicted: {predicted_label}')
    plt.show()


def save_loss_curve(train_losses, save_path):
    epochs = len(train_losses)
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch-wise Loss Curve')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def save_actual_vs_predicted_images(images, actual_labels, predicted_labels, save_path):
    num_images = len(images)
    num_rows = 2
    num_cols = (num_images + 1) // num_rows  # Ensure at least 2 rows
    plt.figure(figsize=(5 * num_cols, 5 * num_rows))
    for i in range(num_images):
        # Reshape the flattened image to [3, 32, 32]
        image = images[i].reshape(3, 32, 32)

        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(np.transpose(image, (1, 2, 0)).astype('uint8'))  # Transpose to (32, 32, 3)
        plt.title(f'Actual: {actual_labels[i]}, Predicted: {predicted_labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()





def visualize(image, save_name='test.png'):
    """
    Visualize a single test image.

    Args:
        image: An array of shape [3072]
        save_name: A file name to save your visualization.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    # Reshape the image array to [32, 32, 3]
    image = np.reshape(image, (32, 32, 3))
    image = image.astype(int)

    plt.imshow(image)
    plt.savefig(save_name)
    return image

