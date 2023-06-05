import torch, torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import cv2, random


def plot_data(X, y):
    color_map = {
        0: "red",
        1: "green",
        2: "blue",
        3: "yellow",
        4: "purple",
        5: "orange",
        6: "black",
        7: "pink",
        8: "gray",
        9: "brown",
    }

    # plot data points
    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], marker="o", color=color_map[y[i]])

    # naming the x axis
    plt.xlabel("x - axis")
    # naming the y axis
    plt.ylabel("y - axis")

    # function to show the plot
    plt.show()


def draw_image(image):
    plt.imshow(image, cmap="gray")
    plt.show()


class DataHandler:
    def __init__(self):
        pass

    @staticmethod
    def get_hand_drawn_letters(batch_size, augment_data=False, only_test=False):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        if augment_data:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                    transforms.RandomAffine(
                        degrees=(-30, 30),
                        translate=(0.32, 0.32),
                        scale=(0.4, 1),
                        shear=(-15, 15, -15, 15),
                    ),
                ]
            )

        # load EMNIST dataset with only letters
        # split="letters" has 26 classes (upper and lowercase combined into one)
        # split="balanced" has 47 classes (upper and lowercase both)
        kwargs = {"num_workers": 1, "pin_memory": True}
        # kwargs = {}
        if not only_test:
            dataset_train = datasets.EMNIST(
                root="./datasets",
                split="letters",
                train=True,
                download=True,
                transform=transform,
            )

            dataset_train.data = torch.flip(
                DataHandler.rotate_images(dataset_train.data, -90), [2]
            )

            trainset = DataLoader(
                dataset_train, batch_size=batch_size, shuffle=True, **kwargs
            )

        dataset_test = datasets.EMNIST(
            root="./datasets",
            split="letters",
            train=False,
            download=True,
            transform=transform,
        )
        # rotate and flip images

        dataset_test.data = torch.flip(
            DataHandler.rotate_images(dataset_test.data, -90), [2]
        )

        # create dataloaders
        testset = DataLoader(
            dataset_test, batch_size=batch_size, shuffle=True, **kwargs
        )

        if not only_test:
            return trainset, testset
        return testset

    @staticmethod
    def get_hand_drawn_digits(batch_size, augment_data=False):
        """return mnist hand-drawn digits (both training and testing sets)"""

        # NOTE, these transformations are applied only during training and testing
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        if augment_data:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.RandomAffine(
                        degrees=(-30, 30),
                        translate=(0.33, 0.33),
                        scale=(0.3, 1.1),
                        shear=(-30, 30, -30, 30),
                    ),
                ]
            )

        training_data = datasets.MNIST(
            "./datasets",
            train=True,
            download=True,
            transform=transform,
        )
        # kwargs = {"num_workers": 1, "pin_memory": True}
        kwargs = {}
        trainset = DataLoader(training_data, batch_size, shuffle=True, **kwargs)

        test_data = datasets.MNIST(
            "./datasets",
            train=False,
            download=True,
            transform=transform,
        )
        testset = DataLoader(test_data, batch_size, shuffle=False, **kwargs)

        return trainset, testset

    @staticmethod
    def translate_images(images, dx, dy):
        # Convert dx and dy to integers
        dx, dy = int(dx), int(dy)

        # Create an output tensor with the same shape as the input tensor
        output = torch.zeros_like(images)

        # Iterate over each image in the batch
        for i in range(images.shape[0]):
            # Create a zero tensor with the same shape as the input image
            shifted = torch.zeros_like(images[i])

            # Copy the pixels from the input image to the shifted image, applying the dx and dy translations
            shifted[
                max(0, dy) : min(28, 28 + dy), max(0, dx) : min(28, 28 + dx)
            ] = images[i][
                max(0, -dy) : min(28, 28 - dy), max(0, -dx) : min(28, 28 - dx)
            ]

            # Store the shifted image in the output tensor
            output[i] = shifted

        return output

    @staticmethod
    def scale_images(images, scale_factor):
        images = images.numpy()

        # scaled_images = torch.ones_like(images)
        scaled_images = np.ndarray((images.shape[0], 28, 28))
        for idx, image in enumerate(images):
            # Resize the image using the scale factor
            # scaled = resize(
            #     image,
            #     (int(28 * scale_factor), int(28 * scale_factor)),
            #     anti_aliasing=True,
            #     preserve_range=True,
            # )
            scaled = cv2.resize(
                image,
                (int(28 * scale_factor), int(28 * scale_factor)),
                # interpolation=cv2.INTER_NEAREST,
            )

            height = scaled.shape[0]
            width = scaled.shape[1]

            if height < 28 or width < 28:
                # If the scaled image is smaller than 28x28, create a new 28x28 image and center the scaled image in it
                new_image = np.zeros((28, 28))
                start_h = (28 - height) // 2
                start_w = (28 - width) // 2
                new_image[
                    start_h : start_h + height, start_w : start_w + width
                ] = scaled
                scaled_images[idx] = new_image
            else:
                center_h = height // 2
                center_w = width // 2
                cropped = scaled[
                    center_h - 14 : center_h + 14, center_w - 14 : center_w + 14
                ]
                scaled_images[idx] = cropped

        return torch.tensor(scaled_images)

    @staticmethod
    def rotate_images(images, degrees):
        rotated_images = torch.ones_like(images)
        for idx, image in enumerate(images):
            rotated = ndimage.rotate(image, degrees, reshape=False)
            rotated_images[idx] = torch.from_numpy(rotated)

        return rotated_images

    @staticmethod
    def apply_random_transformation(
        images,
        translation_ranges=[-5, 5],
        scale_ranges=[0.5, 1.4],
        rotation_ranges=[-45, 45],
    ):
        """applies a each of the following transformations to the input images with a random parameters:
        translation, scaling, rotation, and noise"""

        dx = np.random.randint(*translation_ranges)
        dy = np.random.randint(*translation_ranges)

        scale_factor = np.random.uniform(*scale_ranges)

        degrees = np.random.randint(*rotation_ranges)

        # noise_strength = np.random.uniform(0.1, 0.6)
        # noise_probability = np.random.uniform(0, 0.04)

        rotated = DataHandler.rotate_images(images, degrees)
        translated = DataHandler.translate_images(rotated, dx, dy)
        scaled = DataHandler.scale_images(translated, scale_factor)
        # noised = add_noise(scaled, noise_strength, noise_probability)

        return scaled


# trainset, testset = get_hand_drawn_digits_data(64)


# images = testset.dataset.data[0:10]

# draw_image(images[3])
# # new_images = translate_images(images, -6, 6)
# # new_images = rotate_images(images, 35)
# new_images = scale_images(images, 1.5)
# # new_images = add_noise(images, 0.6, 0.05)

# draw_image(new_images[3])

# trainset, testset = DataHandler.get_hand_drawn_digits(200, augment_data=True)
# x, y = next(iter(testset))
# grid = torchvision.utils.make_grid(x, pad_value=1)
# torchvision.utils.save_image(grid, "augmented.png")
