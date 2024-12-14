import torch
from torch.utils.data import Dataset
import os
import numpy as np

class LoadDataset_from_numpy(Dataset):
    def __init__(self, np_dataset, window_size=15, step_size=15):
        """
        Initialize the dataset with sliding or fixed windows.
        
        Args:
            np_dataset (list): List of numpy file paths.
            window_size (int): Number of timesteps in each window.
            step_size (int): Number of timesteps to slide the window.
        """
        super(LoadDataset_from_numpy, self).__init__()

        # Load files
        X_train = np.load(np_dataset[0])["x"]
        y_train = np.load(np_dataset[0])["y"]

        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)["x"]))
            y_train = np.append(y_train, np.load(np_file)["y"])

        # Create windows
        self.x_data, self.y_data = self._create_windows(X_train, y_train, window_size, step_size)
        self.len = len(self.x_data)

        # Convert to torch tensors
        self.x_data = torch.from_numpy(self.x_data).float()
        self.y_data = torch.from_numpy(self.y_data).long()

        # Correct the shape to (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def _create_windows(self, X, y, window_size, step_size):
        """
        Create windows from the dataset with the specified window and step size.

        Args:
            X (numpy.ndarray): Input features of shape (num_samples, seq_len).
            y (numpy.ndarray): Corresponding labels of shape (num_samples,).
            window_size (int): Size of each window.
            step_size (int): Step size for sliding the window.

        Returns:
            numpy.ndarray: Windowed input features.
            numpy.ndarray: Windowed labels (majority label in the window).
        """
        X_windows = []
        y_windows = []
        for i in range(0, X.shape[0] - window_size + 1, step_size):
            X_window = X[i:i + window_size]  # Extract window
            y_window = y[i:i + window_size]  # Extract corresponding labels
            X_windows.append(X_window)
            # Assign the majority label in the window as the window's label
            y_windows.append(np.bincount(y_window).argmax())
        
        return np.array(X_windows), np.array(y_windows)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator_np(training_files, subject_files, batch_size, window_size=30, step_size=15):
    """
    Create data loaders for training and testing with windowing.

    Args:
        training_files (list): List of training numpy file paths.
        subject_files (list): List of testing numpy file paths.
        batch_size (int): Batch size for the data loader.
        window_size (int): Size of each window.
        step_size (int): Step size for sliding the window.

    Returns:
        train_loader, test_loader, counts: Training loader, testing loader, and class distribution counts.
    """
    train_dataset = LoadDataset_from_numpy(training_files, window_size=window_size, step_size=step_size)
    test_dataset = LoadDataset_from_numpy(subject_files, window_size=window_size, step_size=step_size)

    # Calculate class distribution for all labels
    all_ys = np.concatenate((train_dataset.y_data, test_dataset.y_data))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader, counts
