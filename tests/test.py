import pytest
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from grader_utils import load_notebook_functions

# Load student's notebook
hw = load_notebook_functions("lab.ipynb")


def test_flatten_images():
    """Task 1: Flatten images from (N, 28, 28) to (N, 784)."""
    assert hasattr(hw, 'flatten_images'), "Function flatten_images not found!"

    # Create random batch of 10 images
    N = 10
    X_input = np.random.rand(N, 28, 28)

    X_flat = hw.flatten_images(X_input)

    # Check 1: Shape
    assert X_flat.shape == (N, 784), f"Expected shape ({N}, 784), got {X_flat.shape}"

    # Check 2: Data integrity (sum should be identical)
    assert np.isclose(X_input.sum(), X_flat.sum()), "Sum of pixel values changed after flattening!"

    # Check 3: Check specific value
    assert X_input[0, 5, 5] == X_flat[0, 5 * 28 + 5], "Pixel values got shuffled incorrectly!"


def test_normalize_data():
    """Task 2: Normalize data from [0, 255] to [0, 1]."""
    assert hasattr(hw, 'normalize_data'), "Function normalize_data not found!"

    # Create dummy data with specific boundaries
    X_input = np.array([[0, 127.5, 255], [255, 0, 10]])

    X_norm = hw.normalize_data(X_input)

    # Check 1: Bounds
    assert X_norm.max() <= 1.0, f"Max value {X_norm.max()} is > 1.0"
    assert X_norm.min() >= 0.0, f"Min value {X_norm.min()} is < 0.0"

    # Check 2: Specific calculations
    expected = np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 10 / 255]])
    np.testing.assert_allclose(X_norm, expected, atol=1e-5, err_msg="Normalization calculation incorrect")


def test_train_and_evaluate():
    """Task 3: Train KNN and evaluate accuracy (Updated for 3 return values)."""
    assert hasattr(hw, 'train_and_evaluate'), "Function train_and_evaluate not found!"

    # Create a simple toy dataset (2 separable clusters)
    # Cluster 1: Around (0,0), Cluster 2: Around (10,10)
    X_train = np.array([[0, 0], [0.1, 0.1], [10, 10], [10.1, 10.1]])
    y_train = np.array([0, 0, 1, 1])

    X_test = np.array([[0.05, 0.05], [10.05, 10.05]])
    y_test = np.array([0, 1])

    # UPDATED: Unpack 3 values (model, train_acc, test_acc)
    try:
        model, train_acc, test_acc = hw.train_and_evaluate(X_train, y_train, X_test, y_test, k=1)
    except ValueError:
        pytest.fail("Function train_and_evaluate should return 3 values: (model, train_accuracy, test_accuracy)")

    # Check 1: Returns a model
    assert isinstance(model, KNeighborsClassifier), "Did not return a KNeighborsClassifier instance"

    # Check 2: Returns valid accuracies
    assert isinstance(train_acc, float), "Train accuracy should be a float"
    assert isinstance(test_acc, float), "Test accuracy should be a float"

    # Check 3: Check values (should be perfect 1.0 for this simple case)
    assert train_acc == 1.0, f"Expected 1.0 train accuracy, got {train_acc}"
    assert test_acc == 1.0, f"Expected 1.0 test accuracy, got {test_acc}"


def test_shift_image_flat():
    """Task 5: Pixel shift with zero-padding logic."""
    assert hasattr(hw, 'shift_image_flat'), "Function shift_image_flat not found!"

    # Create a simple 28x28 image with a single vertical line at col 10
    img_2d = np.zeros((28, 28))
    img_2d[:, 10] = 1
    img_flat = img_2d.flatten()

    # Shift Right by 2
    shifted_flat = hw.shift_image_flat(img_flat, shift_amount=2)
    shifted_2d = shifted_flat.reshape(28, 28)

    # Check 1: The line should now be at col 12
    assert shifted_2d[:, 12].sum() == 28, "The vertical line did not move to the correct position"

    # Check 2: The old position should be 0
    assert shifted_2d[:, 10].sum() == 0, "Old position was not cleared"

    # Check 3: Check edge handling (zero padding)
    # Create image with line at far right (col 27)
    img_edge_2d = np.zeros((28, 28))
    img_edge_2d[:, 27] = 1
    img_edge_flat = img_edge_2d.flatten()

    # Shift Right by 1 -> Should disappear (or wrap if implemented simply,
    # but prompt asked for masking/handling wrap-around usually implies zeroing in this context
    # OR simple roll.
    # Based on the solution provided in chat, the logic explicitly zeroed out the wrap-around.
    shifted_edge = hw.shift_image_flat(img_edge_flat, shift_amount=1)
    shifted_edge_2d = shifted_edge.reshape(28, 28)

    # The wrapped pixels (now at col 0) should be 0
    assert shifted_edge_2d[:, 0].sum() == 0, "Wrap-around pixels were not zeroed out!"


def test_find_best_k():
    """Task 4: Hyperparameter tuning."""
    assert hasattr(hw, 'find_best_k'), "Function find_best_k not found!"

    # Dataset where k=1 is better than k=3
    # A point surrounded by noise of different class
    # Train:
    # Class 0: (0,0)
    # Class 1: (2,2), (2.1, 2.1), (2.2, 2.2)
    # Test: (0.1, 0.1) -> Should be Class 0 (nearest is (0,0))
    # If k=3, neighbors are (0,0) [Class 0] and two from Class 1 -> Predicts Class 1 (Wrong)

    X_train = np.array([[0, 0], [2, 2], [2.1, 2.1], [2.2, 2.2]])
    y_train = np.array([0, 1, 1, 1])

    X_test = np.array([[0.1, 0.1]])
    y_test = np.array([0])

    best_k = hw.find_best_k(X_train, y_train, X_test, y_test, [1, 3])

    assert best_k == 1, f"For this dataset, k=1 gives accuracy 1.0, k=3 gives 0.0. Expected best_k=1, got {best_k}"