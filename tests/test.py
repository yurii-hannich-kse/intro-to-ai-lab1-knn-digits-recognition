import pytest
import numpy as np
import sys
from unittest.mock import MagicMock

# --- OPTIMIZATION: MOCK SLOW LIBRARIES ---
# We mock 'umap' so that when the notebook imports it and runs
# fit_transform, it takes 0 seconds instead of 1 minute.
sys.modules["umap"] = MagicMock()

from sklearn.neighbors import KNeighborsClassifier
from grader_utils import load_notebook_functions


# --- FIXTURE: LOAD NOTEBOOK ONCE ---
@pytest.fixture(scope="module")
def hw():
    """
    Loads the notebook only once for the entire test session.
    """
    # This executes the notebook.
    # Because we mocked umap above, the heavy visualization cells will run instantly.
    return load_notebook_functions("lab.ipynb")


# --- TESTS ---

def test_flatten_images(hw):
    """Task 1: Flatten images from (N, 28, 28) to (N, 784)."""
    if not hasattr(hw, 'flatten_images'):
        pytest.fail("Function flatten_images not found!")

    N = 10
    X_input = np.random.rand(N, 28, 28)
    X_flat = hw.flatten_images(X_input)

    assert X_flat.shape == (N, 784)
    assert np.isclose(X_input.sum(), X_flat.sum())


def test_normalize_data(hw):
    """Task 2: Normalize data from [0, 255] to [0, 1]."""
    if not hasattr(hw, 'normalize_data'):
        pytest.fail("Function normalize_data not found!")

    X_input = np.array([[0, 127.5, 255], [255, 0, 10]])
    X_norm = hw.normalize_data(X_input)

    assert X_norm.max() <= 1.0
    assert X_norm.min() >= 0.0
    expected = np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 10 / 255]])
    np.testing.assert_allclose(X_norm, expected, atol=1e-5)


def test_train_and_evaluate(hw):
    """Task 3: Train KNN and evaluate."""
    if not hasattr(hw, 'train_and_evaluate'):
        pytest.fail("Function train_and_evaluate not found!")

    X_train = np.array([[0, 0], [0.1, 0.1], [10, 10], [10.1, 10.1]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.05, 0.05], [10.05, 10.05]])
    y_test = np.array([0, 1])

    try:
        model, train_acc, test_acc = hw.train_and_evaluate(X_train, y_train, X_test, y_test, k=1)
    except ValueError:
        pytest.fail("Function train_and_evaluate should return 3 values")

    assert isinstance(model, KNeighborsClassifier)
    assert train_acc == 1.0


def test_find_best_k(hw):
    """Task 4: Hyperparameter tuning."""
    if not hasattr(hw, 'find_best_k'):
        pytest.fail("Function find_best_k not found!")

    X_train = np.array([[0, 0], [2, 2], [2.1, 2.1], [2.2, 2.2]])
    y_train = np.array([0, 1, 1, 1])
    X_test = np.array([[0.1, 0.1]])
    y_test = np.array([0])

    # We pass a mocked pyplot so it doesn't try to generate a UI
    # Note: We can't easily stop the student's plt.show() from running,
    # but since we are in a headless CI environment, it usually just passes.
    best_k = hw.find_best_k(X_train, y_train, X_test, y_test, [1, 3])

    assert best_k == 1


def test_shift_image_flat(hw):
    """Task 5: Pixel shift."""
    if not hasattr(hw, 'shift_image_flat'):
        pytest.fail("Function shift_image_flat not found!")

    img_2d = np.zeros((28, 28))
    img_2d[:, 10] = 1
    img_flat = img_2d.flatten()

    shifted_flat = hw.shift_image_flat(img_flat, shift_amount=2)
    shifted_2d = shifted_flat.reshape(28, 28)

    assert shifted_2d[:, 12].sum() == 28
    assert shifted_2d[:, 10].sum() == 0