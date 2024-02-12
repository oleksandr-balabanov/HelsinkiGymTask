import logging
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculate common regression metrics between true values and predictions.

    Parameters:
    - y_true (array-like): Ground truth (correct) target values with shape (n_samples,) or (n_samples, n_outputs).
    - y_pred (array-like): Estimated target values with shape (n_samples,) or (n_samples, n_outputs).

    Returns:
    - dict: Dictionary containing MSE, RMSE, MAE, and R^2 score.
    """
    try:
        if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
            raise ValueError("y_true and y_pred must be numpy arrays.")
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            "Mean Squared Error (MSE)": mse,
            "Root Mean Squared Error (RMSE)": rmse,
            "Mean Absolute Error (MAE)": mae,
            "R^2 Score": r2
        }

        for metric, value in metrics.items():
            logging.info(f"{metric}: {value}")

        return metrics
    except Exception as e:
        logging.error(f"Failed to calculate metrics: {e}")
        return {}

def evaluate_predictions(model, X_test, y_test):
    """
    Evaluate the model on the test set and return metrics.

    Parameters:
    - model: The trained model for predictions.
    - X_test (array-like): Test samples with shape (n_samples, n_features).
    - y_test (array-like): True values for X_test with shape (n_samples,) or (n_samples, n_outputs).

    Returns:
    - dict: Dictionary containing calculated metrics.
    """
    try:
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        return metrics
    except Exception as e:
        logging.error(f"Failed to evaluate predictions: {e}")
        return {}