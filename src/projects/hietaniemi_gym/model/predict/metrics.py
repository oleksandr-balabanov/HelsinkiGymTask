from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculate and print out common regression metrics between true values and predictions.

    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs), Ground truth (correct) target values.
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs), Estimated target values.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")

def evaluate_predictions(model, X_test, y_test):
    """
    Evaluate the model on the test set and print out metrics.

    :param model: The trained model that will make predictions.
    :param X_test: array-like of shape (n_samples, n_features), Test samples.
    :param y_test: array-like of shape (n_samples,) or (n_samples, n_outputs), True values for X_test.
    """
    y_pred = model.predict(X_test)
    calculate_metrics(y_test, y_pred)
