from logger import logger
from predictor import predict_usage  # Assuming predict_usage is a function in predictor.py

def predict_pipeline(data):
    """
    Applies a pretrained model to the data to make predictions.
    
    :param data: pandas DataFrame, The cleaned and transformed dataset.
    :return: pandas DataFrame, The dataset with added prediction columns.
    """
    logger.info("Starting prediction pipeline")

    try:
        # Make predictions
        logger.info("Making predictions on the dataset")
        predictions = predict_usage(data)  # Adapt this call to match your predictor.py function signature

        # Merge predictions back to the dataset
        logger.info("Merging predictions with the original dataset")
        data_with_predictions = data.copy()
        data_with_predictions['predictions'] = predictions  # Adjust as necessary based on your model's output

        logger.info("Prediction pipeline completed successfully")
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}")
        raise

    return data_with_predictions
