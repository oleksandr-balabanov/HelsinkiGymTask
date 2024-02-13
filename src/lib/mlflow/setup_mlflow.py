import mlflow
import logging

def setup_mlflow(tracking_uri: str, experiment_name: str, artifact_location: str = None):
    """
    Sets up MLflow with a given tracking URI, experiment name, and optional artifact location.

    Parameters:
    - tracking_uri: The URI to the MLflow tracking server. If using a local file system, use 'file:///absolute/path/to/mlruns'.
    - experiment_name: The name of the experiment under which runs will be logged. If the experiment does not exist, it will be created.
    - artifact_location: Optional. A path to a directory where artifacts will be stored if you are using a file-based store. 
      If using a remote tracking server with remote storage (e.g., S3, Azure Blob Storage), configure the server accordingly and omit this parameter.

    Example usage:
    setup_mlflow("file:///absolute/path/to/mlruns", "My Experiment")
    """
    # Set the tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # Check if the experiment already exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        # Create the experiment if it does not exist and get the experiment ID
        experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)

    # Set the experiment to use for logging
    mlflow.set_experiment(experiment_name)

    logging.info(f"MLflow setup completed. Tracking URI: {tracking_uri}, Experiment ID: {experiment_id}")