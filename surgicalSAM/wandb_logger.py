import wandb


def initialize_wandb(project_name, config):
    """
    Initialize a new wandb run.

    :param project_name: Name of the wandb project.
    :param config: Dictionary containing run configurations.
    """
    wandb.init(
        project=project_name,
        config=config,
    )


def log_results(endovis_results):
    """
    Log the results to wandb.

    :param endovis_results: Dictionary containing the results to log.
    """
    # Split up the dict and create a new one for logging
    log_data = {
        "Validation IoU": endovis_results.get("IoU"),
        "Validation Challenge IoU": endovis_results.get("challengIoU"),
        "mcIoU": endovis_results.get("mcIoU"),
        "mIoU": endovis_results.get("mIoU"),
    }

    # Handle cIoU_per_class separately if it's a list
    cIoU_per_class = endovis_results.get("cIoU_per_class")
    if isinstance(cIoU_per_class, list):
        for idx, value in enumerate(cIoU_per_class):
            log_data[f"cIoU_class_{idx}"] = value

    # Log the results to wandb
    wandb.log(log_data)
