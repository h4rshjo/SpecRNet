import argparse
import logging
from pathlib import Path

import torch
import yaml

import train_models
import evaluate_models
from src.commons import set_seed
from clearml import Task

task = Task.init(project_name="Thesis_ADD", task_name="Training Model - 128")

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
LOGGER.addHandler(ch)


def parse_args():
    parser = argparse.ArgumentParser()

    # Updated dataset paths
    BONAFIDE_DATASET_PATH = "./Bonafide_Dataset/train"
    FAKE_DATASET_PATH = "./Fake Dataset/train"

    parser.add_argument(
        "--bonafide_path",
        type=str,
        default=BONAFIDE_DATASET_PATH,
        help="Path to Bonafide dataset directory",
    )
    parser.add_argument(
        "--fake_path",
        type=str,
        default=FAKE_DATASET_PATH,
        help="Path to Fake dataset directory",
    )
    default_model_config = "./configs/training/specrnet.yaml"
    parser.add_argument(
        "--config",
        help="Model config file path (default: config.yaml)",
        type=str,
        default=default_model_config,
    )

    default_train_amount = None
    parser.add_argument(
        "--train_amount",
        "-a",
        help=f"Amount of files to load for training.",
        type=int,
        default=default_train_amount,
    )

    default_valid_amount = None
    parser.add_argument(
        "--valid_amount",
        "-va",
        help=f"Amount of files to load for validation.",
        type=int,
        default=default_valid_amount,
    )

    default_test_amount = None
    parser.add_argument(
        "--test_amount",
        "-ta",
        help=f"Amount of files to load for testing.",
        type=int,
        default=default_test_amount,
    )

    default_batch_size = 8
    parser.add_argument(
        "--batch_size",
        "-b",
        help=f"Batch size (default: {default_batch_size}).",
        type=int,
        default=default_batch_size,
    )

    default_epochs = 10  # it was 5 originally
    parser.add_argument(
        "--epochs",
        "-e",
        help=f"Epochs (default: {default_epochs}).",
        type=int,
        default=default_epochs,
    )

    default_model_dir = "trained_models"
    parser.add_argument(
        "--ckpt",
        help=f"Checkpoint directory (default: {default_model_dir}).",
        type=str,
        default=default_model_dir,
    )

    parser.add_argument("--cpu", "-c", help="Force using cpu?", action="store_true")

    return parser.parse_args()
task.close()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config["data"].get("seed", 42)
    # fix all seeds
    set_seed(seed)

    if not args.cpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_dir = Path(args.ckpt)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Adjusted for both bonafide and fake datasets

    ### THIS IS 1
    evaluation_config_path, model_path = train_models.train_nn(
        datasets_paths=[
            args.bonafide_path,
            args.fake_path,
        ],
        device=device,
        amount_to_use=(args.train_amount, args.valid_amount),
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_dir=model_dir,
        config=config,
    )

    with open(evaluation_config_path, "r") as f:
        config = yaml.safe_load(f)

    #evaluate_models.evaluate_nn(
        #model_paths=config["checkpoint"].get("path", []),
        #batch_size=args.batch_size,
        #datasets_paths=[args.bonafide_path, args.fake_path],  # Use both datasets during evaluation as well
        #model_config=config["model"],
        #amount_to_use=args.test_amount,
        #device=device,
    #)

    # After training, pass the model_path directly to the evaluation phase
    #evaluate_models.evaluate_nn(
        #model_paths=[model_path],  # Pass the trained model path here
        #batch_size=args.batch_size,
        #datasets_paths=[args.bonafide_path, args.fake_path],  # Use both datasets during evaluation as well
        #model_config=config["model"],
        #amount_to_use=args.test_amount,
        #device=device,
#)
    evaluate_models.evaluate_nn(
        model_paths=config["checkpoint"].get("path", []),
        batch_size=args.batch_size,
        datasets_paths=[args.bonafide_path, args.fake_path],  # Evaluate on both datasets
        model_config=config["model"],
        amount_to_use=args.test_amount,
        device=device,
    )
