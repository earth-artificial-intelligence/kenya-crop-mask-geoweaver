import pytorch_lightning as pl
from pathlib import Path
from argparse import ArgumentParser
import os

import sys

sys.path.append("..")
from src_models_model import Model


def get_checkpoint(data_folder: Path) -> str:

    log_folder = data_folder / "lightning_logs/" 
    list_of_checkpoints = list(log_folder.glob('version*/checkpoints/*.ckpt'))
    print(log_folder.absolute())
    return str(max(list_of_checkpoints, key=os.path.getctime))


def test_model():
    parser = ArgumentParser()

    parser.add_argument("--version", type=int, default=0)

    args = parser.parse_args()

    model_path = get_checkpoint(Path("../data"))

    print(f"Using model {model_path}")
	
    model = Model.load_from_checkpoint(model_path)

    trainer = pl.Trainer()
    trainer.test(model)


if __name__ == "__main__":
    print("Starting...test.py")
    test_model()

