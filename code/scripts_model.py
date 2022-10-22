import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append("..")

from src_models_model import Model
from src_models_train_funcs import train_model

"""
This requires: 
mypy
black
pandas
xarray
Netcdf4
rasterio
jupyter
matplotlib
pip
pytorch=1.6.0
pyproj
Tqdm
geopandas
cartopy
scikit-learn
pytorch-lightning==0.7.1
"""

if __name__ == "__main__":
    print("Starting...model.py")
    parser = ArgumentParser()

    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)

    model_args = Model.add_model_specific_args(parser).parse_args()
    model = Model(model_args)

    train_model(model, model_args)

