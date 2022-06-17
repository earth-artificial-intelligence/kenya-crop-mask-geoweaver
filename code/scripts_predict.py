from pathlib import Path
import sys
import os

sys.path.append("..")

from src_models_model import Model
from src_analysis import plot_results


def kenya_crop_type_mapper():
    data_dir = "../data"

    test_folder = Path("../data/raw/earth_engine_plant_village_kenya/")
    test_files = test_folder.glob("*.tif")
    print(test_files)

    list_of_models = list(Path('../data/lightning_logs/').glob('version*/checkpoints/*.ckpt'))
    latest_model_path = str(max(list_of_models, key=os.path.getctime))
    print(f"Using model {latest_model_path}")

    model = Model.load_from_checkpoint(latest_model_path)

    for test_path in test_files:

        save_dir = Path(data_dir) / "Autoencoder"
        save_dir.mkdir(exist_ok=True)

        print(f"Running for {test_path}")

        savepath = save_dir / f"preds_{test_path.name}"
        if savepath.exists():
            print("File already generated. Skipping")
            continue

        out_forecasted = model.predict(test_path, with_forecaster=True)
        plot_results(out_forecasted, test_path, savepath=save_dir, prefix="forecasted_")

        out_normal = model.predict(test_path, with_forecaster=False)
        plot_results(out_normal, test_path, savepath=save_dir, prefix="full_input_")

        out_forecasted.to_netcdf(save_dir / f"preds_forecasted_{test_path.name}.nc")
        out_normal.to_netcdf(save_dir / f"preds_normal_{test_path.name}.nc")


if __name__ == "__main__":
    print("Starting...predict.py")
    kenya_crop_type_mapper()

