
from pathlib import Path
from typing import Any, Dict
import urllib.request
import zipfile

class BaseExporter:
    r"""Base for all exporter classes. It creates the appropriate
    directory in the data dir (``data_dir/raw/{dataset}``).

    All classes which extend this should implement an export function.

    :param data_folder (pathlib.Path, optional)``: The location of the data folder.
            Default: ``pathlib.Path("data")``
    """

    dataset: str
    default_args_dict: Dict[str, Any] = {}

    def __init__(self, data_folder: Path = Path("data")) -> None:

        self.data_folder = data_folder

        self.raw_folder = self.data_folder / "raw"
        self.output_folder = self.raw_folder / self.dataset
        self.output_folder.mkdir(parents=True, exist_ok=True)



class GeoWikiExporter(BaseExporter):
    r"""
    Download the GeoWiki labels
    """

    dataset = "geowiki_landcover_2017"

    download_urls = [
        "http://store.pangaea.de/Publications/See_2017/crop_all.zip",
        "http://store.pangaea.de/Publications/See_2017/crop_con.zip",
        "http://store.pangaea.de/Publications/See_2017/crop_exp.zip",
        "http://store.pangaea.de/Publications/See_2017/loc_all.zip",
        "http://store.pangaea.de/Publications/See_2017/loc_all_2.zip",
        "http://store.pangaea.de/Publications/See_2017/loc_con.zip",
        "http://store.pangaea.de/Publications/See_2017/loc_exp.zip",
    ]

    @staticmethod
    def download_file(url: str, output_folder: Path, remove_zip: bool = True) -> None:

        filename = url.split("/")[-1]
        output_path = output_folder / filename

        if output_path.exists():
            print(f"{filename} already exists! Skipping")
            return None

        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, output_path)

        if filename.endswith("zip"):

            print(f"Downloaded! Unzipping to {output_folder}")
            with zipfile.ZipFile(output_path, "r") as zip_file:
                zip_file.extractall(output_folder)

            if remove_zip:
                print("Deleting zip file")
                (output_path).unlink()

    def export(self, remove_zip: bool = False) -> None:
        r"""
        Download the GeoWiki labels
        :param remove_zip: Whether to remove the zip file once it has been expanded
        """
        for file_url in self.download_urls:
            self.download_file(file_url, self.output_folder, remove_zip)

            
def export_geowiki():
    print("export_geowiki() will be executed now")
    exporter = GeoWikiExporter(Path("../data"))
    exporter.export()
    
export_geowiki()
