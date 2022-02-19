import sys
from pathlib import Path
from datetime import date

sys.path.append("..")

from src_exporters_geowiki import *
from src_exporters_sentinel_geowiki import *
from src_exporters_sentinel_pv_kenya import *
from src_exporters_sentinel_kenya_non_crop import *
from src_exporters_sentinel_region import *
from src_exporters_sentinel_utils import *

from scripts_process import *

def export_geowiki():
    exporter = GeoWikiExporter(Path("../data"))
    exporter.export()


def export_geowiki_sentinel_ee():
    exporter = GeoWikiSentinelExporter(Path("../data"))
    exporter.export_for_labels(
        num_labelled_points=None, monitor=False, checkpoint=True)


def export_plant_village_sentinel_ee():
    exporter = KenyaPVSentinelExporter(Path("../data"))
    exporter.export_for_labels(
        num_labelled_points=None, monitor=False, checkpoint=True)


def export_kenya_non_crop():
    exporter = KenyaNonCropSentinelExporter(Path("../data"))
    exporter.export_for_labels(
        num_labelled_points=None, monitor=False, checkpoint=True)


def export_region():
    exporter = RegionalExporter(Path("../data"))
    exporter.export_for_region(
        region_name="Busia",
        end_date=date(2020, 9, 13),
        num_timesteps=5,
        monitor=False,
        checkpoint=True,
        metres_per_polygon=None,
        fast=False,
    )


if __name__ == "__main__":
	export_geowiki()
	process_geowiki()
	export_geowiki_sentinel_ee()
	#export_plant_village_sentinel_ee()
	#export_kenya_non_crop()
	#export_region()

