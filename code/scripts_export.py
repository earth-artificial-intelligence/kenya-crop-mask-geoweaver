import sys
from pathlib import Path
from datetime import date
import os

sys.path.append("..")

from src_exporters_geowiki import *
from src_exporters_sentinel_geowiki import *
from src_exporters_sentinel_pv_kenya import *
from src_exporters_sentinel_kenya_non_crop import *
from src_exporters_sentinel_region import *
from src_exporters_sentinel_utils import *




def export_geowiki():
    if len(os.listdir('../data/raw/geowiki_landcover_2017')) == 0:
        exporter = GeoWikiExporter(Path("../data"))
        exporter.export()


def export_geowiki_sentinel_ee():
    if len(os.listdir('../data/raw/earth_engine_geowiki')) == 0:
        exporter = GeoWikiSentinelExporter(Path("../data"))
        exporter.export_for_labels(
            num_labelled_points=10, monitor=False, checkpoint=True)


def export_plant_village_sentinel_ee():
    if len(os.listdir('../data/raw/earth_engine_plant_village_kenya')) == 0:
        exporter = KenyaPVSentinelExporter(Path("../data"))
        exporter.export_for_labels(
            num_labelled_points=10, monitor=False, checkpoint=True)


def export_kenya_non_crop():
    if len(os.listdir('../data/raw/earth_engine_kenya_non_crop')) == 0:
        exporter = KenyaNonCropSentinelExporter(Path("../data"))
        exporter.export_for_labels(
            num_labelled_points=10, monitor=False, checkpoint=True)


def export_region():
    if len(os.listdir('../data/raw/earth_engine_region_busia_partial_slow_cloudfree')) == 0:
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
    print("starting export_geowiki()...")
    export_geowiki()
    print("Done export_geowiki()!")
    print("starting process_geowiki()...")
    #process_geowiki()
    print("Done process_geowiki()!")
    print("starting export_geowiki_sentinel_ee()...this could take a while")
    export_geowiki_sentinel_ee()
    print("Done export_geowiki_sentinel_ee()!")
    print("starting process_plantvillage()...")
    #process_plantvillage()
    print("Done process_plantvillage()!")
    print("starting export_plant_village_sentinel_ee()...")
    export_plant_village_sentinel_ee()
    print("Done export_plant_village_sentinel_ee()!")
    print("starting process_kenya_noncrop()...")
    #process_kenya_noncrop()
    print("Done process_kenya_noncrop()!")
    print("starting export_kenya_non_crop()...")
    #export_kenya_non_crop()
    print("Done export_kenya_non_crop()!")
    print("starting export_region()...")
    #export_region()
    print("Done export_region()!")

