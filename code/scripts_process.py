import sys
from pathlib import Path

sys.path.append("..")

from src_processors_geowiki import *
from src_processors_kenya_non_crop import *
from src_processors_pv_kenya import *

def process_geowiki():
    processor = GeoWikiProcessor(Path("../data"))
    processor.process()


def process_plantvillage():
    processor = KenyaPVProcessor(Path("../data"))
    processor.process()


def process_kenya_noncrop():
    processor = KenyaNonCropProcessor(Path("../data"))
    processor.process()


if __name__ == "__main__":
    print("Starting...process.py")
    process_geowiki()
    process_plantvillage()
    #process_kenya_noncrop()

