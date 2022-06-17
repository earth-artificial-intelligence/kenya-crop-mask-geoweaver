import sys
from pathlib import Path

sys.path.append("..")

from src_engineer_geowiki import GeoWikiEngineer
from src_engineer_pv_kenya import PVKenyaEngineer
from src_engineer_kenya_non_crop import KenyaNonCropEngineer


def engineer_geowiki():
    engineer = GeoWikiEngineer(Path("../data"))
    engineer.engineer(val_set_size=0.2)


def engineer_kenya():
    engineer = PVKenyaEngineer(Path("../data"))
    engineer.engineer(val_set_size=0.1, test_set_size=0.1)


def engineer_kenya_noncrop():
    engineer = KenyaNonCropEngineer(Path("../data"))
    engineer.engineer(val_set_size=0.1, test_set_size=0.1)


if __name__ == "__main__":
    print("Starting...engineer.py")  
    engineer_geowiki()
    engineer_kenya()
    #engineer_kenya_noncrop()
