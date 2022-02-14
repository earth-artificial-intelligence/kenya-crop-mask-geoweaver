import sys
import subprocess
import pkg_resources

# Required packages to run this process.
required = {'pandas', 'numpy', 'xarray', 'pytorch-lightning', 'tqdm', 'earthengine-api'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print("Packages missing and will be installed: ", missing)
    python = sys.executable
    subprocess.check_call(
        [python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

################################
#  END OF PACKAGES VALIDATION  #
################################

'''
This process requires Google Earth Engine creds on machine running it.

This process requires Python 3.7+

'''
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import xarray as xr
from datetime import date, timedelta
import torch
from tqdm import tqdm
from pathlib import Path
import urllib.request
import zipfile
import random
from typing import Optional, List, Any, Dict, Union, Tuple
from dataclasses import dataclass
from math import cos, radians
import math
import ee

####################
#   constants.py   #
####################

# These are algorithm settings for the cloud filtering algorithm
image_collection = "COPERNICUS/S2"

# Ranges from 0-1.Lower value will mask more pixels out.
# Generally 0.1-0.3 works well with 0.2 being used most commonly
cloudThresh = 0.2
# Height of clouds to use to project cloud shadows
cloudHeights = [200, 10000, 250]
# Sum of IR bands to include as shadows within TDOM and the
# shadow shift method (lower number masks out less)
irSumThresh = 0.3
ndviThresh = -0.1
# Pixels to reduce cloud mask and dark shadows by to reduce inclusion
# of single-pixel comission errors
erodePixels = 1.5
dilationPixels = 3

# images with less than this many cloud pixels will be used with normal
# mosaicing (most recent on top)
cloudFreeKeepThresh = 3

BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]

###########################
#   END of constants.py   #
###########################



####################
#   src/utils.py   #
####################
@dataclass
class BoundingBox:

    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
      
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
###########################
#   END of src/utils.py   #
###########################





################
#   utils.py   #
################
@dataclass
class EEBoundingBox(BoundingBox):
    r"""
    A bounding box with additional earth-engine specific
    functionality
    """

    def to_ee_polygon(self) -> ee.Geometry.Polygon:
        return ee.Geometry.Polygon(
            [
                [
                    [self.min_lon, self.min_lat],
                    [self.min_lon, self.max_lat],
                    [self.max_lon, self.max_lat],
                    [self.max_lon, self.min_lat],
                ]
            ]
        )

    def to_metres(self) -> Tuple[float, float]:
        r"""
        :return: [lat metres, lon metres]
        """
        # https://gis.stackexchange.com/questions/75528/understanding-terms-in-length-of-degree-formula
        mid_lat = (self.min_lat + self.max_lat) / 2.0
        m_per_deg_lat, m_per_deg_lon = metre_per_degree(mid_lat)

        delta_lat = self.max_lat - self.min_lat
        delta_lon = self.max_lon - self.min_lon

        return delta_lat * m_per_deg_lat, delta_lon * m_per_deg_lon

    def to_polygons(self, metres_per_patch: int = 3300) -> List[ee.Geometry.Polygon]:

        lat_metres, lon_metres = self.to_metres()

        num_cols = int(lon_metres / metres_per_patch)
        num_rows = int(lat_metres / metres_per_patch)

        print(f"Splitting into {num_cols} columns and {num_rows} rows")

        lon_size = (self.max_lon - self.min_lon) / num_cols
        lat_size = (self.max_lat - self.min_lat) / num_rows

        output_polygons: List[ee.Geometry.Polygon] = []

        cur_lon = self.min_lon
        while cur_lon < self.max_lon:
            cur_lat = self.min_lat
            while cur_lat < self.max_lat:
                output_polygons.append(
                    ee.Geometry.Polygon(
                        [
                            [
                                [cur_lon, cur_lat],
                                [cur_lon, cur_lat + lat_size],
                                [cur_lon + lon_size, cur_lat + lat_size],
                                [cur_lon + lon_size, cur_lat],
                            ]
                        ]
                    )
                )
                cur_lat += lat_size
            cur_lon += lon_size

        return output_polygons

def metre_per_degree(mid_lat: float) -> Tuple[float, float]:
    # https://gis.stackexchange.com/questions/75528/understanding-terms-in-length-of-degree-formula
    # see the link above to explain the magic numbers
    m_per_deg_lat = 111132.954 - 559.822 * cos(2.0 * mid_lat) + 1.175 * cos(radians(4.0 * mid_lat))
    m_per_deg_lon = (3.14159265359 / 180) * 6367449 * cos(radians(mid_lat))

    return m_per_deg_lat, m_per_deg_lon

def bounding_box_from_centre(
    mid_lat: float, mid_lon: float, surrounding_metres: Union[int, Tuple[int, int]]
) -> EEBoundingBox:

    m_per_deg_lat, m_per_deg_lon = metre_per_degree(mid_lat)

    if isinstance(surrounding_metres, int):
        surrounding_metres = (surrounding_metres, surrounding_metres)

    surrounding_lat, surrounding_lon = surrounding_metres

    deg_lat = surrounding_lat / m_per_deg_lat
    deg_lon = surrounding_lon / m_per_deg_lon

    max_lat, min_lat = mid_lat + deg_lat, mid_lat - deg_lat
    max_lon, min_lon = mid_lon + deg_lon, mid_lon - deg_lon

    return EEBoundingBox(max_lon=max_lon, min_lon=min_lon, max_lat=max_lat, min_lat=min_lat)
#######################
#   END of utils.py   #
#######################
  
  
  


##########################
#   cloudfree/utils.py   #
##########################
def combine_bands(current, previous):
    # Transforms an Image Collection with 1 band per Image into a single Image with items as bands
    # Author: Jamie Vleeshouwer

    # Rename the band
    previous = ee.Image(previous)
    current = current.select(BANDS)
    # Append it to the result (Note: only return current item on first element/iteration)
    return ee.Algorithms.If(
        ee.Algorithms.IsEqual(previous, None), current, previous.addBands(ee.Image(current)),
    )


def export(
    image: ee.Image, region: ee.Geometry, filename: str, drive_folder: str, monitor: bool = False,
) -> ee.batch.Export:

    task = ee.batch.Export.image(
        image.clip(region),
        filename,
        {"scale": 10, "region": region, "maxPixels": 1e13, "driveFolder": drive_folder},
    )

    try:
        task.start()
    except ee.ee_exception.EEException as e:
        print(f"Task not started! Got exception {e}")
        return task

    if monitor:
        monitor_task(task)

    return task


def date_to_string(input_date: Union[date, str]) -> str:
    if isinstance(input_date, str):
        return input_date
    else:
        assert isinstance(input_date, date)
        return input_date.strftime("%Y-%m-%d")


def monitor_task(task: ee.batch.Export) -> None:

    while task.status()["state"] in ["READY", "RUNNING"]:
        print(task.status())
        # print(f"Running: {task.status()['state']}")


def rescale(img, exp, thresholds):
    return (
        img.expression(exp, {"img": img})
        .subtract(thresholds[0])
        .divide(thresholds[1] - thresholds[0])
    )
#################################
#   END of cloudfree/utils.py   #
#################################
  
  
  
  
####################
#   cloudfree.py   #
####################
def calcCloudStats(img):
    imgPoly = ee.Algorithms.GeometryConstructors.Polygon(
        ee.Geometry(img.get("system:footprint")).coordinates()
    )

    roi = ee.Geometry(img.get("ROI"))

    intersection = roi.intersection(imgPoly, ee.ErrorMargin(0.5))
    cloudMask = img.select(["cloudScore"]).gt(cloudThresh).clip(roi).rename("cloudMask")

    cloudAreaImg = cloudMask.multiply(ee.Image.pixelArea())

    stats = cloudAreaImg.reduceRegion(
        **{"reducer": ee.Reducer.sum(), "geometry": roi, "scale": 10, "maxPixels": 1e12}
    )

    cloudPercent = ee.Number(stats.get("cloudMask")).divide(imgPoly.area()).multiply(100)
    coveragePercent = ee.Number(intersection.area()).divide(roi.area()).multiply(100)
    cloudPercentROI = ee.Number(stats.get("cloudMask")).divide(roi.area()).multiply(100)

    img = img.set("CLOUDY_PERCENTAGE", cloudPercent)
    img = img.set("ROI_COVERAGE_PERCENT", coveragePercent)
    img = img.set("CLOUDY_PERCENTAGE_ROI", cloudPercentROI)

    return img


def computeQualityScore(img):
    score = img.select(["cloudScore"]).max(img.select(["shadowScore"]))

    score = score.reproject("EPSG:4326", None, 20).reduceNeighborhood(
        **{"reducer": ee.Reducer.mean(), "kernel": ee.Kernel.square(5)}
    )

    score = score.multiply(-1)

    return img.addBands(score.rename("cloudShadowScore"))


def computeS2CloudScore(img):
    toa = img.select(
        ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12",]
    ).divide(10000)

    toa = toa.addBands(img.select(["QA60"]))

    # ['QA60', 'B1','B2',    'B3',    'B4',   'B5','B6','B7', 'B8','  B8A',
    #  'B9',          'B10', 'B11','B12']
    # ['QA60','cb', 'blue', 'green', 'red', 're1','re2','re3','nir', 'nir2',
    #  'waterVapor', 'cirrus','swir1', 'swir2']);

    # Compute several indicators of cloudyness and take the minimum of them.
    score = ee.Image(1)

    # Clouds are reasonably bright in the blue and cirrus bands.
    score = score.min(rescale(toa, "img.B2", [0.1, 0.5]))
    score = score.min(rescale(toa, "img.B1", [0.1, 0.3]))
    score = score.min(rescale(toa, "img.B1 + img.B10", [0.15, 0.2]))

    # Clouds are reasonably bright in all visible bands.
    score = score.min(rescale(toa, "img.B4 + img.B3 + img.B2", [0.2, 0.8]))

    # Clouds are moist
    ndmi = img.normalizedDifference(["B8", "B11"])
    score = score.min(rescale(ndmi, "img", [-0.1, 0.1]))

    # However, clouds are not snow.
    ndsi = img.normalizedDifference(["B3", "B11"])
    score = score.min(rescale(ndsi, "img", [0.8, 0.6]))

    # Clip the lower end of the score
    score = score.max(ee.Image(0.001))

    # score = score.multiply(dilated)
    score = score.reduceNeighborhood(
        **{"reducer": ee.Reducer.mean(), "kernel": ee.Kernel.square(5)}
    )

    return img.addBands(score.rename("cloudScore"))


def projectShadows(image):
    meanAzimuth = image.get("MEAN_SOLAR_AZIMUTH_ANGLE")
    meanZenith = image.get("MEAN_SOLAR_ZENITH_ANGLE")

    cloudMask = image.select(["cloudScore"]).gt(cloudThresh)

    # Find dark pixels
    darkPixelsImg = image.select(["B8", "B11", "B12"]).divide(10000).reduce(ee.Reducer.sum())

    ndvi = image.normalizedDifference(["B8", "B4"])
    waterMask = ndvi.lt(ndviThresh)

    darkPixels = darkPixelsImg.lt(irSumThresh)

    # Get the mask of pixels which might be shadows excluding water
    darkPixelMask = darkPixels.And(waterMask.Not())
    darkPixelMask = darkPixelMask.And(cloudMask.Not())

    # Find where cloud shadows should be based on solar geometry
    # Convert to radians
    azR = ee.Number(meanAzimuth).add(180).multiply(math.pi).divide(180.0)
    zenR = ee.Number(meanZenith).multiply(math.pi).divide(180.0)

    # Find the shadows
    def getShadows(cloudHeight):
        cloudHeight = ee.Number(cloudHeight)

        shadowCastedDistance = zenR.tan().multiply(cloudHeight)  # Distance shadow is cast
        x = azR.sin().multiply(shadowCastedDistance).multiply(-1)  # /X distance of shadow
        y = azR.cos().multiply(shadowCastedDistance).multiply(-1)  # Y distance of shadow
        return image.select(["cloudScore"]).displace(
            ee.Image.constant(x).addBands(ee.Image.constant(y))
        )

    shadows = ee.List(cloudHeights).map(getShadows)
    shadowMasks = ee.ImageCollection.fromImages(shadows)
    shadowMask = shadowMasks.mean()

    # Create shadow mask
    shadowMask = dilatedErossion(shadowMask.multiply(darkPixelMask))

    shadowScore = shadowMask.reduceNeighborhood(
        **{"reducer": ee.Reducer.max(), "kernel": ee.Kernel.square(1)}
    )

    image = image.addBands(shadowScore.rename(["shadowScore"]))

    return image


def dilatedErossion(score):
    # Perform opening on the cloud scores
    score = (
        score.reproject("EPSG:4326", None, 20)
        .focal_min(**{"radius": erodePixels, "kernelType": "circle", "iterations": 3})
        .focal_max(**{"radius": dilationPixels, "kernelType": "circle", "iterations": 3})
        .reproject("EPSG:4326", None, 20)
    )

    return score


def mergeCollection(imgC):
    # Select the best images, which are below the cloud free threshold, sort them in reverse order
    # (worst on top) for mosaicing
    best = imgC.filterMetadata("CLOUDY_PERCENTAGE", "less_than", cloudFreeKeepThresh).sort(
        "CLOUDY_PERCENTAGE", False
    )
    filtered = imgC.qualityMosaic("cloudShadowScore")

    # Add the quality mosaic to fill in any missing areas of the ROI which aren't covered by good
    # images
    newC = ee.ImageCollection.fromImages([filtered, best.mosaic()])

    return ee.Image(newC.mosaic())

def get_single_image_fast(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:

    dates = ee.DateRange(date_to_string(start_date), date_to_string(end_date),)

    startDate = ee.DateRange(dates).start()
    endDate = ee.DateRange(dates).end()
    imgC = ee.ImageCollection(image_collection).filterDate(startDate, endDate).filterBounds(region)

    imgC = (
        imgC.map(lambda x: x.clip(region))
        .map(lambda x: x.set("ROI", region))
        .map(computeS2CloudScore)
        .map(projectShadows)
        .map(computeQualityScore)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    cloudFree = mergeCollection(imgC)

    return cloudFree
  
def get_single_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:

    dates = ee.DateRange(date_to_string(start_date), date_to_string(end_date),)

    startDate = ee.DateRange(dates).start()
    endDate = ee.DateRange(dates).end()
    imgC = ee.ImageCollection(image_collection).filterDate(startDate, endDate).filterBounds(region)

    imgC = (
        imgC.map(lambda x: x.clip(region))
        .map(lambda x: x.set("ROI", region))
        .map(computeS2CloudScore)
        .map(calcCloudStats)
        .map(projectShadows)
        .map(computeQualityScore)
        .sort("CLOUDY_PERCENTAGE")
    )

    cloudFree = mergeCollection(imgC)

    return cloudFree

###########################
#   END of cloudfree.py   #
###########################
  
  
  

#########################
#   exporters/base.py   #
#########################
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
################################
#   END of exporters/base.py   #
################################




##################################
#   exporters/sentinel/base.py   #
##################################
class BaseSentinelExporter(BaseExporter, ABC):

    r"""
    Download cloud free sentinel data for countries,
    where countries are defined by the simplified large scale
    international boundaries.
    """

    dataset: str
    min_date = date(2017, 3, 28)

    def __init__(self, data_folder: Path = Path("data")) -> None:
        super().__init__(data_folder)
        try:
            ee.Initialize()
        except Exception:
            print("This code doesn't work unless you have authenticated your earthengine account")

        self.labels = self.load_labels()

    @abstractmethod
    def load_labels(self) -> pd.DataFrame:
        raise NotImplementedError

    def _export_for_polygon(
        self,
        polygon: ee.Geometry.Polygon,
        polygon_identifier: Union[int, str],
        start_date: date,
        end_date: date,
        days_per_timestep: int,
        checkpoint: bool,
        monitor: bool,
        fast: bool,
    ) -> None:

        if fast:
            export_func = get_single_image_fast
        else:
            export_func = get_single_image

        cur_date = start_date
        cur_end_date = cur_date + timedelta(days=days_per_timestep)

        image_collection_list: List[ee.Image] = []

        print(
            f"Exporting image for polygon {polygon_identifier} from "
            f"aggregated images between {str(cur_date)} and {str(end_date)}"
        )
        filename = f"{polygon_identifier}_{str(cur_date)}_{str(end_date)}"

        if checkpoint and (self.output_folder / f"{filename}.tif").exists():
            print("File already exists! Skipping")
            return None

        while cur_end_date <= end_date:

            image_collection_list.append(
                export_func(region=polygon, start_date=cur_date, end_date=cur_end_date)
            )
            cur_date += timedelta(days=days_per_timestep)
            cur_end_date += timedelta(days=days_per_timestep)

        # now, we want to take our image collection and append the bands into a single image
        imcoll = ee.ImageCollection(image_collection_list)
        img = ee.Image(imcoll.iterate(combine_bands))

        # and finally, export the image
        export(
            image=img,
            region=polygon,
            filename=filename,
            drive_folder=self.dataset,
            monitor=monitor,
        )
#########################################
#   END of exporters/sentinel/base.py   #
#########################################





############################
#   exporters/geowiki.py   #
############################
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
###################################
#   END of exporters/geowiki.py   #
###################################




#####################################
#   exporters/sentinel/geowiki.py   #
#####################################
class GeoWikiSentinelExporter(BaseSentinelExporter):

    dataset = "earth_engine_geowiki"

    def load_labels(self) -> pd.DataFrame:
        # right now, this just loads geowiki data. In the future,
        # it would be neat to merge all labels together
        geowiki = self.data_folder / "processed" / GeoWikiExporter.dataset / "data.nc"
        assert geowiki.exists(), "GeoWiki processor must be run to load labels"
        return xr.open_dataset(geowiki).to_dataframe().dropna().reset_index()

    def labels_to_bounding_boxes(
        self, num_labelled_points: Optional[int], surrounding_metres: int
    ) -> List[EEBoundingBox]:

        output: List[EEBoundingBox] = []

        for idx, row in tqdm(self.labels.iterrows()):
            output.append(
                bounding_box_from_centre(
                    mid_lat=row["lat"], mid_lon=row["lon"], surrounding_metres=surrounding_metres,
                )
            )

            if num_labelled_points is not None:
                if len(output) >= num_labelled_points:
                    return output
        return output

    def export_for_labels(
        self,
        days_per_timestep: int = 1,
        start_date: date = date(2017, 3, 28),
        end_date: date = date(2017, 3, 29),
        num_labelled_points: Optional[int] = None,
        surrounding_metres: int = 80,
        checkpoint: bool = True,
        monitor: bool = False,
        fast: bool = True,
    ) -> None:
        r"""
        Run the GeoWiki exporter. For each label, the exporter will export
        int( (end_date - start_date).days / days_per_timestep) timesteps of data,
        where each timestep consists of a mosaic of all available images within the
        days_per_timestep of that timestep.
        :param days_per_timestep: The number of days of data to use for each mosaiced image.
        :param start_date: The start data of the data export
        :param end_date: The end date of the data export
        :param num_labelled_points: (Optional) The number of labelled points to export.
        :param surrounding_metres: The number of metres surrounding each labelled point to export
        :param checkpoint: Whether or not to check in self.data_folder to see if the file has
            already been exported. If it has, skip it
        :param monitor: Whether to monitor each task until it has been run
        :param fast: Whether to use the faster cloudfree exporter. This function is considerably
            faster, but cloud artefacts can be more pronounced. Default = True
        """
        assert start_date >= self.min_date, f"Sentinel data does not exist before {self.min_date}"

        bounding_boxes_to_download = self.labels_to_bounding_boxes(
            num_labelled_points=num_labelled_points, surrounding_metres=surrounding_metres,
        )

        for idx, bounding_box in enumerate(bounding_boxes_to_download):
            self._export_for_polygon(
                polygon=bounding_box.to_ee_polygon(),
                polygon_identifier=idx,
                start_date=start_date,
                end_date=end_date,
                days_per_timestep=days_per_timestep,
                checkpoint=checkpoint,
                monitor=monitor,
                fast=fast,
            )
############################################
#   END of exporters/sentinel/geowiki.py   #
############################################




##############################
#   src/processors/base.py   #
##############################
class BaseProcessor:
    r"""Base for all processor classes. It creates the appropriate
    directory in the data dir (``data_dir/processed/{dataset}``).

    :param data_folder (pathlib.Path, optional)``: The location of the data folder.
            Default: ``pathlib.Path("data")``
    """

    dataset: str

    def __init__(self, data_folder: Path) -> None:

        set_seed()
        self.data_folder = data_folder
        self.raw_folder = self.data_folder / "raw" / self.dataset
        assert self.raw_folder.exists(), f"{self.raw_folder} does not exist!"

        self.output_folder = self.data_folder / "processed" / self.dataset
        self.output_folder.mkdir(exist_ok=True, parents=True)
#####################################
#   END of src/processors/base.py   #
#####################################
        
  
  
  
#################################
#   src/processors/geowiki.py   #
#################################
class GeoWikiProcessor(BaseProcessor):

    dataset = "geowiki_landcover_2017"

    def load_raw_data(self, participants: str) -> pd.DataFrame:

        participants_to_file_labels = {
            "all": "all",
            "students": "con",
            "experts": "exp",
        }

        file_label = participants_to_file_labels.get(participants, participants)
        assert (
            file_label in participants_to_file_labels.values()
        ), f"Unknown participant {file_label}"

        return pd.read_csv(
            self.raw_folder / f"loc_{file_label}{'_2' if file_label == 'all' else ''}.txt",
            sep="\t",
        )

    def process(self, participants: str = "all") -> None:

        location_data = self.load_raw_data(participants)

        # first, we find the mean sumcrop calculated per location
        mean_per_location = (
            location_data[["location_id", "sumcrop", "loc_cent_X", "loc_cent_Y"]]
            .groupby("location_id")
            .mean()
        )

        # then, we rename the columns
        mean_per_location = mean_per_location.rename(
            {"loc_cent_X": "lon", "loc_cent_Y": "lat", "sumcrop": "mean_sumcrop"},
            axis="columns",
            errors="raise",
        )
        # then, we turn it into an xarray with x and y as indices
        output_xr = (
            mean_per_location.reset_index().set_index(["lon", "lat"])["mean_sumcrop"].to_xarray()
        )

        # and save
        output_xr.to_netcdf(self.output_folder / "data.nc")
########################################
#   END of src/processors/geowiki.py   #
########################################



##########################
#   scripts/process.py   #
##########################
def process_geowiki():
    processor = GeoWikiProcessor(Path("../data"))
    processor.process()
#################################
#   END of scripts/process.py   #
#################################



#########################
#   scripts/export.py   #
#########################
def export_geowiki_sentinel_ee():
    exporter = GeoWikiSentinelExporter(Path("../data"))
    exporter.export_for_labels(
        num_labelled_points=10, monitor=False, checkpoint=True)
################################
#   END of scripts/export.py   #
################################
    
process_geowiki()
export_geowiki_sentinel_ee()
