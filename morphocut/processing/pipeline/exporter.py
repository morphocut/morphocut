from morphocut.processing.pipeline import NodeBase
import cv2 as cv
import pandas as pd
import os
from parse import *
import sys
from etaprogress.progress import ProgressBar
import matplotlib.pyplot as plt
import datetime
from io import StringIO
from zipfile import ZipFile
import zipfile
import morphocut.processing.functional as proc
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_dilation, disk
from skimage import img_as_ubyte
import math
import random
import string


class Exporter(NodeBase):
    """
    Ecotaxa Export

    Writes a .zip-Archive with the following entries:
        - ecotaxa_segmentation.csv
        - <objid>.png
        - <objid>_contours.png

    TODO: Make exported images configurable

    Input:

    {
        object_id: ...
        raw_img: {
            id: ...
            meta: {region props...},
            image: <np.array of shape = [h,w,c]>
        },
        contour_img: {
            image: <np.array of shape = [h,w,c]>
        }
    }

    """

    def __init__(self, archive_fn):
        self.archive_fn = archive_fn

    def __call__(self, input=None):
        with ZipFile(self.archive_fn, 'w', zipfile.ZIP_DEFLATED) as myzip:
            prop_list = []
            prop_list.append(self.get_property_column_types())

            for data_object in input:

                for key in data_object['export_keys']:

                    filename = "{}_{}.png".format(
                        data_object['object_id'], key)
                    img_str = cv.imencode('.png', data_object[key]['image'])[
                        1].tostring()
                    myzip.writestr(filename, img_str)
                    prop_list.append(
                        self.get_properties_list(data_object, filename, img_rank=data_object[key]['img_rank']))

                # filename = "{}.png".format(
                #     data_object['object_id'])
                # img_str = cv.imencode('.png', data_object['raw_img']['image'])[
                #     1].tostring()
                # myzip.writestr(filename, img_str)
                # prop_list.append(
                #     self.get_properties_list(data_object, filename, img_rank=1))

                # filename = "{}_{}.png".format(
                #     data_object['object_id'], 'contours')
                # img_str = cv.imencode('.png', data_object['contour_img']['image'])[
                #     1].tostring()
                # myzip.writestr(filename, img_str)
                # prop_list.append(
                #     self.get_properties_list(data_object, filename, img_rank=2))

            # Transform the list into a dataframe
            prop_frame = pd.DataFrame(prop_list)

            # Write the dataframe as a tsv file to the zip file
            sio = StringIO()
            filepath = 'ecotaxa_segmentation.tsv'
            prop_frame.to_csv(sio, sep='\t', encoding='utf-8', index=False)
            myzip.writestr(filepath, sio.getvalue())

        # Exporter has no output
        yield None

    def get_property_column_types(self):
        '''
        Returns the column types for the columns in the tsv file for the ecotaxa export
        '''
        propDict = {'img_file_name': '[t]',
                    'img_rank': '[f]',
                    'object_id': '[t]',
                    'object_date': '[t]',
                    'object_time': '[t]',
                    'object_width': '[f]',
                    'object_height': '[f]',
                    'object_bx': '[f]',
                    'object_by': '[f]',
                    'object_circ.': '[f]',
                    'object_area_exc': '[f]',
                    'object_area': '[f]',
                    'object_%area': '[f]',
                    'object_major': '[f]',
                    'object_minor': '[f]',
                    'object_y': '[f]',
                    'object_x': '[f]',
                    'object_convex_area': '[f]',
                    'object_min': '[f]',
                    'object_max': '[f]',
                    'object_mean': '[f]',
                    'object_intden': '[f]',
                    'object_perim.': '[f]',
                    'object_elongation': '[f]',
                    'object_range': '[f]',
                    'object_perimareaexc': '[f]',
                    'object_perimmajor': '[f]',
                    'object_circex': '[f]',
                    'object_angle': '[f]',
                    'object_xstart': '[f]',
                    'object_ystart': '[f]',
                    # 'object_feret': '[f]',
                    # 'object_feretareaexc': '[f]',
                    # 'object_perimferet': '[f]',

                    'object_bounding_box_area': '[f]',
                    'object_eccentricity': '[f]',
                    'object_equivalent_diameter': '[f]',
                    'object_euler_number': '[f]',
                    'object_extent': '[f]',
                    'object_local_centroid_row': '[f]',
                    'object_local_centroid_col': '[f]',
                    'object_solidity': '[f]',
                    }
        return propDict

    def get_properties_list(self, data_object, filename, img_rank):
        '''
        Transforms the region properties of the data object into the tsv export format.
        '''
        property = data_object['raw_img']['meta']['properties']
        propDict = {
            # filename of the exported object in the zip
            'img_file_name': filename,
            # rank of the image, in case there are multiple images for each object. starts at 1
            'img_rank': img_rank,
            # the object id of the exported object
            'object_id': data_object['object_id'],
            # date when the object was exported
            'object_date': str(datetime.datetime.now().strftime('%Y%m%d')),
            # time when the object was exported
            'object_time': str(datetime.datetime.now().strftime('%H%M%S')),
            # width of the smallest rectangle enclosing the object
            'object_width': property.bbox[3] - property.bbox[1],
            # height of the smallest rectangle enclosing the object
            'object_height': property.bbox[2] - property.bbox[0],
            # X coordinates of the top left point of the smallest rectangle enclosing the object
            'object_bx': property.bbox[1],
            # Y coordinates of the top left point of the smallest rectangle enclosing the object
            'object_by': property.bbox[0],
            # circularity : (4∗π ∗Area)/Perim^2 a value of 1 indicates a perfect circle, a value approaching 0 indicates an increasingly elongated polygon
            'object_circ.': (4 * math.pi * property.filled_area) / math.pow(property.perimeter, 2),
            # Surface area of the object excluding holes, in square pixels (=Area*(1-(%area/100))
            'object_area_exc': property.area,
            # Surface area of the object in square pixels
            'object_area': property.filled_area,
            # Percentage of object’s surface area that is comprised of holes, defined as the background grey level
            'object_%area': 1 - (property.area / property.filled_area),
            # Primary axis of the best tting ellipse for the object
            'object_major': property.major_axis_length,
            # Secondary axis of the best tting ellipse for the object
            'object_minor': property.minor_axis_length,
            # Y position of the center of gravity of the object
            'object_y': property.centroid[0],
            # X position of the center of gravity of the object
            'object_x': property.centroid[1],
            # The area of the smallest polygon within which all points in the objet t
            'object_convex_area': property.convex_area,
            # Minimum grey value within the object (0 = black)
            'object_min': property.min_intensity,
            # Maximum grey value within the object (255 = white)
            'object_max': property.max_intensity,
            # Average grey value within the object ; sum of the grey values of all pixels in the object divided by the number of pixels
            'object_mean': property.mean_intensity,
            # Integrated density. The sum of the grey values of the pixels in the object (i.e. = Area*Mean)
            'object_intden': property.filled_area * property.mean_intensity,
            # The length of the outside boundary of the object
            'object_perim.': property.perimeter,
            # major/minor
            'object_elongation': property.major_axis_length / property.minor_axis_length,
            # max-min
            'object_range': property.max_intensity - property.min_intensity,
            # perim/area_exc
            'object_perimareaexc': property.perimeter / property.area,
            # perim/major
            'object_perimmajor': property.perimeter / property.major_axis_length,
            # (4 ∗ π ∗ Area_exc)/perim 2
            'object_circex': (4 * math.pi * property.area) / math.pow(property.perimeter, 2),
            # Angle between the primary axis and a line parallel to the x-axis of the image
            'object_angle': property.orientation / math.pi * 180 + 90,
            # X coordinate of the top left point of the image
            'object_xstart': data_object['raw_img']['meta']['xstart'],
            # Y coordinate of the top left point of the image
            'object_ystart': data_object['raw_img']['meta']['ystart'],
            # Maximum feret diameter, i.e. the longest distance between any two points along the object boundary
            # 'object_feret': data_object['raw_img']['meta']['feret'],
            # feret/area_exc
            # 'object_feretareaexc': data_object['raw_img']['meta']['feret'] / property.area,
            # perim/feret
            # 'object_perimferet': property.perimeter / data_object['raw_img']['meta']['feret'],



            # object_bounding_box_area
            'object_bounding_box_area': property.bbox_area,
            # object_eccentricity
            'object_eccentricity': property.eccentricity,
            # object_equivalent_diameter
            'object_equivalent_diameter': property.equivalent_diameter,
            # object_euler_number
            'object_euler_number': property.euler_number,
            # object_extent
            'object_extent': property.extent,
            # object_local_centroid_row
            'object_local_centroid_row': property.local_centroid[0],
            # object_local_centroid_col
            'object_local_centroid_col': property.local_centroid[1],
            # object_solidity
            'object_solidity': property.solidity,
        }

        return propDict
