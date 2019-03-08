import math

from skimage import img_as_ubyte, measure
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_dilation, disk, convex_hull_image
import datetime

from scipy.spatial.distance import pdist

import numpy as np
import cv2 as cv
import morphocut.processing.functional as F
from morphocut.processing.pipeline import NodeBase
from morphocut.server.helpers import Timer


class ExtractRegions(NodeBase):
    def __init__(self, mask_facet, image_facets, output_facet, min_object_area=0):
        self.mask_facet = mask_facet
        self.image_facets = image_facets
        self.output_facet = output_facet
        self.min_object_area = min_object_area

    def __call__(self, input):
        for obj in input:
            mask = obj['facets']['binary_image']['image']
            intensity_img = data_object['facets']['corrected_data']['image']

            # Find connected components in the masked image to identify individual objects
            _, markers = cv.connectedComponents(mask)

            regionprops = [p for p in measure.regionprops(
                markers, intensity_image=intensity_img, coordinates='rc') if p.area > self.min_object_area]

			for i, p in enumerate(regionprops):

                facettes = {
                    self.output_facet: {
                        "data": self.regionprop2zooprocess(p)
                    }
                }

                # Adopt image facets that correspond to each object
                for facet_name in self.image_facets:
                    facettes[facet_name] = {
                        "image": obj[facet_name]["image"][p.slice]
                    }

               	new_obj = {
                    "object_id": '{}_{}'.format(obj['object_id'], i),
                    "facettes": facettes,
                }

                yield new_obj


    def regionprop2zooprocess(self, property):
        propDict = {
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
            # # X coordinate of the top left point of the image
            # 'object_xstart': data_object['raw_img']['meta']['xstart'],
            # # Y coordinate of the top left point of the image
            # 'object_ystart': data_object['raw_img']['meta']['ystart'],
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
