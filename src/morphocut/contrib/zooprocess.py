"""
Feature calculation like in ZooProcess.

    `Zooprocess`_ is a suite of routines in ImageJ macro language
    for Plankton image analysis.

.. _Zooprocess: https://sites.google.com/view/piqv/zooprocess
"""

from typing import Optional

import numpy as np
import skimage.io
import scipy
import scipy.spatial
import skimage.segmentation
from morphocut import Node, Output, RawOrVariable, ReturnOutputs


def regionprop2zooprocess(prop):
    """
    Calculate zooprocess features from skimage regionprops.

    Notes:
        - date/time specify the time of the sampling, not of the processing.
    """
    min_feret, max_feret = calculateFeret(prop.image)
    return {
        "label": prop.label,
        # width of the smallest rectangle enclosing the object
        "width": prop.bbox[3] - prop.bbox[1],
        # height of the smallest rectangle enclosing the object
        "height": prop.bbox[2] - prop.bbox[0],
        # X coordinates of the top left point of the smallest rectangle enclosing the object
        "bx": prop.bbox[1],
        # Y coordinates of the top left point of the smallest rectangle enclosing the object
        "by": prop.bbox[0],
        # circularity : (4∗π ∗Area)/Perim^2 a value of 1 indicates a perfect circle, a value approaching 0 indicates an increasingly elongated polygon
        "circ.": (4 * np.pi * prop.filled_area) / prop.perimeter**2,
        # Surface area of the object excluding holes, in square pixels (=Area*(1-(%area/100))
        "area_exc": prop.area,
        # Surface area of the object in square pixels
        "area": prop.filled_area,
        # Percentage of object’s surface area that is comprised of holes, defined as the background grey level
        "%area": 1 - (prop.area / prop.filled_area),
        # Primary axis of the best fitting ellipse for the object
        "major": prop.major_axis_length,
        # Secondary axis of the best fitting ellipse for the object
        "minor": prop.minor_axis_length,
        # Y position of the center of gravity of the object
        "y": prop.centroid[0],
        # X position of the center of gravity of the object
        "x": prop.centroid[1],
        # The area of the smallest polygon within which all points in the objet fit
        "convex_area": prop.convex_area,
        # Minimum grey value within the object (0 = black)
        "min": prop.min_intensity,
        # Maximum grey value within the object (255 = white)
        "max": prop.max_intensity,
        # Average grey value within the object ; sum of the grey values of all pixels in the object divided by the number of pixels
        "mean": prop.mean_intensity,
        # Integrated density. The sum of the grey values of the pixels in the object (i.e. = Area*Mean)
        "intden": prop.filled_area * prop.mean_intensity,
        # The length of the outside boundary of the object
        "perim.": prop.perimeter,
        # major/minor
        "elongation": np.divide(prop.major_axis_length,
                                prop.minor_axis_length),
        # max-min
        "range": prop.max_intensity - prop.min_intensity,
        # perim/area_exc
        "perimareaexc": prop.perimeter / prop.area,
        # perim/major
        "perimmajor": prop.perimeter / prop.major_axis_length,
        # (4 ∗ π ∗ Area_exc)/perim 2
        "circex": np.divide(4 * np.pi * prop.area, prop.perimeter**2),
        # Angle between the primary axis and a line parallel to the x-axis of the image
        "angle": prop.orientation / np.pi * 180 + 90,
        # # X coordinate of the top left point of the image
        # 'xstart': data_object['raw_img']['meta']['xstart'],
        # # Y coordinate of the top left point of the image
        # 'ystart': data_object['raw_img']['meta']['ystart'],
        # Maximum feret diameter, i.e. the longest distance between any two points along the object boundary
        'feret': max_feret,
        # feret/area_exc
        #'feretareaexc': prop.image / prop.area,
        # perim/feret
        #'perimferet': prop.perimeter / prop.image,
        "bounding_box_area": prop.bbox_area,
        "eccentricity": prop.eccentricity,
        "equivalent_diameter": prop.equivalent_diameter,
        "euler_number": prop.euler_number,
        "extent": prop.extent,
        "local_centroid_col": prop.local_centroid[1],
        "local_centroid_row": prop.local_centroid[0],
        "solidity": prop.solidity,
    }


def calculateFeret(image):
    boundaries = skimage.segmentation.find_boundaries(
        image, mode="inner")  #find the boundary
    border_coordinates = np.nonzero(boundaries)  #find the boundary coordinates
    min_diam, max_diam = calculate_diameter(border_coordinates)

    return min_diam, max_diam


def calculateConvexHulls(border_coordinates):
    #hulls = ConvexHull(border_coordinates)
    u, l = (border_coordinates)
    i = 0
    j = len(l) - 1
    while i < len(u) - 1 or j > 0:
        yield u[i], l[j]

        # if all the way through one side of hull, advance the other side
        if i == len(u) - 1: j -= 1
        elif j == 0:
            i += 1
        else:
            j -= 1

            # still points left on both lists, compare slopes of next hull edges
        '''elif (u[i+1][1]-u[i][1])*(l[j][0]-l[j-1][0]) > \
                (l[j][1]-l[j-1][1])*(u[i+1][0]-u[i][0]):
            i += 1
            '''


def calculate_diameter(border_coordinates):
    #Calculate the maximum feret diameter
    '''max_diam, pair = max([((p[0] - q[0])**2 + (p[1] - q[1])**2, (p, q))
                          for p, q in calculateConvexHulls(border_coordinates)
                          ])
    '''
    max_diam, pair = max([((p - q)**2, (p, q))
                          for p, q in calculateConvexHulls(border_coordinates)
                          ])
    #Calculate the minimum feret diameter
    '''min_diam, pair = min([((p[0] - q[0])**2 + (p[1] - q[1])**2, (p, q))
                          for p, q in calculateConvexHulls(border_coordinates)
                          ])
    '''
    min_diam, pair = min([((p - q)**2, (p, q))
                          for p, q in calculateConvexHulls(border_coordinates)
                          ])
    return min_diam, max_diam


@ReturnOutputs
@Output("meta")
class CalculateZooProcessFeatures(Node):
    """
    Calculate descriptive features similar to ZooProcess using :py:func:`skimage.measure.regionprops`.

    Args:
        regionprops (RegionProperties or Variable): :py:class:`~skimage.measure._regionprops.RegionProperties`
            instance returned by :py:class:`FindRegions`.
        meta (dict or Variable, optional): Meta-data dictionary to update.
        prefix (str or Variable, optional): Prefix for all keys.

    Example:
        .. code-block:: python

            with Pipeline() as p:
                image = ...
                mask = ...

                regionprops = FindRegions(mask, image)

                features = CalculateZooProcessFeatures(regionprops)
    """
    def __init__(
            self,
            regionprops: RawOrVariable,
            meta: Optional[RawOrVariable[dict]] = None,
            prefix: Optional[RawOrVariable[str]] = None,
    ):
        super().__init__()

        self.regionprops = regionprops
        self.meta = meta
        self.prefix = prefix

    def transform(self, regionprops, meta, prefix):
        if meta is None:
            meta = {}

        features = regionprop2zooprocess(regionprops)

        if prefix is not None:
            features = {
                "{}{}".format(self.prefix, k): v
                for k, v in features.items()
            }

        return {**meta, **features}


image = skimage.data.camera()
calculateFeret(image)