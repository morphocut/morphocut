
"""
Extract regions.
"""

import math

import numpy as np
from morphocut.pipeline import NodeBase
from skimage import measure


def pad_slice(slc, padding, size):
    """
    Parameters:
        slc: original slice
        padding: padding to be added
        size: maximum size of the dimension
    """

    start, stop, step = slc.start, slc.stop, slc.step

    assert start >= 0 and stop >= 0

    start = max(start - padding, 0)
    stop = min(stop + padding, size)

    return slice(start, stop, step)


class ExtractRegions(NodeBase):
    """
    Parameters:
        mask_facet: The label image is calculate on this image.
        image_facets: These images are taken over to newly created objects.
        output_facet: Name of the output facet.
        min_area: Minimal area of an object.
        padding: Number of context pixels around an extracted object.

    BUG: perimeter is sometimes zero. We might have to calculate it manually.

    Output Facet:
        data: ZooProcess-compatible object data.
    """

    def __init__(self, mask_facet, intensity_facet, image_facets, output_facet, min_area=0, padding=0):
        self.mask_facet = mask_facet
        self.image_facets = image_facets
        self.output_facet = output_facet
        self.min_area = min_area
        self.padding = padding
        self.intensity_facet = intensity_facet

    def __call__(self, input):
        for obj in input:
            mask = obj['facets'][self.mask_facet]['image']
            intensity_img = obj['facets'][self.intensity_facet]['image']

            label_image = measure.label(mask)

            regionprops = measure.regionprops(
                label_image, intensity_image=intensity_img, coordinates='rc')

            for i, prop in enumerate(regionprops):
                if prop.area < self.min_area:
                    continue

                # Calculate padded slice
                padded_slice = tuple(pad_slice(slc, self.padding, mask.shape[i])
                                     for i, slc in enumerate(prop._slice))

                facets = {
                    self.output_facet: {
                        "image": label_image[padded_slice] == prop.label,
                        "data": self.regionprop2zooprocess(prop)
                    }
                }

                # Adopt image facets that correspond to each object
                for facet_name in self.image_facets:
                    img = obj["facets"][facet_name]["image"]

                    facets[facet_name] = {
                        "image": img[padded_slice]
                    }

                new_obj = {
                    "object_id": '{}_{}'.format(obj['object_id'], i),
                    "parent": obj,
                    "parent_slice": padded_slice,
                    "facets": facets,
                }

                yield new_obj

    def regionprop2zooprocess(self, prop):
        """
        Calculate zooprocess features from skimage regionprops.

        Notes:
            - date/time specify the time of the sampling, not of the processing.
        """
        propDict = {
            # width of the smallest rectangle enclosing the object
            'width': prop.bbox[3] - prop.bbox[1],
            # height of the smallest rectangle enclosing the object
            'height': prop.bbox[2] - prop.bbox[0],
            # X coordinates of the top left point of the smallest rectangle enclosing the object
            'bx': prop.bbox[1],
            # Y coordinates of the top left point of the smallest rectangle enclosing the object
            'by': prop.bbox[0],
            # circularity : (4∗π ∗Area)/Perim^2 a value of 1 indicates a perfect circle, a value approaching 0 indicates an increasingly elongated polygon
            'circ.': (4 * math.pi * prop.filled_area) / math.pow(prop.perimeter, 2),
            # Surface area of the object excluding holes, in square pixels (=Area*(1-(%area/100))
            'area_exc': prop.area,
            # Surface area of the object in square pixels
            'area': prop.filled_area,
            # Percentage of object’s surface area that is comprised of holes, defined as the background grey level
            '%area': 1 - (prop.area / prop.filled_area),
            # Primary axis of the best fitting ellipse for the object
            'major': prop.major_axis_length,
            # Secondary axis of the best fitting ellipse for the object
            'minor': prop.minor_axis_length,
            # Y position of the center of gravity of the object
            'y': prop.centroid[0],
            # X position of the center of gravity of the object
            'x': prop.centroid[1],
            # The area of the smallest polygon within which all points in the objet fit
            'convex_area': prop.convex_area,
            # Minimum grey value within the object (0 = black)
            'min': prop.min_intensity,
            # Maximum grey value within the object (255 = white)
            'max': prop.max_intensity,
            # Average grey value within the object ; sum of the grey values of all pixels in the object divided by the number of pixels
            'mean': prop.mean_intensity,
            # Integrated density. The sum of the grey values of the pixels in the object (i.e. = Area*Mean)
            'intden': prop.filled_area * prop.mean_intensity,
            # The length of the outside boundary of the object
            'perim.': prop.perimeter,
            # major/minor
            'elongation': np.divide(prop.major_axis_length, prop.minor_axis_length),
            # max-min
            'range': prop.max_intensity - prop.min_intensity,
            # perim/area_exc
            'perimareaexc': prop.perimeter / prop.area,
            # perim/major
            'perimmajor': prop.perimeter / prop.major_axis_length,
            # (4 ∗ π ∗ Area_exc)/perim 2
            'circex': np.divide(4 * math.pi * prop.area, prop.perimeter**2),
            # Angle between the primary axis and a line parallel to the x-axis of the image
            'angle': prop.orientation / math.pi * 180 + 90,
            # # X coordinate of the top left point of the image
            # 'xstart': data_object['raw_img']['meta']['xstart'],
            # # Y coordinate of the top left point of the image
            # 'ystart': data_object['raw_img']['meta']['ystart'],
            # Maximum feret diameter, i.e. the longest distance between any two points along the object boundary
            # 'feret': data_object['raw_img']['meta']['feret'],
            # feret/area_exc
            # 'feretareaexc': data_object['raw_img']['meta']['feret'] / property.area,
            # perim/feret
            # 'perimferet': property.perimeter / data_object['raw_img']['meta']['feret'],



            'bounding_box_area': prop.bbox_area,
            'eccentricity': prop.eccentricity,
            'equivalent_diameter': prop.equivalent_diameter,
            'euler_number': prop.euler_number,
            'extent': prop.extent,
            'local_centroid_col': prop.local_centroid[1],
            'local_centroid_row': prop.local_centroid[0],
            'solidity': prop.solidity,
        }
        return propDict
