import math

from skimage import img_as_ubyte, measure
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_dilation, disk, convex_hull_image

from scipy.spatial.distance import pdist

import numpy as np
import cv2 as cv
from morphocut.pipeline import NodeBase


class Processor(NodeBase):
    """
    DEPRECATED?!?

    A processing node. Performs segmentation on images to find objects and their region properties

    Input:

    {
        object_id: ...
        facets: {
            input_data: {
                meta: {filename: ...},
                image: <np.array of shape = [h,w,c]>
            }
            corrected_data: {
                image: <np.array of shape = [h,w,c]>
            }
        }
    }

    Output:

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

    def __init__(self, min_object_area=None, padding=None, image_manipulators=[], raw_img_rank=1):
        self.min_object_area = min_object_area
        self.padding = padding
        self.image_manipulators = image_manipulators
        self.raw_img_rank = raw_img_rank

    def __call__(self, input=None):
        # print('processor call')
        for step, data_object in enumerate(input):
            print('current step: {}\n'.format(step))
            print('Processing file '
                  + data_object['facets']['input_data']['meta']['filepath'])
            yield from self.process_single_image(data_object)

    def process_single_image(self, data_object):
        src = data_object['facets']['corrected_data']['image']

        # Segment foreground objects from background objects using thresholding with the otsu method
        _, mask = cv.threshold(src, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        mask = cv.bitwise_not(mask)

        # Find connected components in the masked image to identify individual objects
        _, markers = cv.connectedComponents(mask)

        # Retrieve regionprops of the connected components and filter out those who are smaller than 30 pixels in area
        properties = measure.regionprops(
            markers, intensity_image=src, coordinates='rc')
        if (self.min_object_area):
            properties = [p for p in properties if p.area
                          > self.min_object_area]

        yield from self.export_image_regions(data_object, properties)

    def export_image_regions(self, data_object, properties):
        '''
        Iterates through the region properties and exports images containing each object
        '''

        for i, property in enumerate(properties):

            src_img = data_object['facets']['input_data']['image']

            # Define bounding box and position of the object
            x = property.bbox[0]
            y = property.bbox[1]
            w = property.bbox[2] - property.bbox[0]
            h = property.bbox[3] - property.bbox[1]

            # Define bordersize based on the width and height of the object. The border size specifies how much of the image around the object is shown in its image.
            if (self.padding):
                bordersize_w = int(w * self.padding)
                bordersize_h = int(h * self.padding)
            else:
                bordersize_w = 0
                bordersize_h = 0

            # Calculate min and max values for the border around the object, so that there are no array errors (no value below 0, no value above max width/height).
            xmin = max(0, x - bordersize_w)
            xmax = min(src_img.shape[0], x + w + bordersize_w)
            ymin = max(0, y - bordersize_h)
            ymax = min(src_img.shape[1], y + h + bordersize_h)

            border_top = y - ymin
            border_bottom = ymax - (y + h)
            border_left = x - xmin
            border_right = xmax - (x + w)

            # Create the masked and the masked contour image of the object
            original_masked = src_img[xmin:xmax, ymin:ymax]

            # with Timer() as t:
            #     feret_diameter = self.feret_diameter_maximum(property)
            # print('processing one object took %.03f sec.' % t.interval)
            feret_diameter = 1

            new_object = dict(
                object_id='{}_{}'.format(data_object['object_id'], i),
                raw_img=dict(
                    id=i,
                    meta=dict(
                        properties=property,
                        xstart=ymin,
                        ystart=xmin,
                        feret=feret_diameter,
                    ),
                    image=original_masked,
                    img_rank=self.raw_img_rank,
                ),
            )

            dimensions = dict(
                x=x,
                y=y,
                w=w,
                h=h,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                border_top=border_top,
                border_bottom=border_bottom,
                border_left=border_left,
                border_right=border_right,
            )

            export_keys = ['raw_img']
            for im in self.image_manipulators:
                key, img_dict = im(original_masked, property, dimensions)
                new_object[key] = img_dict
                export_keys.append(im.key)
            new_object['export_keys'] = export_keys

            yield new_object

    def feret_diameter_maximum(self, property):
        # property: a RegionProp object
        # from Steven Brown
        # https://github.com/scikit-image/scikit-image/issues/2320#issuecomment-256057683
        label_image = property._label_image
        label = property.label
        identity_convex_hull = convex_hull_image(label_image == label)
        coordinates = np.vstack(measure.find_contours(identity_convex_hull, 0.5,
                                                      fully_connected='high'))
        distances = pdist(coordinates, 'sqeuclidean')
        return math.sqrt(np.max(distances))
