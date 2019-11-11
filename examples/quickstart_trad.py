import glob
import os.path

import numpy as np
import PIL
from skimage.measure import label, regionprops

for i, base_path in enumerate(["/path/a", "/path/b", "/path/c"]):
    pattern = os.path.join(base_path, "subpath/to/input/files/*.jpg")
    for path in glob.iglob("subpath/to/input/files/*.jpg"):
        source_basename = os.path.splitext(os.path.basename(path))[0]

        image = np.array(PIL.Image.open(path))
        mask = image < 128

        label_image = label(mask)

        for region in regionprops(label_image):

            roi_image = region.intensity_image

            output_fn = "/path/to/output/{:d}-{}-{:d}.png".format(
                i, source_basename, region.label
            )

            img = PIL.Image.fromarray(roi_image)
            img.save(output_fn)
