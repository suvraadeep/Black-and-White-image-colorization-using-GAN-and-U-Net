import os
import glob
import numpy as np
from fastai.data.external import untar_data, URLs
coco_path = untar_data(URLs.COCO_SAMPLE)
coco_path = str(coco_path) + "/train_sample"

path = coco_path
paths = glob.glob(path + "/*.jpg")
np.random.seed(123)
