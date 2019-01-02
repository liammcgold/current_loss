import tifffile

import methods

import numpy as np


def test_brute_force():

    gt=tifffile.imread("real_data/constricted_gt.tif")

    output=methods.brute_force(gt)

    tifffile.imsave("real_data/output_brute.tif",np.asarray(output,dtype=np.float32))

    output_masked=output
    output_masked[np.where(gt!=0)]=0

    tifffile.imsave("real_data/output_masked_brute.tif", np.asarray(output, dtype=np.float32))

