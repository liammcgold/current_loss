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




def test_random(k=10):

    gt=tifffile.imread("real_data/constricted_gt.tif")

    output=methods.random_method(gt,k)

    tifffile.imsave("real_data/output_random.tif",np.asarray(output,dtype=np.float32))

    output_masked=output
    output_masked[np.where(gt!=0)]=0

    tifffile.imsave("real_data/output_masked_random.tif", np.asarray(output, dtype=np.float32))

def gen_bigger_test_data():

    gt = tifffile.imread("real_data/constricted_gt.tif")

    gt_mask=np.asarray(gt!=0,dtype=np.int)

    gt2=np.zeros((np.shape(gt)[0]*10+10*2,np.shape(gt)[1]*10+10*2,np.shape(gt)[2]*10+10*2))

    for i in range(0,10):
        for j in range(0,10):
            for k in range(0,10):
                x_start = i * (np.shape(gt)[0] + 2)
                x_stop = i * (np.shape(gt)[0] + 2) + np.shape(gt)[0]
                y_start = j * (np.shape(gt)[1] + 2)
                y_stop = j * (np.shape(gt)[1] + 2) + np.shape(gt)[1]
                z_start = k * (np.shape(gt)[2] + 2)
                z_stop = k * (np.shape(gt)[2] + 2) + np.shape(gt)[2]
                gt2[x_start:x_stop,y_start:y_stop,z_start:z_stop]=np.multiply(gt+(np.max(gt2)+1),gt_mask)
                print((x_start,y_start,z_start))


    tifffile.imsave("real_data/large_gt.tif", np.asarray(gt2, dtype=np.float32))
