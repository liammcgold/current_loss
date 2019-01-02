import numpy as np
from scipy.ndimage.filters import convolve,sobel
import tifffile


def brute_force(gt):

    '''
      phi_i corresponds to fields

      the boundary conditions will be defined for each supervoxel and laplace will be solved

      this will then be used to generate omega which is the vector containing max currents at each point


    '''



    fields=__get_fields(gt)

    currents=__get_currents(fields)

    return currents

def __solve_poison(boundary_conds, n=100):

    kernel = [[[0, 0, 0],
               [0, 1, 0],
               [0, 0, 0]],
              [[0, 1, 0],
               [1, 0, 1],
               [0, 1, 0]],
              [[0, 0, 0],
               [0, 1, 0],
               [0, 0, 0]]]

    output=boundary_conds.copy()


    for i in range(0,n):

        output=convolve(output,kernel)*(1/6)

        output[np.where(boundary_conds==1)]=1
        output[np.where(boundary_conds==-1)]=-1

    return output


def __get_fields(gt):

    uniques = np.unique(gt)

    # Number of IDs
    N = np.shape(uniques)[0]

    # (N,x,y,z) to hold all potential values
    fields = np.zeros((N,) + np.shape(gt))

    i = 0

    # get PHI_i (fields)
    for num in uniques:
        boundary_conds = np.zeros(np.shape(gt))
        boundary_conds[np.where(gt != 0)] = -1
        boundary_conds[np.where(gt == num)] = 1

        tifffile.imsave("debug/boundary_conds" + str(i) + ".tif", np.asarray(boundary_conds, dtype=np.float32))

        output = __solve_poison(boundary_conds)

        fields[i] = output

        tifffile.imsave("debug/field_"+str(i)+".tif",np.asarray(output,dtype=np.float32))

        i += 1

    return fields


def __get_currents(fields):


    total_field=np.sum(fields+1    , axis = 0)

    tifffile.imsave("debug/total_fields.tif", np.asarray(total_field, dtype=np.float32))

    # get divergence mag of each point for each field

    #get grads in all directions
    x_grads_saquared = np.square(sobel(fields, axis=1))
    y_grads_saquared = np.square(sobel(fields, axis=2))
    z_grads_saquared = np.square(sobel(fields, axis=3))

    grad_mag_squared = x_grads_saquared + y_grads_saquared + z_grads_saquared

    grad_mags = np.sqrt(grad_mag_squared)

    #find max current at each point
    currents = np.sum(grad_mags, axis=0)

    return currents


def __get_currents2(fields):


    # get divergence mag of each point for each field

    #get grads in all directions
    x_grads_saquared = np.square(sobel(fields, axis=1))
    y_grads_saquared = np.square(sobel(fields, axis=1))
    z_grads_saquared = np.square(sobel(fields, axis=1))

    grad_mag_squared = x_grads_saquared + y_grads_saquared + z_grads_saquared

    grad_mags = np.sqrt(grad_mag_squared)

    #find max current at each point
    currents = np.argmax(grad_mags, axis=0)

    return currents