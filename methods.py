import numpy as np
from scipy.ndimage.filters import convolve,sobel
import tifffile



def brute_force(gt,n=100):

    '''
      phi_i corresponds to fields

      the boundary conditions will be defined for each supervoxel and poison will be solved

      this will then be used to generate omega which is the vector containing max currents at each point


    '''



    fields=__get_fields(gt,n=n)

    currents=__get_currents(fields)

    return currents

def random_method(gt,k):

    #due to probability calculations this value minimizes error always
    n=3

    uniques=np.unique(gt)
    uniques=np.delete(uniques,np.where(uniques==0))

    m=int(k/n)

    fields = np.zeros((k,) + np.shape(gt))

    i=0
    for i in range(0,m):
        new_gt=np.zeros(np.shape(gt))

        i=1
        for val in uniques:
            rand=np.random.randint(1,4)

            #ensures all values used
            if i<4:
                new_gt[np.where(gt==val)]=i
                i+=1

            else:
                new_gt[np.where(gt == val)] = rand



        tifffile.imsave("debug/rand_gt"+str(i),np.asarray(new_gt,dtype=np.float32))



        fields[i*n:(i+1)*n]=__get_fields(new_gt)

        i+=1

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

def __get_fields(gt,n=100):

    uniques = np.unique(gt)
    uniques = np.delete(uniques, np.where(uniques == 0))

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

        output = __solve_poison(boundary_conds,n=n)

        fields[i] = output

        tifffile.imsave("debug/field_"+str(i)+".tif",np.asarray(output,dtype=np.float32))

        i += 1

    return fields

def __get_currents(fields):


    # get divergence mag of each point for each field

    #get grads in all directions
    x_grads_saquared = np.square(sobel(fields, axis=1))
    y_grads_saquared = np.square(sobel(fields, axis=2))
    z_grads_saquared = np.square(sobel(fields, axis=3))*(1/16)

    grad_mag_squared = x_grads_saquared + y_grads_saquared + z_grads_saquared

    grad_mags = np.sqrt(grad_mag_squared)

    #find max current at each point
    currents = np.max(grad_mags, axis=0)

    return currents

# class state(object):
#
#     def __init__(self,gt,n):
#
#
#         # two n chanel volumes, one for theta and one for phi bar
#         # first pass theta chanel 0 is equivalent to gt and phi bar is equivalent to the
#         # the coresponding
#
#         # first lets generate theta for first pass
#
#         # theta will be in the form (j_1,j_2...j_n,len) where len indicates how many
#         # values are not null
#         theta = np.zeros(np.shape(gt) + (n,), dtype=np.int)
#
#         # first element is ID
#         theta[:, :, :, 0] = gt
#
#         # last element is 1 because only one value is not null in first pass
#         theta[:, :, :, n] = 1
#
#         # phi bar defined similarly where last value is  list length
#         phi_bar = np.zeros(np.shape(gt) + (n,), dtype=np.float32)
#
#         # populate first section with theta values
#         phi_bar[:, :, :, 0] = np.asarray(gt != 0, dtype=np.float32)
#
#
#         self.theta=theta
#         self.phi=phi_bar
# def approximation(gt,n=6):
#
#     '''
#
#     -j corresponds to unique IDs
#     -phi_i corresponds to all fields
#     -theta corresponds to gt indicies that return the top n values at each point
#     -phi_bar contains n tuples for each point, each containing the index and corresponding phi i value of the points contained in theta at that point
#     -psi(j) returns the corresponding phi values at that point for provided j only if j is contained in theta for that point, otherwise returns -1
#     -theta_tilda union of theta values at voxel and neighboring voxels
#     -phi_tilda average of psi(j) values correspondiong to theta_tilda list at each point with neighboring voxels
#
#     Steps:
#             1)set theta for first pass with boundary conditions used for phi_i
#             2)obtain phi_bar,psi(J),theta_tilda and phi_tilda using theta
#             3)Loop for given itterations:
#                     -recalculate theta, this time using phi_tilda instead of phi_i
#                     -obtain phi_bar,psi(J),theta_tilda and phi_tilda using theta
#             4)use phi_bar to get currents
#             5)return sum of currents
#     '''
# def __get_fields_aprox(gt,n):
#     '''
#
#         -j corresponds to unique IDs
#         -phi_i corresponds to all fields
#         -theta corresponds to gt indicies that return the top n values at each point
#         -phi_bar contains n tuples for each point, each containing the index and corresponding phi i value of the points contained in theta at that point
#         -psi(j) returns the corresponding phi values at that point for provided j only if j is contained in theta for that point, otherwise returns -1
#         -theta_tilda union of theta values at voxel and neighboring voxels
#         -phi_tilda average of psi(j) values correspondiong to theta_tilda list at each point with neighboring voxels
#
#         Steps:
#                 1)set theta for first pass with boundary conditions used for phi_i
#                 2)obtain phi_bar,psi(J),theta_tilda and phi_tilda using theta
#                 3)Loop for given itterations:
#                         -recalculate theta, this time using phi_tilda instead of phi_i
#                         -obtain phi_bar,psi(J),theta_tilda and phi_tilda using theta
#                 4)use phi_bar to get currents
#                 5)return sum of currents
#     '''
#
#     st=state(gt,n)
#
#     st=aprox_iteration(st,n,gt)

# def aprox_iteration(state,n,gt):
#
#     #pad for loop
#     padded_theta=np.pad(state.theta,(1,1,1,0))
#     padded_phi=np.pad(state.phi,(1,1,1,0))
#
#     new_theta=np.zeros(np.shape(padded_theta))
#     new_phi=np.zeros(np.shape(padded_phi))
#
#     theta_til=np.zeros(n*7)
#
#     for i in range(1,np.shape(gt)[0]+1):
#         for j in range(1,np.shape(gt)[1]+1):
#             for k in range(1,np.shape(gt)[2]+1):
#
#
#                #collect neighboring values
#                 theta_til[0:n]=padded_theta[i,j,k]
#
#                 theta_til[n:n*2]=padded_theta[i-1,j,k]
#                 theta_til[n*2:n*3]=padded_theta[i+1,j,k]
#                 theta_til[n*3:n*4]=padded_theta[i,j-1,k]
#                 theta_til[n*4:n*5]=padded_theta[i,j+1,k]
#                 theta_til[n*5:n*6]=padded_theta[i,j,k-1]
#                 theta_til[n*6:n*7]=padded_theta[i,j,k+1]
#
#                 #remove duplicates
#                 relevant_vals=np.unique(theta_til)
#
#                 averages=np.zeros(np.shape(relevant_vals)+(2,))
#                 averages[:,1]=relevant_vals
#
#                 #calc averages
#                 i=0
#                 for val in relevant_vals:
#
#                     if val==0:
#                         averages[i]=-1
#                         i+=1
#                     else:
#
#                         where=np.where(padded_theta[i-1,j,k]==val)
#                         if where.shape[0]==1:
#                             averages[i,0]+=padded_phi[i-1,j,k,where]*(1/6)
#                         else:
#                             averages[i,0]+=-1/6
#
#                         where = np.where(padded_theta[i + 1, j, k] == val)
#                         if where.shape[0] == 1:
#                             averages[i,0] += padded_phi[i + 1, j, k,where] * (1 / 6)
#                         else:
#                             averages[i,0] += -1 / 6
#
#                         where = np.where(padded_theta[i, j-1, k] == val)
#                         if where.shape[0] == 1:
#                             averages[i,0] += padded_phi[i, j-1, k,where] * (1 / 6)
#                         else:
#                             averages[i,0] += -1 / 6
#
#                         where = np.where(padded_theta[i, j + 1, k] == val)
#                         if where.shape[0] == 1:
#                             averages[i, 0] += padded_phi[i, j + 1, k,where] * (1 / 6)
#                         else:
#                             averages[i, 0] += -1 / 6
#
#                         where = np.where(padded_theta[i, j, k-1] == val)
#                         if where.shape[0] == 1:
#                             averages[i,0] += padded_phi[i - 1, j, k+1] * (1 / 6)
#                         else:
#                             averages[i,0] += -1 / 6
#
#                         i+=1
#
#                         sort=np.argsort(averages[:,0])
#
#                         averages[:]=averages[sort]
#
#                         new_phi[i,j,k]=averages[-5:]

x=1