import numpy as np

from dpwnets import tools
from dpwnets import dp_mechanisms

def constraned_inference(arr):
    S = []
    for k in range(1,len(arr)+1):
        i = np.argmax(arr[0:k])
        j = np.argmin(arr[i:len(arr)]) + i
        sk = int( np.sum(arr[i:j+1])/len(arr[i:j+1]) )
        S.append(sk)
    return S

def proposed(arr):
    return tools.min_l2_norm(arr, np.sum(arr))

def l2(arr1, arr2):
    return np.linalg.norm(arr1 - arr2, ord=2) 
    # return np.sqrt( np.sum(np.power((arr1-arr2),2)) )

if __name__ == "__main__":
    arr_orig = np.array([2, 3, 3, 3, 3, 3, 5, 6, 7, 10])
    # arr_orig = np.array([7, 9, 11, 12])

    arr_pert = dp_mechanisms.geometric_mechanism(arr_orig, 2)
    
    constraned_inference_arr = constraned_inference(arr_pert)
    print('constraned inference: %s' % constraned_inference_arr)
    print('l2 constraned inference: %s' % l2(arr_orig, constraned_inference_arr))
    
    proposed_arr = sorted(proposed(arr_pert))
    print('proposed inference: %s' % proposed_arr)
    print('l2 proposed inference: %s' % l2(arr_orig, proposed_arr))
