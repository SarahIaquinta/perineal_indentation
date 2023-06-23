"""
This file is the initial version of the routine for extracting data
from the laset acquisition and then to apply a median filter to it. 
This code was reshaped to fit into the main read_file.py file. 
This code is kept for possible further interest in conducting similar analyses.
"""

import numpy as np
from matplotlib import pyplot as plt
from math import nan

## Select folder
# fime_name = '230407_FF2A.csv'
fime_name = '230403_FF1C.csv'
## Extract raw data
cur_file = open(fime_name,'r')
all_line = cur_file.readlines()

flag_continue = False
l_vec_Z = []
l_time = []
for cur_line in all_line:
    if cur_line[0:12] == 'Frame,Source':
        II = cur_line.find('Axis')
        cur_line0 = cur_line[II+5:]
        II = [pos for pos, char in enumerate(cur_line0) if char == ',']
        vec_pos_axis = np.zeros(len(II)+1)
        vec_pos_axis[0] = float(cur_line0[0:II[0]])
        for it_c in range(len(II)-1):
            vec_pos_axis[it_c+1] = float(cur_line0[II[it_c]+1:II[it_c+1]])
        
        vec_pos_axis[-1] = float(cur_line0[II[-1]+1:])
        flag_continue = True
    
    elif flag_continue and (len(cur_line)>10):
        II = [pos for pos, char in enumerate(cur_line) if char == ',']
        l_time.append(float(cur_line[II[1]+1:II[2]]))
        II = cur_line.find('Z')
        cur_line0 = cur_line[II+1:]
        II = [pos for pos, char in enumerate(cur_line0) if char == ',']
        vec_Z = np.zeros(len(II)+1)
        if II[0]==0:
            vec_Z[0] = nan
        else:
            vec_Z[0] = float(cur_line0[0:II[0]])
        
        for it_c in range(len(II)-1):
            if (II[it_c]+1)==(II[it_c+1]):
                vec_Z[it_c+1] = nan
            else:    
                vec_Z[it_c+1] = float(cur_line0[II[it_c]+1:II[it_c+1]])
        
        if (II[-1]+2)==len(cur_line0):
            vec_Z[0] = nan
        else:
            vec_pos_axis[-1] = float(cur_line0[II[-1]+1:])
        
        l_vec_Z.append(vec_Z) 

mat_Z = np.vstack(l_vec_Z)
vec_time = np.array(l_time)

## Reduce area of interest
plt.imshow(mat_Z)
plt.xlabel('space')
plt.ylabel('time')
plt.title('Select top left and bottom right corner')
A = plt.ginput(2,timeout=-1)
plt.close()

mat_Z = mat_Z[int(A[0][1]):int(A[1][1]),int(A[0][0]):int(A[1][0])]
vec_time = vec_time[int(A[0][1]):int(A[1][1])]
vec_pos_axis = vec_pos_axis[int(A[0][0]):int(A[1][0])]

## Save raw data
np.savetxt('time.txt',vec_time)
np.savetxt('axis.txt',vec_pos_axis)
np.savetxt('Z_raw.txt',mat_Z)

#!DEBUG!
# ~ plt.imshow(mat_Z)
# ~ plt.show()
# ~ plt.close()

# ~ plt.plot(mat_Z[int(mat_Z.shape[0]/2),:])
# ~ plt.show()
# ~ plt.close()

## Apply median filter
mat_Z_n = np.zeros(mat_Z.shape)
cnt = 0
for it_i in range(mat_Z.shape[0]):
    if (it_i/mat_Z.shape[0]*100)>cnt:
        print(str(cnt).zfill(2)+'%')
        cnt += 1
    
    for it_j in range(mat_Z.shape[1]):
        nn = 2
        flag_continue = True
        while flag_continue:
            vec_cur = mat_Z[max(0,it_i-nn):min(mat_Z.shape[0],it_i+nn),max(0,it_j-nn):min(mat_Z.shape[1],it_j+nn)].flatten()
            II = np.where(np.isnan(vec_cur)<1)[0]
            if len(II)>3:
                vec_cur = vec_cur[II]
                flag_continue = False
            else:
                nn += 1
        mat_Z_n[it_i,it_j] = np.median(vec_cur)

#!DEBUG!
# ~ plt.imshow(mat_Z_n)
# ~ plt.show()
# ~ plt.close()

## Save smoothed data
np.savetxt('Z_median.txt',mat_Z_n)

## Select where to measure depth
plt.plot(mat_Z[int(mat_Z.shape[0]/2),:],'-k')
plt.plot(mat_Z_n[int(mat_Z_n.shape[0]/2),:],'-b')
plt.xlabel('space')
plt.ylabel('height')
plt.title('select abscissa where to measure depth')
A = plt.ginput(1,timeout=-1)
plt.close()

## Get depth evolution
vec_d = mat_Z_n[:,int(A[0][0])]

## Remove outliers
II = np.min(np.where(np.abs((vec_d-np.median(vec_d))/np.median(vec_d))<0.5)[0])
vec_time_d = vec_time[II:]
vec_d = vec_d[II:]

## Save data
np.savetxt('time_depth.txt',vec_time_d)
np.savetxt('depth_raw.txt',vec_d)

## Smoothn depth
nn = 150
vec_d_smth = np.convolve(vec_d,np.ones(nn)/nn,'same')
vec_d_smth[0:int(nn/2)+1] = vec_d_smth[int(nn/2)+2]
vec_d_smth[-int(nn/2)+1:] = vec_d_smth[-int(nn/2)+2]

#!DEBUG!
# ~ plt.plot(vec_time_d,vec_d,'-k')
# ~ plt.plot(vec_time_d,vec_d_smth,'-b')
# ~ plt.xlabel('time')
# ~ plt.ylabel('depth')
# ~ plt.show()
# ~ plt.close()

## Save data
np.savetxt('depth_smooth.txt',vec_d_smth)

