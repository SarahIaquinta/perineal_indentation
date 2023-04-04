import numpy as np
from matplotlib import pyplot as plt
from math import nan

fime_name = 'FF1 2A.csv'

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

np.savetxt('time.txt',vec_time)
np.savetxt('axis.txt',vec_pos_axis)
np.savetxt('Z.txt',mat_Z)

plt.imshow(mat_Z)
plt.show()
plt.close()

