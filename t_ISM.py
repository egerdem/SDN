import os

import numpy as np
from numpy import linalg as LA
from scipy.io import savemat
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5

c  = 343;
Fs = 44100;
Nt = round(Fs/2);
xs = np.array([4.5, 3.5, 2]);

rAng = [0,1.04719755119660,2.09439510239320,3.14159265358979,4.18879020478639,5.23598775598299];
locOffset = [2, 2, 1.5]; #source?
# xr = np.tile(locOffset, (6, 1)) + [0.1*np.cos(rAng), 0.1*np.sin(rAng), np.zeros(len(rAng))];
# xr = np.transpose(xr);

# xr=np.array([[3,3.4, 3.3] ,[2.95,3.48,3.3],[2.85,3.48,3.3],[2.8,3.4,3.3],[2.85,3.31,3.3],[2.95,3.31,3.3]]);
xr=np.array([[2, 2, 1.5]]); #observation
xr = np.transpose(xr);

L  = np.array([9, 7, 4]);
N =  np.array([0, 0, 0]);
T60 = 0.6;

Tw = 11;
Fc = 0.9;

Rd = 0.08;

t = np.linspace(0,Nt*1/Fs,Nt);
count = 0;
RIR = np.zeros((8000,6),dtype=object);
#RIR = np.full((256,1,8000,6), 0)
sloc = np.zeros((256, 3));




res = {'field1': RIR, 'field2': sloc};

import os
import sys
sys.path.append(os.getcwd())

from ISM import ISM

# while (count < 1):
#
#     xs = np.multiply([L[0]-0.4, L[1]-0.4, L[2]-0.4],np.random.rand(3)) + [0.2, 0.2, 0.2];
#     if (np.linalg.norm(np.transpose(locOffset)-xs)<1):
#         continue
#     else:
#         print('hello')
#         count = count + 1;
#         B=ISM(xr, xs, L, T60, N, Nt, Rd, [], Tw, Fc, Fs, c);
#         RIR[count-1,0]  = B[0];
#         sloc[count-1, :] = xs;
#         print(count)

B=ISM(xr, xs, L, T60, N, Nt, Rd, [], Tw, Fc, Fs, c);
RIR  = B[0];
sloc= xs;


# res['field1'][count-1,:,:] = RIR[count-1,0];
# res['field2'][count-1,:] = sloc[count-1,:];


#plt.plot(RIR[0,0,:,:])
plt.plot(RIR)
plt.title('Room Impulse Response')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()