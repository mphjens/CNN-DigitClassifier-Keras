import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import scipy.misc
from mpl_toolkits.mplot3d import Axes3D

a = np.loadtxt("slider_code.txt", skiprows=1,usecols=(0,1,2,3))
Xs, Ys, Zs = a[:,1] * - 1, a[:,0] * -1, a[:,3]

outSize = 1500

MaxX = np.max(Xs)
MinX = np.min(Xs)
xRange = (MaxX - MinX)

MaxY = np.max(Ys)
MinY = np.min(Ys)
yRange = (MaxY - MinY)

MaxZ = np.max(Zs)
MinZ = np.min(Zs)
ZRange = (MaxZ - MinZ)

aspectratio = yRange/xRange

extent=(MinX, MaxX ,MinY, MaxY)
N = 90j
xs,ys =np.linspace(MinX, MaxX, outSize), np.linspace(MinY,  MaxY, outSize * aspectratio)
zi = griddata(Xs,Ys, Zs, xs, ys, interp="linear")

pixelVals = np.array((zi - MinZ) / ZRange * 255, dtype=np.int)
pixelVals[pixelVals > 65] = 255
pixelVals[pixelVals <= 65] = 0

print pixelVals
scipy.misc.imsave('outfile.jpg',pixelVals )

plt.imshow(pixelVals, extent=extent)
plt.show()

#zi = griddata(a[:,0],a[:,1], a[:,3], np.linspace(MinX, MaxX, 100), np.linspace(MinY, MaxY, 100), interp="linear")



