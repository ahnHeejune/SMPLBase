import numpy as np
import cv2 

srcPtsList = [ [0., 0.], [799., 0.], [0., 599.], [799., 599.]]
tgtPtsList = [ [100., 0.], [799., 0.], [0., 599.],  [799., 599.]]

srcPtsNp = np.array(srcPtsList).reshape([1,-1,2]) #.astype('float32')
tgtPtsNp = np.array(tgtPtsList).reshape([1,-1,2]) #.astype('float32')
print("dtype", srcPtsNp.dtype)  # float32

matches = list()
for i in range(srcPtsNp.shape[1]):
    matches.append(cv2.DMatch(i,i,0))

tps = cv2.createThinPlateSplineShapeTransformer()    
tps.estimateTransformation(srcPtsNp, tgtPtsNp, matches)

print("float64 makes trouble with applyTransformation")
_, transPtsNp = tps.applyTransformation(srcPtsNp)
for i in range(srcPtsNp.shape[1]):
    print(srcPtsNp[0,i,:], "->", transPtsNp[0, i,:], ":", tgtPtsNp[0,i,:])


print("float32 works with applyTransformation")
_, transPtsNp = tps.applyTransformation(srcPtsNp.astype('float32'))
for i in range(srcPtsNp.shape[1]):
    print(srcPtsNp[0,i,:], "->", transPtsNp[0, i,:], ":", tgtPtsNp[0,i,:])
  
    