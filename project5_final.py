import cv2, os, sys, copy, random
import numpy as np
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import matplotlib.pyplot as plt

#=====================================================================
#=====================================================================

def fundamental_matrix(points1,points2):
	F_b,_ = cv2.findFundamentalMat(points1,points2,cv2.FM_8POINT)

	mat = []
	mass_cent = [0.,0.]
	mass_cent_p = [0.,0.]
	for i in range(len(points1)):
		mass_cent[0] += points1[i,0]
		mass_cent[1] += points1[i,1]
		mass_cent_p[0] += points2[i,0]
		mass_cent_p[1] += points2[i,1]
	mass_cent = np.divide(mass_cent,float(len(points1)))
	mass_cent_p = np.divide(mass_cent_p,float(len(points1)))

	scale1 = 0.
	scale2 = 0.
	for i in range(len(points1)):
		scale1 += np.sqrt((points1[i][0]-mass_cent[0])**2+(points1[i][1]-mass_cent[1])**2)
		scale2 += np.sqrt((points2[i][0]-mass_cent_p[0])**2+(points2[i][1]-mass_cent_p[1])**2)
	
	scale1 = scale1/len(points1)
	scale2 = scale2/len(points1)

	scale1 = np.sqrt(2.)/scale1
	scale2 = np.sqrt(2.)/scale2
	A = np.zeros((8,9))
	for i in range(8):
		x1 = (points1[i][0]-mass_cent[0])*scale1
		y1 = (points1[i][1]-mass_cent[1])*scale1
		x2 = (points2[i][0]-mass_cent_p[0])*scale2
		y2 = (points2[i][1]-mass_cent_p[1])*scale2

		row = np.array([x2*x1,x2*y1,x2,y2*x1,y2*y1,y2,x1,y1,1])
		A[i]=row

	U,S,V = np.linalg.svd(A)

	F = V[-1]
	F = np.reshape(F,(3,3))
	U,S,V = np.linalg.svd(F)
	S[2] = 0
	F = U@np.diag(S)@V

	T1 = np.array([scale1,0,-scale1*mass_cent[0],0,scale1,-scale1*mass_cent[1],0,0,1])
	T1 = T1.reshape((3,3))
	T2 = np.array([scale2,0,-scale2*mass_cent_p[0],0,scale2,-scale2*mass_cent_p[1],0,0,1])
	T2 = T2.reshape((3,3))
	F = np.transpose(T2)@F@T1
	F = F / F[-1,-1]
	return F,F_b

#=====================================================================

def random_points(points1,points2):
	pts_index = np.random.randint(len(points1), size=8)
	
	X1 = []
	X2 = []
	for i in range(8):
		X1.append(points1[pts_index[i]])
		X2.append(points2[pts_index[i]])

	return np.array(X1), np.array(X2)

#=====================================================================

def ransac_8_points(M,epsilon,pts1,pts2,matches):
	n = 0
	F_best = []
	S_in = []

	confidence = 0.99

	N = sys.maxsize
	count = 0
	while N > count:
		x_1,x_2 = random_points(pts1,pts2)

		F,F_b = fundamental_matrix(x_1,x_2)

		if count%100 == 0:
			print(count)

		S = []
		inliers = 0
		for j in range(len(pts1)):
			x1 = np.array([pts1[j,0], pts1[j,1], 1]).reshape(-1,1)
			x2 = np.array([pts2[j,0], pts2[j,1], 1]).reshape(-1,1)

			ep1 = F @ x1
			ep2 = np.transpose(F) @ x2

			numerator = (np.transpose(x2) @ F @ x1)**2
			denominator = ep1[0]**2 + ep1[1]**2 + ep2[0]**2 + ep2[1]**2
			e = numerator/denominator

			if e[0,0] <= epsilon:
				inliers += 1
				S.append(matches[j])

		if len(S) > n:
			n = len(S)
			F_best = F 
			S_in = S 

		ratio = len(S)/len(pts1)
		if np.log(1-(ratio**8)) == 0:
			continue

		N = np.log(1-confidence)/np.log(1-(ratio**8))
		count += 1

	# print(len(S_in),len(pts1),len(S_in)/len(pts1))
	return F_best,S_in

#=====================================================================

def EssentialMatrix(K,pnts1,pnts2,matches):
	M = 500
	epsilon = 0.05

	F,inliers = ransac_8_points(M,epsilon,pnts1,pnts2,matches)

	E = np.transpose(K) @ F @ K
	U,S,V = np.linalg.svd(E)
	S_new = [[1,0,0],[0,1,0],[0,0,0]]
	E = U@S_new@V
	return E,inliers

#============================================================

def LinearTriangulation(K,C1,R1,C2,R2,pts1,pts2):
	P1 = K @ np.hstack((R1, -R1 @ C1))
	P2 = K @ np.hstack((R2, -R2 @ C2))
	# P1 = np.hstack((R1,C1))
	# P2 = np.hstack((R2,C2))
	
	X = []

	for i in range(len(pts1)):
		x1 = pts1[i]
		x2 = pts2[i]

		a1 = x1[0]*P1[2,:]-P1[0,:]
		a2 = x1[1]*P1[2,:]-P1[1,:]
		a3 = x2[0]*P2[2,:]-P2[0,:]
		a4 = x2[1]*P2[2,:]-P2[1,:]
		
		A = [a1,a2,a3,a4]		

		# print(A)

		# print(np.shape(A))

		U,S,V = np.linalg.svd(A)
		V_out = V[3]
		# V_out = V_out.reshape(-1,1)
		V_out = V_out/V_out[-1]
		X.append(V_out)

	return X

#=====================================================================

def CameraMatrix(fx,fy,cx,cy,s):
	return [[fx, s, cx],[0, fy, cy],[0, 0, 1]]

#=====================================================================

def CameraPose(K,E):
	W = [[0,-1,0],[1,0,0],[0,0,1]]
	U,S,V = np.linalg.svd(E)

	poses = {}

	poses['C1'] = U[:,2].reshape(3,1)
	poses['C2'] = -U[:,2].reshape(3,1)
	poses['C3'] = U[:,2].reshape(3,1)
	poses['C4'] = -U[:,2].reshape(3,1)

	poses['R1'] = U @ W @ V
	poses['R2'] = U @ W @ V 
	poses['R3'] = U @ np.transpose(W) @ V
	poses['R4'] = U @ np.transpose(W) @ V

	for i in range(4):
		C = poses['C'+str(i+1)]
		R = poses['R'+str(i+1)]
		if np.linalg.det(R) < 0:
			C = -C 
			R = -R 
			poses['C'+str(i+1)] = C 
			poses['R'+str(i+1)] = R
		I = np.eye(3,3)
		M = np.hstack((I,C.reshape(3,1)))
		poses['P'+str(i+1)] = K @ R @ M

	return poses

#=====================================================================

def findZ(R1,t,pts1,pts2,K):
	Co = [[0],[0],[0]]
	Ro = np.eye(3,3)
	P1 = np.eye(3,4)
	P2 = np.hstack((R1,t))

	X1 = LinearTriangulation(K,Co,Ro,t,R1,pts1,pts2)
	X1 = np.array(X1)

	check = 0
	for i in range(X1.shape[0]):
		x = X1[i,:].reshape(-1,1)
		if R1[2]@np.subtract(x[0:3],t) > 0 and x[2] > 0:
			check += 1

	return check

#=====================================================================

current_pos = np.zeros((3, 1))
print(current_pos)
current_rot = np.eye(3)
print(current_rot)
# current_pos = np.array([[-130.65236835],   [11.74805102], [-140.5268776 ]])
# print(current_pos)
# current_rot = np.array([[0.99663235, 0.01788091, 0.0800264 ],[-0.02138265,  0.99884122,  0.04311639],[-0.07916271, -0.04468238,  0.9958598 ]])
# print(current_rot)


model_path = "model"
fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(model_path)

images = [img for img in os.listdir('undistorted') if img.endswith(".png")]

fig = plt.figure('Figure 1',figsize=(7,5)) #10,7
fig.suptitle('Project 5 - Visual Odometry')

gs = plt.GridSpec(2,3)
ax1 = fig.add_subplot(gs[:,0:-1])
ax1.set_title('Odometry Map')
ax2 = fig.add_subplot(gs[0,-1])
ax2.set_title('Original Image')
ax2.axis('off')
ax3 = fig.add_subplot(gs[1,-1])
ax3.set_title('Image With Features')
ax3.axis('off')

init_frame = 20
for i in range(len(images)):
	if i < init_frame:
		continue

	print('frame '+str(i))

	j = i+1
	current_img = cv2.imread("undistorted/frame_"+str(i)+".png",0)
	next_img = cv2.imread("undistorted/frame_"+str(j)+".png",0)

	sift = cv2.xfeatures2d.SIFT_create()
	
	kp_current,des_current = sift.detectAndCompute(current_img,None)
	kp_next,des_next = sift.detectAndCompute(next_img,None)

	bf = cv2.BFMatcher()

	matches = bf.knnMatch(des_current,des_next,k=2)

	good = []
	for m,n in matches:
		if m.distance < 0.5*n.distance:
			good.append(m)

	pts1 = np.array([kp_current[m.queryIdx].pt for m in good]).reshape(-1,2)
	pts2 = np.array([kp_next[m.trainIdx].pt for m in good]).reshape(-1,2)

	pts1 = np.array(pts1)
	pts2 = np.array(pts2)

	K = CameraMatrix(fx,fy,cx,cy,0)
	E,inliers = EssentialMatrix(K,pts1,pts2,matches)
	
	poses = CameraPose(K,E)

	depth = [0,'0']
	for p in range(4):
		r_op = poses['R'+str(p+1)]
		t_op = poses['C'+str(p+1)]
		zs = findZ(r_op,t_op,pts1,pts2,K)

		if depth[0] < zs:
			depth[0] = zs
			depth[1] = str(p+1)

	R = poses['R'+depth[1]]
	t = -1*poses['C'+depth[1]]

	if t[2] > 0:
		t = t*-1

	x_curr = current_pos[0]
	z_curr = current_pos[2]

	current_pos += current_rot.dot(t)
	current_rot = R.dot(current_rot)
	
	x_new = current_pos[0]
	z_new = current_pos[2]

	ax1.plot([x_curr,x_new],[-z_curr,-z_new],'o')
	orig = cv2.cvtColor(current_img, cv2.COLOR_GRAY2RGB)
	kp_orig = cv2.drawKeypoints(current_img,kp_current,None,color=(0,255,0))
	ax2.imshow(orig)

	ax3.imshow(kp_orig)

	# cv2.imwrite('matches/frame_'+str(i)+'.png',kp_orig)

	# inlier_matches = []
	# for i in inliers:
	# 	inlier_matches.append(i[0])

	# inlier_img = cv2.drawMatches(current_img,kp_current,next_img,kp_next,inlier_matches,None,matchColor=(0,0,255))	
	# cv2.imshow('matches',cv2.resize(inlier_img,(0,0),fx=0.5,fy=0.5))
	# cv2.imwrite('match_example.png',inlier_img)

	plt.pause(0.001)

	f = open("more_vo_data.txt", "a")
	st = str(i)+","+str(x_curr)+","+str(-z_curr)+","+str(current_pos.reshape(1,-1))+","+str(current_rot[0].reshape(1,-1))+","+str(current_rot[1].reshape(1,-1))+","+str(current_rot[2].reshape(1,-1))
	print(st)
	f.write(st)
	f.write("\n")

	print('========================')
	cv2.waitKey(1)

cv2.waitKey(0)