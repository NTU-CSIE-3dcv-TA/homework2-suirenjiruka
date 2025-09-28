from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from sympy import *
import pandas as pd
import numpy as np
import open3d as o3d
import random
import cv2
import time
import math

from tqdm import tqdm

np.random.seed(1428) # do not change this seed
random.seed(1428) # do not change this seed

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def distortion(point, distCoeffs):
    x, y, z = point
    k1, k2, p1, p2 = distCoeffs
    r_sqr = x**2 + y**2
    x_hat = x * (1 + k1 * r_sqr + k2*r_sqr**2) + p2 * (r_sqr + 2 * x**2) + 2* p1 * x *y
    y_hat = y * (1 + k1 * r_sqr + k2*r_sqr**2) + 2 * p2 *x *y + p1 * (r_sqr + 2 * y**2)
    dis_point = np.array([x_hat, y_hat, z])
    return dis_point



def get_cosine(P1, P2, cameraMatrix, distCoeffs):
    k1, k2, p1, p2 = distCoeffs
    x1 = (P1[0] - cameraMatrix[0][2]) / cameraMatrix[0][0]  #先算x1, y1 = K^-1 P1，取消camera intrinsic parameter影響，並轉換成實際CCS座標
    y1 = (P1[1] - cameraMatrix[1][2]) / cameraMatrix[1][1]
    x2 = (P2[0] - cameraMatrix[0][2]) / cameraMatrix[0][0]
    y2 = (P2[1] - cameraMatrix[1][2]) / cameraMatrix[1][1]
    def undistort(x, y):       #估計扭取誤差，消除扭曲誤差
        x_o = x
        y_o = y
        for i in range(10):
            r_sqr = x**2 +y**2
            div = 1 + k1 * r_sqr + k2 * r_sqr**2
            x_sub = p2 * (r_sqr + 2 * x**2) + 2* p1 * x *y
            y_sub = 2 * p2 *x *y + p1 * (r_sqr + 2 * y**2)
            x = (x_o - x_sub) / div
            y = (y_o - y_sub) / div
        return x, y
    x1, y1 = undistort(x1, y1)
    x2, y2 = undistort(x2, y2)
    d1 = np.array([x1, y1, 1])
    d1 = d1 / np.linalg.norm(d1)   #normalize 至單位長度，cosine會恰好等於兩線内積
    d2 = np.array([x2, y2, 1])
    d2 = d2 / np.linalg.norm(d2)
    return np.dot(d1, d2), d1

def solveP3P(three_points, two_points, cameraMatrix, distCoeffs):
    #計算夾角cos
    cos_12, d1 = get_cosine(two_points[0], two_points[1], cameraMatrix, distCoeffs)
    cos_13, d3 = get_cosine(two_points[2], two_points[0], cameraMatrix, distCoeffs)
    cos_23, d2 = get_cosine(two_points[1], two_points[2], cameraMatrix, distCoeffs)
    dis12 = math.sqrt(np.sum((three_points[0] - three_points[1])**2))
    dis13 = math.sqrt(np.sum((three_points[0] - three_points[2])**2))
    dis23 = math.sqrt(np.sum((three_points[1] - three_points[2])**2))
    #算4次方程
    K1 = (dis23 / (dis13 + 1e-9))**2
    K2 = (dis23 / (dis12 + 1e-9))**2
    G4 = (K1*K2 - K1 - K2)**2 - 4* K1 * K2 * (cos_23**2)
    G3 = 4*(K1*K2 - K1 - K2)*K2*(1-K1)*cos_12 + 4 * K1*cos_23*((K1*K2 - K1 + K2) *cos_13 + 2* K2 *cos_12*cos_23)
    G2 = (2*K2*(1-K1)*cos_12)**2 + 2 * (K1*K2 - K1 - K2) * (K1*K2 + K1 - K2) + 4* K1 *((K1 -K2)* (cos_23**2) + K1*(1-K2)*(cos_13**2) -2 * (1+K1) *K2 *cos_12*cos_13*cos_23)
    G1 = 4*(K1*K2 + K1 - K2)*K2*(1-K1)*cos_12 + 4 * K1*((K1*K2 - K1 + K2) *cos_13*cos_23 + 2* K1* K2 *cos_12* (cos_13**2))
    G = (K1*K2 + K1 - K2)**2 - 4* (K1 **2) * K2 * (cos_13**2)
    #解4次方程
    x = np.roots([G4, G3, G2, G1, G])
    X = []
    for xi in x:
        try:
            if abs(np.imag(xi)) > 1e-19:
                continue
            else:
                if(float(np.real(xi)) >= 0):
                    X.append(float(np.real(xi)))
        except:
            continue
    # 計算ABC
    A = []
    for xi in X.copy():
        temp = np.roots([(1 +xi**2 - 2*xi*cos_12),0,- dis12**2])   
        if abs(np.imag(temp[0])) > 1e-19:
            X.remove(xi)
        else:
            A.append(abs(float(np.real(temp[0]))))   #A的2次是沒有一次項，解必定正負成對
 
    y = symbols('y')
    Y = []
    for xi, ai in zip(X.copy(), A.copy()):
        y = ((xi**2 - K1) - ((xi **2) * (1 -K2) + 2 *xi *K2 *cos_12 - K2) * (1 - K1) ) / (2 * K1 * (xi* cos_23 - cos_13))
        if y < 0:
            X.remove(xi)
            A.remove(ai)
        else:
            Y.append(float(y))
    A = np.array(A, dtype=np.float64)
    B = np.array(X, dtype=np.float64) * A
    C = np.array(Y, dtype=np.float64) * A
    #計算R 和 T
    rvecs = []
    tvecs = []
    Camera_cord = np.array([[ai * d1, bi* d2, ci*d3] for ai, bi, ci in zip (A, B, C)])
    for ccs in Camera_cord:
        centroid_X = np.mean(three_points, axis=0)
        centroid_c = np.mean(ccs, axis=0)
        x_shift = three_points - centroid_X
        c_shift = ccs -centroid_c
        H = np.dot(np.transpose(x_shift), c_shift)
        H = np.array(H, dtype=np.float64) 
        U, S, V = np.linalg.svd(H)
        R = np.dot(np.transpose(V), np.transpose(U))
        if np.linalg.det(R) < 0:
            V[-1,:] *= -1
            R = np.dot(np.transpose(V), np.transpose(U))
        T = centroid_c - np.dot(R, centroid_X)
        rvecs.append(R)
        tvecs.append(T)
    return np.array(rvecs), np.array(tvecs)


def pnpsolver(query,model,cameraMatrix=0):
    kp_query, desc_query = query  # 2D 照片的 (X, Y)  + 照片descriptor
    kp_model, desc_model = model  # 3D 模型的 (X, Y, Z)  + 模型descriptor
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    # TODO: solve PnP problem using OpenCV
    # Hint: you may use "Descriptors Matching and ratio test" first
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(desc_query, desc_model, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.25 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)

    two_points= np.array([kp_query[m.queryIdx] for m in good_matches])
    three_points = np.array([kp_model[m.trainIdx] for m in good_matches])
    print(f"good match: {len(good_matches)}")
    #success, rvec_cv, tvec_cv, inliers = cv2.solvePnPRansac(three_points, two_points, cameraMatrix, distCoeffs, None, None, iterationsCount=100, reprojectionError=0.5, flags= cv2.SOLVEPNP_P3P)
    #RANSAC 實作
    iter = 100
    threshold = 0.5
    max_len = 0
    max_r = []
    max_t = []
    max_inliers = []
    for i in range(iter):
        samples = random.sample(range(len(two_points)), k=3)
        rvec, tvec = solveP3P(three_points[samples], two_points[samples], cameraMatrix, distCoeffs)
        #success, rvec_1, tvec_1 = cv2.solveP3P( three_points[samples], two_points[samples], cameraMatrix, distCoeffs, rvecs=None, tvecs=None, flags= cv2.SOLVEPNP_P3P)
        #print(f"rvec {cv2.Rodrigues(rvec_1[0])}, tvec : {tvec_1}, myrvec {rvec}, mytvec : {tvec},")
        for r, t in zip(rvec, tvec):
            inliers = []
            for id, (three_point, two_point) in enumerate(zip(three_points, two_points)):
                three_point.reshape(1, 3)
                project = np.dot(r, three_point)
                project = project + t
                project = project / project[2]
                project = distortion(project, distCoeffs)
                project = np.dot(cameraMatrix, project)
                error = math.sqrt(np.sum((project[:2] - two_point)**2))
                if(error < 0.35):
                    inliers.append(id)
            if(len(inliers) > max_len):
                max_len = len(inliers)
                max_r = r
                max_t = t
                max_inliers = inliers
    print(f"max_len:{max_len}, id: {max_inliers}.")
    para = np.hstack([np.reshape(max_r, (9)) , np.reshape(max_t, (3))])
    #LSE 最終優化
    def project(para, three_point, two_point):
        r = np.array(para[:9]).reshape(3,3)
        t = np.array(para[9:12]).reshape(3)
        error = 0
        for three, two in zip(three_point, two_point):
            project = np.dot(r, three)
            project = project + t
            project = project / project[2]
            project = distortion(project, distCoeffs)
            project = np.dot(cameraMatrix, project)
            error += math.sqrt(np.sum((project[:2] - two)**2))
        return error
    results = least_squares(project, para,args=(three_points[max_inliers], two_points[max_inliers]))
    rvec = np.array(results.x[:9]).reshape(3,3)
    tvec = np.array(results.x[9:12]).reshape(3)
    print(rvec, tvec)
    #print(rvec_cv, tvec_cv)
    return None, rvec, tvec, inliers


def rotation_error(R1, R2):
    #TODO: calculate rotation error
    R2_inv = [-R2[0], -R2[1], -R2[2], R2[3]]
    #w = R1[0]*R2_inv[0] - R1[1]*R2_inv[1] - R1[2]*R2_inv[2] - R1[3]*R2_inv[3]
    #x = R1[0]*R2_inv[1] + R1[1]*R2_inv[0] + R1[2]*R2_inv[3] - R1[3]*R2_inv[2]
    #y = R1[0]*R2_inv[2] - R1[1]*R2_inv[3] + R1[2]*R2_inv[0] + R1[3]*R2_inv[1]
    #z  =R1[0]*R2_inv[3] + R1[1]*R2_inv[2] - R1[2]*R2_inv[1] + R1[3]*R2_inv[0]
    r1 = R.from_quat(R1)
    r2_inv = R.from_quat(R2_inv)
    r_diff = r1 * r2_inv
    q_diff = R.as_quat(r_diff)
    R_diff = np.linalg.norm(q_diff) + 1e-9
    print(f"R: {R1}, GT: {R2}")
    return 2 * np.arccos(np.clip(abs(q_diff[3] / R_diff), 0, 1.0))

def translation_error(t1, t2):
    #TODO: calculate translation error
    diff = t1 - t2
    error = np.sum(diff**2)
    print(f"T: {t1}, GT: {t2}")
    return math.sqrt(error)

def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat

def painter(img, r, t):
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])
    cube_point = []
    colors = []
    height, width, channel = img.shape
    camera_pos = -t
    for x in np.arange (0, 1.1, 0.15):   #設置cube voxel座標
        for y in np.arange (0, 1.1, 0.15):
            for z in np.arange (0, 1.1, 0.15):
                if (x == 1.05 or y == 1.05 or z == 1.05 or x == 0 or y == 0 or z == 0):
                    cube_point.append([x, y, z])
                    if x == 0:     #指定不同面的voxel顏色
                        colors.append([255,0,0])
                    elif y == 0:
                        colors.append([25,255,40])
                    elif y == 1.05:
                        colors.append([180,25,230])
                    elif z == 0:
                        colors.append([204,230,40])
                    elif z == 1.05:
                        colors.append([30,230,230])
                    else:
                        colors.append([0,0,255])
    cube_point = np.array(cube_point)
    colors = np.array(colors)
    #調整cube位置
    transform_mat = get_transform_mat(np.array([0., 0., 0.]), np.array([ 1.07, -0.48,  0.83]),0.42)
    cube_point = (transform_mat @ np.concatenate([
                            cube_point.transpose(), 
                            np.ones([1, cube_point.shape[0]])
                            ], axis=0)).transpose()
    cube_point_pair = []
    #排序point
    for p, c in zip(cube_point, colors):
        dis = math.sqrt(np.sum((p - camera_pos) ** 2))
        cube_point_pair.append((p, c, dis))
    cube_point_pair = np.array(cube_point_pair, dtype=[("point", float, (3,)), ("color", float, (3,)), ("distance", float)])
    cube_point_pair = np.sort(cube_point_pair, order = "distance")[::-1]
    processed_img = img.copy()
    #投影至照片並著色
    for point_pair in cube_point_pair:
        coordinate = point_pair[0]
        project = np.dot(r, coordinate)
        project = project + t
        z = project[2]
        project = project / project[2]
        project = distortion(project, distCoeffs)
        project = np.dot(cameraMatrix, project)
        if(z > 0 and project[0] < width and project[0] > 0 and project[1] < height and project[1] > 0):
            h = math.floor(project[1])+4 if math.floor(project[1])+4 < height else height
            w = math.floor(project[0])+4 if math.floor(project[0])+4 < height else width
            processed_img[math.floor(project[1]) - 1 : h, math.floor(project[0])-1 : w] = point_pair[1]
    return processed_img

def visualization(Camera2World_Transform_Matrixs, points3D_df):
    #TODO: visualize the camera pose
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    base_point = [[0, 0, 0, 1], [0.15, 0.1, 0.4, 1],  [-0.15, 0.1, 0.4, 1],  [0.15, -0.1, 0.4, 1],  [-0.15, -0.1, 0.4, 1]]
    lines = [
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 4]]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    vis = o3d.visualization.Visualizer() 
    vis.create_window()
    vis.add_geometry(pcd)
    for matrix in Camera2World_Transform_Matrixs:
        point = np.transpose(np.dot(matrix, np.transpose(base_point)))
        point = point[:, :-1]
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(point),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)
    vis.run()
    return

def video_out(img_list):
    height, width, channel = img_list[0].shape
    fps = 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video_writer = cv2.VideoWriter('./AR_text.mp4', fourcc, fps, ( width, height))
    for img in img_list:
        #img = cv2.resize(img, (width, height))
        video_writer.write(img)

    video_writer.release()

if __name__ == "__main__":
    # Load data
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)


    IMAGE_ID_LIST = range(170, 220)
    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []
    img_list = []

    for idx in tqdm(IMAGE_ID_LIST):
        # Load quaery image
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_COLOR)
        if(len(fname) < 16):
            continue
        img_list.append(rimg)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))
        # rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat() # Convert rotation vector to quaternion
        # tvec = tvec.reshape(1,3) # Reshape translation vector
        r_list.append(rvec)
        t_list.append(tvec)

        # Get camera pose groudtruth
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values
        # Calculate error
        print(type(rotq_gt[0]))
        r_error = rotation_error(R.as_quat(R.from_matrix(rvec)), rotq_gt[0])
        t_error = translation_error(tvec, tvec_gt[0])
        print(r_error, t_error)
        rotation_error_list.append(r_error)
        translation_error_list.append(t_error)

    # TODO: calculate median of relative rotation angle differences and translation differences and print them
    median_r_error =  np.median(rotation_error_list)
    median_t_error =  np.median(translation_error_list)
    print(f"median rotation error: {median_r_error}, median translation error: {median_t_error}.")

    # TODO: result visualization
    Camera2World_Transform_Matrixs = []
    for r, t in zip(r_list, t_list):
        # TODO: calculate camera pose in world coordinate system
        c2w = np.eye(4)
        t = -np.dot(np.linalg.inv(r), t)
        c2w[:3,:3] = r       
        c2w[:3,3]  = t
        c2w[3,3] = 1
        print(c2w)
        Camera2World_Transform_Matrixs.append(c2w)
    visualization(Camera2World_Transform_Matrixs, points3D_df)
    processed_img = []
    for r, t, img in zip(r_list, t_list, img_list):
         processed_img.append(painter(img, np.array(r), np.array(t)))
    video_out(processed_img)
