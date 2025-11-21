'''
z크레인에 대한 Z값 까지 추정하는 코드

'''

import cv2
import numpy as np
import copy
from scipy.optimize import least_squares




def affine_matrix(param):
    tx, ty, theta = param[:3]
    """
    tx, ty: translation (pixel or world unit)
    theta: rotation (radian)
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    M = np.array([
        [cos_t, -sin_t, tx],
        [sin_t,  cos_t, ty],
        [0,      0,     1 ]
    ], dtype=float)

    return M

def crane_pose(param, crane_point):
    pose_matrix = affine_matrix(param)
    Z_from_magenet = param[3]

    crane_point_ = copy.deepcopy(crane_point)
    crane_point_[:,2] = 1.0
    crane_point_ = (pose_matrix @ crane_point_.T).T
    crane_point_[:,2] = Z_from_magenet
    return pose_matrix, Z_from_magenet ,crane_point_

def projection(param, crane_point, intrinsic, distortion):
    pose_matrix = affine_matrix(param)
    Z_from_magenet = param[3]

    crane_point_ = copy.deepcopy(crane_point)
    crane_point_[:,2] = 1.0
    crane_point_ = (pose_matrix @ crane_point_.T).T
    crane_point_[:,2] = Z_from_magenet

    # R, T (월드 → 카메라)
    rvec = np.array([[0], [0], [0]], dtype=np.float32)   # 회전 (Rodrigues)
    tvec = np.array([[0], [0], [5]], dtype=np.float32)   # z=5m 앞

    # 3D → 2D 투영
    image_points, _ = cv2.projectPoints(
        crane_point_,
        rvec,
        tvec,
        intrinsic,
        distortion
    )
    image_points = image_points.reshape(-1,2)
    return image_points


def cost_function(param, crane_point, corner_point, intrinsic, distortion):

    Z_from_magenet = param[3]
    crane_point[:,2] = Z_from_magenet

    image_points = projection(param, crane_point, intrinsic, distortion)

    cost = (corner_point.astype(np.float64) - image_points).reshape(-1)
    return cost


####################################

# 입력


intrinsic = np.array([
    [ 1.3856356e+03, -8.9599198e-01,  9.4715491e+02],
    [ 0.0000000e+00,  1.3882871e+03,  6.0742163e+02],
    [ 0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
])
distortion = np.array([-0.085307,  0.079377,  0.000436, -0.000662])

# 
# 0-------3------> X
# |       |
# |       |
# |       | 2.25
# |       |
# |       |
# |       |
# 1-------2
# |  0.45
# V
# Y

crane_point = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 2.25, 0.0],
    [0.45, 2.25, 0.0],
    [0.45, 0.0, 0.0]
])


# 이미지 로드
image = cv2.imread("sample.jpg")
image = cv2.undistort(image, intrinsic, distortion)

corner_point = np.array([
    [768, 322],
    [767, 782],
    [857, 782],
    [859, 323],
])

########################################################################
# 최적화

param = [0.0, 0.0, 0.0, 0.0] # tx, ty, theta, Z

result = least_squares(cost_function, param, args=(crane_point, corner_point, intrinsic, distortion))
result_translate_param = result.x
print(result,'\n')
print(result_translate_param.tolist(),'\n')


#############################
# 검산

image_points = projection(result_translate_param, crane_point, intrinsic, distortion)
image_points = image_points.astype(int)

# 각 코너에 초록 점 찍기(GT 정답)
for (x, y) in corner_point:
    image = cv2.circle(image, (x, y), 5, (0, 255, 0), -1)   # BGR = 빨강

# 각 코너에 빨간 점 찍기
for (x, y) in image_points:
    image = cv2.circle(image, (x, y), 5, (0, 0, 255), -1)   # BGR = 빨강

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow("image", image)
key = cv2.waitKey(0)


#############################

# 결과

pose_matrix, Z_from_magenet ,crane_point_ = crane_pose(result_translate_param, crane_point)



print(crane_point_)
print()
print(pose_matrix)
print(Z_from_magenet)

