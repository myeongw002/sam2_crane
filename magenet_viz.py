'''
z크레인에 대한 Z값 까지 추정하는 코드

'''

import cv2
import numpy as np
import copy
from scipy.optimize import least_squares
import open3d as o3d
import os




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


def create_crane_mesh(param, crane_point, crane_height=0.119, color=[1.0, 0.5, 0.0]):
    """
    크레인 메쉬 생성 (상단면 + 하단면으로 3D 박스 형태)
    
    Args:
        param: [tx, ty, theta, Z] - 추정된 pose
        crane_point: 크레인 상단 4개 코너 (2D 좌표, 2.25m x 0.45m)
        crane_height: 크레인 높이 (Z 방향, 기본값 0.119m)
        color: 메쉬 색상
    
    Returns:
        mesh, wireframe: Open3D TriangleMesh와 LineSet
    """
    pose_matrix, Z_from_magenet, crane_point_top = crane_pose(param, crane_point)
    
    # 하단면 생성 (Z 값을 높이만큼 증가)
    crane_point_bottom = crane_point_top.copy()
    crane_point_bottom[:, 2] += crane_height
    
    # 8개 정점 (상단 4개 + 하단 4개)
    vertices = np.vstack([crane_point_top, crane_point_bottom])
    
    # 삼각형 면 정의 (12개 삼각형 = 6개 사각형 면)
    triangles = np.array([
        # 상단면 (Top Face - indices 0,1,2,3)
        [0, 1, 2], [0, 2, 3],
        # 하단면 (Bottom Face - indices 4,5,6,7)
        [4, 6, 5], [4, 7, 6],
        # 옆면들
        [0, 3, 7], [0, 7, 4],  # Side 1
        [3, 2, 6], [3, 6, 7],  # Side 2
        [2, 1, 5], [2, 5, 6],  # Side 3
        [1, 0, 4], [1, 4, 5]   # Side 4
    ])
    
    # 메쉬 생성
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    
    # 와이어프레임 생성
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    wireframe.paint_uniform_color([0, 0, 0])  # 검은색 외곽선
    
    return mesh, wireframe


def visualize_3d(pcd_path, param, crane_point, crane_height=0.119, T_l2c_path=None):
    """
    Point Cloud와 추정된 크레인 메쉬를 함께 시각화 (카메라 좌표계)
    
    Args:
        pcd_path: Point cloud 파일 경로
        param: [tx, ty, theta, Z] - 추정된 pose
        crane_point: 크레인 상단 4개 코너 (2.25m x 0.45m)
        crane_height: 크레인 높이 (기본값 0.119m)
        T_l2c_path: LiDAR to Camera 변환 행렬 파일 경로
    """
    geoms = []
    
    # LiDAR to Camera 변환 행렬 로드
    if T_l2c_path and os.path.exists(T_l2c_path):
        T_l2c = np.loadtxt(T_l2c_path, delimiter=',').reshape(4, 4)
        print(f"\n📐 Loaded T_l2c from {T_l2c_path}")
    else:
        T_l2c = np.eye(4)
        print("\n⚠️  Using identity matrix for T_l2c")
    
    # Camera extrinsic (tvec = [0, 0, 5])
    T_cam_offset = np.eye(4)
    T_cam_offset[2, 3] = 5.0  # Z축으로 5m 이동
    
    # 1. Point Cloud 로드 및 카메라 좌표계로 변환
    print(f"\n📂 Loading point cloud: {pcd_path}")
    pcd_lidar = o3d.io.read_point_cloud(pcd_path)
    print(f"   Loaded {len(pcd_lidar.points)} points")
    
    # LiDAR → Camera 변환
    pcd_camera = copy.deepcopy(pcd_lidar)
    pcd_camera.transform(T_l2c)
    pcd_camera.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(pcd_camera)
    
    # 2. 추정된 크레인 메쉬 생성 (World 좌표계)
    crane_mesh_world, crane_wireframe_world = create_crane_mesh(param, crane_point, crane_height, color=[1.0, 0.5, 0.0])
    
    # World → Camera 변환 (tvec 적용)
    crane_mesh_camera = copy.deepcopy(crane_mesh_world)
    crane_mesh_camera.transform(T_cam_offset)
    crane_wireframe_camera = copy.deepcopy(crane_wireframe_world)
    crane_wireframe_camera.transform(T_cam_offset)
    
    geoms.append(crane_mesh_camera)
    geoms.append(crane_wireframe_camera)
    
    # 3. 좌표계 표시 (카메라 원점)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    geoms.append(coord_frame)
    
    # 4. 크레인 코너 포인트 표시 (빨간색, 카메라 좌표계)
    pose_matrix, Z_from_magenet, crane_corners_world = crane_pose(param, crane_point)
    
    # World → Camera 변환
    crane_corners_h = np.hstack([crane_corners_world, np.ones((len(crane_corners_world), 1))])
    crane_corners_camera = (T_cam_offset @ crane_corners_h.T).T[:, :3]
    
    corner_pcd = o3d.geometry.PointCloud()
    corner_pcd.points = o3d.utility.Vector3dVector(crane_corners_camera)
    corner_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 빨간색
    geoms.append(corner_pcd)
    
    # 시각화
    print("\n🎨 Visualizing 3D scene (Camera Coordinate)...")
    print(f"   Crane Pose (World): tx={param[0]:.3f}, ty={param[1]:.3f}, θ={np.degrees(param[2]):.1f}°, Z={param[3]:.3f}m")
    print(f"   Camera Offset: tvec=[0, 0, 5]")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud + Crane Mesh (Camera Coord)", width=1280, height=720)
    
    for g in geoms:
        vis.add_geometry(g)
    
    # 렌더링 옵션
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.point_size = 2.0
    
    vis.run()
    vis.destroy_window()
    print("✅ Visualization closed")



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
image = cv2.imread("/workspace/sequences_sample/202511201026498851/image/0004.jpg")
image = cv2.undistort(image, intrinsic, distortion)

corner_point = np.array([
    [978, 389],
    [979, 767],
    [1056, 768],
    [1053, 389],
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
# 3D 시각화

pcd_path = "/workspace/sequences_sample/202511201026498851/pcd/0004.pcd"
T_l2c_path = "/workspace/sam2/transform3_tuned_tuned.txt"
visualize_3d(pcd_path, result_translate_param, crane_point, crane_height=0.119, T_l2c_path=T_l2c_path)


#############################

# 결과

pose_matrix, Z_from_magenet ,crane_point_ = crane_pose(result_translate_param, crane_point)



print(crane_point_)
print()
print(pose_matrix)
print(Z_from_magenet)

