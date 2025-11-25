import open3d as o3d
import numpy as np
from matplotlib import cm


def color_by_axis(pcd, axis="z", cmap_name="viridis"):
    """
    PCD의 특정 축(x, y, z)에 따라 색을 지정합니다.

    axis: "x", "y", "z"
    cmap_name: matplotlib colormap 이름
    """
    pts = np.asarray(pcd.points)

    if pts.size == 0:
        return pcd  # 비어있으면 그대로 반환

    # 축 선택
    if axis == "x":
        values = pts[:, 0]
    elif axis == "y":
        values = pts[:, 1]
    else:
        axis = "z"
        values = pts[:, 2]

    # 정규화 (0~1)
    min_v, max_v = values.min(), values.max()
    if max_v - min_v < 1e-6:
        norm_values = np.zeros_like(values)
    else:
        norm_values = (values - min_v) / (max_v - min_v)

    # colormap 적용
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(norm_values)[:, :3]  # RGB only

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_pcds(pcd_paths, axis="z", cmap_name="viridis"):
    """
    주어진 PCD 리스트를 axis 값 기반 colormap으로 시각화합니다.
    """
    geometries = []

    for idx, path in enumerate(pcd_paths):
        pcd = o3d.io.read_point_cloud(path)
        pcd = color_by_axis(pcd, axis=axis, cmap_name=cmap_name)

        print(f"[{idx}] Loaded: {path}")
        print(f"    Points: {len(pcd.points)}")
        print(f"    Axis: {axis}, Colormap: {cmap_name}")

        geometries.append(pcd)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"PCD Color by {axis.upper()}-Axis",
        width=800, height=600,
        left=50, top=50,
        point_show_normal=False
    )


if __name__ == '__main__':
    pcd_paths = [
        '/workspace/sequences_sample/202511201026498853/pcd/0000.pcd',
        '/workspace/sequences_sample/202511201026498853/pcd/0004.pcd'
    ]

    visualize_pcds(
        pcd_paths,
        axis="x",           # "x", "y", "z" 가능
        cmap_name="turbo"   # 예: viridis, plasma, jet, turbo 등
    )

