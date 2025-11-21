import open3d as o3d
import numpy as np

def load_and_color_pcds(pcd_paths, colors):
    pcds = []
    for path, color in zip(pcd_paths, colors):
        pcd = o3d.io.read_point_cloud(path)
        print(f"Loaded : {pcd}")
        if pcd.is_empty():
            raise FileNotFoundError(f"PCD 파일을 불러올 수 없습니다: {path}")
        pcd.paint_uniform_color(color)
        pcds.append(pcd)
    return pcds


def measure_distances(merged_pcd):
    """
    선택된 점 개수에 따라 2개씩 쌍을 만들어 거리 계산.
    2개  → A-B
    3개  → A-B   (마지막 1개는 무시)
    4개  → A-B, C-D
    5개  → A-B, C-D (마지막 1개 무시)
    """
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Select Points (Shift+Click)")
    vis.add_geometry(merged_pcd)
    print("Shift + 좌클릭으로 여러 점을 선택한 뒤 창을 닫으세요.")
    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()

    if len(picked) < 2:
        print(f"선택된 점 개수: {len(picked)} (최소 2점 필요)")
        return 0

    pts = np.asarray(merged_pcd.points)

    results = []
    # 점을 2개씩 묶기
    num_pairs = len(picked) // 2

    for i in range(num_pairs):
        idx_a = picked[2 * i]
        idx_b = picked[2 * i + 1]

        pA = pts[idx_a]
        pB = pts[idx_b]
        dist = np.linalg.norm(pB - pA)
        results.append((idx_a, idx_b, pA, pB, dist))

    return results


def main():
    PCD_PATHS = [
        "/home/myungw00/sam2_ws/sequences_sample/202511191815489071/pcd/frame_0000.pcd"
    ]
    COLORS = [
        [0.5, 0.5, 0.5]
    ]

    pcds = load_and_color_pcds(PCD_PATHS, COLORS)
    merged_pcd = pcds[0]

    while True:
        try:
            results = measure_distances(merged_pcd)
            if results is None:
                pass

            print("=== 거리 계산 결과 ===")
            for (idx_a, idx_b, pA, pB, dist) in results:
                print(f"Point A (index={idx_a}): x={pA[0]:.3f}, y={pA[1]:.3f}, z={pA[2]:.3f}")
                print(f"Point B (index={idx_b}): x={pB[0]:.3f}, y={pB[1]:.3f}, z={pB[2]:.3f}")
                print(f"Distance = {dist:.5f} m\n")

        except:
            pass
            
        inp = input("다시 측정하려면 Enter, 종료하려면 'q' 입력: ")
        if inp.lower() == 'q':
            break
if __name__ == "__main__":
    main()

