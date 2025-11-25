import open3d as o3d
import numpy as np
from matplotlib import cm  # colormap for axis color


# ==================== Common Utils ==================== #

COLOR_PRESETS = {
    '1': ([1.0, 0.0, 0.0], 'red'),
    '2': ([0.0, 1.0, 0.0], 'green'),
    '3': ([0.0, 0.0, 1.0], 'blue'),
    '4': ([1.0, 1.0, 0.0], 'yellow'),
    '5': ([1.0, 0.0, 1.0], 'magenta'),
    '6': ([0.0, 1.0, 1.0], 'cyan'),
    '7': ([0.5, 0.5, 0.5], 'gray'),
    '8': ([1.0, 0.5, 0.0], 'orange'),
}


def print_color_presets():
    print("\n=== Color Presets ===")
    for k, (rgb, name) in COLOR_PRESETS.items():
        print(f"{k}: {name}  (RGB={rgb})")


def get_color_from_preset(num: str, default: str = '7'):
    if num not in COLOR_PRESETS:
        num = default
    color, name = COLOR_PRESETS[num]
    print(f"Selected Color: {name} (index={num}, RGB={color})")
    return color


# ==================== PCD Load & Merge ==================== #

def load_pcds(pcd_paths):
    pcds = []
    for path in pcd_paths:
        pcd = o3d.io.read_point_cloud(path)
        print(f"Loaded: {path} -> {pcd}")
        if pcd.is_empty():
            raise FileNotFoundError(f"Failed to load PCD file: {path}")
        pcds.append(pcd)
    return pcds


def merge_pcds(pcds):
    merged = o3d.geometry.PointCloud()
    for pcd in pcds:
        merged += pcd
    return merged


# ==================== Color Apply ==================== #

def apply_flat_color(merged_pcd, color):
    merged_pcd.paint_uniform_color(color)


def apply_per_pcd_colors_with_input(pcds):
    print_color_presets()
    for i, pcd in enumerate(pcds):
        num = input(f"Select color preset for PCD #{i} [default: 7(gray)]: ").strip()
        color = get_color_from_preset(num if num else '7', default='7')
        pcd.paint_uniform_color(color)


def select_colormap_by_number(num):
    cmap_list = {
        '1': 'viridis',
        '2': 'plasma',
        '3': 'inferno',
        '4': 'magma',
        '5': 'cividis',
        '6': 'turbo',
        '7': 'jet',
    }
    return cmap_list.get(num, 'viridis')


def apply_axis_color(merged_pcd, axis='z', cmap_num='1'):
    axis = axis.lower()
    axis_idx = {'x': 0, 'y': 1, 'z': 2}

    if axis not in axis_idx:
        raise ValueError("Axis must be one of: x, y, z")

    pts = np.asarray(merged_pcd.points)
    if pts.size == 0:
        return

    idx = axis_idx[axis]
    vals = pts[:, idx]
    vmin, vmax = vals.min(), vals.max()

    norm = (vals - vmin) / (vmax - vmin) if vmax != vmin else np.zeros_like(vals)

    cmap_name = select_colormap_by_number(cmap_num)
    cmap = cm.get_cmap(cmap_name)
    colors_rgb = cmap(norm)[:, :3]

    merged_pcd.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64))
    print(f"Applied Axis Color: axis={axis}, colormap={cmap_name}, range=({vmin:.3f}, {vmax:.3f})")


# ==================== Distance Measurement ==================== #

def measure_distances(merged_pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Select Points (Shift+Click)")
    vis.add_geometry(merged_pcd)
    print("Shift + Left Click to select points → Close window to measure distances.")
    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()

    if len(picked) < 2:
        print(f"Selected points: {len(picked)} (Need at least 2)")
        return []

    pts = np.asarray(merged_pcd.points)
    results = []
    pairs = len(picked) // 2

    for i in range(pairs):
        idx_a = picked[2 * i]
        idx_b = picked[2 * i + 1]

        pA = pts[idx_a]
        pB = pts[idx_b]
        dist = np.linalg.norm(pB - pA)

        results.append((idx_a, idx_b, pA, pB, dist))

    return results


# ==================== MAIN ==================== #

def main():
    PCD_PATHS = [
        '/workspace/sequences_sample/202511201026498853/pcd/0000.pcd',
        # '/workspace/sequences_sample/202511201026498853/pcd/0004.pcd'
    ]

    # ---- Single or Multi PCD selection ----
    if len(PCD_PATHS) == 1:
        selected = [PCD_PATHS[0]]
    else:
        print("=== PCD Mode ===")
        print("1: Single PCD (use first file only)")
        print("2: Multi PCD (use all files)")
        mode = input("Select option [default: 1]: ").strip()

        selected = PCD_PATHS if mode == '2' else [PCD_PATHS[0]]
        print(f"Using {len(selected)} PCD file(s)")

    pcds = load_pcds(selected)

    # ---- Color Mode ----
    print("\n=== Color Mode ===")
    print("1: Flat Color (single color)")
    print("2: Axis Color (axis + colormap)")
    print("3: Per-PCD Colors (each PCD has different color)")
    color_mode = input("Select option [default: 1]: ").strip()

    # ---- Apply Color ----
    if color_mode == '2':
        axis = input("Select axis (x / y / z) [default: z]: ").strip().lower()
        if axis not in ['x', 'y', 'z']:
            axis = 'z'

        print("\n=== Colormap List ===")
        print("1: viridis")
        print("2: plasma")
        print("3: inferno")
        print("4: magma")
        print("5: cividis")
        print("6: turbo")
        print("7: jet")
        cmap_num = input("Select colormap number [default: 1]: ").strip()
        cmap_num = cmap_num if cmap_num else '1'

        merged_pcd = merge_pcds(pcds)
        apply_axis_color(merged_pcd, axis, cmap_num)

    elif color_mode == '3':
        if len(pcds) == 1:
            print("Only one PCD loaded → per-PCD color mode not necessary but applied.")
        apply_per_pcd_colors_with_input(pcds)
        merged_pcd = merge_pcds(pcds)

    else:
        print_color_presets()
        num = input("Select flat color preset [default: 7(gray)]: ").strip()
        color = get_color_from_preset(num if num else '7', default='7')

        merged_pcd = merge_pcds(pcds)
        apply_flat_color(merged_pcd, color)

    # ---- Measurement Loop ----
    while True:
        results = measure_distances(merged_pcd)

        print("\n=== Distance Results ===")
        if not results:
            print("No valid point pairs.\n")
        else:
            for (i, j, pA, pB, dist) in results:
                print(f"A(index={i}): x={pA[0]:.3f}, y={pA[1]:.3f}, z={pA[2]:.3f}")
                print(f"B(index={j}): x={pB[0]:.3f}, y={pB[1]:.3f}, z={pB[2]:.3f}")
                print(f"Distance = {dist:.5f} m\n")

        if input("Press Enter to measure again, or 'q' to quit: ").lower() == 'q':
            break


if __name__ == "__main__":
    main()

