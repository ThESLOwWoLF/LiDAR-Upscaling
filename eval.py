import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from potts_sr import potts_super_resolve, chamfer_distance


def estimate_edge_mask(pc, k=16, threshold=0.02):
    nn = NearestNeighbors(n_neighbors=k).fit(pc)
    _, idx = nn.kneighbors(pc)
    local_var = np.var(pc[idx], axis=1).mean(axis=1)
    return local_var > threshold


def avg_point_to_point_distance(A, B):
    nn = NearestNeighbors(n_neighbors=1).fit(B)
    dists, _ = nn.kneighbors(A)
    return np.mean(dists)


def density_consistency(A, B, radius=0.2):
    def avg_neighbors(pc):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        densities = []
        for pt in pc:
            [_, idx, _] = pcd_tree.search_radius_vector_3d(pt, radius)
            densities.append(len(idx))
        return np.mean(densities)

    return avg_neighbors(A), avg_neighbors(B)


def edge_preservation_score(pred, gt, edge_mask, threshold=0.05):
    pred_edge = pred[edge_mask]
    gt_edge = gt[edge_mask]
    nn = NearestNeighbors(n_neighbors=1).fit(gt_edge)
    dists, _ = nn.kneighbors(pred_edge)
    return np.mean(dists < threshold)


def interactive_viewer(pc_gt, pc_restored, label1="GT", label2="Restored", offset=2.0):
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    mode = {"status": "merge"}

    def create_colored_pcd(pc, cmap, x_offset=0):
        pc = pc.copy()
        pc[:, 0] += x_offset
        z = pc[:, 2]
        norm = (z - z.min()) / (z.max() - z.min())
        colors = plt.get_cmap(cmap)(norm)[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def get_geometry():
        if mode["status"] == "gt":
            return [create_colored_pcd(pc_gt, 'plasma', 0)]
        elif mode["status"] == "restored":
            return [create_colored_pcd(pc_restored, 'viridis', 0)]
        else:
            return [create_colored_pcd(pc_gt, 'plasma', 0), create_colored_pcd(pc_restored, 'viridis', offset)]

    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("LiDAR Super-Resolution Viewer", 1280, 720)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    scene.scene.set_background([0, 0, 0, 1])

    def update_scene():
        scene.scene.clear_geometry()
        geometries = get_geometry()
        for i, geo in enumerate(geometries):
            scene.scene.add_geometry(f"pcd{i}", geo, rendering.MaterialRecord())
        bbox = geometries[0].get_axis_aligned_bounding_box()
        for geo in geometries[1:]:
            bbox += geo.get_axis_aligned_bounding_box()
        scene.setup_camera(60, bbox, bbox.get_center())

    panel = gui.Vert(0.5, gui.Margins(10, 10, 10, 10))

    btn_gt = gui.Button("Show GT (128)")
    btn_restored = gui.Button("Show Restored")
    btn_merge = gui.Button("Merge View")

    def show_gt(): mode["status"] = "gt"; update_scene()
    def show_restored(): mode["status"] = "restored"; update_scene()
    def show_merge(): mode["status"] = "merge"; update_scene()

    btn_gt.set_on_clicked(show_gt)
    btn_restored.set_on_clicked(show_restored)
    btn_merge.set_on_clicked(show_merge)

    panel.add_child(btn_gt)
    panel.add_child(btn_restored)
    panel.add_child(btn_merge)

    window.add_child(scene)
    window.add_child(panel)

    def on_layout(ctx):
        r = window.content_rect
        panel.frame = gui.Rect(r.get_right() - 220, r.y, 220, r.height)
        scene.frame = gui.Rect(r.x, r.y, r.width - 220, r.height)

    window.set_on_layout(on_layout)
    update_scene()
    gui.Application.instance.run()


def apply_global_icp(source, target, voxel_size=0.1, icp_threshold=0.5):
    def preprocess(pcd, voxel_size):
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
        )
        return pcd, fpfh

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target)

    src_down, src_fpfh = preprocess(source_pcd, voxel_size)
    tgt_down, tgt_fpfh = preprocess(target_pcd, voxel_size)

    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, True,
        voxel_size * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)
    )

    icp = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, icp_threshold, ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    aligned = source_pcd.transform(icp.transformation)
    return np.asarray(aligned.points)


if __name__ == "__main__":
    pc_32_full = np.load("processed/output_32_raw.npy")
    pc_64_full = np.load("processed/output_64_000_raw.npy")
    pc_128_full = np.load("processed/output_128_000_raw.npy")

    print("Estimating edge mask...")
    edge_full = estimate_edge_mask(pc_64_full)

    min_len = min(len(pc_64_full), len(pc_128_full), len(edge_full))
    step = 20
    mask = np.arange(0, min_len, step)

    pc_32 = pc_32_full[mask]
    pc_64 = pc_64_full[mask]
    pc_128 = pc_128_full[mask]
    edge = edge_full[mask]

    print("Running super-resolution...")
    restored = potts_super_resolve(pc_64, pc_128, edge, iters=100)

    print("Aligning using global + ICP...")
    restored = apply_global_icp(restored, pc_128)

    print("\nAccuracy Metrics:")
    cd = chamfer_distance(restored, pc_128)
    print(f"Chamfer Distance: {cd:.6f}")

    p2p_gt_to_restored = avg_point_to_point_distance(pc_128, restored)
    p2p_restored_to_gt = avg_point_to_point_distance(restored, pc_128)
    print(f"Avg point to point distance (raw → Restored):   {p2p_gt_to_restored:.6f}")
    print(f"Avg point to point distance (Restored → raw):   {p2p_restored_to_gt:.6f}")

    dens_gt, dens_restored = density_consistency(pc_128, restored)
    print(f"Density (RAW): {dens_gt:.2f} | Density (Restored): {dens_restored:.2f}")

    edge_score = edge_preservation_score(restored, pc_128, edge)
    print(f"Edge Preservation Score: {edge_score * 100:.2f}%")

    interactive_viewer(pc_128, restored)

    print("\nFinal Point Counts:")
    print("Input (64-beam):", len(pc_64))
    print("Restored:", len(restored))
    print("Raw 128-beam:", len(pc_128))
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from potts_sr import potts_super_resolve, chamfer_distance


def estimate_edge_mask(pc, k=16, threshold=0.02):
    nn = NearestNeighbors(n_neighbors=k).fit(pc)
    _, idx = nn.kneighbors(pc)
    local_var = np.var(pc[idx], axis=1).mean(axis=1)
    return local_var > threshold


def avg_point_to_point_distance(A, B):
    nn = NearestNeighbors(n_neighbors=1).fit(B)
    dists, _ = nn.kneighbors(A)
    return np.mean(dists)


def density_consistency(A, B, radius=0.2):
    def avg_neighbors(pc):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        densities = []
        for pt in pc:
            [_, idx, _] = pcd_tree.search_radius_vector_3d(pt, radius)
            densities.append(len(idx))
        return np.mean(densities)

    return avg_neighbors(A), avg_neighbors(B)


def edge_preservation_score(pred, gt, edge_mask, threshold=0.05):
    pred_edge = pred[edge_mask]
    gt_edge = gt[edge_mask]
    nn = NearestNeighbors(n_neighbors=1).fit(gt_edge)
    dists, _ = nn.kneighbors(pred_edge)
    return np.mean(dists < threshold)


def interactive_viewer(pc_gt, pc_restored, label1="GT", label2="Restored", offset=2.0):
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    mode = {"status": "merge"}

    def create_colored_pcd(pc, cmap, x_offset=0):
        pc = pc.copy()
        pc[:, 0] += x_offset
        z = pc[:, 2]
        norm = (z - z.min()) / (z.max() - z.min())
        colors = plt.get_cmap(cmap)(norm)[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def get_geometry():
        if mode["status"] == "gt":
            return [create_colored_pcd(pc_gt, 'plasma', 0)]
        elif mode["status"] == "restored":
            return [create_colored_pcd(pc_restored, 'viridis', 0)]
        else:
            return [create_colored_pcd(pc_gt, 'plasma', 0), create_colored_pcd(pc_restored, 'viridis', offset)]

    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("LiDAR Super-Resolution Viewer", 1280, 720)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    scene.scene.set_background([0, 0, 0, 1])

    def update_scene():
        scene.scene.clear_geometry()
        geometries = get_geometry()
        for i, geo in enumerate(geometries):
            scene.scene.add_geometry(f"pcd{i}", geo, rendering.MaterialRecord())
        bbox = geometries[0].get_axis_aligned_bounding_box()
        for geo in geometries[1:]:
            bbox += geo.get_axis_aligned_bounding_box()
        scene.setup_camera(60, bbox, bbox.get_center())

    panel = gui.Vert(0.5, gui.Margins(10, 10, 10, 10))

    btn_gt = gui.Button("Show GT (128)")
    btn_restored = gui.Button("Show Restored")
    btn_merge = gui.Button("Merge View")

    def show_gt(): mode["status"] = "gt"; update_scene()
    def show_restored(): mode["status"] = "restored"; update_scene()
    def show_merge(): mode["status"] = "merge"; update_scene()

    btn_gt.set_on_clicked(show_gt)
    btn_restored.set_on_clicked(show_restored)
    btn_merge.set_on_clicked(show_merge)

    panel.add_child(btn_gt)
    panel.add_child(btn_restored)
    panel.add_child(btn_merge)

    window.add_child(scene)
    window.add_child(panel)

    def on_layout(ctx):
        r = window.content_rect
        panel.frame = gui.Rect(r.get_right() - 220, r.y, 220, r.height)
        scene.frame = gui.Rect(r.x, r.y, r.width - 220, r.height)

    window.set_on_layout(on_layout)
    update_scene()
    gui.Application.instance.run()


def apply_global_icp(source, target, voxel_size=0.1, icp_threshold=0.5):
    def preprocess(pcd, voxel_size):
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
        )
        return pcd, fpfh

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target)

    src_down, src_fpfh = preprocess(source_pcd, voxel_size)
    tgt_down, tgt_fpfh = preprocess(target_pcd, voxel_size)

    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, True,
        voxel_size * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)
    )

    icp = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, icp_threshold, ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    aligned = source_pcd.transform(icp.transformation)
    return np.asarray(aligned.points)


if __name__ == "__main__":
    pc_32_full = np.load("processed/output_32_raw.npy")
    pc_64_full = np.load("processed/output_64_000_raw.npy")
    pc_128_full = np.load("processed/output_128_000_raw.npy")

    print("Estimating edge mask...")
    edge_full = estimate_edge_mask(pc_64_full)

    min_len = min(len(pc_64_full), len(pc_128_full), len(edge_full))
    step = 20
    mask = np.arange(0, min_len, step)

    pc_32 = pc_32_full[mask]
    pc_64 = pc_64_full[mask]
    pc_128 = pc_128_full[mask]
    edge = edge_full[mask]

    print("Running super-resolution...")
    restored = potts_super_resolve(pc_64, pc_128, edge, iters=100)

    print("Aligning using global + ICP...")
    restored = apply_global_icp(restored, pc_128)

    print("\nAccuracy Metrics:")
    cd = chamfer_distance(restored, pc_128)
    print(f"Chamfer Distance: {cd:.6f}")

    p2p_gt_to_restored = avg_point_to_point_distance(pc_128, restored)
    p2p_restored_to_gt = avg_point_to_point_distance(restored, pc_128)
    print(f"Avg point to point distance (raw → Restored):   {p2p_gt_to_restored:.6f}")
    print(f"Avg point to point distance (Restored → raw):   {p2p_restored_to_gt:.6f}")

    dens_gt, dens_restored = density_consistency(pc_128, restored)
    print(f"Density (RAW): {dens_gt:.2f} | Density (Restored): {dens_restored:.2f}")

    edge_score = edge_preservation_score(restored, pc_128, edge)
    print(f"Edge Preservation Score: {edge_score * 100:.2f}%")

    interactive_viewer(pc_128, restored)

    print("\nFinal Point Counts:")
    print("Input (64-beam):", len(pc_64))
    print("Restored:", len(restored))
    print("Raw 128-beam:", len(pc_128))
