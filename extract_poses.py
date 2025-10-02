import numpy as np
import open3d as o3d
from ouster.sdk.client import SensorInfo, XYZLut, Scans
from ouster.sdk.pcap import Pcap


json_file = "32 beam/OS-1-32-U0_v3.0.1_1024x10_20230510_100110.json"
pcap_file = "32 beam/OS-1-32-U0_v3.0.1_1024x10_20230510_100110-000.pcap"

with open(json_file, 'r') as f:
    metadata_str = f.read()

info = SensorInfo(metadata_str)
pcap = Pcap(pcap_file, info)
xyzlut = XYZLut(info)

# Iterate through scans
for scan in Scans(pcap):
    xyz = xyzlut(scan)  # shape: (H, W, 3)
    pc = xyz.reshape(-1, 3)

    pc = pc[~np.isnan(pc).any(axis=1)]
    pc = pc[~np.isinf(pc).any(axis=1)]

    print(f"Extracted {len(pc)} points")
    np.save("processed/aligned_32_beam_from_pcap.npy", pc)

    # Optional visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])
    break  # remove this to extract more than 1 frame
