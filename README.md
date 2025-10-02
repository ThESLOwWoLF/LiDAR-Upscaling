# LiDAR-Upscaling


# LiDAR Point Cloud Super-Resolution 

This project implements a **LiDAR point cloud super-resolution pipeline** that upscales low-resolution LiDAR scans (e.g., 32/64-beam) into higher-resolution representations (128-beam). The method leverages **Potts model optimization, edge preservation, density consistency checks, and ICP alignment** to generate enhanced point clouds for applications such as **autonomous driving, mapping, and robotics**.  

---

##  Features  

- **Edge-aware Upscaling**: Uses local variance to detect and preserve sharp edges.  
- **Potts Model Super-Resolution**: Improves point cloud density and structural fidelity.  
- **Chamfer Distance & Metrics**: Quantitative evaluation with Chamfer distance, point-to-point distances, and density statistics.  
- **Edge Preservation Score**: Measures how well fine-grained edges are retained.  
- **Global Registration + ICP Alignment**: Aligns restored point clouds with ground truth for fair evaluation.  
- **Interactive 3D Viewer (Open3D GUI)**: Toggle between **Ground Truth**, **Restored**, and **Merged View** with color maps.  

---

## Metrics Implemented  

- **Chamfer Distance** (Restored ↔ Ground Truth)  
- **Avg. Point-to-Point Distance**  
- **Density Consistency** (neighborhood density comparison)  
- **Edge Preservation Score**  

## Visualization  

Interactive Open3D viewer allows you to compare:  
- **Ground Truth (128-beam LiDAR)**  
- **Restored Super-Resolution Output**  
- **Side-by-Side / Merged View**  

Color maps:  
- **GT** → Plasma  
- **Restored** → Viridis  

---

## Project Structure  

```plaintext
├── processed/
│   ├── output_32_raw.npy
│   ├── output_64_000_raw.npy
│   ├── output_128_000_raw.npy
├── potts_sr.py        # Potts model SR + Chamfer distance utilities
├── lidar_sr.py        # Main pipeline (this code)
