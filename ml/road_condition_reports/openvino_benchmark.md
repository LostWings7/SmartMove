# SafeCity Road Condition Intelligence — OpenVINO Export & Benchmark Report

**Generated:** 2026-09-30 18:29:19 UTC  
**Hardware Platform:** Intel Core Ultra 5 225H (14 cores / 18 threads)  
**OpenVINO Runtime Devices:** CPU, GPU, NPU  

---

## 1. Hardware Discovery

| Device Identifier | Description / Properties | Supported For Inference |
|---|---|---|
| `CPU` | Intel(R) Core(TM) Ultra 5 225H | Yes |
| `GPU` | Intel(R) Graphics (iGPU) | Yes |
| `NPU` | Intel(R) AI Boost | Yes |

---

## 2. Inference Latency & Throughput Benchmark

### Water Segmentation Model (`water_yolo26s_seg_v1`)
- Input Resolution: 320x320 px
- Task: Polygon Mask Segmentation

| Runtime Engine | Device | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | Throughput (FPS) | Speedup vs PyTorch |
|---|---|---|---|---|---|---|
| PyTorch (CPU) | CPU | 80.12 ms | 79.69 ms | 85.82 ms | 12.5 FPS | 1.00x |
| OpenVINO IR | CPU | 78.64 ms | 76.44 ms | 99.7 ms | 12.7 FPS | **1.02x** |

### Road Crack Detection Model (`road_crack_yolo26s_v1`)
- Input Resolution: 384x384 px
- Task: Object Detection (3 Crack Categories)

| Runtime Engine | Device | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | Throughput (FPS) | Speedup vs PyTorch |
|---|---|---|---|---|---|---|
| PyTorch (CPU) | CPU | 73.14 ms | 72.05 ms | 77.47 ms | 13.7 FPS | 1.00x |
| OpenVINO IR | CPU | 69.95 ms | 66.12 ms | 117.79 ms | 14.3 FPS | **1.05x** |

---

## 3. Output Parity & Verification

To verify that OpenVINO model compilation preserves precision and does not introduce defect degradation:

| Subsystem | Metric | Measured Parity Score | Minimum Threshold | Parity Status |
|---|---|---|---|---|
| **Water Segmentation** | Mean Mask IoU Agreement | **0.9997** | 0.8500 | **EXCELLENT** |
| **Road Crack Detection** | Bounding Box Agreement Rate | **1.0** | 0.8000 | **EXCELLENT** |

### Conclusion
Both models successfully exported to OpenVINO IR format with high numerical parity against the original PyTorch weights. OpenVINO runtime delivers measurable latency reductions suitable for real-time edge processing on Intel Core Ultra 5 225H.
