# SafeCity Loop V2 — OpenVINO Hardware Deployment Benchmark

- **Exported Model Artifact:** `models\pothole_yolo26s_openvino_model\`
- **Source Weights:** `models/pothole_yolo26s.pt`

## 1. Multi-Device Hardware Performance

| Device | Hardware Description | Mean Latency | Median (P50) | P95 Latency | FPS | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU** | Intel(R) Core(TM) Ultra 5 225H | **100.65 ms** | 97.62 ms | 149.25 ms | **9.9** | Active |
| **GPU** | Intel(R) Graphics (iGPU) | **23.9 ms** | 24.02 ms | 26.13 ms | **41.8** | Active |
| **NPU** | Intel(R) AI Boost | **18.94 ms** | 18.7 ms | 21.87 ms | **52.8** | Active |

## 2. Accuracy Parity Check (PyTorch vs. OpenVINO)

- Evaluated Test Images: 50
- PyTorch Detections: 20
- OpenVINO Detections: 20
- Mean Bounding Box IoU: **0.9933**
- Mean Confidence Delta: `0.0009`
- **Parity Status:** ✅ PASSED (Zero Material Degradation)
