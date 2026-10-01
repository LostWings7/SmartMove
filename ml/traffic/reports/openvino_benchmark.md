# SafeCity Loop V2 — OpenVINO Export & Hardware Benchmark Report

**Exported Model:** `models\traffic_yolo26s_v2_openvino_model\`  
**Source Weights:** `models/traffic_yolo26s_v2.pt`  
**Evaluation Date:** 2026-09-30 20:01:22  
**Image Size:** 512x512 px  

---

## 1. Hardware Benchmark Summary

| Device | Status | Mean Latency | P50 Latency | P95 Latency | Throughput |
|---|---|---|---|---|---|
| **CPU** | `SUCCESS` | `8.92 ms` | `8.82 ms` | `10.82 ms` | `112.1 FPS` |
| **GPU** | `UNSUPPORTED` | N/A | N/A | N/A | N/A |
| **NPU** | `UNSUPPORTED` | N/A | N/A | N/A | N/A |

---

## 2. Video Temporal Parity Analysis (PyTorch vs OpenVINO)

| Parity Metric | Result | Target Standard |
|---|---|---|
| **Frames Tested** | `7` | Consecutive temporal frames |
| **PyTorch Detections** | `115` | Baseline |
| **OpenVINO Detections** | `113` | Quantized/IR Inference |
| **Detection Discrepancy** | `1.74%` | $< 8.0\%$ allowable |
| **Mean Confidence Delta** | `0.0228` | $< 0.05$ |
| **Parity Status** | `VERIFIED` | Strict consistency |
