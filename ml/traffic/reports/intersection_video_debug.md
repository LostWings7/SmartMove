# SafeCity Video Processing Pipeline — Intersection Video Debug Report

- **Video File**: `10065568-849_orig.mp4`
- **Resolution**: `1280x720`
- **Video FPS**: `29.970`
- **Total Frames**: `2173`
- **Duration**: `72.51s` (`1:12`)
- **Analysis Wall Time**: `55.68s`
- **Model**: `SafeCity Traffic YOLO26s v2`
- **Tracker**: `bytetrack.yaml`
- **Camera Mode**: `aerial_intersection`
- **Calibration Status**: `NOMINAL_PIXEL_ESTIMATE`
- **Tracking Health Status**: `ANALYSIS_HEALTHY`

## 1. Tracking Health & Trajectory Summary

| Metric | Value |
|---|---|
| Total Confirmed Tracks | **225** |
| Sustained Trajectories (>=3 pts) | **152** |
| Long-Lived Trajectories (>=10 pts) | **83** |
| Median Track Duration | **0.9s** |
| Mean Samples per Track | **10.7** |
| Single-Point Track Percentage | **17.3%** |
| Processed Frames | **1087** |
| Identified Near-Miss Candidates | **65** |

## 2. Top Sustained Trajectory Telemetry (Confirmed Road Users)

| Track ID | Object Type | Est. Speed | Direction | Trajectory Length |
|---|---|---|---|---|
| #1365 | Car | 14.4 km/h | Westbound | 53 points |
| #91 | Car | 0.1 km/h | Southbound | 51 points |
| #786 | Car | 32.4 km/h | Westbound | 50 points |
| #901 | Car | 61.6 km/h | Westbound | 47 points |
| #124 | Pedestrian | 3.5 km/h | Southbound | 47 points |
| #1390 | Car | 9.5 km/h | Westbound | 46 points |
| #881 | Car | 22.0 km/h | Westbound | 45 points |
| #929 | Car | 56.6 km/h | Westbound | 42 points |
| #81 | Car | 13.0 km/h | Southbound | 42 points |
| #1038 | Pedestrian | 6.1 km/h | Westbound | 42 points |
| #261 | Pedestrian | 11.2 km/h | Westbound | 40 points |
| #295 | Car | 14.0 km/h | Westbound | 38 points |
| #1092 | Pedestrian | 17.4 km/h | Westbound | 37 points |
| #833 | Car | 81.1 km/h | Westbound | 36 points |
| #476 | Car | 28.7 km/h | Westbound | 34 points |

## 3. Conflict Analysis & Surrogate Safety

Identified 65 near-miss candidate(s) meeting TTC <= 1.05s.
- **NM-b6fb-139-132** at `00:14`: car vs car, TTC=1.05s, Clearance=5.4m, Severity=HIGH
- **NM-b6fb-124-207** at `00:16`: pedestrian vs car, TTC=0.62s, Clearance=6.7m, Severity=CRITICAL
- **NM-b6fb-255-256** at `00:18`: car vs car, TTC=0.25s, Clearance=4.2m, Severity=CRITICAL
- **NM-b6fb-344-261** at `00:20`: car vs pedestrian, TTC=1.35s, Clearance=7.2m, Severity=CRITICAL
- **NM-b6fb-406-385** at `00:23`: car vs car, TTC=0.19s, Clearance=3.7m, Severity=CRITICAL
- **NM-b6fb-413-418** at `00:23`: car vs car, TTC=0.82s, Clearance=5.4m, Severity=CRITICAL
- **NM-b6fb-418-406** at `00:23`: car vs car, TTC=1.82s, Clearance=11.0m, Severity=MODERATE
- **NM-b6fb-406-413** at `00:25`: car vs car, TTC=1.02s, Clearance=4.3m, Severity=HIGH
- **NM-b6fb-447-476** at `00:27`: car vs car, TTC=0.60s, Clearance=10.8m, Severity=CRITICAL
- **NM-b6fb-464-455** at `00:27`: car vs car, TTC=0.89s, Clearance=8.4m, Severity=CRITICAL
- **NM-b6fb-489-476** at `00:28`: car vs car, TTC=0.32s, Clearance=5.5m, Severity=CRITICAL
- **NM-b6fb-464-476** at `00:28`: car vs car, TTC=1.14s, Clearance=9.8m, Severity=HIGH
- **NM-b6fb-573-584** at `00:33`: car vs pedestrian, TTC=0.44s, Clearance=7.7m, Severity=CRITICAL
- **NM-b6fb-573-648** at `00:33`: car vs pedestrian, TTC=0.64s, Clearance=13.0m, Severity=CRITICAL
- **NM-b6fb-669-584** at `00:34`: car vs pedestrian, TTC=0.60s, Clearance=9.2m, Severity=CRITICAL
- **NM-b6fb-670-727** at `00:35`: pedestrian vs car, TTC=0.61s, Clearance=9.4m, Severity=CRITICAL
- **NM-b6fb-670-776** at `00:36`: pedestrian vs car, TTC=0.44s, Clearance=7.1m, Severity=CRITICAL
- **NM-b6fb-755-727** at `00:36`: pedestrian vs car, TTC=2.00s, Clearance=15.4m, Severity=HIGH
- **NM-b6fb-786-670** at `00:37`: car vs pedestrian, TTC=0.65s, Clearance=11.7m, Severity=CRITICAL
- **NM-b6fb-786-806** at `00:37`: car vs pedestrian, TTC=1.62s, Clearance=8.5m, Severity=CRITICAL
- **NM-b6fb-833-832** at `00:38`: car vs rider, TTC=1.15s, Clearance=8.5m, Severity=CRITICAL
- **NM-b6fb-786-776** at `00:39`: car vs car, TTC=1.30s, Clearance=11.4m, Severity=HIGH
- **NM-b6fb-833-845** at `00:39`: car vs car, TTC=1.90s, Clearance=8.7m, Severity=CRITICAL
- **NM-b6fb-889-913** at `00:41`: rider vs bicycle, TTC=0.10s, Clearance=0.5m, Severity=CRITICAL
- **NM-b6fb-881-924** at `00:41`: car vs car, TTC=1.02s, Clearance=4.7m, Severity=HIGH
- **NM-b6fb-881-883** at `00:43`: car vs car, TTC=0.91s, Clearance=9.0m, Severity=CRITICAL
- **NM-b6fb-929-952** at `00:43`: car vs car, TTC=0.60s, Clearance=5.4m, Severity=CRITICAL
- **NM-b6fb-881-901** at `00:43`: car vs car, TTC=0.90s, Clearance=8.9m, Severity=CRITICAL
- **NM-b6fb-959-985** at `00:43`: car vs car, TTC=1.35s, Clearance=4.6m, Severity=HIGH
- **NM-b6fb-1018-991** at `00:46`: car vs car, TTC=0.70s, Clearance=7.3m, Severity=CRITICAL
- **NM-b6fb-1073-1070** at `00:48`: car vs car, TTC=1.04s, Clearance=11.6m, Severity=CRITICAL
- **NM-b6fb-1076-1073** at `00:48`: car vs car, TTC=1.30s, Clearance=8.5m, Severity=CRITICAL
- **NM-b6fb-1066-1099** at `00:49`: pedestrian vs car, TTC=0.27s, Clearance=3.4m, Severity=CRITICAL
- **NM-b6fb-1066-1070** at `00:49`: pedestrian vs car, TTC=0.74s, Clearance=7.2m, Severity=CRITICAL
- **NM-b6fb-1092-1109** at `00:49`: pedestrian vs car, TTC=0.72s, Clearance=4.2m, Severity=CRITICAL
- **NM-b6fb-1109-1111** at `00:49`: car vs car, TTC=1.10s, Clearance=5.3m, Severity=HIGH
- **NM-b6fb-1076-1066** at `00:49`: car vs pedestrian, TTC=0.69s, Clearance=11.7m, Severity=CRITICAL
- **NM-b6fb-1122-1124** at `00:50`: pedestrian vs car, TTC=0.37s, Clearance=8.3m, Severity=CRITICAL
- **NM-b6fb-1092-1138** at `00:51`: pedestrian vs car, TTC=1.31s, Clearance=5.7m, Severity=CRITICAL
- **NM-b6fb-1146-1122** at `00:51`: car vs pedestrian, TTC=0.67s, Clearance=15.8m, Severity=CRITICAL
- **NM-b6fb-1138-1122** at `00:52`: car vs pedestrian, TTC=0.23s, Clearance=4.9m, Severity=CRITICAL
- **NM-b6fb-1154-1146** at `00:52`: pedestrian vs car, TTC=0.65s, Clearance=5.1m, Severity=CRITICAL
- **NM-b6fb-1161-1166** at `00:52`: car vs car, TTC=0.98s, Clearance=5.4m, Severity=CRITICAL
- **NM-b6fb-1146-1166** at `00:52`: car vs car, TTC=1.29s, Clearance=10.3m, Severity=HIGH
- **NM-b6fb-1161-1122** at `00:52`: car vs pedestrian, TTC=0.27s, Clearance=6.3m, Severity=CRITICAL
- **NM-b6fb-1122-1166** at `00:52`: pedestrian vs car, TTC=0.21s, Clearance=6.1m, Severity=CRITICAL
- **NM-b6fb-1161-1183** at `00:53`: car vs pedestrian, TTC=0.24s, Clearance=4.6m, Severity=CRITICAL
- **NM-b6fb-1183-1166** at `00:53`: pedestrian vs car, TTC=0.44s, Clearance=4.8m, Severity=CRITICAL
- **NM-b6fb-1190-1212** at `00:54`: car vs car, TTC=1.18s, Clearance=8.0m, Severity=HIGH
- **NM-b6fb-1380-1372** at `00:58`: car vs car, TTC=0.11s, Clearance=0.8m, Severity=CRITICAL
- **NM-b6fb-1380-1390** at `00:58`: car vs car, TTC=0.82s, Clearance=4.2m, Severity=CRITICAL
- **NM-b6fb-1365-1433** at `00:59`: car vs car, TTC=1.18s, Clearance=3.2m, Severity=HIGH
- **NM-b6fb-1365-1472** at `00:59`: car vs car, TTC=1.42s, Clearance=6.1m, Severity=HIGH
- **NM-b6fb-1365-1390** at `01:00`: car vs car, TTC=0.87s, Clearance=4.2m, Severity=CRITICAL
- **NM-b6fb-1472-1433** at `01:00`: car vs car, TTC=1.24s, Clearance=2.9m, Severity=HIGH
- **NM-b6fb-1365-1519** at `01:00`: car vs car, TTC=1.28s, Clearance=4.7m, Severity=HIGH
- **NM-b6fb-1511-1488** at `01:00`: car vs car, TTC=1.80s, Clearance=5.8m, Severity=MODERATE
- **NM-b6fb-1511-1519** at `01:00`: car vs car, TTC=0.12s, Clearance=2.0m, Severity=CRITICAL
- **NM-b6fb-1481-1511** at `01:00`: car vs car, TTC=1.62s, Clearance=7.0m, Severity=MODERATE
- **NM-b6fb-1517-1487** at `01:00`: car vs car, TTC=1.15s, Clearance=3.9m, Severity=HIGH
- **NM-b6fb-1390-1517** at `01:01`: car vs car, TTC=1.96s, Clearance=3.5m, Severity=MODERATE
- **NM-b6fb-1390-1506** at `01:01`: car vs car, TTC=1.12s, Clearance=1.6m, Severity=HIGH
- **NM-b6fb-1390-1487** at `01:01`: car vs truck, TTC=1.69s, Clearance=7.1m, Severity=MODERATE
- **NM-b6fb-1506-1487** at `01:01`: car vs truck, TTC=1.77s, Clearance=8.6m, Severity=MODERATE
- **NM-b6fb-1511-1557** at `01:01`: car vs car, TTC=1.72s, Clearance=9.1m, Severity=MODERATE
