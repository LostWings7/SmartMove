# SafeCity Loop V2 — Traffic AI Dataset Report (V2 Expansion)

**Dataset Source:** Berkeley DeepDrive (BDD100K) — BSD 3-Clause License  
**Generated:** 2026-09-30 18:31:11  
**Total Frames:** 2271 (vs 704 in V1)  
**Total Video Sequences:** 958 (vs 103 in V1)  
**Zero Temporal Leakage:** Strict Sequence UUID Partitioning  

---

## 1. Class Distribution Comparison: V1 vs V2

| Class ID | Class Name | V1 Count | V2 Count | Absolute Gain | % Increase | V2 Relative Share |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | **pedestrian** | 2,563 | 5,931 | +3,368 | **+131.4%** | 18.16% |
| 1 | **rider** | 104 | 658 | +554 | **+532.7%** | 2.02% |
| 2 | **car** | 7,404 | 22,888 | +15,484 | **+209.1%** | 70.09% |
| 3 | **truck** | 350 | 1,166 | +816 | **+233.1%** | 3.57% |
| 4 | **bus** | 177 | 512 | +335 | **+189.3%** | 1.57% |
| 5 | **motorcycle** | 40 | 460 | +420 | **+1050.0%** | 1.41% |
| 6 | **bicycle** | 201 | 1,039 | +838 | **+416.9%** | 3.18% |
| — | **TOTAL** | **10,839** | **32,654** | **+21,815** | **+201.3%** | 100.0% |

---

## 2. Dataset Split Summary (Strict Sequence Partitioning)

| Split | Sequences | Frames | Total Objects | Motorcycle | Rider | Bicycle | Pedestrian | Car |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 670 | 1593 | 23069 | 319 | 485 | 730 | 4213 | 16113 |
| **VAL** | 142 | 306 | 4502 | 64 | 98 | 173 | 758 | 3184 |
| **TEST** | 146 | 372 | 5083 | 77 | 75 | 136 | 960 | 3591 |

---

## 3. Object Scale & Environmental Diversity

- **Small Objects (< 32x32 px):** 13,467 (41.2%)
- **Medium Objects (32x32 to 96x96 px):** 13,361 (40.9%)
- **Large Objects (> 96x96 px):** 5,826 (17.8%)

### Lighting & Weather Conditions

| Attribute | Categories Present in V2 |
| :--- | :--- |
| **Weather** | clear: 1098, undefined: 389, overcast: 304, partly cloudy: 187, snowy: 153, rainy: 139, foggy: 1 |
| **Time of Day** | daytime: 1436, night: 649, dawn/dusk: 183, undefined: 3 |
| **Scene** | city street: 1639, highway: 354, residential: 264, undefined: 5, parking lot: 5, gas stations: 2, tunnel: 2 |

---

## 4. Key Takeaways for V2 Training
1. **Motorcycle instances increased from 40 to over 400+** (>10x expansion).
2. **Rider instances increased from 104 to over 600+** (>6x expansion).
3. **Bicycle instances increased from 201 to over 1,000+** (>5x expansion).
4. **Strict Sequence Grouping** ensures held-out test split has authentic unseen driving videos with genuine two-wheeler interactions.