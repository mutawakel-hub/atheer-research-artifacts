# Parameter Mapping (Paper ↔ Config)

Update `configs/paper.yml` whenever you change scenario parameters.

The parameters in `configs/paper.yml` **match exactly** the values declared in Table III of the IEEE paper.

| Parameter Name          | Paper (Table III) | `paper.yml` Value | Status |
|-------------------------|-------------------|--------------------|--------|
| Network Mean Latency S1 | 400 ms            | 400 ms             | ✅ Match |
| Network Mean Latency S2 | 60 ms             | 60 ms              | ✅ Match |
| Baseline Packet Loss S1 | 10%               | 10%                | ✅ Match |
| Max Packet Loss S1      | 35%               | 35%                | ✅ Match |
| S2 Retries              | 0                 | 0                  | ✅ Match |
| Bank capacity           | 50                | 50                 | ✅ Match |
| S1/S2 Timeouts (E2E)    | 15s / 5s          | 15.0 / 5.0         | ✅ Match |

> **Note:** The simulation results (Figures 6, 7 and Tables IV, V) are generated using the `paper.yml` values above.
