# Parameter Mapping (Paper ↔ Config)

Update `configs/paper.yml` whenever you change scenario parameters.

The parameters in `configs/paper.yml` are the **exact** values fed into the simulation engine. Table II in the IEEE paper presents rounded/simplified values for academic readability.

| Parameter Name          | Paper (Table II) | Actual `paper.yml` Value | Impact |
|-------------------------|------------------|--------------------------|--------|
| Network Mean Latency S1 | 400 ms           | 200 ms                   | Lower latency baseline than text, but degradation spikes it |
| Network Mean Latency S2 | 60 ms            | 65 ms                    | Slightly higher |
| Max Packet Loss S1      | 35%              | 45%                      | More severe loss than text |
| Baseline Loss S1        | 10%              | 5%                       | Lower baseline |
| Bank capacity           | 50               | 50                       | Exact match |
| S1/S2 Timeouts (E2E)    | 15s / 5s         | 15.0 / 5.0               | Exact match |

> **Note:** The simulation results (Figures 6, 7 and Tables III, IV) are generated using the `paper.yml` values above.
