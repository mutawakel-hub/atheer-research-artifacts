

Update this document if you change the model.

# Model Assumptions (Section VII)

- Discrete-event simulation using SimPy.
- Arrivals: Poisson process (exponential inter-arrival) per offered load (TPS).
- **4-Layer End-to-End Model:**
        1. **Edge Layer (SDK):** Local NFC tap and HCE cryptogram generation (local_time_s).
        2. **Network Layer (Transport):** Uplink/Downlink via S1 (Public Internet) or S2 (Private APN), with packet loss and latency.
        3. **Processing Layer (Atheer Switch):** Python-simulated switch with:
                 - *Idempotency Check* (Redis micro-latency)
                 - *Transaction Storage* (PostgreSQL micro-latency)
                 - Centralized filter to prevent double-spending.
        4. **Integration Layer (Core Bank):** Bank processing via API adapters (queue + service).
- **Switch Overhead:** Sum of Redis and PostgreSQL micro-latencies, measured per transaction.
- **E2E timeout:** fail if total latency exceeds configured threshold.
- **Success Rate:** Includes transactions dropped due to network or E2E timeouts.
- **Reproducibility:** All random seeds fixed for academic reproducibility.

