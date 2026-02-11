# Model Assumptions (Section VI)

- Discrete-event simulation using SimPy.
- Arrivals: Poisson process (exponential inter-arrival) per offered load (TPS).
- Network latency: LogNormal (positive, right-skewed tail).
- Packet loss: Bernoulli per attempt, with retries up to configured limit.
- Bank/back-end: finite-capacity server (SimPy Resource) with queue timeout.
- E2E timeout: fail if total latency exceeds configured threshold.
- Network-only comparison: bank capacity/service/local times identical across S1 and S2.

Update this document if you change the model.
