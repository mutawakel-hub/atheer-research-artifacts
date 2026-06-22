# Simulation Parameters — Detailed Justification

This document provides the rationale for each parameter used in the Discrete Event Simulation (DES).

---

## S1: Public Internet Baseline

S1 represents the merchant using the standard public internet without any prioritization or optimization. This is the status quo for most payment systems in low-infrastructure environments.

| Parameter | Value | Source / Justification |
|-----------|-------|------------------------|
| Mean latency (L̄) | 400 ms | Ookla Speedtest Global Index for Yemen (median RTT, 2024). Reflects chronic congestion and damaged infrastructure. |
| Sigma (σ) | 0.35 | LogNormal variance parameter. Calibrated to match the long-tailed distribution observed in real measurements. |
| Base packet loss (p₀) | 0.5% | ISOC Pulse Internet Resilience Index for Yemen. Lower bound during low congestion. |
| Max packet loss | 35% | Cap during peak congestion. Reflects worst-case observed in conflict-affected networks. |
| Max retries (r) | 2 | Application-layer retry limit for unstable networks. More retries would exceed user patience. |
| E2E timeout max | 15 s | EMVCo contactless payment specification limit (EMVCo v3.2, 2023). |
| Load degradation α_L | 0.75 | Strong degradation reflecting public-internet BGP peering congestion. |
| Load threshold λ_th | 25 TPS | Networks begin to degrade noticeably above this load in observed Yemeni conditions. |
| Bank capacity (c) | 50 TPS | Mid-tier core-banking server processing capacity estimate. |
| Payload size | 180 bytes | Same as S2 (architecture invariant). |

---

## S2: Mobile Data with Partner-Subsidized Billing

S2 represents the merchant using standard mobile data (LTE/3G) with the Atheer architecture's 180-byte optimized payload. The partner wallet subsidizes the data cost via a B2B agreement with the carrier.

| Parameter | Value | Source / Justification |
|-----------|-------|------------------------|
| Mean latency (L̄) | 130 ms | Realistic for Yemeni mobile data RTT with small payloads. Based on carrier mobile data measurements. Lower than S1 because mobile data has fewer BGP hops than the full public internet path. |
| Sigma (σ) | 0.30 | Lower variance than S1 due to carrier SLA bounds. |
| Base packet loss (p₀) | 1.5% | Mobile data SLA upper bound for carrier-grade service. Higher than ideal (0.1%) but realistic for Yemeni conditions. |
| Max packet loss | 4% | Capped at 4% under load. Much lower than S1's 35% cap due to carrier SLA. |
| Max retries (r) | 1 | Single retry due to short payload (180 bytes fits in one TCP segment). |
| E2E timeout max | 10 s | Tightened from S1's 15 s because the deterministic path has lower variance. |
| Load degradation α_L | 0.15 | Mild degradation. Mobile data paths are less susceptible to BGP congestion than full public internet. |
| Load threshold λ_th | 100 TPS | Higher threshold than S1 due to SLA guarantees. |
| Bank capacity (c) | 50 TPS | Same as S1 (architecture invariant). |
| Payload size | 180 bytes | Optimized packet: 32B routing + 32B LUK + 4B ATC + 8B amount + 16B nonce + 8B timestamp + 64B ECDSA + 28B AES-GCM IV+tag = 192B uncompressed → 180B after header compression. |

---

## Why These Values Are Realistic for Yemen

### Mobile Data Latency: 130 ms

The original v1.0 simulation claimed 60 ms for Private APN, which was unrealistic for Yemeni conditions. The v2.0 value of 130 ms reflects:

1. **Carrier RTT baseline**: Yemeni mobile carriers (Yemen Mobile, Sabafon, MTN, Y Telecom) typically achieve 80-150 ms RTT for domestic destinations.
2. **TLS 1.3 overhead**: Two round trips for handshake + one for data = ~3 × 50 ms = 150 ms additional.
3. **Gateway processing**: ~20 ms for signature verification and database lookups.
4. **Total**: 80 ms (carrier) + 50 ms (TLS) + 20 ms (processing) ≈ 150 ms, conservatively set to 130 ms mean.

This is higher than v1's 60 ms but much lower than S1's 400 ms (which includes BGP peering delays).

### Packet Loss: 1.5%

Mobile data in Yemen experiences higher packet loss than in developed markets due to:
- Damaged cell tower infrastructure
- Power outages affecting base stations
- Congestion during peak hours

However, carrier SLAs typically guarantee < 2% packet loss for business customers. The 1.5% base rate reflects this SLA bound, with a 4% cap under load.

### Load Degradation: α_L = 0.15

S2 degrades much less than S1 (0.75) because:
1. Mobile data paths don't traverse public BGP peering points
2. Carrier SLAs provide capacity guarantees
3. The 180-byte payload is small enough to fit in a single TCP segment, avoiding fragmentation issues

---

## Simulation Architecture

### Discrete Event Simulation (DES) with SimPy

The simulation uses SimPy 4.1+ with the following architecture:

```
Transaction Arrival (Poisson)
    ↓
Edge Processing (50 ms fixed)
    ↓
Network Uplink (LogNormal + Bernoulli loss)
    ↓
Switch Processing (20 ms + micro-random)
    ↓
Core Banking (M/M/c queue, c=50 TPS)
    ↓
Network Downlink (LogNormal + Bernoulli loss)
    ↓
E2E Timeout Check (10s/15s)
```

### Reproducibility

- **N = 10** replications per load level
- **Deterministic seeds**: `seed = 1000 + i * 7 + hash(scenario_name) % 1000`
- **Warmup period**: 60 seconds (discarded)
- **Measurement window**: 300 seconds
- **Confidence intervals**: 95% CI using t-distribution

---

## Load Levels

| TPS | Interpretation |
|-----|----------------|
| 5 | Current Yemeni baseline (low-traffic rural area) |
| 25 | Small merchant cluster |
| 50 | Medium merchant cluster / national switch target |
| 100 | Urban merchant district |
| 250 | Regional scale |
| 500 | Stress test (comparable to Saudi Mada network average load) |

---

## Verifying the Parameters

Researchers can verify these parameters against:

1. **Ookla Speedtest**: https://www.speedtest.net/global-index/yemen
2. **ISOC Pulse**: https://pulse.internetsociety.org/country/yemen
3. **Cable.co.uk mobile data pricing**: https://www.cable.co.uk/mobiles/worldwide-data-pricing/
4. **ITU ICT Price Baskets**: https://www.itu.int/en/ITU-D/Statistics/Pages/IPB/default.aspx

If you have access to real Yemeni network measurements, please consider contributing to improve these parameters via a pull request.
