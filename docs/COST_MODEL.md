# Cost Model — Derivation and Sensitivity Analysis

This document provides the full derivation of the partner-subsidized billing cost model and the sensitivity analysis.

---

## 1. Problem Statement

The central economic question of the v2 architecture is:

> **Can a partner wallet absorb the merchant's mobile data cost while remaining profitable?**

The answer depends on three factors:
1. **Payload size** (P_size): How many bytes per transaction?
2. **Data price** (P_data): How much does mobile data cost per MB?
3. **MDR revenue** (R_MDR): How much does the wallet earn per transaction?

---

## 2. Notation

| Symbol | Meaning | Units |
|--------|---------|-------|
| N_txn | Number of daily transactions | count |
| P_size | Payload size per transaction | bytes |
| P_data | Mobile data price | $/MB |
| A_avg | Average transaction amount | $ |
| r_MDR | Merchant Discount Rate | fraction |
| C_daily | Daily data cost | $ |
| R_daily | Daily MDR revenue | $ |
| ρ | Cost ratio (C_daily / R_daily) | % |

---

## 3. Equations

### Daily Data Cost

Each transaction transmits P_size bytes. With N_txn transactions per day:

```
Total bytes per day = N_txn × P_size
Total MB per day    = (N_txn × P_size) / 1,048,576
Daily cost          = (N_txn × P_size / 1,048,576) × P_data
```

### Daily MDR Revenue

Each transaction generates MDR revenue equal to the transaction amount times the MDR rate:

```
Daily revenue = N_txn × A_avg × r_MDR
```

### Cost Ratio

The percentage of MDR revenue consumed by data cost:

```
ρ = (C_daily / R_daily) × 100%
```

---

## 4. Numerical Example (Yemen)

### Base Case Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| N_txn | 100,000 transactions/day | National-scale partner wallet assumption |
| P_size | 180 bytes | Architecture specification (Section IV-C) |
| P_data | $1.00 / MB | Cable.co.uk upper bound for Yemen (2024) |
| A_avg | $5.00 | Typical Yemeni micro-transaction |
| r_MDR | 1% | Emerging markets standard MDR |

### Calculation

```
C_daily = (100,000 × 180 / 1,048,576) × $1.00
        = (18,000,000 / 1,048,576) × $1.00
        = 17.17 MB × $1.00
        = $17.17

R_daily = 100,000 × $5.00 × 0.01
        = $5,000

ρ = ($17.17 / $5,000) × 100%
  = 0.343%
```

**Result**: Data cost consumes only **0.34%** of MDR revenue.

---

## 5. Sensitivity Analysis

### 5.1 Varying Mobile Data Price

| P_data ($/MB) | C_daily ($) | ρ (%) |
|---------------|-------------|-------|
| 0.50 | 8.58 | 0.17% |
| 1.00 | 17.17 | 0.34% |
| 2.00 | 34.33 | 0.69% |
| 5.00 | 85.83 | 1.72% |
| 10.00 | 171.66 | 3.43% |

**Interpretation**: Even at $10/MB (10× the Yemen upper bound), the cost ratio remains below 4%.

### 5.2 Varying Transaction Volume

| N_txn | C_daily ($)| R_daily ($) | ρ (%) |
|-------|------------|-------------|-------|
| 1,000 | 0.17 | 50 | 0.34% |
| 10,000 | 1.72 | 500 | 0.34% |
| 100,000 | 17.17 | 5,000 | 0.34% |
| 1,000,000 | 171.66 | 50,000 | 0.34% |

**Interpretation**: The cost ratio is **invariant** to transaction volume — it depends only on P_size, P_data, A_avg, and r_MDR.

### 5.3 Varying Average Transaction Amount

| A_avg ($) | R_daily ($) | ρ (%) |
|-----------|-------------|-------|
| 1 | 1,000 | 1.72% |
| 5 | 5,000 | 0.34% |
| 10 | 10,000 | 0.17% |
| 50 | 50,000 | 0.03% |

**Interpretation**: Higher-value transactions make the model more favorable (data cost is fixed per transaction; revenue scales with amount).

### 5.4 Varying MDR Rate

| r_MDR (%) | R_daily ($) | ρ (%) |
|-----------|-------------|-------|
| 0.5 | 2,500 | 0.69% |
| 1.0 | 5,000 | 0.34% |
| 1.5 | 7,500 | 0.23% |
| 2.0 | 10,000 | 0.17% |

**Interpretation**: Lower MDR rates (more competitive markets) increase the cost ratio but it remains below 1% at typical rates.

---

## 6. Break-Even Analysis

The model breaks even (ρ = 100%) when:

```
C_daily = R_daily
(N_txn × P_size / 1,048,576) × P_data = N_txn × A_avg × r_MDR
P_size × P_data / 1,048,576 = A_avg × r_MDR
```

Solving for the break-even data price:

```
P_data_break_even = (A_avg × r_MDR × 1,048,576) / P_size
                  = ($5 × 0.01 × 1,048,576) / 180
                  = $291.27 / MB
```

**Interpretation**: Data price would need to be **$291/MB** (291× the Yemen upper bound) for the model to break even. The model is economically robust by a very wide margin.

---

## 7. Comparison with Traditional Models

| Model | Who Pays Data | Cost per Transaction | Sustainability |
|-------|---------------|----------------------|----------------|
| Traditional (merchant pays) | Merchant | ~$0.00017/txn | Burden on small merchants |
| Zero-rated APN (v1, unverifiable) | Carrier | $0 (claimed) | Depends on carrier agreement |
| **Partner-subsidized (v2)** | **Partner wallet** | **~$0.00017/txn** | **Sustainable (0.34% of MDR)** |

---

## 8. Assumptions and Limitations

### Assumptions
1. The partner wallet has a B2B agreement with the carrier to identify and bill Atheer traffic
2. The carrier charges standard mobile data rates (no special discount)
3. All transactions successfully complete (upper bound on data cost)
4. No additional overhead (TCP ACKs, TLS handshake) — conservative, as these are typically amortized

### Limitations
1. The model does not account for the partner wallet's operational costs (server, HSM, staff)
2. The model assumes the carrier can accurately identify Atheer traffic (via SNI/certificate pin)
3. Real-world packet loss would reduce the effective data cost (fewer successful transmissions)
4. The model does not consider potential volume discounts from the carrier

### Conservative Bias
The model is **conservative** (overestimates cost) because:
- It uses the upper bound of Yemen data pricing ($1.00/MB vs. typical $0.30-0.80/MB)
- It does not account for failed transactions (which don't generate MDR revenue but also don't consume data)
- It assumes no carrier volume discounts

---

## 9. Conclusion

The cost model demonstrates that the partner-subsidized billing approach is **economically sustainable by a wide margin**:

- **Base case**: 0.34% cost ratio (291× safety margin)
- **Worst case** (10× data price): 3.43% cost ratio (29× safety margin)
- **Break-even**: $291/MB (291× current upper bound)

This justifies the v2 architecture's shift from Private APN (which required unverifiable zero-rating claims) to Mobile Data Routing (which is economically sustainable based on publicly available pricing data).
