# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] — 2026-06-22

### Major Revision (Post-DTISD 2026 Rejection)

This release addresses all concerns raised by three reviewers at DTISD 2026, fundamentally restructuring the network layer and economic model.

### Added
- **New Network Layer**: Mobile Data Routing with Partner-Subsidized Billing (replaces Private APN)
- **180-byte Payload Proof**: Formal field-by-field breakdown showing optimization from 192 bytes to 180 bytes
- **Cost Model**: Explicit equations (Eq. 6, 7, 8) with sensitivity analysis (Figure 8)
- **Ethical Statement** (Section III-A): Addresses Reviewer 2 concerns about security claims on named banks
- **Dual-Path Provisioning**: In-band (TLS 1.3) + Out-of-band (bank branch LAN) addressing Reviewer 3's contradiction critique
- **Comparison Table** (Table VI): Atheer vs. M-Pesa, Alkuraimi, Kipaad
- **DDoS row** in Threat Model (Table II): Edge rate limiting + WAF + payload minimization
- **Arabic version** of the paper (XeLaTeX + polyglossia + Amiri font)
- **Full GitHub repository structure** with docs, examples, citation file
- **Reproducible SimPy simulation** with N=10 replications and deterministic seeds
- **8 publication-quality figures** at 200 DPI

### Changed
- **Corrected wallet names**: Jeeb → Jaib, Al-Kuraimi → Alkuraimi, M-Flous → MFloos, Jawaly → Jawali
- **Title**: Removed "A Case Study from Yemen" (Reviewer 3 noted it wasn't a true case study)
- **Simulation parameters** (Table III):
  - S2 latency: 60 ms → 130 ms (realistic for Yemeni mobile data)
  - S2 packet loss: 0.1% → 1.5% (realistic SLA upper bound)
  - S2 retries: 0 → 1
  - S2 load degradation: disabled → enabled (α_L = 0.15, λ_th = 100)
  - S2 E2E timeout: 5 s → 10 s (unified closer to S1)
- **Provisioning section** (V-A): Now explicitly addresses the contradiction of using criticized public network
- **Security section** (VI-D): Rewritten as "Network Security: Mobile Data Hardening" (was "Private APN Segmentation")
- **Economic analysis** (Section VIII): Replaced APN-based claims with explicit cost model

### Removed
- **Private APN** references throughout (title, abstract, keywords, body, references)
- **Zero-rated data claim** (was unverifiable — no public Yemeni carrier SLA data)
- **IaaS revenue stream claim** (was a conceptual confusion — APN is NaaS, not IaaS)
- **Unverifiable references**:
  - [24] Yemen Telecom Tech. Rep. (no public URL)
  - [27] Allot white paper (commercial, biased)
  - [31] Adapt IT blog post (weak source)
  - [32] PwC report (general, doesn't support specific claims)
- **Security claims about named Yemeni banks** (Reviewer 2 ethical concern)

### Fixed
- **Wallet name spellings** (Reviewer 1)
- **Methodological contradiction** in provisioning (Reviewer 3)
- **Unrealistic latency assumptions** for Yemeni networks (Reviewer 3)
- **Missing evidence** for digital infrastructure claims (Reviewer 3)

### Results Comparison (v1 vs. v2)

| Metric | v1 (Private APN) | v2 (Mobile Data) |
|--------|------------------|------------------|
| S2 success at 500 TPS | 99.80% | 97.61% |
| S2 P95 latency at 500 TPS | 245 ms | 672 ms |
| S2 latency assumption | 60 ms (unrealistic) | 130 ms (realistic) |
| Cost model | Implicit (zero-rated) | Explicit (180 B × price/MB ÷ MDR) |
| Ethical concerns | Multiple | Resolved |

The v2 results are more conservative but more credible, reflecting realistic Yemeni network conditions.

---

## [1.0.0] — 2024

### Original Submission to DTISD 2026

- Initial architecture with Private APN network layer
- 4-tier design: Edge, Network (APN), Switch, Integration
- DES evaluation across 6 load levels (5-500 TPS)
- Claimed 99.8% success at 500 TPS on APN pathway
- Open-source simulation artifact published on GitHub

### Outcome
**Rejected** by DTISD 2026 Technical Program Committee based on:
- Reviewer 1: Methodological concerns + wallet name errors
- Reviewer 2: Ethical concerns (security claims on real banks without permission)
- Reviewer 3: Methodological issues + unverifiable claims + IaaS confusion

This feedback drove the v2.0 revision.

---

## Versioning Policy

- **Major** (X.0.0): Fundamental architecture changes, paper revisions
- **Minor** (0.X.0): New simulation scenarios, additional figures, documentation
- **Patch** (0.0.X): Bug fixes, typo corrections, parameter tuning
