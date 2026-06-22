# Ethical Statement

## Research Ethics Disclosure for the Atheer Simulation Artifact

---

## 1. Scope of This Document

This document discloses the ethical considerations and responsible research practices applied in the development and evaluation of the Atheer simulation artifact.

---

## 2. Research Subject and Methodology

### 2.1 Research Subject

This artifact is a **Discrete Event Simulation (DES)** for evaluating an offline mobile payment architecture called "Atheer." The architecture is designed for low-infrastructure environments.

### 2.2 Methodology

The simulation uses synthetic data only. It models:
- Poisson-distributed transaction arrivals
- LogNormal-distributed network latency
- Bernoulli-distributed packet loss
- M/M/c queuing for core banking
- Deterministic random seeds (for reproducibility)

### 2.3 No Human Subjects

This research **does not involve human subjects**. No user studies, surveys, or experiments with real users were conducted.

### 2.4 No Real Financial Systems

This research **does not test, evaluate, or assess** any real banking system, payment system, or financial institution. The simulation is fully synthetic.

---

## 3. Treatment of Named Institutions

### 3.1 Mentions of Wallet Services

The documentation mentions wallet services by name (Alkuraimi, Jawali, Jaib, MFloos) **only** in the context of describing the existing market landscape.

### 3.2 No Security Claims

This artifact makes **no claims** about:
- The security posture of any named institution
- The cryptographic strength of any named system
- The vulnerability of any named system to specific attacks
- The business practices or operational quality of any named institution

---

## 4. Data Sources and Provenance

### 4.1 Simulation Data

All simulation data is **fully synthetic**. No real banking data, transaction logs, or user data was used.

### 4.2 Cost Model Data

Cost model parameters are derived from **publicly available** sources only:

| Parameter | Source | URL |
|-----------|--------|-----|
| Mobile data pricing | Cable.co.uk | https://www.cable.co.uk/mobiles/worldwide-data-pricing/ |
| ICT price baskets | ITU | https://www.itu.int/en/ITU-D/Statistics/Pages/IPB/default.aspx |
| Internet penetration | DataReportal | https://datareportal.com/reports/digital-2024-yemen |
| Internet resilience | ISOC Pulse | https://pulse.internetsociety.org/country/yemen |
| Network latency | Ookla Speedtest | https://www.speedtest.net/global-index/yemen |

No proprietary, confidential, or non-public data was used.

### 4.3 No Carrier Disclosures

The cost model does **not** rely on:
- Proprietary carrier SLA documents
- B2B pricing agreements
- Internal technical reports from mobile operators
- Non-disclosure agreements (NDAs)

---

## 5. Permissions and Approvals

### 5.1 No Permissions Required

Because this research:
- Does not test real systems
- Does not use real data
- Does not make claims about named institutions
- Does not involve human subjects

**No permissions** from banks, carriers, regulators, or ethics boards were required.

---

## 6. Conflicts of Interest

### 6.1 Author Affiliations

The author is affiliated with:
- **Sana'a University, Yemen** (Faculty of Computer Science)
- No commercial affiliations with any payment service provider, bank, or carrier

### 6.2 Financial Interests

The author has **no financial interest** in:
- Any wallet service
- Any mobile network operator
- Any banking institution
- Any payment infrastructure vendor

### 6.3 Research Funding

This research received **no specific grant** from any funding agency in the public, commercial, or not-for-profit sectors.

---

## 7. Responsible Disclosure

### 7.1 Vulnerability Disclosure

This artifact does **not disclose any vulnerability** in any real system. The threat model described in the documentation describes general attack vectors and mitigations for the **proposed** architecture, not for any existing system.

### 7.2 No Real-World Testing

No penetration testing, security auditing, or vulnerability assessment was performed on any real payment system, banking system, or mobile network.

---

## 8. Reproducibility and Open Science

### 8.1 Open Source

The complete simulation source code is available at:
https://github.com/mutawakel-hub/atheer-research-artifacts

Released under the **MIT License** to enable independent verification.

### 8.2 Reproducibility

The simulation uses:
- Deterministic random seeds
- Version-controlled parameters
- Documented assumptions
- Public data sources

Any researcher can reproduce the results by running the simulation script.

### 8.3 Transparency

All assumptions are documented in:
- `docs/SIMULATION_PARAMETERS.md`
- `docs/COST_MODEL.md`
- This ethical statement

---

## 9. Potential Misuse and Mitigations

### 9.1 Potential Misuse

The architecture described in this artifact could potentially be:
- Used to bypass legitimate payment systems
- Modified for fraudulent purposes
- Used to criticize existing systems without justification

### 9.2 Mitigations

To mitigate misuse:
1. The artifact is framed as an **architectural proposal**, not an attack tool
2. The system requires cooperation with legitimate partner wallets and banks
3. The cost model and economic analysis demonstrate that the system is designed for **legitimate financial inclusion**, not for circumventing regulations

---

## 10. Contact for Ethical Concerns

If you have any ethical concerns about this research, please contact:
- **Ahmed Al-Mutawakel**: a.almutawakel@su.edu.ye

We take all concerns seriously and will respond promptly.

---

*This ethical statement is part of the Atheer simulation artifact and is licensed under the same MIT License as the rest of the repository.*
