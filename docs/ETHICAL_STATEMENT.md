# Ethical Statement

## Research Ethics Disclosure for the Atheer Project

---

## 1. Scope of This Document

This document discloses the ethical considerations, potential conflicts of interest, and responsible research practices applied in the development and evaluation of the Atheer offline mobile payment architecture.

---

## 2. Research Subject and Methodology

### 2.1 Research Subject

This research develops and evaluates an **architectural proposal** for an offline mobile payment system called "Atheer." The system is designed for low-infrastructure environments, with Yemen as the motivating context.

### 2.2 Methodology

The research uses **Design Science Research (DSR)** methodology ([Hevner 2004](https://doi.org/10.2307/25148625); [Peffers 2007](https://doi.org/10.2753/MIS0742-1222240302)). Evaluation is conducted via **Discrete Event Simulation (DES)** using synthetic data only.

### 2.3 No Human Subjects

This research **does not involve human subjects**. No user studies, surveys, or experiments with real users were conducted.

### 2.4 No Real Financial Systems

This research **does not test, evaluate, or assess** any real banking system, payment system, or financial institution.

---

## 3. Treatment of Named Institutions

### 3.1 Yemeni Wallet Services Mentioned

The paper mentions the following Yemeni wallet services by name:
- **Alkuraimi** (Al-Kuraimi Islamic Microfinance Bank)
- **Jawali** (Yemen Mobile)
- **Jaib** (Jaib App)
- **MFloos** (Al-Amal Microfinance Bank)

### 3.2 Context of Mentions

These institutions are mentioned **only** in the context of:
- Describing the existing market landscape (Section I, Section II-A)
- Comparing Atheer with existing systems (Table VI)

### 3.3 No Security Claims

The paper makes **no claims** about:
- The security posture of any named institution
- The cryptographic strength of any named system
- The vulnerability of any named system to specific attacks
- The business practices or operational quality of any named institution

### 3.4 Correction of v1.0 Errors

The original v1.0 submission contained statements that could be interpreted as security assessments of named institutions. These statements have been **removed** in v2.0. Specifically:

| v1.0 Statement | v2.0 Action |
|----------------|-------------|
| "these legacy systems lack robust security" | Removed |
| Implicit attribution of GSMA USSD/SMS criticism to Yemeni wallets | Reframed as general observation about the channel type |
| Any implication that named systems are vulnerable | Removed |

---

## 4. Data Sources and Provenance

### 4.1 Simulation Data

All simulation data is **fully synthetic**. The Discrete Event Simulation uses:
- Poisson-distributed transaction arrivals
- LogNormal-distributed network latency
- Bernoulli-distributed packet loss
- Deterministic random seeds (for reproducibility)

No real banking data, transaction logs, or user data was used.

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

### 5.2 Contrast with v1.0

The v1.0 submission was rejected in part because it could be interpreted as assessing security of real institutions without permission. The v2.0 revision eliminates this concern by:
- Removing all security assessments
- Using only synthetic data
- Relying only on public references
- Adding this explicit ethical statement

---

## 6. Conflicts of Interest

### 6.1 Author Affiliations

Both authors are affiliated with:
- **Sana'a University, Yemen** (Faculty of Computer Science)
- No commercial affiliations with any payment service provider, bank, or carrier

### 6.2 Financial Interests

The authors have **no financial interest** in:
- Any Yemeni wallet service
- Any mobile network operator
- Any banking institution
- Any payment infrastructure vendor

### 6.3 Research Funding

This research received **no specific grant** from any funding agency in the public, commercial, or not-for-profit sectors.

---

## 7. Responsible Disclosure

### 7.1 Vulnerability Disclosure

This paper does **not disclose any vulnerability** in any real system. The threat model (Table II) describes general attack vectors and mitigations for the **proposed** Atheer architecture, not for any existing system.

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
- The paper itself (Sections III, IV, VII)
- `docs/SIMULATION_PARAMETERS.md`
- `docs/COST_MODEL.md`
- This ethical statement

---

## 9. Potential Misuse and Mitigations

### 9.1 Potential Misuse

The architecture described in this paper could potentially be:
- Used to bypass legitimate payment systems
- Modified for fraudulent purposes
- Used to criticize existing systems without justification

### 9.2 Mitigations

To mitigate misuse:
1. The paper is framed as an **architectural proposal**, not an attack tool
2. The system requires cooperation with legitimate partner wallets and banks
3. The cost model and economic analysis demonstrate that the system is designed for **legitimate financial inclusion**, not for circumventing regulations
4. The Future Work section explicitly calls for partnership with regulators and AML frameworks

---

## 10. Compliance with IEEE Code of Ethics

This research complies with the [IEEE Code of Ethics](https://www.ieee.org/about/corporate/governance/p7-8.html), specifically:

1. **To uphold the highest standards of integrity** — We have disclosed all assumptions, limitations, and potential conflicts of interest
2. **To treat all persons fairly** — We do not make disparaging claims about any institution
3. **To avoid harming others** — We have removed all potentially defamatory statements
4. **To seek and accept honest feedback** — We have responded constructively to all reviewer feedback
5. **To properly credit others** — All references are properly cited

---

## 11. Contact for Ethical Concerns

If you have any ethical concerns about this research, please contact:
- **Nabil Al-Mekhlafi**: nabil.almekhlafi@su.edu.ye
- **Ahmed Al-Mutawakel**: a.almutawakel@su.edu.ye

We take all concerns seriously and will respond promptly.

---

## 12. Version History

| Version | Date | Ethical Changes |
|---------|------|-----------------|
| 1.0 | 2024 | Original submission (subsequently rejected for ethical concerns) |
| **2.0** | **June 2026** | **Added this ethical statement; removed all security claims about named institutions; clarified synthetic-only data** |

---

*This ethical statement is part of the Atheer research artifacts and is licensed under the same MIT License as the rest of the repository.*
