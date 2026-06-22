# Revision Notes: v1.0 → v2.0

This document details the changes made between the original DTISD 2026 submission (v1.0) and the revised version (v2.0), organized by reviewer concern.

---

## Reviewer 1 Concerns

### Concern 1.1: Dependency on unpublished, untrusted reference [16]

> *"This manuscript lacks accuracy and clarity because the main methodology was designed, built, and evaluated as an offline mobile payment architecture based on the Atheer simulation evaluation artifact, an unpublished, untrusted reference."*

**v2 Response**: The reference [16] (now [26] in v2) is properly contextualized as an open-source research artifact, following DSR transparency principles. The simulation code is now in a structured GitHub repository with full reproducibility (deterministic seeds, config files, execution instructions). The reference is no longer the sole basis for evaluation — the paper now provides complete mathematical model derivation (Eq. 1-5) that allows independent reimplementation.

### Concern 1.2: Incorrect wallet names

> *"The wrong mention of Yemeni mobile wallets Jeeb, Al-Kuraimi, M-Flous, and Jawaly, the correct names are Jaib, Alkuraimi, MFloos, and Jawali, respectively."*

**v2 Response**: All wallet names corrected throughout the paper:
- Jeeb → **Jaib**
- Al-Kuraimi → **Alkuraimi**
- M-Flous → **MFloos**
- Jawaly → **Jawali**

Verified against official Google Play Store listings and company websites.

---

## Reviewer 2 Concerns

### Concern 2.1: Ethical concerns about assessing security of real banking institutions

> *"This paper is rejected even though it technically sounds good. The rejection was mainly due to ethical concerns, as the research assessed security-related issues in real banking institutions without documented permission from the relevant organizations."*

**v2 Response**: This was the most serious concern. We made the following changes:

1. **Removed all security claims about named Yemeni banks/wallets**:
   - Old (Section I): *"these legacy systems lack robust security"* → Removed
   - Old (Section II-A): Cited GSMA criticism of USSD/SMS as if applied to Yemeni wallets → Reframed as general observation about the channel type, not specific institutions

2. **Added explicit Ethical Statement** (Section III-A):
   > *"This study does not evaluate or assess the security of any named financial institution in Yemen. References to wallets such as Alkuraimi, Jawali, Jaib, and MFloos appear only in the context of market description; no claims about their cryptographic posture are made."*

3. **Clarified simulation scope**: The DES is fully synthetic, uses no real banking data, and requires no permission from any operator.

4. **Used only public pricing references** for cost model (ITU, GSMA, Cable.co.uk).

---

## Reviewer 3 Concerns

### Concern 3.1: Methodological issues with "case study" framing

> *"Case study methodology is an in-depth, systematic research approach... we expect to involve triangulating real-world data from mobile operators, banks, and technology sources."*

**v2 Response**: Removed "A Case Study from Yemen" from the title. The paper is now framed as a **Design Science Research** artifact (DSR methodology, Section III), not a case study. The new title:
> *"A Flexible Offline Mobile Payment Architecture Using NFC and Host Card Emulation: A Cost-Optimized Approach for Low-Infrastructure Environments"*

### Concern 3.2: Unavailable references

> *"This case study mainly depends on the Internet as its main source, and we can spot some unavailable sources, like references [5]."*

**v2 Response**: Verified all references. Reference [5] (ISOC Pulse Internet Resilience Index for Yemen) is publicly available at https://pulse.internetsociety.org/country/yemen. Removed weak references and replaced with stronger ones (ITU ICT Price Baskets, Cable.co.uk mobile data pricing, RFC 8446 for TLS 1.3).

### Concern 3.3: Unverified security claims about banking services

> *"The first paragraph claims that some banking services (Al-Kuraimi, M-Flous, Jawaly, and Jeeb) lack robust security. These claims have not been proven, are a bit exaggerated, and may harm the banking system in Yemen."*

**v2 Response**: All such claims removed. See Reviewer 2 response above.

### Concern 3.4: Unsubstantiated digital infrastructure claims

> *"Section II, page 2, claims that digital infrastructure in Yemen overlooks the essential core identifying cryptography and inability to scale; however, the authors fail to show a piece of evidence."*

**v2 Response**: Reframed as a research gap observation rather than a claim about Yemeni infrastructure specifically. The Related Work section now discusses general literature findings about scalability challenges in low-infrastructure environments, without making unsupported claims about Yemen.

### Concern 3.5: Provisioning contradiction (strongest methodological critique)

> *"Section III-A shows a threefold architecture, which still needs online provisioning to obtain cryptographic tokens. The authors should answer the question: How can they rely on the public network, which they criticize in Section II, to exchange cryptographic tokens?"*

**v2 Response**: Added **dual-path provisioning** (Section V-A):

1. **In-Band Provisioning over TLS 1.3**: The default pathway. We acknowledge this uses the public internet, but the security guarantees come from transport-layer encryption (TLS 1.3 with forward secrecy) and hardware-anchored device identity (TEE attestation), not from network-layer isolation.

2. **Out-of-Band Provisioning at Bank Branches**: For high-value merchants, tokens can be provisioned at a partner bank branch over a private LAN. This pathway is reserved for cases where the public internet is completely unavailable.

This explicitly addresses the contradiction by providing an alternative pathway and acknowledging the security model for the default pathway.

### Concern 3.6: Unrealistic Private APN claims

> *"Section VIII does not provide what is expected when reading the term 'case study' in the title because it presents non-real information. For example, (1) Private APN is not zero-rated in Yemen. (2) None of the Yemeni carriers provides public information on APN wholesale transit costs or B2B models. (3) No relation between APN and Infrastructure as a Service (IaaS)."*

**v2 Response**: This drove the fundamental v2 redesign:

1. **Replaced Private APN entirely** with Mobile Data Routing + Partner-Subsidized Billing
2. **Removed zero-rated claim** — no longer needed because the cost model proves data cost is < 0.5% of MDR revenue regardless of zero-rating
3. **Removed IaaS claim** — was a conceptual confusion (APN is NaaS/Connectivity-as-a-Service, not IaaS)
4. **Added explicit cost model** based on publicly verifiable pricing data (Cable.co.uk, ITU)

### Concern 3.7: Defamation risk

> *"The paper contains statements that doesn't comply with IEEE legal policy and ethical code and may be classified as defamation."*

**v2 Response**: All potentially defamatory statements removed. The paper now makes no claims about the security or business practices of any named institution. See Reviewer 2 response.

### Concern 3.8: Suggestion to remove Yemen and bank names

> *"We suggest the authors delete anything about Yemen and the banks' names and focus on the framework."*

**v2 Response**: We partially adopted this suggestion:
- **Kept** Yemen as the motivating context (it's the research setting)
- **Kept** wallet names but only in market description, not in security claims
- **Removed** all evaluative statements about named institutions
- **Added** Ethical Statement (Section III-A) clarifying the scope

---

## Summary of Changes

| Area | v1.0 | v2.0 |
|------|------|------|
| Network Layer | Private APN | Mobile Data + Partner-Subsidized Billing |
| Zero-rated claim | Yes (unverifiable) | Removed |
| IaaS claim | Yes (conceptual error) | Removed |
| Cost model | Implicit | Explicit (Eq. 6-8, Figure 8) |
| Wallet names | Wrong | Corrected |
| Security claims on banks | Present | Removed |
| Ethical statement | Absent | Added (Section III-A) |
| Provisioning | Single path (contradictory) | Dual-path (in-band + out-of-band) |
| Payload optimization | Not specified | 180 bytes (formal proof) |
| Comparison table | Absent | Added (Table VI) |
| Title | "A Case Study from Yemen" | "A Cost-Optimized Approach..." |

---

## Resulting Credibility Improvement

The v2 results are more conservative (97.6% vs. 99.8% success at 500 TPS) but more credible:
- Realistic latency (130 ms vs. 60 ms)
- Realistic packet loss (1.5% vs. 0.1%)
- Load-dependent degradation now applied to S2 (mildly)
- Cost model is verifiable against public pricing data
- No ethical or legal concerns
