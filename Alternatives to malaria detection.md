**Ranked by dataset availability & validation (most data → least data)**

| Rank | Method | Key Datasets | Sample Size (Clinical) | Accuracy Data | Refs |
|-------|---------|--------------|------------------------|---------------|------|
| 1 | **HRP2/pLDH RDT (combination)** | Uganda (6,354), Sierra Leone, Ethiopia meta-analysis | 6,354+ | Se 91-99%, Sp 56-96% |  |
| 2 | **Microscopy** | Ethiopia meta-analysis (29 studies), Brazil study | 2,000+ (pooled) | Se 83%, Sp 100% |  |
| 3 | **LAMP** | Brazil (91), Ethiopia meta-analysis | 822 | Se 90-100%, Sp 94-99% |  |
| 4 | **pLDH only RDT** | Sierra Leone (children <5), Uganda | 350+ | Se 99%, Sp 96% |  |
| 5 | **HRP2 only RDT** | Multiple Africa studies | 500+ | Se 99%, Sp 75% (false positive high) |  |
| 6 | **Magnetic hemozoin (MOD)** | Kenya (67 patients) | 67 | Accuracy reported, dataset small |  |
| 7 | **Optical hemozoin (spectroscopy)** | Lab data only (extinction coefficients) | 0 clinical | No clinical validation |  |
| 8 | **Thermal/photoacoustic** | Computational simulation | 0 | No clinical data |  |
| 9 | **Microwave/electromagnetic** | In vitro validation | 0 | Lab only |  |
| 10 | **Mueller ellipsometry** | Synthetic hemozoin suspension | 0 | Lab only |  |

---

**Key insight for caveman box:**

**Biggest datasets = RDT, microscopy, LAMP.**  
**Smallest = pure physics methods (hemozoin, thermal, microwave).**

If building cheap box, magnetic hemozoin has *some* human data (67 patients Kenya).  
Optical hemozoin? Zero clinical validation. Only lab spectra.

**Recommendation:** Use RDT data for benchmark. But build hemozoin box anyway — no consumables. Just know dataset small. Validate yourself.