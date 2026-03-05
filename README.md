# ZEBRA <img src="twemoji--zebra.svg" width="50" valign="middle">

This is the official source code for the paper:  
**"ZEBRA: A Zero-shot Graph Neural Framework for Illicit Account Detection in Large-scale Blockchain Transaction Networks"** 🔗

---

### 📢 Status Update
> [!IMPORTANT]
> 🚧 **Work in Progress:** We are actively updating the **source code** and **datasets** to ensure full reproducibility.
> Stay tuned for more updates! 📦✨

---

### 🚀  Methodology: The ZEBRA Pipeline
The following figure illustrates the overall architecture of ZEBRA. Our framework systematically processes blockchain transaction data through a zero-shot learning paradigm to identify illicit accounts.

<p align="center">
<img src="zebra-5.pdf" width="800" alt="ZEBRA Overall Pipeline">

<em>Figure 1: The overall pipeline of ZEBRA, showcasing the flow from transaction networks to illicit account detection.</em>
</p>

### 📊 Experimental Results on Ethereum-S & Bitcoin-M

| Category | Method | Ethereum-S (AUROC) | Ethereum-S (AUPRC) | Bitcoin-M (AUROC) | Bitcoin-M (AUPRC) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Supervised** | GCN | 57.03 | 53.96 | 54.55 | 20.71 |
| | GAT | 66.19 | 61.93 | 64.44 | 25.10 |
| | SAGE | 66.38 | 59.28 | 63.40 | 21.13 |
| **Semi/Un-sup** | GGAD | 64.56 | 63.70 | 56.76 | 24.60 |
| | GADAM | 56.85 | 53.55 | 44.19 | 17.06 |
| | FreeGAD | 66.00 | 59.28 | 64.74 | 22.99 |
| 🚀 **Ours** | **ZEBRA** | **75.20** | **74.65** | **65.09** | **27.87** |
| | *Improvement* | **+8.82↑** | **+10.95↑** | **+0.35↑** | **+2.77↑** |
