# 📁 CrackDistill Reports & Academic Documentation Hub

This directory is the central repository for all academic reports, LaTeX papers, empirical compendiums, and research briefs produced throughout the **CrackDistill** project lifecycle (from Day 0 to the final deployment SOTA).

---

## 📑 Index of Reports

### 1. 📜 LaTeX Master Paper & Publication Drafts
* **[crackdistill_master_report.tex](file:///home/shahin/distill/reports/crackdistill_master_report.tex)**:
  * **The definitive master LaTeX report** detailing the full project arc from Day 0 through the final verified breakthroughs.
  * *Covers:* Mathematical loss formulations (Bernoulli KL, CWD, Asymmetric Tversky, Prototype Upsampling), Connected Components conversion, Optuna search ($\tau=3.7769, W=0.9612$), PANet Neck CWD, 250-mosaic native teacher extraction, 2D Gaussian apodization TTA engine, all-time record leaderboards, Tesla T4 edge deployment (107.8 FPS), and future research roadmap.
* **[crackdistill_paper_v1.tex](file:///home/shahin/distill/reports/crackdistill_paper_v1.tex)**: Initial LaTeX paper draft from August 2026.
* **[crackdistill_paper_v1.pdf](file:///home/shahin/distill/reports/crackdistill_paper_v1.pdf)**: Compiled PDF of the initial paper draft.

---

### 2. 📊 Empirical Registries & Leaderboards
* **[final_verdict.md](file:///home/shahin/distill/reports/final_verdict.md)**: Master empirical verdict and comparative leaderboard across in-domain and OOD datasets.
* **[nb_exp_results.md](file:///home/shahin/distill/reports/nb_exp_results.md)**: Full numeric metric tables (Dice, Precision, Recall, Box/Mask mAP50, mAP50-95) for all executed Kaggle runs.
* **[final_results_exp.md](file:///home/shahin/distill/reports/final_results_exp.md)**: Comprehensive early experiment registry and loss ablation tables.

---

### 3. 🔬 Deep-Dive Architectural & Strategy Syntheses
* **[good_review.md](file:///home/shahin/distill/reports/good_review.md)**: Senior review of KD literature, hyperparameter dynamics, and cross-domain transfer.
* **[next_moves_forOOD.md](file:///home/shahin/distill/reports/next_moves_forOOD.md)**: Strategic blueprint synthesizing Claude & GPT proposals for resolution-preserving KD, two-stage fine-tuning, and multi-scale TTA.
* **[possibleOODimprovements.md](file:///home/shahin/distill/reports/possibleOODimprovements.md)**: Forensic data audit uncovering the zero-negative pavement crop bias in Crack500 and native $640\times360$ source dimensions.
* **[layerKD.md](file:///home/shahin/distill/reports/layerKD.md)**: Intermediate feature distillation design, ViT-to-CNN capacity gap analysis, and Channel-Wise Distillation (CWD).
* **[layerKDarch.md](file:///home/shahin/distill/reports/layerKDarch.md)**: Multi-scale PANet neck hook specifications, layer pairing matrices (12, 15, 18), and Cross-Architecture Projector (CAP) dimensions.
* **[academic_report_extended.md](file:///home/shahin/distill/reports/academic_report_extended.md)**: Extended narrative detailing chronological phase-by-phase development.
* **[update.md](file:///home/shahin/distill/reports/update.md)**: Executive briefing summarizing the mosaic breakthrough for supervisors and academic committees.

---

## 🏆 Project Empirical Summary

| Recipe / Benchmark Variant | In-Domain Mask mAP50 | Direct OOD Mask mAP50 | Full-Res Tiled Dice | Multi-Scale TTA Precision | Multi-Scale TTA Recall | Distinct Architectural Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline YOLOv11 (No KD)** | 0.5400 | 0.0848 | 0.2414 | 0.2840 | 0.2104 | Standard YOLO reference |
| **`04_affinity` (Spatial Relational KD)** | **0.5569** 👑 | 0.0831 | 0.2594 | 0.3080 | 0.2240 | **In-Domain Segmentation Champion** |
| **`03_mosaic_native` (Wide Composites)** | 0.5440 | 0.1409 | **0.7542** 👑 | 0.7709 | 0.7848 | **Overall Megapixel Dice Champion** |
| **`06_res_preserving` (Upsampled Proto)** | 0.5340 | 0.1130 | 0.7446 | 0.7541 | **0.7945** 👑 | **All-Time Hairline Crack Recall Champion** |
| **`07_asymmetric_tversky` (Soft Tversky $\beta=0.7$)** | 0.5390 | **0.1459** 👑 | 0.7402 | **0.7831** 👑 | 0.7546 | **All-Time Direct OOD & Precision Champion** |
| **`05_twostage_tune` (50 ep Transfer)** | 0.5286 | 0.1130 | 0.7412 | 0.7571 | 0.7664 | **Compute Efficiency Champion** (1.1 hr training) |
