# Spatial Transcriptomic Analysis of the Tumor Microenvironment via a Grid-Level Z-axis Stratification and Molecular Density
# 基于网格级 Z 轴分层与分子密度的肿瘤微环境空间转录组分析

## Overview
## 概述
This repository contains a segmentation-free, grid-based analysis workflow for single-molecule spatial transcriptomics (10x Genomics Xenium). The workflow summarizes transcript coordinates using two grid-level geometric features:
- a density proxy (`log1p` transcript count per grid), and
- the Z-Stacking Index (within-grid standard deviation of transcript Z coordinates), used as a local Z-axis dispersion / layering metric.

These coordinate-derived summaries are used for unsupervised spatial stratification and downstream molecular characterization (grid-level DGE, marker-group scoring, and exploratory pathway enrichment). The notebook also includes interface-centric analysis based on signed distance to an operational boundary between major spatial groups, enabling distance-binned gradient profiling of geometric and molecular features.

本仓库提供一套面向单分子空间转录组（10x Genomics Xenium）的、无需细胞分割、基于空间网格的分析流程。该流程使用两类网格级几何特征对转录本坐标进行摘要：
- 密度代理指标（每网格转录本计数的 `log1p`），以及
- Z-Stacking Index（Z 轴堆叠指数）：每个网格内转录本 Z 坐标的标准差，用作局部 Z 轴离散度/分层度量指标。

上述由坐标计算得到的摘要特征用于无监督空间分层，并结合网格级 DGE、标记组得分与探索性通路富集进行分子刻画。Notebook 还包含界面中心化分析：基于主要空间组之间的操作性边界定义有符号距离，从而对几何与分子特征沿距离轴进行分箱梯度剖析。

## Key Ideas
## 核心思想
- Segmentation-free stratification / 无需分割的分层: avoid dependence on cell segmentation in densely packed regions by operating at fixed spatial grids.
- Local Z-axis layering metric / 局部 Z 轴分层度量: use within-grid Z dispersion (Z-Stacking Index) to complement 2D density summaries when absolute Z coordinates may be influenced by slide-scale geometry.
- Interface-centric profiling / 界面中心化剖析: summarize feature transitions using signed distance to an operational boundary, supporting distance-binned gradient visualization.

- 无需分割的空间分层：在致密区域不依赖细胞分割，通过固定空间网格进行分析。
- 局部 Z 轴分层度量：使用网格内 Z 离散度（Z-Stacking Index）补充 2D 密度摘要；当绝对 Z 坐标可能受玻片尺度几何因素影响时，该指标有助于刻画局部变化。
- 界面中心化剖析：通过到操作性边界的有符号距离汇总跨界面变化，支持距离分箱的梯度可视化。

## Workflow (Notebook)
## 工作流（Notebook）
1. Quality control (QC) / 质量控制: filter low-quality transcripts and technical controls.
2. Global Z diagnostics / 全局 Z 诊断: summarize Z-coordinate distribution and large-scale trends.
3. Grid aggregation / 网格聚合: compute density proxy and Z-Stacking Index per grid.
4. Unsupervised stratification / 无监督分层: cluster grids using geometric features only.
5. Molecular characterization / 分子刻画: grid-level DGE, marker-group scores, exploratory enrichment.
6. Neighborhood and interface analysis / 邻域与界面分析: neighborhood metrics; signed-distance gradients; feature transition summaries.

1. 质量控制：过滤低质量转录本与技术对照。
2. 全局 Z 诊断：汇总 Z 坐标分布与大尺度趋势。
3. 网格聚合：计算每网格密度代理指标与 Z-Stacking Index。
4. 无监督分层：仅基于几何特征对网格聚类。
5. 分子刻画：网格级 DGE、标记组得分、探索性富集分析。
6. 邻域与界面分析：邻域指标；有符号距离梯度；跨界面过渡的定量摘要。

## Data Availability
## 数据可得性
This study uses the publicly available 10x Genomics Xenium FFPE Human Breast Cancer Representative Dataset. To comply with redistribution policies, raw data are not included in this repository.

To reproduce the analysis, download the dataset from the 10x Genomics Datasets Portal and place files under `input/` at the project root.

本研究使用公开的 10x Genomics Xenium FFPE Human Breast Cancer Representative Dataset。为遵守数据再分发政策，本仓库不包含原始数据。

如需复现分析，请从 10x Genomics Datasets Portal 下载数据，并将文件放置于项目根目录的 `input/` 目录下。

Expected files / 需要的文件：
- `input/outs/transcripts.parquet` (subcellular transcript coordinates / 转录本亚细胞坐标)
- `input/*_he_image.ome.tif` or `input/*_he_image.tif` (H&E morphology image / H&E 形态学图像)
- `input/Xenium_FFPE_Human_Breast_Cancer_Rep1_gene_groups.csv` (marker-group definitions / 标记组定义)

Dataset entry keywords (portal search) / 数据集检索关键词：
- “Xenium Human Breast Cancer FFPE”
- “High Resolution Mapping of the Breast Cancer Tumor Microenvironment using Spatial Transcriptomics”

## Reproducibility Notes
## 可复现性说明
- Many steps use explicit analysis settings (grid size, thresholds, neighborhood radius, bin width). Results can vary with these settings; sensitivity checks are included in the notebook.
- Grid-level tests involve spatially correlated observations; interpret p-values with this dependence in mind and prioritize effect sizes and profile shapes when comparing regions.
- Z-Stacking Index summarizes within-grid Z dispersion and may reflect both biological structure and acquisition geometry; the workflow therefore emphasizes localized summaries and interface-centric comparisons.

- 多个步骤依赖明确的分析设置（网格大小、阈值、邻域半径、分箱宽度等）；结果可能随设置变化，Notebook 中包含敏感性检查。
- 网格之间存在空间相关性；解读显著性时需考虑这一依赖关系，比较区域时建议同时关注效应量与轮廓形状。
- Z-Stacking Index 表征网格内 Z 离散度，可能同时受到生物学结构与采集几何因素影响；因此流程强调局部摘要与界面中心化对照。
