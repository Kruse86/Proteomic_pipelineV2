#!/usr/bin/env python3
"""
proteomics_pipeline.py
======================
Unified proteomics analysis pipeline integrating:

  Stage 1 — FASTA ingestion & validation
  Stage 2 — Physicochemical profiling      (protein_analysis.py)
  Stage 3 — Secondary structure propensity (Chou-Fasman)
  Stage 4 — Hydropathy analysis            (Kyte-Doolittle)
  Stage 5 — Sequence complexity            (Shannon entropy)
  Stage 6 — Signal peptide detection       (heuristic scorer)
  Stage 7 — Motif scanning                 (17-motif curated library)
  Stage 8 — ML localisation prediction     (Random Forest, trained model)
  Stage 9 — Report generation              (terminal + JSON + HTML)

Usage
-----
  python proteomics_pipeline.py proteins.fasta
  python proteomics_pipeline.py proteins.fasta --model model.pkl
  python proteomics_pipeline.py proteins.fasta --stages 1 2 3 6 7
  python proteomics_pipeline.py proteins.fasta --json > results.json
  python proteomics_pipeline.py proteins.fasta --html report.html
  python proteomics_pipeline.py proteins.fasta --no-ml

Author : Jason Iles, 2026
"""

import re
import sys
import json
import math
import argparse
import statistics
from dataclasses import dataclass, field, asdict
from collections import Counter
from pathlib import Path
from typing import Optional
from datetime import datetime

# ── optional ML deps ──────────────────────────────────────────────────────────
try:
    import joblib
    import pandas as pd
    import numpy as np
    HAS_ML = True
except ImportError:
    HAS_ML = False

# ── GHMA deps (numpy / scipy / matplotlib) ────────────────────────────────────
try:
    import numpy as _np
    from scipy.linalg    import svd       as _svd
    from scipy.fft       import rfft      as _rfft, rfftfreq as _rfftfreq
    from scipy.stats     import entropy   as _entropy, pearsonr as _pearsonr
    from scipy.ndimage   import gaussian_filter1d as _gf1d
    HAS_GHMA = True
except ImportError:
    HAS_GHMA = False

# ══════════════════════════════════════════════════════════════════════════════
# Reference Data  (inline — no external deps)
# ══════════════════════════════════════════════════════════════════════════════

AA_STANDARD = set("ACDEFGHIKLMNPQRSTVWY")

RESIDUE_MASS = {
    'A':71.03711,'R':156.10111,'N':114.04293,'D':115.02694,'C':103.00919,
    'Q':128.05858,'E':129.04259,'G':57.02146,'H':137.05891,'I':113.08406,
    'L':113.08406,'K':128.09496,'M':131.04049,'F':147.06841,'P':97.05276,
    'S':87.03203,'T':101.04768,'W':186.07931,'Y':163.06333,'V':99.06841,
}
WATER = 18.01056

AA_CLASS = {
    'D':'acidic','E':'acidic','K':'basic','R':'basic','H':'basic',
    'S':'polar','T':'polar','N':'polar','Q':'polar',
    'A':'aliphatic','V':'aliphatic','I':'aliphatic','L':'aliphatic',
    'M':'aliphatic','G':'aliphatic','F':'aromatic','W':'aromatic',
    'Y':'aromatic','C':'special','P':'special',
}
AA_COLOUR = {
    'acidic':'#fb7185','basic':'#60a5fa','polar':'#34d399',
    'aliphatic':'#94a3b8','aromatic':'#c084fc','special':'#fbbf24',
}

KD = {
    'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
    'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
    'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,
}

PKA = {'N_term':8.0,'C_term':3.1,'D':3.9,'E':4.1,'C':8.5,'H':6.5,'K':10.8,'R':12.5,'Y':10.1}

CF = {
    'A':(1.42,0.83,0.66),'R':(0.98,0.93,1.00),'N':(0.67,0.89,1.33),
    'D':(1.01,0.54,1.46),'C':(0.70,1.19,1.19),'Q':(1.11,1.10,0.98),
    'E':(1.51,0.37,0.74),'G':(0.57,0.75,1.56),'H':(1.00,0.87,0.95),
    'I':(1.08,1.60,0.47),'L':(1.21,1.30,0.59),'K':(1.16,0.74,1.01),
    'M':(1.45,1.05,0.60),'F':(1.13,1.38,0.60),'P':(0.57,0.55,1.52),
    'S':(0.77,0.75,1.43),'T':(0.83,1.19,0.96),'W':(1.08,1.37,0.96),
    'Y':(0.69,1.47,1.14),'V':(1.06,1.70,0.50),
}

INST_W = {
    ('D','D'):6.58,('D','G'):5.99,('D','R'):6.58,('E','E'):7.66,
    ('E','K'):4.94,('F','K'):5.54,('H','H'):14.0,('R','H'):3.24,('R','R'):4.35,
}
# ─────────────────────────────────────────────────────────────────────────────
# PTM mass catalogue  (monoisotopic Da shifts)
# ─────────────────────────────────────────────────────────────────────────────
PTM_CATALOGUE = {
    # Phosphorylation
    "Phospho_S":       {"delta":  79.9663, "res": "S",   "cls": "phosphorylation",  "desc": "Phosphoserine"},
    "Phospho_T":       {"delta":  79.9663, "res": "T",   "cls": "phosphorylation",  "desc": "Phosphothreonine"},
    "Phospho_Y":       {"delta":  79.9663, "res": "Y",   "cls": "phosphorylation",  "desc": "Phosphotyrosine"},
    # Oxidation
    "Oxidation_M":     {"delta":  15.9949, "res": "M",   "cls": "oxidation",        "desc": "Oxidised methionine"},
    "Oxidation_W":     {"delta":  15.9949, "res": "W",   "cls": "oxidation",        "desc": "Oxidised tryptophan"},
    "Hydroxy_P":       {"delta":  15.9949, "res": "P",   "cls": "oxidation",        "desc": "Hydroxyproline (collagen)"},
    # Acetylation
    "Acetyl_Nterm":    {"delta":  42.0106, "res": None,  "cls": "acetylation",      "desc": "N-terminal acetylation"},
    "Acetyl_K":        {"delta":  42.0106, "res": "K",   "cls": "acetylation",      "desc": "Acetyllysine"},
    # Methylation
    "Methyl_K":        {"delta":  14.0157, "res": "K",   "cls": "methylation",      "desc": "Monomethyllysine"},
    "Dimethyl_K":      {"delta":  28.0313, "res": "K",   "cls": "methylation",      "desc": "Dimethyllysine"},
    "Trimethyl_K":     {"delta":  42.0470, "res": "K",   "cls": "methylation",      "desc": "Trimethyllysine"},
    "Methyl_R":        {"delta":  14.0157, "res": "R",   "cls": "methylation",      "desc": "Monomethylarginine"},
    # Ubiquitination / UBL
    "GlyGly_K":        {"delta": 114.0429, "res": "K",   "cls": "ubiquitination",   "desc": "Ubiquitination GlyGly tag"},
    "SUMO":            {"delta":11500.0,   "res": "K",   "cls": "ubiquitination",   "desc": "SUMOylation (~11.5 kDa, human SUMO-1)"},
    # Glycosylation
    "N_HexNAc":        {"delta": 203.0794, "res": "N",   "cls": "glycosylation",    "desc": "N-linked HexNAc (GlcNAc)"},
    "N_complex":       {"delta":1444.5,    "res": "N",   "cls": "glycosylation",    "desc": "N-linked complex glycan (biantennary)"},
    "O_HexNAc":        {"delta": 203.0794, "res": "ST",  "cls": "glycosylation",    "desc": "O-linked HexNAc (GalNAc)"},
    # Deamidation
    "Deamid_N":        {"delta":   0.9840, "res": "N",   "cls": "deamidation",      "desc": "Deamidation N to D"},
    "Deamid_Q":        {"delta":   0.9840, "res": "Q",   "cls": "deamidation",      "desc": "Deamidation Q to E"},
    # Disulfide (reductive)
    "Disulfide":       {"delta":  -2.0156, "res": "CC",  "cls": "structural",       "desc": "Disulfide bond (-2H per Cys pair)"},
    # Lipidation
    "Myristoyl":       {"delta": 210.1984, "res": None,  "cls": "lipidation",       "desc": "N-terminal myristoylation"},
    "Palmitoyl_C":     {"delta": 238.2297, "res": "C",   "cls": "lipidation",       "desc": "S-palmitoylation"},
    "Farnesyl_C":      {"delta": 204.1878, "res": "C",   "cls": "lipidation",       "desc": "Farnesylation (CaaX)"},
    # Additional PTMs (2026 additions)
    "Sulfation_Y":     {"delta":  79.9568, "res": "Y",   "cls": "sulfation",        "desc": "Tyrosine O-sulfation"},
    "PyroGlu_Nterm":   {"delta": -17.0265, "res": None,  "cls": "pyroglutamate",    "desc": "N-terminal pyroglutamation (Q/E→pyroGlu)"},
    "Citrull_R":       {"delta":   0.9840, "res": "R",   "cls": "citrullination",   "desc": "Citrullination (Arg→Cit, deimination)"},
    "Carbamyl_Nterm":  {"delta":  43.0058, "res": None,  "cls": "carbamylation",    "desc": "N-terminal carbamylation (urea artefact)"},
    "Propionamide_C":  {"delta":  71.0371, "res": "C",   "cls": "alkylation",       "desc": "Propionamide on Cys (acrylamide adduct)"},
    "IAA_C":           {"delta":  57.0215, "res": "C",   "cls": "alkylation",       "desc": "Iodoacetamide carbamidomethylation (standard)"},
}

# Motif name → PTMs that it evidences
_MOTIF_PTM_MAP = {
    # Phosphorylation
    "PKC_phospho":       ["Phospho_S", "Phospho_T"],
    "CK2_phospho":       ["Phospho_S", "Phospho_T"],
    "PKA_phospho":       ["Phospho_S", "Phospho_T"],
    "Tyr_kinase":        ["Phospho_Y"],
    "CDK_phospho":       ["Phospho_S", "Phospho_T"],
    "CDK_minimal":       ["Phospho_S", "Phospho_T"],
    "ATM_ATR_phospho":   ["Phospho_S", "Phospho_T"],
    "MAPK_phospho":      ["Phospho_S", "Phospho_T"],
    "AURORA_phospho":    ["Phospho_S", "Phospho_T"],
    "DNAPK_phospho":     ["Phospho_S", "Phospho_T"],
    "CAMKII_phospho":    ["Phospho_S", "Phospho_T"],
    # Glycosylation
    "N_glycosylation":   ["N_HexNAc", "N_complex"],
    "O_glycosylation":   ["O_HexNAc"],
    "O_GalNAc_mucin":    ["O_HexNAc"],
    "O_GlcNAc_cytosolic":["O_HexNAc"],
    # Structural
    "Cys_disulfide":     ["Disulfide"],
    "EGF_like_6Cys":     ["Disulfide"],
    # Ubiquitin
    "SUMO_consensus":    ["SUMO"],
    "D_box_degron":      ["GlyGly_K"],
    "KEN_box_degron":    ["GlyGly_K"],
    # Lipidation
    "Myristoylation_Gly2":["Myristoyl"],
    "CaaX_farnesyl":     ["Farnesyl_C"],
    "CaaX_geranyl":      ["Farnesyl_C"],
    "DHHC_palmitoyl":    ["Palmitoyl_C"],
    "CAAX_general":      ["Farnesyl_C"],
}

# Average occupancy per PTM class (evidence-weighted estimate).
# Override per-analysis via stage_mass(ptm_occ_override={...}).
_PTM_OCC = {
    "phosphorylation": 0.30,   # conservative average; regulatory sites often 70-90%
    "glycosylation":   0.50,   # N-linked; O-linked typically lower (~0.20)
    "oxidation":       0.15,   # M oxidation common in aged/stressed samples
    "acetylation":     0.20,
    "methylation":     0.10,
    "ubiquitination":  0.08,
    "deamidation":     0.03,   # spontaneous; higher in older samples
    "lipidation":      0.05,
    "structural":      1.00,   # disulfides: closed if motif evidence present
    "sulfation":       0.25,   # tissue-specific; high in extracellular/secreted
    "pyroglutamate":   0.40,   # common N-term artefact in recombinant proteins
    "citrullination":  0.05,   # low baseline; elevated in inflammatory contexts
    "carbamylation":   0.10,   # sample prep artefact (urea gels)
    "alkylation":      0.90,   # IAA/propionamide: nearly complete in prep
}

# Average amino acid masses (for PAGE/gel comparisons)
AA_AVG_MASS = {
    'A':89.094,'R':174.203,'N':132.119,'D':133.104,'C':121.159,
    'Q':146.146,'E':147.130,'G':75.032,'H':155.156,'I':131.174,
    'L':131.174,'K':146.189,'M':149.208,'F':165.192,'P':115.132,
    'S':105.093,'T':119.119,'W':204.228,'Y':181.191,'V':117.148,
}


MOTIF_LIB = {
    # ══ GLYCOSYLATION ═══════════════════════════════════════════════════════════
    "N_glycosylation":      (r"N[^P][ST][^P]",             "glycosylation",      "N-linked glycosylation sequon (NxS/T, no Pro at +1 or +3)"),
    "O_glycosylation":      (r"[ST][ACDEFGHIKLMNQRSTVWY]",  "glycosylation",      "O-glycosylation Ser/Thr (not followed by Pro)"),
    "O_GalNAc_mucin":       (r"[ST][^P][^P]",              "glycosylation",      "O-GalNAc mucin-type (loose S/T-xP consensus)"),
    "O_GlcNAc_cytosolic":   (r"[ST][AVG]",                 "glycosylation",      "O-GlcNAc transferase: S/T followed by A, V, or G"),
    "C_mannosylation":      (r"W.{2}W",                    "glycosylation",      "C-mannosylation on Trp: WXXW motif"),

    # ══ PHOSPHORYLATION ══════════════════════════════════════════════════════════
    "PKC_phospho":          (r"[ST][^P][KR]",              "phosphorylation",    "Protein kinase C: S/T-x-K/R"),
    "CK2_phospho":          (r"[ST].{2}[DE]",              "phosphorylation",    "Casein kinase 2: S/T-x-x-D/E"),
    "PKA_phospho":          (r"[RK]{2}[^P][ST]",           "phosphorylation",    "Protein kinase A: R/K-R/K-x-S/T"),
    "Tyr_kinase":           (r"[RK].{2,3}[DE].{2,3}Y",    "phosphorylation",    "Tyrosine kinase substrate consensus"),
    "CDK_phospho":          (r"[ST]PK|[ST]PR",             "phosphorylation",    "CDK full: S/T-P-K/R (Ser/Thr-Pro-basic)"),
    "CDK_minimal":          (r"[ST]P[^$]",                 "phosphorylation",    "CDK minimal: S/T-P (proline-directed)"),
    "ATM_ATR_phospho":      (r"[ST]Q",                     "phosphorylation",    "ATM/ATR DNA damage: S/T-Q motif"),
    "MAPK_phospho":         (r"P[ST]P",                    "phosphorylation",    "MAPK: Pro-S/T-Pro docking"),
    "PLK1_polo_box":        (r"[ST][ST]P",                 "phosphorylation",    "PLK1 polo-box docking: S/T-S/T-P"),
    "CK1_phospho":          (r"[ST].{3}[ST]",              "phosphorylation",    "CK1: S/T-x-x-x-S/T (priming site 4 upstream)"),
    "AURORA_phospho":       (r"[RK]R[ST]",                 "phosphorylation",    "Aurora A/B kinase: R-R/K-S/T"),
    "DNAPK_phospho":        (r"[ST]Q[DE]",                 "phosphorylation",    "DNA-PK: S/T-Q-E/D, subset of ATM/ATR sites"),
    "NEK2_phospho":         (r"[MFLIY][RK].{0,1}[ST]",    "phosphorylation",    "NEK2: hydrophobic-basic-S/T"),
    "CAMKII_phospho":       (r"R.{2}[ST][LIVMF]",         "phosphorylation",    "CaMKII: R-x-x-S/T-hydrophobic"),

    # ══ UBIQUITIN / UBL SYSTEM ══════════════════════════════════════════════════
    "SUMO_consensus":       (r"[VILMF]K.E",                "ubiquitin_ubl",      "SUMOylation consensus: ψ-K-x-E"),
    "SUMO_inverted":        (r"E.K[VILMF]",                "ubiquitin_ubl",      "Inverted SUMO: E-x-K-ψ (reverse consensus)"),
    "UFM1_site":            (r"[KR][KR]KK",                "ubiquitin_ubl",      "UFMylation: basic cluster K-K-K"),
    "PCNA_PIP_box":         (r"Q..[ILM]..[FA][FY]",        "ubiquitin_ubl",      "PCNA PIP-box: Q-x-x-I/L/M-x-x-F/A-F/Y"),
    "D_box_degron":         (r"R.{2}L",                    "ubiquitin_ubl",      "APC/C D-box (destruction box): R-x-x-L"),
    "KEN_box_degron":       (r"KEN[^P]",                   "ubiquitin_ubl",      "APC/C KEN-box: K-E-N not followed by Pro"),
    "ABBA_APC_motif":       (r"F.{2}[FY]",                 "ubiquitin_ubl",      "ABBA motif: F-x-x-F/Y (Cdh1-binding degron)"),
    "F_box_substrate":      (r"[LIVMF].{3}[LIVMF].{2}.{4}[LIVMF]", "ubiquitin_ubl", "F-box substrate: hydrophobic anchors"),

    # ══ N-TERMINAL ACETYLATION ══════════════════════════════════════════════════
    "NatA_acetyl":          (r"^[SAGCTN][^P]",             "acetylation",        "NatA N-terminal acetylation (after Met removal): S/A/G/C/T/N at position 2"),
    "NatB_acetyl":          (r"^M[DN]",                    "acetylation",        "NatB: Met-Asp or Met-Asn N-terminus"),
    "NatC_acetyl":          (r"^M[LFIVYW]",                "acetylation",        "NatC: Met followed by hydrophobic residue"),

    # ══ METHYLATION ══════════════════════════════════════════════════════════════
    "PRMT_GAR_motif":       (r"GG[AR]|[AR]GG",             "methylation",        "PRMT GAR motif: Gly-Gly-Arg/Arg-Gly-Gly (asymmetric dimethylation)"),
    "PRMT5_sDMA":           (r"G[AR]G",                    "methylation",        "PRMT5 symmetric dimethylarginine: G-R/A-G"),
    "H3K4_SET_context":     (r"ARTKQ",                     "methylation",        "H3 tail ARTK: context for K4 methylation by SET1/MLL"),

    # ══ LIPIDATION ══════════════════════════════════════════════════════════════
    "Myristoylation_Gly2":  (r"^MG[NQHSTAGD]",            "lipidation",         "N-myristoylation: Met-Gly at N-term (Gly-2 exposed after Met removal)"),
    "CaaX_farnesyl":        (r"C[ACVILMF]{2}[ACQSM]$",    "lipidation",         "CaaX farnesylation: Cys-aal-aal-S/C/A/Q/M at C-terminus"),
    "CaaX_geranyl":         (r"C[ACVILMF]{2}[LM]$",       "lipidation",         "CaaX geranylgeranylation: Cys-aal-aal-L/M at C-terminus"),
    "DHHC_palmitoyl":       (r"C[^P]{1,3}C",              "lipidation",         "DHHC palmitoylation substrate: adjacent Cys residues"),

    # ══ PROTEOLYTIC CLEAVAGE ════════════════════════════════════════════════════
    "Furin_cleavage":       (r"[RK].{0,1}[KR]R(?=[^P])",  "cleavage",           "Furin/PCSK: R/K-x-K/R-R (not followed by Pro)"),
    "Caspase_3_6":          (r"[DE]..D(?=[AGSVC])",        "cleavage",           "Caspase-3/6: D/E-x-x-D cleavage"),
    "Caspase_1":            (r"[WYF][EH]HD(?=[AG])",       "cleavage",           "Caspase-1 (IL-1β processing): W/Y/F-E/H-H-D"),
    "Caspase_8_9":          (r"[ILV][EQ]TD(?=[SGNA])",     "cleavage",           "Caspase-8/9: I/L/V-E/Q-T-D"),
    "Granzyme_B":           (r"[IVL]EPD",                  "cleavage",           "Granzyme B: I/V/L-E-P-D cleavage site"),
    "ADAM_sheddase":        (r"HE.{2}H",                   "cleavage",           "ADAM metalloprotease zinc-binding: H-E-x-x-H"),
    "MMP_cleavage":         (r"P[^P].{3}[LIVMA]",          "cleavage",           "Matrix metalloprotease: P-x-x-x-x-hydrophobic"),
    "TEV_protease":         (r"ENL[YF]FQ[SGP]",            "cleavage",           "TEV protease recognition: ENLYFQS/G (biotech tool)"),
    "Signal_peptidase_AXA": (r"[LIVMF]{3,}A[^P]A",        "cleavage",           "Signal peptidase I AXA rule: hydrophobic(3+)-A-x-A"),
    "Thrombin_PAR":         (r"LDPR[^P]",                  "cleavage",           "Thrombin PAR receptor cleavage: L-D-P-R"),

    # ══ LOCALISATION ════════════════════════════════════════════════════════════
    "NLS_basic":            (r"K{3,}|[KR]{4,}",            "localisation",       "Monopartite NLS: K-K-K+ or 4+ consecutive basic residues"),
    "NLS_bipartite":        (r"[KR]{2}.{7,12}[KR]{3,}",   "localisation",       "Bipartite NLS: two basic clusters separated by 7–12 residues"),
    "NES_leucine_rich":     (r"L.{2,3}[LIVMF].{2,3}L.{2,3}[LIVMF]", "localisation", "Nuclear export signal (CRM1/XPO1): L-x(2-3)-hydrophobic-x(2-3)-L"),
    "CRM1_NES":             (r"L[^P].{2}[LIVMF][^P].{2,3}L", "localisation",   "CRM1-dependent NES: strict L-non-P-x-x-hydrophobic pattern"),
    "NoLS_nucleolar":       (r"[KR]{3}.{5,12}[KR]{3}",    "localisation",       "Nucleolar localisation signal (NoLS): two basic clusters"),
    "KDEL_retention":       (r"[KRHQSA]DEL$",              "localisation",       "ER retention: C-terminal KDEL or HDEL"),
    "ER_retrieval_KKXX":    (r"KK.{2}$",                   "localisation",       "ER retrieval KKXX: C-terminal dilysine (Golgi-to-ER)"),
    "ER_retrieval_RR":      (r"RR.{1,4}$",                 "localisation",       "ER retrieval: C-terminal diarginine"),
    "Dilysine_KXKXX":       (r"K.[KR].{2}$",              "localisation",       "KXKXX dilysine ER/Golgi retrieval"),
    "Mito_targeting":       (r"^M[^DE]{0,5}[RK]",          "localisation",       "Mitochondrial presequence: positive N-terminus (M-x-K/R)"),
    "Peroxisome_PTS1":      (r"[SACQ][KRH][LM]$",         "localisation",       "Peroxisome targeting signal 1: C-terminal S/A/C/Q-K/R/H-L/M"),
    "Peroxisome_PTS2":      (r"[RK][LIVMF]{5,6}[QHNS][LIVMF]", "localisation", "Peroxisome targeting signal 2: N-terminal R/K-x5/6-Q/H/N-hydrophobic"),

    # ══ PROTEIN-PROTEIN INTERACTION ═════════════════════════════════════════════
    "RGD_integrin":         (r"RGD",                       "interaction",        "Integrin-binding RGD tripeptide (fibronectin, vitronectin)"),
    "LDV_integrin":         (r"LDV",                       "interaction",        "Integrin LDV-binding site (fibronectin CS-1, VCAM-1)"),
    "SH3_binding":          (r"P.{2}P",                    "interaction",        "SH3 domain binding: P-x-x-P core"),
    "SH3_class_I":          (r"[RK].{2,3}PxxP",           "interaction",        "SH3 class I: R/K at -3 or -4 of P-x-x-P"),
    "SH2_binding":          (r"Y.{1,3}[IVLM]",            "interaction",        "SH2 domain: pY-x-x-I/V/L/M"),
    "PTB_NPXY":             (r"NPX[YF]",                   "interaction",        "PTB domain: N-P-x-Y/F (phosphotyrosine-independent)"),
    "14_3_3_mode1":         (r"R[^EDKR].{1,2}[ST][LFYW]P", "interaction",      "14-3-3 mode 1: R-S/T-x-P phosphoserine docking"),
    "14_3_3_mode2":         (r"R[KR].{1,2}[ST][LFYW]P",   "interaction",       "14-3-3 mode 2: R-R/K-x-x-pS/T-x-P"),
    "PDZ_class_I":          (r"[ST]..[VILFC]$",            "interaction",        "PDZ class I: C-terminal S/T-x-x-hydrophobic"),
    "PDZ_class_II":         (r"[VILFC]..[VILFC]$",         "interaction",        "PDZ class II: C-terminal hydrophobic-x-x-hydrophobic"),
    "EVH1_FPPPP":           (r"FPPPP",                     "interaction",        "EVH1 domain (Ena/VASP): F-P-P-P-P proline-rich"),
    "WD40_binding":         (r"[DE]{2}..[LIVMF]",          "interaction",        "WD40 repeat: acidic patch D/E-D/E-x-x-hydrophobic"),
    "LXXLL_NR_box":         (r"L.{2}LL",                   "interaction",        "LXXLL nuclear receptor coactivator box"),
    "RXL_cyclin_dock":      (r"R.L",                       "interaction",        "Cyclin docking R-x-L (CDK substrate recruitment)"),
    "EF_hand_Ca2":          (r"[DE].{1}[LIVMF].{2}[DE].{1}[LIVMF]", "interaction", "EF-hand Ca2+-binding loop core pattern"),
    "Cadherin_HAV":         (r"HAV",                       "interaction",        "Cadherin homophilic adhesion HAV tripeptide (EC1 domain)"),
    "GFOGER_collagen":      (r"GF.GER",                    "interaction",        "Collagen GFOGER integrin-binding motif"),
    "EGF_like_6Cys":        (r"C.{3,7}C.{3,7}C.{6,15}C.{3,7}C.{3,7}C", "interaction", "EGF-like domain: 6-Cys disulfide pattern"),
    "TSP_WSR_motif":        (r"W[ST]R",                    "interaction",        "Thrombospondin type-1 W-S/T-R cell-binding motif"),
    "Laminin_G_bind":       (r"[KR].{3}[DE].{2}[KR]",     "interaction",        "Laminin G-domain: K/R-x-x-x-D/E-x-x-K/R"),

    # ══ CELL CYCLE / DEGRADATION ════════════════════════════════════════════════
    "PCNA_PIP_strict":      (r"Q.{2}[ILM].{2}[FA][FY]",   "cell_cycle",         "PCNA PIP-box strict: Q-x-x-I/L/M-x-x-F/A-F/Y"),
    "RXL_strict_cyclin":    (r"[RK][VILMF]L",              "cell_cycle",         "Cyclin D-box (strict): R/K-hydrophobic-L"),
    "BRCT_pSer_bind":       (r"[ST].{2}[KR]",              "cell_cycle",         "BRCT phosphoserine docking: S/T-x-x-K/R"),
    "Rb_LxCxE":             (r"L.C.E",                     "cell_cycle",         "Retinoblastoma LxCxE: L-x-C-x-E (viral oncoproteins)"),
    "Cyclin_box_MRAIL":     (r"MR[AG][ILV]L",              "cell_cycle",         "Cyclin box MRAIL hydrophobic cleft contact"),

    # ══ AUTOPHAGY ═══════════════════════════════════════════════════════════════
    "LIR_AIM_LC3":          (r"[WFY].{2}[LIV]",            "autophagy",          "LIR/AIM motif: W/F/Y-x-x-L/I/V (Atg8/LC3 binding)"),
    "KFERQ_CMA":            (r"[KQNR][VILMF].{2,4}[KQNR]", "autophagy",         "KFERQ-like CMA targeting: basic-hydrophobic-x(2-4)-basic"),
    "UIM_ubiquitin":        (r"[LIVMF]A[^P]L[AG][^P]",    "autophagy",          "Ubiquitin-interacting motif (UIM): LALAL core"),
    "ENTH_PH_PIP2":         (r"[KR]{2}.{1}[KR]",           "autophagy",          "PIP2-binding basic patch: K/R-K/R-x-K/R"),

    # ══ MEMBRANE / TOPOLOGY ═════════════════════════════════════════════════════
    "GxxxG_TM_dimer":       (r"G.{3}G",                    "membrane",           "GxxxG TM helix dimerisation interface (glycophorin-like)"),
    "VxP_reticulon":        (r"V[^P]P",                    "membrane",           "VxP membrane curvature (reticulon/REEP family)"),
    "CAAX_general":         (r"C[ACVILMF]{2}[ACQSM]$",    "membrane",           "CaaX prenylation: Cys-aal-aal-S/C/A/Q/M C-terminus"),

    # ══ SIGNALLING MOTIFS ═══════════════════════════════════════════════════════
    "ITAM_immunoreceptor":  (r"Y.{2}[LI].{6,8}Y.{2}[LI]", "signalling",        "ITAM: Y-x-x-L/I-x(6-8)-Y-x-x-L/I (activating immunoreceptors)"),
    "ITIM_inhibitory":      (r"[SLIVY].Y.{2}[LIVMT]",     "signalling",         "ITIM: S/L/I/V/Y-x-Y-x-x-L/I/V/M/T (inhibitory receptors)"),
    "TRAF_binding_6":       (r"[PSAT].Q[EK]",              "signalling",         "TRAF6 binding: P/S/A/T-x-Q-E/K"),
    "DFG_kinase_loop":      (r"DFG[^P]",                   "signalling",         "Kinase activation loop DFG motif (inactive: DFG-out)"),
    "HRD_kinase_loop":      (r"[AG]HRD",                   "signalling",         "Kinase catalytic loop: A/G-H-R-D"),
    "P_loop_Walker_A":      (r"G.{4}GK[ST]",               "signalling",         "Walker A P-loop NTPase: G-x-x-x-x-G-K-S/T"),
    "Walker_B_NTPase":      (r"[LIVMF]{4}DE",              "signalling",         "Walker B: hydrophobic(4)-D-E (Mg2+ coordination)"),
    "DEAD_helicase":        (r"DEAD|DEAH",                  "signalling",         "DEAD/DEAH-box RNA helicase motif II"),
    "SAM_domain_core":      (r"[HYF].{3}[YHF].{3}[KRE]",  "signalling",         "SAM domain aromatic-x(3)-aromatic-x(3)-charged"),

    # ══ REDOX / METAL ════════════════════════════════════════════════════════════
    "CxxC_zinc_finger":     (r"C.{2}C",                    "redox_metal",        "CXXC zinc-finger: Cys-x-x-Cys coordination"),
    "HxxH_zinc":            (r"H.{2}H",                    "redox_metal",        "H-x-x-H zinc coordination (carbonic anhydrase, etc.)"),
    "Thioredoxin_CXXC":     (r"C[GPA].{0,1}C",            "redox_metal",        "Thioredoxin active site: C-G/P/A-x-C"),
    "Glutaredoxin_CPYC":    (r"C[PV][YFW]C",               "redox_metal",        "Glutaredoxin CPYC/CPFC active site"),
    "Iron_sulfur_C4":       (r"C.{2}C.{2}C.{3}CP",        "redox_metal",        "Ferredoxin [4Fe-4S] cluster: C-x-x-C-x-x-C-x-x-x-C-P"),
    "Peroxidase_Cys":       (r"C[^P].{3}[LIVMF]",         "redox_metal",        "Peroxidase active-site Cys: C-non-P-x-x-x-hydrophobic"),
    "SOD_Cu_Zn":            (r"H.{3}H.{33,38}H",          "redox_metal",        "Cu/Zn SOD copper ligand: H-x-x-x-H-x(33-38)-H"),

    # ══ RNA BINDING ══════════════════════════════════════════════════════════════
    "RGG_box_RNA":          (r"RGG",                       "rna_binding",        "RGG box: Arg-Gly-Gly (RNA binding, often tandem)"),
    "KH_domain_GXXG":       (r"G.{2}G",                   "rna_binding",        "KH domain: G-x-x-G RNA-binding GXXG loop"),
    "RRM_RNP1":             (r"[RK].{0,1}GF[IV]",         "rna_binding",        "RRM RNP1 octamer: K/R-x-G-F-I/V"),
    "dsRBD_basic":          (r"[KR].{3}[KR].{2}[KR]",    "rna_binding",        "dsRBD: basic helix contacts (K/R-x-x-x-K/R-x-x-K/R)"),
    "CCCH_zinc_finger":     (r"C.{1,3}C.{1,3}C.{3,9}H",  "rna_binding",        "CCCH zinc finger: C-C-C-H RNA-binding zinc coordination"),

    # ══ STRUCTURAL ══════════════════════════════════════════════════════════════
    "PEST_signal":          (r"[PEST]{8,}",                "structural",         "PEST degradation signal: P/E/S/T-rich region (rapid turnover)"),
    "Cys_disulfide":        (r"C",                         "structural",         "Cysteine residue (disulfide bond potential)"),
    "Leucine_zipper":       (r"L.{6}L.{6}L.{6}L",        "structural",         "Leucine zipper: L-x(6)-L-x(6)-L-x(6)-L heptad repeat"),
    "Coiled_coil_heptad":   (r"[LIVMF].{2}[LIVMF].{3}",   "structural",         "Coiled-coil heptad: hydrophobic at positions a and d"),
    "Proline_kink":         (r"[LIVMF]{3,}P[LIVMF]{3,}",  "structural",         "Proline kink in hydrophobic stretch (TM helix disruption)"),
    "GlyGly_flex":          (r"GG",                        "structural",         "Gly-Gly flexible linker / loop motif"),
    "Ankyrin_repeat":       (r"[DE].{1,2}[GS].{1,2}[KR]", "structural",        "Ankyrin repeat surface: acidic-small-basic triad"),
    "TPR_repeat_core":      (r"[LIVMF].{2}.{3}[AEG].{3}[LIVMF]", "structural", "Tetratricopeptide repeat (TPR) hydrophobic core"),

    # ══ VIRAL / PATHOGEN ════════════════════════════════════════════════════════
    "Integrase_DDE":        (r"D.{35,50}D.{2,5}[EQ]",     "viral_pathogen",     "Retroviral integrase DDE catalytic triad"),
    "Serine_protease_HDS":  (r"H.{30,60}D.{40,80}[ST]",   "viral_pathogen",     "Serine protease catalytic triad H-D-S/T"),
    "Cysteine_protease_HC": (r"H.{20,50}C",                "viral_pathogen",     "Cysteine protease catalytic pair H-C"),
    "Zinc_metalloprot":     (r"HExxH",                     "viral_pathogen",     "Zinc metalloprotease: H-E-x-x-H active site"),
    "Viral_PPXY":           (r"PP.Y",                      "viral_pathogen",     "Viral late domain PPXY: budding from host membrane"),
    "Viral_PTAP":           (r"PT[AS]P",                   "viral_pathogen",     "Viral late domain PTAP/PSAP: TSG101 binding (HIV, Ebola)"),
    "Viral_YPX3L":          (r"Y[^P].{3}L",               "viral_pathogen",     "Viral late domain YPxnL: ALIX binding motif"),
    "Zoonotic_furin":       (r"RR[AR]R",                   "viral_pathogen",     "Polybasic furin site: R-R-A/R-R (SARS-CoV-2 S1/S2)"),
}


POSITIVE_AA    = set("KR")
HYDROPHOBIC_AA = set("AILMFWV")
SMALL_AA       = set("AVSTG")
AROMATIC_AA    = set("FWY")

# ══════════════════════════════════════════════════════════════════════════════
# Data Models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MotifHit:
    motif: str
    category: str
    start: int
    end: int
    matched: str
    context: str
    description: str

@dataclass
class PipelineResult:
    # Identity
    accession:    str
    description:  str
    sequence:     str
    length:       int
    timestamp:    str = field(default_factory=lambda: datetime.now().isoformat())

    # Stage outputs
    physicochemical:    Optional[dict] = None
    composition:        Optional[dict] = None
    hydropathy:         Optional[dict] = None
    secondary_structure:Optional[dict] = None
    complexity:         Optional[dict] = None
    signal_peptide:     Optional[dict] = None
    motifs:             Optional[dict] = None
    disorder:           Optional[dict] = None
    mass:               Optional[dict] = None
    maldi:              Optional[dict] = None
    cleavage:           Optional[dict] = None
    ml_prediction:      Optional[dict] = None
    feature_vector:     Optional[dict] = None
    ghma:               Optional[dict] = None

    # Pipeline metadata
    stages_run:   list = field(default_factory=list)
    warnings:     list = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — FASTA Parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_fasta(filepath: str) -> list[tuple[str,str,str]]:
    records, acc, desc, parts = [], None, "", []
    with open(filepath) as fh:
        for line in fh:
            line = line.rstrip()
            if not line or line.startswith(";"): continue
            if line.startswith(">"):
                if acc: records.append((acc, desc, "".join(parts).upper()))
                hdr   = line[1:]
                sp    = hdr.split(None, 1)
                acc   = sp[0]
                desc  = sp[1] if len(sp) > 1 else ""
                parts = []
            else:
                parts.append(line.upper().replace(" ",""))
    if acc: records.append((acc, desc, "".join(parts).upper()))
    if not records:
        raise ValueError(f"No FASTA records in '{filepath}'")
    return records

def clean_seq(seq: str) -> str:
    return "".join(c for c in seq if c in AA_STANDARD)

# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2c — Mass Prediction (raw AA mass + PTM-modified range)
# ══════════════════════════════════════════════════════════════════════════════

def stage_mass(seq: str, sp_cleavage_pos: int = 0,
               motif_hits: dict = None) -> dict:
    """
    Computes:
      raw_monoisotopic_da  — unmodified sequence, monoisotopic scale
      raw_average_da       — unmodified sequence, average mass (for SDS-PAGE)
      ptm_min_da           — raw + disulfide reductions only
      ptm_max_da           — raw + every eligible PTM applied once per site
      ptm_expected_da      — raw + occupancy-weighted estimate (uses motif evidence)
      per_ptm              — per-modification contribution table
      esi_charge_states    — predicted m/z for z=1..8 (protonated, expected mass)
      sp_peptide_mass_da   — mass of cleaved signal peptide (if sp_cleavage_pos > 0)
    """
    mature = seq[sp_cleavage_pos:] if sp_cleavage_pos > 0 else seq
    ct = Counter(mature)
    n  = len(mature)

    # Raw monoisotopic
    raw = sum(RESIDUE_MASS.get(aa, 0.0) for aa in mature) + WATER
    # Raw average
    avg = sum(AA_AVG_MASS.get(aa, 0.0) for aa in mature) + 18.015
    # SP peptide
    sp_mass = None
    if sp_cleavage_pos > 0:
        sp_mass = round(sum(RESIDUE_MASS.get(aa, 0.0) for aa in seq[:sp_cleavage_pos]) + WATER, 4)

    # Per-PTM contributions
    per_ptm = {}
    total_pos_delta = 0.0
    disulf_delta    = 0.0

    for name, ptm in PTM_CATALOGUE.items():
        res = ptm["res"]
        if res is None:
            count = 1
        elif res == "CC":
            # Conservative: only pair Cys if motif evidence (Cys_disulfide hits)
            # or if Cys density > 3% (strongly suggestive of disulfide-rich protein)
            cys_n = ct.get("C", 0)
            cys_density = cys_n / max(n, 1)
            if (motif_hits and "Cys_disulfide" in motif_hits
                    and motif_hits["Cys_disulfide"].get("count", 0) >= 2):
                count = cys_n // 2
            elif cys_density >= 0.03:
                count = cys_n // 2
            else:
                count = 0  # assume free Cys; no disulfide delta
        elif res == "ST":
            count = ct.get("S", 0) + ct.get("T", 0)
        elif len(res) == 1:
            count = ct.get(res, 0)
        else:
            count = sum(ct.get(r, 0) for r in res)
        if count == 0:
            continue
        shift = ptm["delta"] * count
        per_ptm[name] = {
            "delta_per_site": round(ptm["delta"], 4),
            "eligible_sites": count,
            "total_delta":    round(shift, 4),
            "description":    ptm["desc"],
            "class":          ptm["cls"],
        }
        if name == "Disulfide":
            disulf_delta = shift
        elif shift > 0:
            total_pos_delta += shift

    ptm_min = round(raw + disulf_delta, 4)
    ptm_max = round(raw + total_pos_delta, 4)

    # Evidence-weighted expected mass
    exp_delta = disulf_delta
    if motif_hits:
        for motif, ptm_names in _MOTIF_PTM_MAP.items():
            if motif not in motif_hits: continue
            nhits = motif_hits[motif].get("count", 0)
            for pname in ptm_names:
                if pname not in per_ptm: continue
                c   = per_ptm[pname]
                occ = _PTM_OCC.get(c["class"], 0.10)
                exp_delta += c["delta_per_site"] * min(nhits, c["eligible_sites"]) * occ
    else:
        for name, c in per_ptm.items():
            if name == "Disulfide": continue
            exp_delta += c["delta_per_site"] * c["eligible_sites"] * _PTM_OCC.get(c["class"], 0.05)

    ptm_expected = round(raw + exp_delta, 4)

    # By-class summary
    by_class: dict = {}
    for name, c in per_ptm.items():
        cls = c["class"]
        by_class.setdefault(cls, {"ptms": [], "max_delta": 0.0})
        by_class[cls]["ptms"].append(name)
        by_class[cls]["max_delta"] = round(by_class[cls]["max_delta"] + c["total_delta"], 4)

    # ESI m/z — protonated [M+zH]z+, sodium [M+Na]+, potassium [M+K]+
    PROTON =  1.007276
    SODIUM  = 22.989218
    POTASS  = 38.963158
    esi = {
        **{f"H_z{z}": round((ptm_expected + z * PROTON) / z, 4) for z in range(1, 9)},
        "Na_z1":  round(ptm_expected - PROTON + SODIUM + PROTON, 4),   # [M+Na]+
        "K_z1":   round(ptm_expected - PROTON + POTASS  + PROTON, 4),  # [M+K]+
        "Na2_z2": round((ptm_expected + SODIUM + PROTON) / 2, 4),      # [M+Na+H]2+
    }

    # Isotopic scale recommendation
    if raw < 3000:
        iso_hint = "< 3 kDa: monoisotopic mass dominant; use for peptide MS"
    elif raw < 15000:
        iso_hint = "3–15 kDa: monoisotopic resolvable on Orbitrap/FT-ICR; average for MALDI/SDS-PAGE"
    else:
        iso_hint = "> 15 kDa: average mass for MALDI/SDS-PAGE; monoisotopic requires FT-ICR or native MS"

    # Recommended mass to report depending on analysis type
    recommended = {
        "bottom_up_ms":   round(raw, 4),           # peptide-level: monoisotopic
        "intact_protein":  round(ptm_expected, 4), # intact MS: PTM-adjusted expected
        "sds_page":        round(avg / 1000, 1),   # gel: average in kDa (rough)
        "native_ms":       round(ptm_expected, 4), # native: expected with modifications
    }

    return {
        "sequence_length":     n,
        "includes_sp_removal": sp_cleavage_pos > 0,
        "sp_cleavage_pos":     sp_cleavage_pos,
        "sp_peptide_mass_da":  sp_mass,
        "raw_monoisotopic_da": round(raw, 4),
        "raw_average_da":      round(avg, 4),
        "ptm_min_da":          ptm_min,
        "ptm_max_da":          ptm_max,
        "ptm_expected_da":     ptm_expected,
        "ptm_range_da":        round(ptm_max - ptm_min, 2),
        "ptm_delta_expected":  round(exp_delta, 4),
        "per_ptm":             per_ptm,
        "by_ptm_class":        by_class,
        "esi_charge_states":   esi,
        "raw_kda":             round(raw / 1000, 4),
        "ptm_expected_kda":    round(ptm_expected / 1000, 4),
        "isotopic_hint":       iso_hint,
        "recommended_masses":  recommended,
    }



# ══════════════════════════════════════════════════════════════════════════════
# Stage 2d — MALDI-TOF Intact Native Mass Simulation
# ══════════════════════════════════════════════════════════════════════════════

def _build_maldi_svg(native_avg, reduced_avg, env_n, env_r,
                     adducts_n, adducts_r, n_ss, matrix):
    W, H = 460, 130
    PAD_L, PAD_R, PAD_T, PAD_B = 36, 12, 14, 28
    all_m = [p[0] for p in env_n + env_r]
    xmin, xmax = min(all_m), max(all_m)
    xspan = xmax - xmin or 1.0
    def xp(m): return PAD_L + ((m - xmin) / xspan) * (W - PAD_L - PAD_R)
    def yp(r): return PAD_T + (1.0 - r) * (H - PAD_T - PAD_B)
    y0 = yp(0)
    parts = [
        f'<rect width="{W}" height="{H}" fill="#08111c" rx="4"/>',
        f'<line x1="{PAD_L}" y1="{y0:.1f}" x2="{W-PAD_R}" y2="{y0:.1f}" stroke="#1a2535" stroke-width="1"/>',
        f'<text x="{PAD_L+2}" y="{PAD_T+9}" font-family="monospace" font-size="8" fill="#2a3a4a">MALDI \u00b7 {matrix} \u00b7 linear +ve mode</text>',
    ]
    # X-axis ticks
    mag = 10 ** math.floor(math.log10(max(xspan / 6, 1)))
    tick_step = max(1, round(xspan / 6 / mag) * mag)
    t = math.ceil(xmin / tick_step) * tick_step
    while t <= xmax:
        tx = xp(t)
        parts.append(f'<line x1="{tx:.1f}" y1="{y0}" x2="{tx:.1f}" y2="{y0+4}" stroke="#2a3a4a" stroke-width="1"/>')
        parts.append(f'<text x="{tx:.1f}" y="{H-4}" text-anchor="middle" font-family="monospace" font-size="8" fill="#3a4a5a">{t:.0f}</text>')
        t += tick_step
    # Envelopes
    for env, col, fcol in [(env_n, "#60a5fa", "#3b82f6"), (env_r, "#fb7185", "#f43f5e")]:
        pts  = ' '.join(f'{xp(m):.1f},{yp(r):.1f}' for m, r in env)
        fill = f'{xp(env[0][0]):.1f},{y0:.1f} ' + pts + f' {xp(env[-1][0]):.1f},{y0:.1f}'
        parts.append(f'<polygon points="{fill}" fill="{fcol}" opacity="0.12"/>')
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.8" stroke-linejoin="round" opacity="0.9"/>')
    # Peak labels
    for mass, col, lbl in [(native_avg, "#93c5fd", f"Native {native_avg:.1f}"),
                            (reduced_avg, "#fca5a5", f"+DTT {reduced_avg:.1f}")]:
        px = xp(mass)
        anchor = "start" if mass < (xmin+xmax)/2 else "end"
        off = 4 if anchor == "start" else -4
        parts.append(f'<line x1="{px:.1f}" y1="{PAD_T+12}" x2="{px:.1f}" y2="{yp(0.92):.1f}" stroke="{col}" stroke-width="0.8" stroke-dasharray="3 2" opacity="0.6"/>')
        parts.append(f'<text x="{px+off:.1f}" y="{PAD_T+22}" text-anchor="{anchor}" font-family="monospace" font-size="8" font-weight="500" fill="{col}">{lbl} Da</text>')
    # Delta annotation
    if n_ss > 0:
        delta = reduced_avg - native_avg
        mx = (xp(native_avg) + xp(reduced_avg)) / 2
        my = PAD_T + (H - PAD_T - PAD_B) * 0.55
        parts.append(
            '<defs><marker id="arr" markerWidth="5" markerHeight="5" refX="2.5" refY="2.5" orient="auto">'
            '<path d="M0,0 L5,2.5 L0,5 Z" fill="#fbbf24"/></marker></defs>'
        )
        parts.append(f'<line x1="{xp(native_avg):.1f}" y1="{my:.1f}" x2="{xp(reduced_avg):.1f}" y2="{my:.1f}" stroke="#fbbf24" stroke-width="1" marker-end="url(#arr)"/>')
        parts.append(f'<text x="{mx:.1f}" y="{my-4:.1f}" text-anchor="middle" font-family="monospace" font-size="8" fill="#fbbf24">+{delta:.2f} Da ({n_ss} \u00d7 SS)</text>')
    # Legend
    lx = W - PAD_R - 2
    parts.append(
        f'<rect x="{lx-82}" y="{H-PAD_B+3}" width="9" height="5" rx="1" fill="#60a5fa" opacity="0.85"/>'
        f'<text x="{lx-69}" y="{H-PAD_B+9}" font-family="monospace" font-size="8" fill="#93c5fd">Native</text>'
        f'<rect x="{lx-28}" y="{H-PAD_B+3}" width="9" height="5" rx="1" fill="#fb7185" opacity="0.85"/>'
        f'<text x="{lx-15}" y="{H-PAD_B+9}" font-family="monospace" font-size="8" fill="#fca5a5">+DTT</text>'
    )
    return (f'<svg viewBox="0 0 {W} {H}" class="maldi-svg" xmlns="http://www.w3.org/2000/svg">'
            + ''.join(parts) + '</svg>')


def stage_maldi(seq: str, sp_cleavage_pos: int = 0,
                n_cys_pairs: int = None,
                matrix: str = 'sinapinic') -> dict:
    """
    MALDI-TOF simulation: intact native vs DTT-reduced average masses.
    Gaussian isotope envelope (Yergey 1983). Linear positive mode adducts.
    """
    mature = seq[sp_cleavage_pos:] if sp_cleavage_pos > 0 else seq
    ct     = Counter(mature)
    n_cys  = ct.get('C', 0)
    n_ss   = min(n_cys_pairs if n_cys_pairs is not None else n_cys // 2, n_cys // 2)
    raw_avg     = sum(AA_AVG_MASS.get(aa, 0.0) for aa in mature) + 18.015
    native_avg  = raw_avg - n_ss * 2.01565
    reduced_avg = raw_avg
    def _env(center, n_pts=300):
        sigma = 0.00055 * math.sqrt(center) * center
        span  = max(4.0 * sigma, 80.0)
        return [(round(center - span + i*(2*span/n_pts), 2),
                 round(math.exp(-0.5*((center - span + i*(2*span/n_pts) - center)/sigma)**2), 6))
                for i in range(n_pts + 1)]
    H_ = 1.007276; Na = 22.989218; K = 38.963158
    def _adu(m): return {
        '[M+H]+':    round(m + H_, 2),
        '[M+Na]+':   round(m + Na, 2),
        '[M+K]+':    round(m + K,  2),
        '[M+2H]2+':  round((m + 2*H_) / 2, 2),
        '[M+2Na]2+': round((m + 2*Na) / 2, 2),
    }
    env_n = _env(native_avg);  env_r = _env(reduced_avg)
    an = _adu(native_avg);     ar = _adu(reduced_avg)
    return {
        'mature_length':    len(mature),
        'n_cys_total':      n_cys,
        'n_disulfides':     n_ss,
        'matrix':           matrix,
        'raw_avg_da':       round(raw_avg,     4),
        'native_avg_da':    round(native_avg,  4),
        'reduced_avg_da':   round(reduced_avg, 4),
        'delta_dtt_da':     round(reduced_avg - native_avg, 4),
        'sigma_native_da':  round(0.00055 * math.sqrt(native_avg)  * native_avg,  2),
        'sigma_reduced_da': round(0.00055 * math.sqrt(reduced_avg) * reduced_avg, 2),
        'adducts_native':   an,
        'adducts_reduced':  ar,
        'envelope_native':  env_n,
        'envelope_reduced': env_r,
        'spectrum_svg':     _build_maldi_svg(native_avg, reduced_avg, env_n, env_r, an, ar, n_ss, matrix),
    }

# Stage 2 — Physicochemical
# ══════════════════════════════════════════════════════════════════════════════

def _net_charge(seq, pH):
    c  = 1/(1+10**(pH-PKA['N_term']))
    c -= 1/(1+10**(PKA['C_term']-pH))
    for aa in seq:
        if aa in ('K','R','H'): c += 1/(1+10**(pH-PKA[aa]))
        elif aa in ('D','E','C','Y'): c -= 1/(1+10**(PKA[aa]-pH))
    return c

def _pi(seq):
    lo, hi = 0.0, 14.0
    for _ in range(60):
        m = (lo+hi)/2
        if _net_charge(seq,m) > 0: lo = m
        else: hi = m
    return (lo+hi)/2

def _instability(seq):
    if len(seq) < 2: return 0.0
    return (10/len(seq)) * sum(INST_W.get((seq[i],seq[i+1]),1.0) for i in range(len(seq)-1))

def stage_physicochemical(seq: str) -> dict:
    n  = len(seq)
    ct = Counter(seq)
    mw = sum(RESIDUE_MASS.get(aa,0) for aa in seq) + WATER
    return {
        "length":               n,
        "molecular_weight_da":  round(mw,2),
        "isoelectric_point":    round(_pi(seq),2),
        "gravy":                round(sum(KD.get(aa,0) for aa in seq)/n, 4),
        "aromaticity":          round(sum(ct.get(aa,0) for aa in "FWY")/n, 4),
        "instability_index":    round(_instability(seq),2),
        "aliphatic_index":      round((ct.get('A',0)*100 + ct.get('V',0)*290 +
                                      (ct.get('I',0)+ct.get('L',0))*390)/n, 2),
        "net_charge_ph74":      round(_net_charge(seq,7.4),2),
        "extinction_coeff":     ct.get('W',0)*5500 + ct.get('Y',0)*1490 + ct.get('C',0)*125,
        "stable":               _instability(seq) < 40.0,
    }

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2b — Composition
# ══════════════════════════════════════════════════════════════════════════════

def stage_composition(seq: str) -> dict:
    n  = len(seq)
    ct = Counter(seq)
    mono = {aa: round(ct.get(aa,0)/n,5) for aa in sorted(AA_STANDARD)}
    AA_LIST = sorted(AA_STANDARD)
    dp_counts = {a+b: 0 for a in AA_LIST for b in AA_LIST}
    for i in range(len(seq)-1):
        p = seq[i:i+2]
        if p in dp_counts: dp_counts[p] += 1
    tot = sum(dp_counts.values()) or 1
    return {
        "monomer_freq":   mono,
        "dipeptide_freq": {k: round(v/tot, 5) for k, v in sorted(dp_counts.items())},
        "charged_positive":  round(sum(ct.get(a,0) for a in "KR")/n,4),
        "charged_negative":  round(sum(ct.get(a,0) for a in "DE")/n,4),
        "hydrophobic_frac":  round(sum(ct.get(a,0) for a in HYDROPHOBIC_AA)/n,4),
        "aromatic_frac":     round(sum(ct.get(a,0) for a in AROMATIC_AA)/n,4),
        "polar_frac":        round(sum(ct.get(a,0) for a in "STNQ")/n,4),
    }

# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Secondary Structure (Chou-Fasman)
# ══════════════════════════════════════════════════════════════════════════════

def stage_secondary_structure(seq: str, window: int = 7) -> dict:
    n     = len(seq)
    half  = window//2
    alpha, beta, coil = [], [], []
    for i in range(n):
        seg = seq[max(0,i-half):min(n,i+half+1)]
        pa  = sum(CF.get(a,(1,1,1))[0] for a in seg)/len(seg)
        pb  = sum(CF.get(a,(1,1,1))[1] for a in seg)/len(seg)
        pc  = sum(CF.get(a,(1,1,1))[2] for a in seg)/len(seg)
        alpha.append(round(pa,4)); beta.append(round(pb,4)); coil.append(round(pc,4))
    calls = []
    for pa,pb,_ in zip(alpha,beta,coil):
        if pa>1.03 and pa>=pb: calls.append('H')
        elif pb>1.05 and pb>pa: calls.append('E')
        else: calls.append('C')
    cc = Counter(calls)
    nt = min(35,n)
    regions, cur, s0 = [], calls[0], 0
    for i,c in enumerate(calls[1:],1):
        if c != cur:
            if i-s0>=4: regions.append({"type":cur,"start":s0,"end":i-1,"length":i-s0})
            cur,s0 = c,i
    if n-s0>=4: regions.append({"type":cur,"start":s0,"end":n-1,"length":n-s0})
    return {
        "p_alpha": alpha, "p_beta": beta, "p_coil": coil,
        "calls": "".join(calls), "regions": regions,
        "helix_fraction": round(cc.get('H',0)/n,4),
        "sheet_fraction": round(cc.get('E',0)/n,4),
        "coil_fraction":  round(cc.get('C',0)/n,4),
        "n_term_helix_score": round(calls[:nt].count('H')/nt,4),
        "n_helix_regions": sum(1 for r in regions if r['type']=='H'),
        "n_sheet_regions": sum(1 for r in regions if r['type']=='E'),
    }

# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Hydropathy
# ══════════════════════════════════════════════════════════════════════════════

def stage_hydropathy(seq: str, window: int = 9) -> dict:
    n = len(seq)
    if n < window:
        return {"window":window,"profile":[],"hydrophobic_regions":[],"tm_helices":[],
                "mean_score":0,"max_score":0,"n_hydrophobic_regions":0,"n_tm_helices":0}
    # O(n) cumulative-sum sliding window — 4× faster than naive re-sum per window
    kd_arr  = [KD.get(aa, 0.0) for aa in seq]
    win_sum = sum(kd_arr[:window])
    scores  = [round(win_sum / window, 4)]
    for i in range(1, n - window + 1):
        win_sum += kd_arr[i + window - 1] - kd_arr[i - 1]
        scores.append(round(win_sum / window, 4))
    threshold, TM_MIN = 1.6, 18
    regions, in_r, s0 = [], False, 0
    for i, v in enumerate(scores):
        if v >= threshold and not in_r:  in_r, s0 = True, i
        elif v < threshold and in_r:
            seg = scores[s0:i]; ln = i - s0 + window - 1
            regions.append({"start":s0,"end":i+window-2,"length":ln,
                            "mean_score":round(sum(seg)/max(1,len(seg)),3),
                            "is_tm": ln >= TM_MIN})
            in_r = False
    if in_r:
        seg = scores[s0:]; ln = len(scores) - s0 + window - 1
        regions.append({"start":s0,"end":len(scores)+window-2,"length":ln,
                        "mean_score":round(sum(seg)/max(1,len(seg)),3),
                        "is_tm": ln >= TM_MIN})
    tm_helices = [r for r in regions if r["is_tm"]]
    pk = scores.index(max(scores))
    return {
        "window":window,"threshold":threshold,"tm_min_length":TM_MIN,
        "profile":scores,
        "mean_score":round(sum(scores)/len(scores),4),
        "max_score":round(max(scores),4),"min_score":round(min(scores),4),
        "peak_position":pk,
        "hydrophobic_regions":regions,"n_hydrophobic_regions":len(regions),
        "tm_helices":tm_helices,"n_tm_helices":len(tm_helices),
    }

# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Complexity
# ══════════════════════════════════════════════════════════════════════════════

def stage_complexity(seq: str, window: int = 12) -> dict:
    max_h, lc_thr = math.log2(20), 2.0
    entropies = []
    for i in range(len(seq)-window+1):
        ct = Counter(seq[i:i+window])
        h  = -sum((c/window)*math.log2(c/window) for c in ct.values() if c>0)
        entropies.append(round(h,4))
    if not entropies:
        return {"window":window,"profile":[],"low_complexity_regions":[],"mean_entropy":0,"is_low_complexity":False}
    lc_regions = []
    in_r,s0 = False,0
    for i,h in enumerate(entropies):
        if h<lc_thr and not in_r: in_r,s0=True,i
        elif h>=lc_thr and in_r:
            lc_regions.append({"start":s0,"end":i+window-2,"sequence":seq[s0:i+window-1]})
            in_r=False
    if in_r: lc_regions.append({"start":s0,"end":len(entropies)+window-2,"sequence":seq[s0:len(entropies)+window-1]})
    mean_h = sum(entropies)/len(entropies)
    return {
        "window":window,"max_entropy_bits":round(max_h,4),"lc_threshold":lc_thr,
        "profile":entropies,"mean_entropy":round(mean_h,4),"min_entropy":round(min(entropies),4),
        "low_complexity_regions":lc_regions,"n_lc_regions":len(lc_regions),
        "is_low_complexity":mean_h<lc_thr,
    }

# ══════════════════════════════════════════════════════════════════════════════

# Stage 5b — Disorder prediction (heuristic IUPred-like)
# ══════════════════════════════════════════════════════════════════════════════

def stage_disorder(seq: str, window: int = 21) -> dict:
    """
    Per-residue disorder score. Disordered: high charge + low hydrophobicity + high entropy.
    Score = charge_density - hydro_density + normalised_entropy. Threshold >= 0.5.
    """
    n, max_h = len(seq), math.log2(20)
    half     = window // 2
    scores   = []
    for i in range(n):
        seg  = seq[max(0, i-half):min(n, i+half+1)]
        slen = len(seg)
        ct   = Counter(seg)
        h    = -sum((c/slen)*math.log2(c/slen) for c in ct.values() if c > 0)
        chg  = sum(1 for aa in seg if aa in "DEKR") / slen
        hyd  = sum(1 for aa in seg if aa in HYDROPHOBIC_AA) / slen
        scores.append(round(chg - hyd + h / max_h, 4))
    threshold = 0.5
    regions, in_r, s0 = [], False, 0
    for i, v in enumerate(scores):
        if v >= threshold and not in_r: in_r, s0 = True, i
        elif v < threshold and in_r:
            if i - s0 >= 10:
                regions.append({"start":s0+1,"end":i,"length":i-s0,
                                "mean_score":round(sum(scores[s0:i])/(i-s0),3),
                                "sequence":seq[s0:i]})
            in_r = False
    if in_r and n - s0 >= 10:
        regions.append({"start":s0+1,"end":n,"length":n-s0,
                        "mean_score":round(sum(scores[s0:])/max(1,n-s0),3),
                        "sequence":seq[s0:]})
    frac = sum(1 for v in scores if v >= threshold) / n
    return {
        "window":window,"threshold":threshold,"profile":scores,
        "disordered_frac":round(frac,4),
        "disordered_regions":regions,
        "n_disordered_regions":len(regions),
        "is_disordered": frac > 0.35,
    }

# Stage 6 — Signal Peptide
# ══════════════════════════════════════════════════════════════════════════════

def stage_signal_peptide(seq: str, max_scan: int = 45) -> dict:
    best, best_score = None, -float('inf')
    limit = min(max_scan, len(seq)-5)
    for cl in range(10, limit+1):
        n_reg = seq[:5]
        h_reg = seq[5:cl-3]
        c_reg = seq[cl-3:cl]
        if len(h_reg)<4 or len(c_reg)<3: continue
        pd  = sum(a in POSITIVE_AA    for a in n_reg)/5
        hd  = sum(a in HYDROPHOBIC_AA for a in h_reg)/len(h_reg)
        ok  = c_reg[0] in SMALL_AA and c_reg[2] in SMALL_AA
        pen = max(0,(len(h_reg)-15)*0.25)
        sc  = 2.0*pd + 4.0*hd + (2.5 if ok else -1.5) - pen
        if sc > best_score:
            best_score = sc
            best = {"predicted_cleavage_pos":cl,"score":round(sc,4),
                    "pos_density":round(pd,4),"hydro_density":round(hd,4),
                    "cleavage_ok":ok,"h_region_len":len(h_reg),
                    "signal_peptide":seq[:cl],"mature_start":seq[cl:cl+10],
                    "n_region":n_reg,"h_region":h_reg,"c_region":c_reg}
    if best and best["score"] >= 2.0:
        best["detected"] = True
        return best
    return {"detected":False,"score":round(best_score,4) if best else None,
            "reason":"Score below threshold (2.0)"}

# ══════════════════════════════════════════════════════════════════════════════
# Stage 7 — Motif Scanner
# ══════════════════════════════════════════════════════════════════════════════

_COMPILED_MOTIFS = {k: re.compile(v[0]) for k,v in MOTIF_LIB.items()}

def stage_motifs(seq: str, ctx: int = 5) -> dict:
    hits_by_type: dict[str,list] = {}
    for name, compiled in _COMPILED_MOTIFS.items():
        _pat, cat, desc = MOTIF_LIB[name]
        hits = []
        for m in compiled.finditer(seq):
            s,e = m.start(), m.end()
            fl  = seq[max(0,s-ctx):s]
            fr  = seq[e:min(len(seq),e+ctx)]
            hits.append(MotifHit(motif=name,category=cat,start=s+1,end=e,
                                 matched=m.group(),context=f"{fl}[{m.group()}]{fr}",
                                 description=desc))
        if hits:
            hits_by_type[name] = {
                "category":cat,"description":desc,
                "count":len(hits),
                "hits":[{"start":h.start,"end":h.end,"matched":h.matched,
                          "context":h.context} for h in hits]
            }
    cat_counts: dict[str,int] = {}
    for v in hits_by_type.values():
        cat_counts[v["category"]] = cat_counts.get(v["category"],0) + v["count"]
    return {
        "n_motif_types": len(hits_by_type),
        "total_hits":    sum(v["count"] for v in hits_by_type.values()),
        "by_category":   cat_counts,
        "motifs":        hits_by_type,
    }

# ══════════════════════════════════════════════════════════════════════════════
# Stage 8 — ML Feature Vector + Prediction
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_vector(r: PipelineResult) -> dict:
    fv = {}
    if r.physicochemical:
        p = r.physicochemical
        fv.update({"mw_kda":round(p["molecular_weight_da"]/1000,3),"pi":p["isoelectric_point"],
                   "gravy":p["gravy"],"aromaticity":p["aromaticity"],
                   "instability_idx":p["instability_index"],"aliphatic_idx":p["aliphatic_index"],
                   "net_charge_74":p["net_charge_ph74"],"length":p["length"]})
    if r.composition:
        c = r.composition
        for aa in sorted(AA_STANDARD): fv[f"aa_{aa}"] = c["monomer_freq"].get(aa,0)
        fv.update({"charged_pos":c["charged_positive"],"charged_neg":c["charged_negative"],
                   "hydrophobic":c["hydrophobic_frac"],"aromatic_frac":c["aromatic_frac"],
                   "polar_frac":c["polar_frac"]})
    if r.secondary_structure:
        ss = r.secondary_structure
        fv.update({"helix_frac":ss["helix_fraction"],"sheet_frac":ss["sheet_fraction"],
                   "coil_frac":ss["coil_fraction"],"n_term_helix_score":ss["n_term_helix_score"],
                   "n_helix_regions":ss["n_helix_regions"],"n_sheet_regions":ss["n_sheet_regions"]})
    if r.hydropathy:
        h = r.hydropathy
        fv.update({"hydropathy_mean":h["mean_score"],"hydropathy_max":h["max_score"],
                   "n_hydro_regions":h["n_hydrophobic_regions"]})
    if r.complexity:
        fv.update({"mean_entropy":r.complexity["mean_entropy"],
                   "min_entropy":r.complexity["min_entropy"],
                   "n_lc_regions":r.complexity["n_lc_regions"]})
    if r.signal_peptide:
        sp = r.signal_peptide
        fv.update({"sp_detected":1 if sp.get("detected") else 0,
                   "sp_score":sp.get("score",0) or 0,
                   "sp_hydro":sp.get("hydro_density",0) or 0,
                   "sp_cleavage_ok":1 if sp.get("cleavage_ok") else 0})
    if r.hydropathy:
        fv["n_tm_helices"] = r.hydropathy.get("n_tm_helices", 0)
    if r.disorder:
        d = r.disorder
        fv.update({"disordered_frac":d.get("disordered_frac",0),
                   "n_disordered_regions":d.get("n_disordered_regions",0),
                   "is_disordered":int(d.get("is_disordered",False))})
    if r.motifs:
        m = r.motifs["motifs"]; n = r.length or 1
        fv.update({"n_glyco_sites":m.get("N_glycosylation",{}).get("count",0),
                   "n_cys":m.get("Cys_disulfide",{}).get("count",0),
                   "has_kdel":1 if "KDEL_retention" in m else 0,
                   "has_nls":1 if ("NLS_basic" in m or "NLS_bipartite" in m) else 0})
        nls_hits = (m.get("NLS_basic",{}).get("hits",[]) +
                    m.get("NLS_bipartite",{}).get("hits",[]))
        fv["nls_n_terminal"] = int(any(h["start"] <= n//3 for h in nls_hits))
        fv["nls_c_terminal"] = int(any(h["start"] >= (2*n)//3 for h in nls_hits))
        kdel_hits = m.get("KDEL_retention",{}).get("hits",[])
        fv["kdel_c_terminal"] = int(any(h["end"] >= n-5 for h in kdel_hits))
    return fv

def stage_ml_predict(fv: dict, model_path: str) -> dict:
    if not HAS_ML:
        return {"error":"scikit-learn / pandas not installed"}
    try:
        bundle   = joblib.load(model_path)
        pipeline = bundle["pipeline"]
        le       = bundle["label_encoder"]
        df       = pd.DataFrame([fv])
        proba    = pipeline.predict_proba(df)[0]
        classes  = list(le.classes_)
        idx      = int(proba.argmax())
        conf_bands = [(0.90,"HIGH"),(0.70,"MODERATE"),(0.50,"LOW"),(0.0,"UNCERTAIN")]
        band = next(b for t,b in conf_bands if proba[idx] >= t)
        return {
            "prediction":      classes[idx],
            "confidence":      round(float(proba[idx]),4),
            "confidence_band": band,
            "per_class_proba": {c:round(float(p),4) for c,p in zip(classes,proba)},
        }
    except Exception as e:
        return {"error":str(e)}

# ══════════════════════════════════════════════════════════════════════════════
# Stage 10 — GHMA  (Geometric Harmonic Manifold Analysis)
# ══════════════════════════════════════════════════════════════════════════════
#
# Multichannel physicochemical + Chou-Fasman embedding → SVD manifold.
# Computes global spectral/geometric metrics and a local sliding-window
# profile.  Co-peak detection flags residue positions of structural interest.
#
# All functions prefixed _ghma_ to avoid namespace collisions with the
# pipeline's own tables (which share some names).
# ══════════════════════════════════════════════════════════════════════════════

# Physicochemical channels (z-scored per sequence)
_GHMA_PROPS = {
    "hydrophobicity": {
        'A': 1.8, 'C': 2.5, 'D':-3.5, 'E':-3.5, 'F': 2.8, 'G':-0.4,
        'H':-3.2, 'I': 4.5, 'K':-3.9, 'L': 3.8, 'M': 1.9, 'N':-3.5,
        'P':-1.6, 'Q':-3.5, 'R':-4.5, 'S':-0.8, 'T':-0.7, 'V': 4.2,
        'W':-0.9, 'Y':-1.3,
    },
    "charge": {
        'A': 0,  'C': 0,  'D':-1,  'E':-1,  'F': 0,  'G': 0,
        'H': 0.5,'I': 0,  'K': 1,  'L': 0,  'M': 0,  'N': 0,
        'P': 0,  'Q': 0,  'R': 1,  'S': 0,  'T': 0,  'V': 0,
        'W': 0,  'Y': 0,
    },
    "flexibility": {
        'A':0.36,'C':0.35,'D':0.51,'E':0.50,'F':0.31,'G':0.54,
        'H':0.32,'I':0.46,'K':0.47,'L':0.37,'M':0.30,'N':0.46,
        'P':0.51,'Q':0.49,'R':0.53,'S':0.51,'T':0.44,'V':0.39,
        'W':0.31,'Y':0.42,
    },
    "polarity": {
        'A': 8.1,'C': 5.5,'D':13.0,'E':12.3,'F': 5.2,'G': 9.0,
        'H':10.4,'I': 5.2,'K':11.3,'L': 4.9,'M': 5.7,'N':11.6,
        'P': 8.0,'Q':10.5,'R':10.5,'S': 9.2,'T': 8.6,'V': 5.9,
        'W': 5.4,'Y': 6.2,
    },
    "mass": {
        'A': 89,'C':121,'D':133,'E':147,'F':165,'G': 75,
        'H':155,'I':131,'K':146,'L':131,'M':149,'N':132,
        'P':115,'Q':146,'R':174,'S':105,'T':119,'V':117,
        'W':204,'Y':181,
    },
}

# Chou-Fasman propensity channels — replaces fixed phi/psi priors.
# Values: Pa (helix), Pb (sheet), Pt (turn). Source: Chou & Fasman 1978.
_GHMA_CF_HELIX = {
    'A':1.42,'C':0.70,'D':1.01,'E':1.51,'F':1.13,'G':0.57,
    'H':1.00,'I':1.08,'K':1.16,'L':1.21,'M':1.45,'N':0.67,
    'P':0.57,'Q':1.11,'R':0.98,'S':0.77,'T':0.83,'V':1.06,
    'W':1.08,'Y':0.69,
}
_GHMA_CF_SHEET = {
    'A':0.83,'C':1.19,'D':0.54,'E':0.37,'F':1.38,'G':0.75,
    'H':0.87,'I':1.60,'K':0.74,'L':1.30,'M':1.05,'N':0.89,
    'P':0.55,'Q':1.10,'R':0.93,'S':0.75,'T':1.19,'V':1.70,
    'W':1.37,'Y':1.47,
}
_GHMA_CF_TURN  = {
    'A':0.66,'C':1.19,'D':1.46,'E':0.74,'F':0.60,'G':1.56,
    'H':0.95,'I':0.47,'K':1.01,'L':0.59,'M':0.60,'N':1.56,
    'P':1.52,'Q':0.98,'R':0.95,'S':1.43,'T':0.96,'V':0.50,
    'W':0.96,'Y':1.14,
}


def _ghma_zscore(arr):
    return (arr - arr.mean()) / (arr.std() + 1e-12)


def _ghma_encode(seq):
    """Build z-scored channel dict for a sequence string."""
    ch = {}
    for name, table in _GHMA_PROPS.items():
        arr = _np.array([table.get(a, 0.0) for a in seq])
        ch[name] = _ghma_zscore(arr)
    for name, table in [
        ("cf_helix", _GHMA_CF_HELIX),
        ("cf_sheet", _GHMA_CF_SHEET),
        ("cf_turn",  _GHMA_CF_TURN),
    ]:
        arr = _np.array([table.get(a, 1.0) for a in seq])
        ch[name] = _ghma_zscore(arr)
    return ch


def _ghma_drop_redundant(channels, r_thresh=0.92):
    """Remove channels whose |Pearson r| > r_thresh against a kept channel."""
    keys   = list(channels.keys())
    kept   = [keys[0]]
    dropped = []
    for k in keys[1:]:
        redundant = False
        for kk in kept:
            a, b = channels[k], channels[kk]
            if a.std() < 1e-10 or b.std() < 1e-10:
                redundant = True   # constant channel — carries no info
                break
            r, _ = _pearsonr(a, b)
            if abs(r) > r_thresh:
                redundant = True
                break
        if redundant:
            dropped.append(k)
        else:
            kept.append(k)
    return {k: channels[k] for k in kept}, dropped


def _ghma_embed(channels, n_components=3):
    """SVD PCA on N×C channel matrix. Returns (embedded N×n, var_fracs)."""
    X = _np.vstack([channels[k] for k in channels]).T
    X = X - X.mean(axis=0)
    _, S, Vt = _svd(X, full_matrices=False)
    var_fracs = S**2 / (S**2).sum()
    return X @ Vt.T[:, :n_components], var_fracs[:n_components]


def _ghma_curvature(x, sigma=2.0):
    x   = _gf1d(x, sigma)
    dx  = _np.gradient(x)
    ddx = _np.gradient(dx)
    return _np.abs(ddx) / ((1 + dx**2)**1.5 + 1e-12)


def _ghma_spectrum(x):
    f = _rfftfreq(len(x), d=1)
    p = _np.abs(_rfft(x))
    return f, p


def _ghma_phase_coherence(x, f_lo=0.05, f_hi=0.45):
    """Mean resultant length restricted to [f_lo, f_hi] band."""
    f    = _rfftfreq(len(x), d=1)
    mask = (f >= f_lo) & (f <= f_hi)
    if not mask.any():
        return 0.0
    phases = _np.angle(_np.fft.rfft(x)[mask])
    return float(_np.abs(_np.mean(_np.exp(1j * phases))))


def _ghma_spectral_entropy(p):
    p = p / (p.sum() + 1e-12)
    return float(_entropy(p))


def _ghma_resonance_stability(p):
    norm = p / (p.max() + 1e-12)
    return float(1.0 / (_np.var(norm) + 1e-12))


def _ghma_higuchi_fd(x, k_max=20):
    """Higuchi (1988) fractal dimension with correct normalisation."""
    N     = len(x)
    k_max = min(k_max, N // 2)
    ks    = _np.arange(2, k_max + 1)
    Lk    = []
    for k in ks:
        Lm_list = []
        for m in range(1, k + 1):
            idx   = _np.arange(m - 1, N, k)
            n_seg = len(idx) - 1
            if n_seg < 1:
                continue
            raw  = _np.sum(_np.abs(_np.diff(x[idx])))
            norm = raw * (N - 1) / (n_seg * k * k)
            Lm_list.append(norm)
        Lk.append(_np.mean(Lm_list) if Lm_list else _np.nan)
    Lk = _np.array(Lk)
    ok = _np.isfinite(Lk) & (Lk > 0)
    if ok.sum() < 3:
        return float('nan')
    slope, _ = _np.polyfit(_np.log(1.0 / ks[ok]), _np.log(Lk[ok]), 1)
    return float(slope)


def _ghma_anisotropy(X):
    cov     = _np.cov(X.T)
    _, S, _ = _svd(cov)
    return float(S[0] / (_np.mean(S[1:]) + 1e-12))


def _ghma_discrete_torsion(X3, smoothing=2.0):
    """
    Frenet-Serret torsion of the 3-D manifold curve.

        τ = [(r′ × r″) · r‴] / ‖r′ × r″‖²

    Computed via successive np.gradient calls on the N×3 position matrix.

    Stability notes
    ---------------
    * When ‖r′ × r″‖² ≈ 0 the curve is locally straight or planar and
      torsion is geometrically undefined.  We mask those points with NaN
      rather than trusting the 1e-12 floor, which would produce enormous
      but finite spikes that distort the smoothed field.
    * Smoothing via Gaussian diffusion is applied before returning so
      isolated instabilities don't propagate into co-peak detection.
    """
    r   = X3                                      # N × 3
    dr  = _np.gradient(r,  axis=0)
    d2r = _np.gradient(dr,  axis=0)
    d3r = _np.gradient(d2r, axis=0)

    cross      = _np.cross(dr, d2r)               # N × 3
    numerator  = _np.sum(cross * d3r, axis=1)     # N
    denom      = _np.sum(cross**2,    axis=1)     # N

    # Mask near-degenerate points (locally straight / planar curve)
    valid      = denom > 1e-6
    torsion    = _np.full(len(r), _np.nan)
    torsion[valid] = numerator[valid] / denom[valid]

    # Interpolate NaNs with linear fill before smoothing so Gaussian
    # filter doesn't spread NaN contamination
    if not valid.all():
        idx    = _np.arange(len(torsion))
        finite = _np.isfinite(torsion)
        if finite.sum() > 2:
            torsion = _np.interp(idx, idx[finite], torsion[finite])
        else:
            return _np.zeros(len(r))              # degenerate — return flat

    return _gf1d(torsion, smoothing)


def _ghma_torsion_pde(torsion, iterations=8, alpha=0.08):
    """
    Heat-equation PDE relaxation on the torsion field:
        ∂τ/∂t = α ∇²τ

    Uses a proper centred finite-difference Laplacian
        ∇²τ[i] = τ[i-1] - 2τ[i] + τ[i+1]
    with zero-flux (Neumann) boundary conditions — avoids the boundary
    artefacts that accumulate with two successive np.gradient calls.
    """
    tau = torsion.copy()
    for _ in range(iterations):
        lap        = _np.empty_like(tau)
        lap[1:-1]  = tau[:-2] - 2*tau[1:-1] + tau[2:]   # interior
        lap[0]     = tau[1]   - tau[0]                    # Neumann BC
        lap[-1]    = tau[-2]  - tau[-1]                   # Neumann BC
        tau       += alpha * lap
    return tau


def _ghma_manifold_torsion(channels):
    """
    Build the full 3-D embedding, compute discrete torsion, apply PDE
    relaxation, and return a z-scored 1-D torsion signal.

    Returns None if the embedding is degenerate (< 3 components, or
    fewer than 15 residues after cleaning).
    """
    X, var_fracs = _ghma_embed(channels, n_components=3)
    if X.shape[1] < 3 or X.shape[0] < 15:
        return None

    raw     = _ghma_discrete_torsion(X)
    relaxed = _ghma_torsion_pde(raw)

    std = relaxed.std()
    if std < 1e-10:
        return _np.zeros(len(relaxed))
    return (relaxed - relaxed.mean()) / std


def _ghma_find_copeaks(positions, local, min_metrics=3, percentile=75.0):
    flags = _np.zeros(len(positions), dtype=int)
    for arr in local.values():
        ok  = _np.isfinite(arr)
        thr = _np.percentile(arr[ok], percentile) if ok.any() else _np.inf
        flags += (arr >= thr).astype(int)
    return positions[flags >= min_metrics]


def _ghma_sliding(x, n, channels_full, win=35, step=8):
    half = win // 2
    centres, local = [], {m: [] for m in
                          ("entropy", "coherence", "curvature", "fractal", "anisotropy", "torsion")}
    for c in range(half, n - half, step):
        s, e = max(0, c - half), min(n, c + half)
        if (e - s) < 15:
            continue
        ch_w      = {k: channels_full[k][s:e] for k in channels_full}
        X_w, _    = _ghma_embed(ch_w)
        xw        = X_w[:, 0]
        _, p      = _ghma_spectrum(xw)
        local["entropy"].append(_ghma_spectral_entropy(p))
        local["coherence"].append(_ghma_phase_coherence(xw))
        local["curvature"].append(float(_np.mean(_ghma_curvature(xw))))
        local["fractal"].append(_ghma_higuchi_fd(xw))
        local["anisotropy"].append(_ghma_anisotropy(X_w))
        # torsion: use 3-D window embedding; fall back to NaN if degenerate
        tau_w = _ghma_manifold_torsion(ch_w)
        local["torsion"].append(float(_np.mean(_np.abs(tau_w)))
                                if tau_w is not None else _np.nan)
        centres.append(c)
    pos = _np.array(centres)
    return pos, {k: _np.array(v) for k, v in local.items()}


def stage_ghma(seq: str, win: int = 35, step: int = 8,
               r_thresh: float = 0.92) -> dict:
    """
    Stage 10 — GHMA analysis.

    Returns a dict suitable for JSON serialisation and terminal/HTML
    reporting.  No matplotlib calls — plotting is handled separately
    via ghma_plot() for standalone / interactive use.
    """
    if not HAS_GHMA:
        return {"error": "numpy/scipy not available"}
    if len(seq) < 10:
        return {"error": "Sequence too short for GHMA (< 10 aa)"}

    channels          = _ghma_encode(seq)
    channels, dropped = _ghma_drop_redundant(channels, r_thresh)
    X, var_fracs      = _ghma_embed(channels)
    x                 = X[:, 0]
    f, p              = _ghma_spectrum(x)

    # Global torsion field on the full 3-D embedding
    tau_global = _ghma_manifold_torsion(channels)
    if tau_global is not None:
        tau_abs    = _np.abs(tau_global)
        # Exclude first/last 5 residues — np.gradient boundary errors
        # compound through three successive derivative estimates and
        # produce artefactually large values at sequence termini.
        boundary   = max(5, len(tau_global) // 20)
        tau_masked = tau_abs.copy()
        tau_masked[:boundary]  = 0.0
        tau_masked[-boundary:] = 0.0
        tau_mean   = round(float(_np.mean(tau_abs)), 4)
        tau_max    = round(float(_np.max(tau_masked)), 4)
        tau_argmax = int(_np.argmax(tau_masked))
    else:
        tau_mean = tau_max = tau_argmax = None

    positions, local  = _ghma_sliding(x, len(seq), channels, win, step)
    copeaks           = _ghma_find_copeaks(positions, local)

    # Serialise local profiles as plain lists
    local_ser = {k: [None if _np.isnan(v) else round(float(v), 6)
                     for v in arr]
                 for k, arr in local.items()}

    return {
        "channels_retained"  : len(channels),
        "channels_dropped"   : dropped,
        "pc1_var_fraction"   : round(float(var_fracs[0]), 4),
        "pc2_var_fraction"   : round(float(var_fracs[1]), 4),
        "pc1_low_variance"   : bool(var_fracs[0] < 0.25),
        "phase_coherence"    : round(_ghma_phase_coherence(x), 4),
        "spectral_entropy"   : round(_ghma_spectral_entropy(p), 4),
        "resonance_stability": round(_ghma_resonance_stability(p), 4),
        "higuchi_fd"         : round(_ghma_higuchi_fd(x), 4),
        "anisotropy"         : round(_ghma_anisotropy(X), 4),
        "dominant_period"    : round(float(1.0 / (f[_np.argmax(p)] + 1e-12)), 4),
        "torsion_mean_abs"   : tau_mean,
        "torsion_max_abs"    : tau_max,
        "torsion_peak_pos"   : tau_argmax,
        "copeak_positions"   : copeaks.tolist(),
        "window_positions"   : positions.tolist(),
        "local"              : local_ser,
    }


def ghma_plot(name: str, seq: str, ghma_result: dict = None,
              win: int = 35, step: int = 8):
    """
    Standalone GHMA plot (matplotlib).  Call this interactively; it is
    never called from pipeline workers.  If ghma_result is pre-computed
    (e.g. from r.ghma) it is reused; otherwise stage_ghma() is called.
    """
    import matplotlib.pyplot as plt

    if ghma_result is None:
        ghma_result = stage_ghma(seq, win=win, step=step)
    if "error" in ghma_result:
        print(f"GHMA plot skipped: {ghma_result['error']}")
        return

    channels          = _ghma_encode(seq)
    channels, _       = _ghma_drop_redundant(channels)
    X, var_fracs      = _ghma_embed(channels)
    x, y              = X[:, 0], X[:, 1]
    f, p              = _ghma_spectrum(x)

    pos      = _np.array(ghma_result["window_positions"])
    local    = {k: _np.array(v, dtype=float)
                for k, v in ghma_result["local"].items()}
    copeaks  = _np.array(ghma_result["copeak_positions"])

    fig, axs = plt.subplots(4, 2, figsize=(14, 15))
    fig.suptitle(
        f"{name}  |  PC1 var={ghma_result['pc1_var_fraction']:.2f}"
        f"  PC2 var={ghma_result['pc2_var_fraction']:.2f}",
        fontsize=12,
    )

    axs[0, 0].plot(x, lw=0.9, color="#2c7bb6")
    for cp in copeaks:
        axs[0, 0].axvline(cp, color="tomato", lw=0.7, alpha=0.6)
    axs[0, 0].set_title("PC1 Trajectory  (red = co-peak)")
    axs[0, 0].set_xlabel("Residue index")

    axs[0, 1].plot(f, p, lw=0.9, color="#1a9641")
    axs[0, 1].set_title("Global Power Spectrum")
    axs[0, 1].set_xlabel("Frequency (residues⁻¹)")

    axs[1, 0].plot(x, y, lw=0.5, alpha=0.7, color="#7b2d8b")
    axs[1, 0].set_title("Manifold  PC1 vs PC2")
    axs[1, 0].set_xlabel("PC1"); axs[1, 0].set_ylabel("PC2")

    axs[1, 1].plot(pos, local["entropy"], color="#d7191c")
    axs[1, 1].set_title("Local Spectral Entropy")
    axs[1, 1].set_xlabel("Residue position")

    axs[2, 0].plot(pos, local["coherence"], color="#fdae61")
    axs[2, 0].set_title("Local Phase Coherence  (band-limited)")
    axs[2, 0].set_xlabel("Residue position")

    fd = local["fractal"]
    ok = _np.isfinite(fd)
    axs[2, 1].plot(pos[ok], fd[ok], color="#404040")
    axs[2, 1].set_title("Local Higuchi Fractal Dimension")
    axs[2, 1].set_xlabel("Residue position")

    # Global torsion field
    tau_global = _ghma_manifold_torsion(channels)
    if tau_global is not None:
        axs[3, 0].plot(tau_global, lw=0.8, color="#7b2d8b")
        if ghma_result.get("torsion_peak_pos") is not None:
            axs[3, 0].axvline(ghma_result["torsion_peak_pos"],
                              color="tomato", lw=0.8, alpha=0.7, ls="--")
        axs[3, 0].set_title("Global Manifold Torsion τ  (PDE-relaxed)")
        axs[3, 0].set_xlabel("Residue index")
        axs[3, 0].axhline(0, color="#aaa", lw=0.5)
    else:
        axs[3, 0].text(0.5, 0.5, "Torsion: degenerate embedding",
                       ha="center", va="center", transform=axs[3, 0].transAxes)

    # Local mean |torsion|
    tau_local = local.get("torsion", _np.array([]))
    ok_t = _np.isfinite(tau_local)
    if ok_t.any():
        axs[3, 1].plot(pos[ok_t], tau_local[ok_t], color="#2c7bb6")
        axs[3, 1].set_title("Local Mean |τ|  (window torsion magnitude)")
        axs[3, 1].set_xlabel("Residue position")
    else:
        axs[3, 1].text(0.5, 0.5, "No local torsion data",
                       ha="center", va="center", transform=axs[3, 1].transAxes)

    plt.tight_layout()
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

import time
import csv
from multiprocessing import Pool, cpu_count

# ══════════════════════════════════════════════════════════════════════════════
# Stage 9 — Natural Cleavage Site Detection
# Catalogue of 81 biologically validated cleavage sites across 20 protease
# systems. Integrates with SP position (stage 6) and motif hits (stage 7).
# ══════════════════════════════════════════════════════════════════════════════

# Format: name → (pattern, cut_offset, description, system, specificity, reference)
_CLEAVAGE_SITES: dict = {
    # ── Proprotein convertases ─────────────────────────────────────────────
    "Furin_RXXR":       (r"R[^P]{2}R(?=[^P])",   4, "Furin: R-x-x-R↓",               "Furin/PCSK",    "HIGH"),
    "Furin_RRKR":       (r"RR[KR]R(?=[^P])",      4, "Furin polybasic: R-R-K/R-R↓",   "Furin/PCSK",    "HIGH"),
    "Furin_RXKR":       (r"R[RKAS]KR(?=[^P])",    4, "Furin high-affinity: R-x-K-R↓", "Furin/PCSK",    "HIGH"),
    "PCSK_dibasic":     (r"[KR]{2}(?=[^P])",       2, "PCSK dibasic: K/R-K/R↓",        "Furin/PCSK",    "MODERATE"),
    "PC2_RR":           (r"RR(?=[^P])",            2, "PC2: R-R↓ (glucagon, POMC)",     "Furin/PCSK",    "MODERATE"),
    "PCSK6_PACE4":      (r"R[^P]R[RK](?=[^P])",   4, "PCSK6/PACE4: R-x-R-R/K↓",      "Furin/PCSK",    "MODERATE"),
    "Proinsulin_dibasic":(r"[KR]{2}(?=G)",         2, "Proinsulin/proglucagon: K/R-K/R↓G", "Furin/PCSK","HIGH"),
    "PCSK9_autocatalytic":(r"VFAQ(?=S)",           4, "PCSK9 autocatalytic: VFAQ↓S",   "Furin/PCSK",    "HIGH"),
    # ── Signal peptidase ──────────────────────────────────────────────────
    "Signal_pep_I":     (r"[LIVMF]{3,}[^DE]{2,6}[AGSCT][^DEKRP][AGSCT](?=[^DE])", 8,
                         "Signal peptidase I: AXA↓ (use SP from stage 6)",           "Signal peptidase","LOW"),
    "Signal_pep_II":    (r"[LIVMF]{3,}[ACGST]{0,2}C(?=[ADEST])", -1,
                         "Signal peptidase II: h(3+)-C↓ (lipoprotein)",              "Signal peptidase","MODERATE"),
    # ── Caspases ──────────────────────────────────────────────────────────
    "Caspase_1":        (r"[WYLF][EH]HD(?=[AG])",  4, "Caspase-1: W/Y/L/F-E/H-H-D↓A/G", "Caspase",    "HIGH"),
    "Caspase_2":        (r"[DN]EVD(?=[AGST])",      4, "Caspase-2: D/N-E-V-D↓",           "Caspase",    "HIGH"),
    "Caspase_3_7":      (r"D[EQ]VD(?=[AGST])",      4, "Caspase-3/7: D-E/Q-V-D↓ (DEVD)", "Caspase",    "HIGH"),
    "Caspase_4_5":      (r"[LW]EHD(?=[GA])",        4, "Caspase-4/5: L/W-E-H-D↓",         "Caspase",    "HIGH"),
    "Caspase_6":        (r"VE[HI]D(?=[AGST])",      4, "Caspase-6: V-E-H/I-D↓",           "Caspase",    "HIGH"),
    "Caspase_8_10":     (r"[ILV][EQ]TD(?=[SGNA])", 4, "Caspase-8/10: I/L/V-E/Q-T-D↓",   "Caspase",    "HIGH"),
    "Caspase_9":        (r"LEHD(?=[FA])",            4, "Caspase-9: L-E-H-D↓F/A",          "Caspase",    "HIGH"),
    "Caspase_14":       (r"WEVE(?=[HQ])",            4, "Caspase-14: W-E-V-E↓",            "Caspase",    "HIGH"),
    # ── Granzymes ─────────────────────────────────────────────────────────
    "Granzyme_B":       (r"[IVL]EPD(?=[SGANT])",   4, "Granzyme B: I/V/L-E-P-D↓",        "Granzyme",   "HIGH"),
    "Granzyme_A":       (r"[KR](?=[^P])",           1, "Granzyme A: K/R↓",                "Granzyme",   "LOW"),
    "Granzyme_H":       (r"[FYW](?=[^P])",          1, "Granzyme H: F/Y/W↓",              "Granzyme",   "LOW"),
    "Granzyme_K":       (r"[KR](?=[^P])",           1, "Granzyme K: K/R↓",                "Granzyme",   "LOW"),
    "Granzyme_M":       (r"[ML](?=[^P])",           1, "Granzyme M: M/L↓",                "Granzyme",   "LOW"),
    # ── Matrix metalloproteases ───────────────────────────────────────────
    "MMP_1_3":          (r"GPQ[GA](?=[LIVMA])",     4, "MMP-1/3: G-P-Q-G/A↓h",           "MMP",        "HIGH"),
    "MMP_2_9":          (r"G[PA][LI]G(?=[LIVMA])",  4, "MMP-2/9: G-P/A-L/I-G↓h",         "MMP",        "HIGH"),
    "MMP_7":            (r"RPLAL(?=[WAL])",          5, "MMP-7: R-P-L-A-L↓",              "MMP",        "HIGH"),
    "MMP_generic":      (r"P[^P]{2}[LIVMA](?=[LIVMAFGVS])", 4, "MMP consensus: P-x-x-h↓h'",  "MMP",    "MODERATE"),
    "MMP_14_MT1":       (r"[RK]PL[LIVMA](?=[LIVMA])",5, "MT1-MMP: K/R-P-L-h↓h",          "MMP",        "MODERATE"),
    # ── ADAM / ADAMTS ─────────────────────────────────────────────────────
    "ADAM_17_TACE":     (r"[LIVMA]{2}[AV](?=[QEST])",3, "ADAM-17/TACE: h-h-A/V↓Q/E/S/T", "ADAM",       "MODERATE"),
    "ADAM_10":          (r"H[LIVMA]{2}SH",           3, "ADAM-10: H-h-h-S-H zinc site",   "ADAM",        "LOW"),
    "ADAMTS_4_5":       (r"EGE(?=[AG])",             3, "ADAMTS-4/5: E-G-E↓A/G (aggrecan)","ADAM",      "HIGH"),
    # ── Coagulation ───────────────────────────────────────────────────────
    "Thrombin":         (r"[LI]?PR(?=[^P])",         3, "Thrombin: (L/I-)P-R↓",           "Coagulation","HIGH"),
    "Factor_Xa":        (r"[LIVMA]{3}[RK](?=[^P])", 4, "Factor Xa: h-h-h-R/K↓",          "Coagulation","MODERATE"),
    "Factor_VIIa":      (r"RKVG(?=[LIVMA])",         4, "Factor VIIa: R-K-V-G↓h",         "Coagulation","HIGH"),
    "Plasmin":          (r"[KR](?=[^P])",            1, "Plasmin: K/R↓ (fibrinolysis)",    "Coagulation","LOW"),
    "Kallikrein_1":     (r"R(?=MK|SS)",             1, "Kallikrein-1: R↓MK/SS",           "Coagulation","HIGH"),
    # ── Complement ────────────────────────────────────────────────────────
    "C1s":              (r"QAR(?=[LIVSF])",          3, "C1s: Q-A-R↓h (classical pathway)", "Complement","HIGH"),
    "C3_convertase":    (r"[KR]{2}[ST](?=[AG])",     3, "C3 convertase: K/R-K/R-S/T↓A/G", "Complement","MODERATE"),
    "Factor_I":         (r"R[AS](?=R[^P])",          2, "Factor I: R-A/S↓R (C3b/C4b)",    "Complement","MODERATE"),
    # ── Cathepsins ────────────────────────────────────────────────────────
    "Cathepsin_B":      (r"[LIVMF]R(?=[^P])",       2, "Cathepsin B: h-R↓ (pH 5)",        "Cathepsin",  "MODERATE"),
    "Cathepsin_D":      (r"[LIVMF][^P][LIVMF](?=[LIVMF])",3,"Cathepsin D: h-x-h↓h (pH 4)","Cathepsin",  "MODERATE"),
    "Cathepsin_K":      (r"[KR][^P]{2}[LIVMF](?=[AGST])",4,"Cathepsin K: K/R-x-x-h↓ (bone)","Cathepsin","MODERATE"),
    "Cathepsin_L":      (r"[LIVMF][^P][RK](?=[^P])", 3, "Cathepsin L: h-x-K/R↓ (lysosome)","Cathepsin","LOW"),
    "Cathepsin_G":      (r"[YFLM](?=[^P])",          1, "Cathepsin G: F/Y/L/M↓ (neutrophil)","Cathepsin","LOW"),
    "Cathepsin_S":      (r"[LIVMF][^P][LIVMF][ST](?=[^P])",4,"Cathepsin S: h-x-h-S/T↓ (APC)","Cathepsin","MODERATE"),
    # ── Hormone processing ────────────────────────────────────────────────
    "Proinsulin_B_C":   (r"[KR]{2}(?=G)",           2, "Proinsulin B-C: K/R-K/R↓G",      "Hormone",    "HIGH"),
    "POMC_PC1_KK":      (r"KK(?=[^P])",             2, "POMC PC1: K-K↓ (ACTH, β-MSH)",   "Hormone",    "HIGH"),
    "POMC_PC2_KR":      (r"[KR]{2}(?=Y)",           2, "POMC PC2: K/R-K/R↓Y (β-endorphin)","Hormone",  "HIGH"),
    "Neuropeptide_RR":  (r"RR(?=[^P])",             2, "Neuropeptide R-R↓ (oxytocin, AVP)","Hormone",   "MODERATE"),
    # ── Ubiquitin / UBL ───────────────────────────────────────────────────
    "Ubiquitin_GG":     (r"GG(?=[^G])",             2, "Ubiquitin GG↓ (UCH-L1/L3)",       "Ubiquitin",  "MODERATE"),
    "SENP_SUMO":        (r"QG(?=[^GP])",            2, "SENP SUMO: Q-G↓",                 "Ubiquitin",  "HIGH"),
    "ATG4_LC3":         (r"G(?=[^GPA])",            1, "ATG4: -G↓ (LC3/ATG8 autophagy)",  "Ubiquitin",  "LOW"),
    # ── Viral proteases ───────────────────────────────────────────────────
    "SARS_CoV2_3CL":    (r"[LI]Q(?=[AGS])",         2, "SARS-CoV-2 3CLpro: L/I-Q↓A/G/S", "Viral",      "HIGH"),
    "SARS_CoV2_PLpro":  (r"[LI].GG(?=[^P])",        4, "SARS-CoV-2 PLpro: L/I-x-G-G↓",  "Viral",      "HIGH"),
    "HCV_NS3":          (r"[EDQST](?=C[SAGT])",     1, "HCV NS3/4A: D/E/Q/S/T↓C",        "Viral",      "HIGH"),
    "HIV1_protease":    (r"[LIVMA][^P][^P][LIVMA](?=[LIVMA])",2,"HIV-1 PR: h-x-x-h↓h",   "Viral",      "MODERATE"),
    "EV71_2Apro":       (r"[LIVMA]P(?=[GS])",       2, "Enterovirus 2A: h-P↓G/S",         "Viral",      "MODERATE"),
    "DENV_NS3pro":      (r"[KR]{2}[AGST](?=[LIVMF])",3,"Dengue NS3: K/R-K/R-A/G/S/T↓h", "Viral",      "HIGH"),
    # ── Neuropeptide processing ───────────────────────────────────────────
    "Neprilysin_NEP":   (r"[^DE][LIVMFYW](?=[LIVMFYW])",1,"Neprilysin: x-h↓h (enkephalin, ANP)","Neuropeptide","MODERATE"),
    "DPP4_XP":          (r"^[^P]P",                 2, "DPP-IV: X-Pro↓ N-term (GLP-1, GIP)", "Neuropeptide","HIGH"),
    "DPP4_XA":          (r"^[^A]A",                 2, "DPP-IV: X-Ala↓ N-term (BNP)",      "Neuropeptide","HIGH"),
    "Carboxypeptidase_E":(r"[KR]$",                 0, "CPE: C-terminal K/R removal (post-PC)", "Neuropeptide","HIGH"),
    # ── Kallikreins ───────────────────────────────────────────────────────
    "KLK_trypsin":      (r"[KR](?=[^P])",           1, "KLK (trypsin-like): K/R↓",        "Kallikrein", "LOW"),
    "KLK3_PSA":         (r"[KR]SL(?=[^P])",         1, "KLK3/PSA: K/R-S-L↓ (semenogelin)","Kallikrein", "HIGH"),
    "KLK5_KLK7":        (r"[KR](?=[QLA])",          1, "KLK5/7 skin: K/R↓Q/L/A",          "Kallikrein", "MODERATE"),
    # ── Mast cell ─────────────────────────────────────────────────────────
    "Tryptase":         (r"[KR](?=[LIV])",          1, "Mast cell tryptase: K/R↓L/I/V",   "Mast cell",  "MODERATE"),
    "Chymase":          (r"[FYW](?=[^P])",          1, "Mast cell chymase: F/Y/W↓",        "Mast cell",  "LOW"),
    # ── Neutrophil ────────────────────────────────────────────────────────
    "Neutrophil_Elastase":(r"[AV](?=[^P])",         1, "Neutrophil elastase: A/V↓",        "Neutrophil", "LOW"),
    "Proteinase_3":     (r"[AVSIL][^P](?=[^P])",   2, "Proteinase 3: A/V/S/I/L-x↓",      "Neutrophil", "LOW"),
    # ── Renin-angiotensin ─────────────────────────────────────────────────
    "Renin":            (r"[LIVMA]H[LIVMA](?=[^P])",3, "Renin: L/I/V/M/A-H-h↓",          "Renin-Angiotensin","HIGH"),
    "ACE_C_term":       (r"[^P]P$",                 0, "ACE: C-terminal -xP removal",      "Renin-Angiotensin","MODERATE"),
    "ACE2_C_term":      (r"[^P][LIVMF]$",           0, "ACE2: C-terminal hydrophobic",     "Renin-Angiotensin","MODERATE"),
    # ── Meprin ────────────────────────────────────────────────────────────
    "Meprin_alpha":     (r"E[^P]{2}[DE](?=[^P])",  4, "Meprin-α: E-x-x-D/E↓ (IL-18)",   "Meprin",     "MODERATE"),
    "Meprin_beta":      (r"[^DE][LIVMF][KR](?=[^P])",3,"Meprin-β: x-h-K/R↓ (TNF-α)",    "Meprin",     "LOW"),
    # ── Intramembrane ─────────────────────────────────────────────────────
    "BACE1_beta":       (r"KM(?=[DA])",             2, "BACE1: K-M↓D/A (APP β-site)",     "Intramembrane","HIGH"),
    "Gamma_secretase":  (r"[LIVMF]{14,}",          -1, "γ-secretase: within TM (APP, Notch)","Intramembrane","LOW"),
    # ── Propeptide / BMP ──────────────────────────────────────────────────
    "BMP1_tolloid":     (r"[DE]D(?=[^P])",          2, "BMP-1/Tolloid: D-D↓ (pro-collagen)","Propeptide","MODERATE"),
    "Subtilisin_PCSK":  (r"[LIVMA]{2}[KR](?=[^P])",3, "Subtilisin/PCSK: h-h-K/R↓",       "Propeptide", "LOW"),
}

_CLEAVAGE_SPECIFICITY = {"HIGH": 0, "MODERATE": 1, "LOW": 2}

# Motif hits → cleavage site names (for evidence cross-referencing)
_MOTIF_CLEAVAGE_EVIDENCE: dict = {
    "Furin_cleavage":    ["Furin_RXXR", "Furin_RRKR", "Furin_RXKR"],
    "Zoonotic_furin":    ["Furin_RRKR"],
    "Caspase_3_6":       ["Caspase_3_7", "Caspase_6"],
    "Caspase_1":         ["Caspase_1"],
    "Caspase_8_9":       ["Caspase_8_10", "Caspase_9"],
    "Granzyme_B":        ["Granzyme_B"],
    "MMP_cleavage":      ["MMP_generic", "MMP_1_3", "MMP_2_9"],
    "ADAM_sheddase":     ["ADAM_17_TACE", "ADAM_10"],
    "Thrombin_PAR":      ["Thrombin"],
    "Ubiquitin_GG":      ["Ubiquitin_GG", "ATG4_LC3"],
    "SUMO_consensus":    ["SENP_SUMO"],
    "POMC_PC1":          ["POMC_PC1_KK"],
    "Proinsulin_C_A":    ["Proinsulin_B_C", "Proinsulin_dibasic"],
}

_COMPILED_CLEAVAGE = {k: re.compile(v[0]) for k, v in _CLEAVAGE_SITES.items() if v[0]}


def stage_natural_cleavage(
    seq: str,
    sp_cleavage_pos: int = 0,
    motif_hits: dict | None = None,
    n_tm_helices: int = 0,
    min_specificity: str = "MODERATE",
    allowed_systems: set | None = None,
    min_fragment: int = 8,
) -> dict:
    """
    Stage 9 — Natural cleavage site detection and daughter fragment mass calculation.

    Parameters
    ----------
    seq              : cleaned amino acid sequence (mature or full preprotein)
    sp_cleavage_pos  : signal peptide cleavage position from stage 6 (0 = none)
    motif_hits       : motifs dict from stage 7 (for evidence cross-referencing)
    n_tm_helices     : from stage 4 (suppresses intramembrane proteases when 0)
    min_specificity  : "HIGH", "MODERATE", or "LOW" (default: MODERATE)
    allowed_systems  : restrict to named systems (None = all)
    min_fragment     : minimum daughter fragment length in aa (default: 8)

    Returns
    -------
    dict with:
      sites          — list of cleavage site dicts (sorted by cut position)
      fragments      — list of daughter fragment dicts (sorted by mass)
      n_sites        — total site count
      n_fragments    — fragment count
      n_confirmed    — fragments with HIGH-specificity + motif-evidence on both ends
      systems_found  — set of protease systems detected
    """
    min_rank = _CLEAVAGE_SPECIFICITY[min_specificity]
    n = len(seq)

    # Build evidence set from motif hits
    sites_with_evidence: set = set()
    if motif_hits:
        for motif_name, site_names in _MOTIF_CLEAVAGE_EVIDENCE.items():
            if motif_hits.get(motif_name, {}).get("count", 0) > 0:
                sites_with_evidence.update(site_names)

    # ── Scan for cleavage sites ───────────────────────────────────────────
    sites: list[dict] = []
    for name, (pat, offset, desc, system, spec, *_) in _CLEAVAGE_SITES.items():
        if allowed_systems and system not in allowed_systems:
            continue
        if _CLEAVAGE_SPECIFICITY[spec] > min_rank:
            continue
        if system == "Intramembrane" and n_tm_helices == 0:
            continue
        compiled = _COMPILED_CLEAVAGE.get(name)
        if compiled is None:
            continue
        for m in compiled.finditer(seq):
            raw_cut = m.start() + min(abs(offset), len(m.group()))
            cut_pos = max(1, min(raw_cut, n))
            has_ev  = name in sites_with_evidence
            conf    = ("confirmed" if (spec == "HIGH" and has_ev)
                       else "supported" if has_ev else "predicted")
            sites.append({
                "site_name":      name,
                "system":         system,
                "specificity":    spec,
                "description":    desc,
                "match_seq":      m.group(),
                "match_start":    m.start() + 1,
                "match_end":      m.end(),
                "cut_pos":        cut_pos,
                "motif_evidence": has_ev,
                "confidence":     conf,
                "context":        seq[max(0, m.start()-5): m.end()+5],
            })

    # Add signal peptide site (always HIGH confidence when from stage 6)
    if sp_cleavage_pos > 0:
        sites.append({
            "site_name":      "Signal_peptidase_I",
            "system":         "Signal peptidase",
            "specificity":    "HIGH",
            "description":    f"Signal peptidase I at pos {sp_cleavage_pos} (stage 6)",
            "match_seq":      seq[max(0, sp_cleavage_pos-4): min(n, sp_cleavage_pos+2)],
            "match_start":    max(1, sp_cleavage_pos - 3),
            "match_end":      min(n, sp_cleavage_pos + 1),
            "cut_pos":        sp_cleavage_pos,
            "motif_evidence": True,
            "confidence":     "confirmed",
            "context":        seq[max(0, sp_cleavage_pos-6): sp_cleavage_pos+6],
        })

    sites.sort(key=lambda x: (x["cut_pos"], x["site_name"]))

    # ── Compute daughter fragments ────────────────────────────────────────
    from collections import defaultdict
    cut_positions = sorted(set([0] + [s["cut_pos"] for s in sites] + [n]))
    site_at: dict = defaultdict(list)
    for s in sites:
        site_at[s["cut_pos"]].append(s)

    rank_map = {"confirmed": 2, "supported": 1, "predicted": 0}

    fragments: list[dict] = []
    for i in range(len(cut_positions) - 1):
        s_idx, e_idx = cut_positions[i], cut_positions[i+1]
        frag = seq[s_idx:e_idx]
        flen = len(frag)
        if flen < min_fragment:
            continue

        mono = round(sum(RESIDUE_MASS.get(aa, 0) for aa in frag) + WATER, 4)
        avg  = round(sum(AA_AVG_MASS.get(aa, 0) for aa in frag) + 18.015, 2)
        maldi_m = avg if avg > 3000 else mono

        n_sites_here = site_at.get(s_idx, [])
        c_sites_here = site_at.get(e_idx, [])

        nc = max((s["confidence"] for s in n_sites_here),
                 key=lambda x: rank_map.get(x, 0), default="n-term") if n_sites_here else "n-term"
        cc = max((s["confidence"] for s in c_sites_here),
                 key=lambda x: rank_map.get(x, 0), default="c-term") if c_sites_here else "c-term"
        best = ("confirmed" if "confirmed" in (nc, cc)
                else "supported" if "supported" in (nc, cc) else "predicted")

        H = 1.007276; Na = 22.989218; K = 38.963158
        has_dib = any(s["match_seq"][-2:] in ("KK","KR","RK","RR") for s in n_sites_here + c_sites_here)

        fragments.append({
            "start":       s_idx + 1,
            "end":         e_idx,
            "length":      flen,
            "seq":         frag,
            "mass_mono":   mono,
            "mass_avg":    avg,
            "mz1":         round(mono + H, 4),
            "mz2":         round((mono + 2*H)/2, 4),
            "mz3":         round((mono + 3*H)/3, 4),
            "maldi_mh":    round(maldi_m + H,  2),
            "maldi_mna":   round(maldi_m + Na, 2),
            "maldi_mk":    round(maldi_m + K,  2),
            "n_cys":       frag.count("C"),
            "n_basic":     sum(1 for aa in frag if aa in "KRH"),
            "n_cut_sites": [s["site_name"] for s in n_sites_here],
            "c_cut_sites": [s["site_name"] for s in c_sites_here],
            "n_systems":   sorted(set(s["system"] for s in n_sites_here)) or ["N-terminus"],
            "c_systems":   sorted(set(s["system"] for s in c_sites_here)) or ["C-terminus"],
            "confidence":  best,
            "cpe_note":    "Pre-CPE: dibasic C-terminus trimmed by carboxypeptidase E in vivo" if has_dib else "",
        })

    fragments.sort(key=lambda x: x["mass_mono"])

    confirmed = [f for f in fragments if f["confidence"] == "confirmed"]
    systems_found = sorted(set(s["system"] for s in sites))

    return {
        "sites":         sites,
        "fragments":     fragments,
        "n_sites":       len(sites),
        "n_fragments":   len(fragments),
        "n_confirmed":   len(confirmed),
        "systems_found": systems_found,
        "min_specificity": min_specificity,
        "sp_pos_used":   sp_cleavage_pos,
    }


ALL_STAGES = [1,2,3,4,5,6,7,8,9,10]


def _process_one(args: tuple) -> PipelineResult:
    """Worker function — must be top-level for multiprocessing pickle."""
    acc, desc, raw_seq, stages, model_path, context_len = args
    seq  = clean_seq(raw_seq)
    warn = []
    if len(seq) != len(raw_seq):
        warn.append(f"Stripped {len(raw_seq)-len(seq)} non-standard residues")
    if len(seq) < 20:
        warn.append("Sequence shorter than 20 aa — results may be unreliable")

    r = PipelineResult(accession=acc, description=desc,
                       sequence=seq, length=len(seq), warnings=warn)

    if 2 in stages:
        r.physicochemical = stage_physicochemical(seq)
        r.composition     = stage_composition(seq)
        r.stages_run.append(2)
    if 3 in stages:
        r.secondary_structure = stage_secondary_structure(seq)
        r.stages_run.append(3)
    if 4 in stages:
        r.hydropathy = stage_hydropathy(seq)
        r.stages_run.append(4)
    if 5 in stages:
        r.complexity = stage_complexity(seq)
        r.disorder   = stage_disorder(seq)
        r.stages_run.append(5)
    if 6 in stages:
        r.signal_peptide = stage_signal_peptide(seq)
        r.stages_run.append(6)
    if 2 in stages:
        sp_pos = (r.signal_peptide.get("predicted_cleavage_pos", 0)
                  if r.signal_peptide and r.signal_peptide.get("detected") else 0)
        motif_hits = r.motifs.get("motifs", {}) if r.motifs else None
        r.mass  = stage_mass(seq, sp_cleavage_pos=sp_pos, motif_hits=motif_hits)
        r.maldi = stage_maldi(seq, sp_cleavage_pos=sp_pos)
    if 7 in stages:
        r.motifs = stage_motifs(seq, ctx=context_len)
        r.stages_run.append(7)
    if 8 in stages and model_path:
        r.feature_vector = build_feature_vector(r)
        r.ml_prediction  = stage_ml_predict(r.feature_vector, model_path)
        r.stages_run.append(8)
    if 9 in stages:
        sp9   = (r.signal_peptide.get("predicted_cleavage_pos", 0)
                 if r.signal_peptide and r.signal_peptide.get("detected") else 0)
        mhits9 = r.motifs.get("motifs", {}) if r.motifs else None
        ntm9   = r.hydropathy.get("n_tm_helices", 0) if r.hydropathy else 0
        r.cleavage = stage_natural_cleavage(
            seq, sp_cleavage_pos=sp9, motif_hits=mhits9, n_tm_helices=ntm9)
        r.stages_run.append(9)
    if 10 in stages:
        if HAS_GHMA:
            r.ghma = stage_ghma(seq)
            r.stages_run.append(10)
        else:
            r.warnings.append("Stage 10 (GHMA) skipped: numpy/scipy not available")

    return r


def run_pipeline(
    filepath:    str,
    stages:      list[int] = None,
    model_path:  str       = None,
    context_len: int       = 5,
    workers:     int       = 1,
    verbose:     bool      = False,
) -> list[PipelineResult]:
    """
    Run the full analysis pipeline.

    Parameters
    ----------
    filepath    : path to input FASTA file
    stages      : list of stage numbers to run (default: all)
    model_path  : path to trained .pkl for stage 8 ML prediction
    context_len : flanking residues shown in motif context (default: 5)
    workers     : parallel worker processes (default: 1; use 0 for cpu_count())
    verbose     : print per-protein timing to stderr

    Returns
    -------
    List of PipelineResult, one per FASTA record, in input order.
    """
    if stages is None: stages = ALL_STAGES
    stages = set(stages)

    records = parse_fasta(filepath)
    n       = len(records)

    if workers == 0:
        workers = cpu_count()

    t_start = time.perf_counter()
    work    = [(acc, desc, seq, stages, model_path, context_len)
               for acc, desc, seq in records]

    if workers > 1 and n > 1:
        # Multiprocessing — stage 8 (ML) uses joblib which is not fork-safe
        # on all platforms; disable ML in worker pool, apply post-hoc if needed
        if 8 in stages and model_path:
            work_no_ml = [(a,d,s,stages-{8},None,ctx)
                          for a,d,s,_,_,ctx in work]
            with Pool(min(workers, n)) as pool:
                results = pool.map(_process_one, work_no_ml)
            # Apply ML predictions serially (joblib load is not fork-safe)
            for r in results:
                r.feature_vector = build_feature_vector(r)
                r.ml_prediction  = stage_ml_predict(r.feature_vector, model_path)
                if 8 not in r.stages_run:
                    r.stages_run.append(8)
        else:
            with Pool(min(workers, n)) as pool:
                results = pool.map(_process_one, work)
    else:
        results = [_process_one(w) for w in work]

    elapsed = time.perf_counter() - t_start

    if verbose or n >= 100:
        rate = n / elapsed if elapsed > 0 else float('inf')
        print(f"  Pipeline: {n} proteins in {elapsed:.2f}s  "
              f"({rate:.1f} proteins/s,  {elapsed/n*1000:.1f} ms/protein)",
              file=sys.stderr)

    # Stage 2c (mass) computed here: needs SP + motif results, and must run in
    # the main process (not _process_one) to avoid forward-reference issues
    if 2 in stages:
        fasta_records = parse_fasta(filepath)
        seq_map = {acc: clean_seq(seq) for acc, _, seq in fasta_records}
        for r in results:
            seq     = seq_map.get(r.accession, r.sequence)
            sp_pos  = (r.signal_peptide.get("predicted_cleavage_pos", 0)
                       if r.signal_peptide and r.signal_peptide.get("detected") else 0)
            mhits   = r.motifs.get("motifs", {}) if r.motifs else None
            r.mass  = stage_mass(seq, sp_cleavage_pos=sp_pos, motif_hits=mhits)
            r.maldi = stage_maldi(seq, sp_cleavage_pos=sp_pos)
    if 9 in stages:
        for r in results:
            if 9 not in r.stages_run:
                sq9   = r.sequence
                sp9   = (r.signal_peptide.get("predicted_cleavage_pos", 0)
                         if r.signal_peptide and r.signal_peptide.get("detected") else 0)
                mh9   = r.motifs.get("motifs", {}) if r.motifs else None
                ntm9  = r.hydropathy.get("n_tm_helices", 0) if r.hydropathy else 0
                r.cleavage = stage_natural_cleavage(
                    sq9, sp_cleavage_pos=sp9, motif_hits=mh9, n_tm_helices=ntm9)
                r.stages_run.append(9)

    return results

# ══════════════════════════════════════════════════════════════════════════════
# Terminal Report (Stage 9)
# ══════════════════════════════════════════════════════════════════════════════

def _bar(v, mx=1.0, w=22):
    f = min(int((v/mx)*w), w) if mx else 0
    return "█"*f + "░"*(w-f)

def _pct(v): return f"{v*100:5.1f}%"

CLS_COLOUR = {
    "secreted":"94","membrane":"93","cytosolic":"92","nuclear":"95","mitochondrial":"91","uncertain":"90"
}

def print_terminal_report(results: list[PipelineResult]):
    W = "═"*72
    for r in results:
        print(f"\n{W}")
        print(f"  {r.accession}  ·  {r.description[:50]}")
        print(f"  {r.length} aa  ·  stages: {r.stages_run}")
        if r.warnings:
            for w in r.warnings: print(f"  ⚠  {w}")
        print(W)

        if r.physicochemical:
            p = r.physicochemical
            print(f"\n{'PHYSICOCHEMICAL':-<42}")
            print(f"  MW              {p['molecular_weight_da']:>10.2f} Da")
            print(f"  pI              {p['isoelectric_point']:>10.2f}")
            print(f"  GRAVY           {p['gravy']:>+10.4f}  "
                  f"({'hydrophobic' if p['gravy']>0 else 'hydrophilic'})")
            print(f"  Instability     {p['instability_index']:>10.2f}  "
                  f"({'unstable' if not p['stable'] else 'stable'})")
            print(f"  Aliphatic idx   {p['aliphatic_index']:>10.2f}")
            print(f"  Net charge 7.4  {p['net_charge_ph74']:>+10.2f}")
            print(f"  Ext coeff       {p['extinction_coeff']:>10}  M⁻¹cm⁻¹")
        if hasattr(r, 'mass') and r.mass:
            m = r.mass
            sp_note = f"  [SP pos {m['sp_cleavage_pos']} removed, mature = {m['sequence_length']} aa]" if m['includes_sp_removal'] else ""
            print(f"\n{'MASS PREDICTION':-<42}")
            print(f"  Raw monoisotopic  {m['raw_monoisotopic_da']:>13.4f} Da{sp_note}")
            print(f"  Raw average       {m['raw_average_da']:>13.4f} Da  (SDS-PAGE scale)")
            if m['sp_peptide_mass_da']:
                print(f"  SP peptide        {m['sp_peptide_mass_da']:>13.4f} Da  (cleaved)")
            print(f"  PTM min           {m['ptm_min_da']:>13.4f} Da  (disulfides only)")
            print(f"  PTM max           {m['ptm_max_da']:>13.4f} Da  (all sites modified)")
            print(f"  PTM window        {m['ptm_range_da']:>13.2f} Da  span")
            print(f"  PTM expected      {m['ptm_expected_da']:>13.4f} Da  (occupancy-weighted, delta {m['ptm_delta_expected']:+.2f})")
            if m['by_ptm_class']:
                print(f"  Modifications (max Δ per class):")
                for cls, data in sorted(m['by_ptm_class'].items(), key=lambda x:-abs(x[1]['max_delta'])):
                    if data['max_delta'] != 0:
                        sites_str = ", ".join(data['ptms'][:3])
                        print(f"    {cls:<20} {data['max_delta']:>+10.2f} Da  [{sites_str}]")
            esi = m['esi_charge_states']
            # Protonated series
            h_ions  = {k: v for k, v in esi.items() if k.startswith("H_")}
            esi_str = "   ".join(f"{k}={v:.2f}" for k, v in list(h_ions.items())[:6])
            print(f"  ESI [M+zH]z+:      {esi_str}")
            # Salt adducts
            na1 = esi.get("Na_z1"); k1 = esi.get("K_z1"); na2 = esi.get("Na2_z2")
            if na1:
                print(f"  ESI adducts:      [M+Na]+ {na1:.2f}   [M+K]+ {k1:.2f}   [M+Na+H]2+ {na2:.2f}")
            # Recommended masses
            if m.get("recommended_masses"):
                rec = m["recommended_masses"]
                print(f"  Recommended:")
                print(f"    Bottom-up MS    {rec['bottom_up_ms']:>13.4f} Da  (monoisotopic, peptide digest)")
                print(f"    Intact MS       {rec['intact_protein']:>13.4f} Da  (PTM-adjusted expected)")
                print(f"    SDS-PAGE gel    {rec['sds_page']:>13.1f} kDa  (average mass)")
            # Isotopic scale note
            if m.get("isotopic_hint"):
                print(f"  Note: {m['isotopic_hint']}")
            # PTM mass interpretation
            delta = m.get("ptm_delta_expected", 0)
            if abs(delta) > 500:
                ptm_hint = f"  ⚠  PTM delta +{delta:.0f} Da — substantial glycosylation or ubiquitination likely"
            elif abs(delta) > 100:
                ptm_hint = f"  ↑  PTM delta +{delta:.0f} Da — significant glycosylation or phosphorylation present"
            elif abs(delta) > 20:
                ptm_hint = f"  ~  PTM delta +{delta:.0f} Da — moderate oxidation or acetylation expected"
            elif abs(delta) > 0.1:
                ptm_hint = f"     PTM delta +{delta:.2f} Da — minimal modification (deamidation / disulfide only)"
            else:
                ptm_hint = "     No PTM mass shift predicted"
            print(f"{ptm_hint}")

        if hasattr(r, 'maldi') and r.maldi:
            md = r.maldi
            print(f"\n{'MALDI-TOF  (intact · linear +ve · ' + md['matrix'] + ')':-<42}")
            print(f"  Cys / SS bonds       : {md['n_cys_total']} Cys  ·  {md['n_disulfides']} disulfides")
            print(f"  Native (SS closed)   : {md['native_avg_da']:>13.4f} Da  (avg)")
            print(f"  Reduced (+DTT)       : {md['reduced_avg_da']:>13.4f} Da  (avg)")
            print(f"  Delta DTT            : +{md['delta_dtt_da']:>12.4f} Da  ({md['n_disulfides']} × 2.016)")
            print(f"  Envelope sigma       : {md['sigma_native_da']:.2f} Da  /  {md['sigma_reduced_da']:.2f} Da")
            an = md['adducts_native'];  ar = md['adducts_reduced']
            print(f"  Native  [M+H]+       : {an['[M+H]+']:>12.2f}   [M+Na]+  {an['[M+Na]+']:.2f}   [M+K]+  {an['[M+K]+']:.2f}")
            print(f"  Reduced [M+H]+       : {ar['[M+H]+']:>12.2f}   [M+Na]+  {ar['[M+Na]+']:.2f}   [M+K]+  {ar['[M+K]+']:.2f}")
            if md['n_cys_total'] >= 6:
                print(f"  ⚠  {md['n_cys_total']} Cys residues — disulfide-rich; confirm reduction is complete before acquiring")
            if md['n_disulfides'] == 0 and md['n_cys_total'] > 0:
                print(f"  ℹ  Cys present but no SS bonds assumed — free thiols or Cys density below pairing threshold")

        if hasattr(r, 'cleavage') and r.cleavage:
            cl = r.cleavage
            print(f"\n{'NATURAL CLEAVAGE SITES':-<42}")
            print(f"  Systems detected     : {len(cl['systems_found'])} → {', '.join(cl['systems_found'][:5])}{'...' if len(cl['systems_found'])>5 else ''}")
            print(f"  Sites found          : {cl['n_sites']:>4}  (min specificity: {cl['min_specificity']})")
            print(f"  Daughter fragments   : {cl['n_fragments']:>4}  (≥ 8 aa)")
            print(f"  Confirmed fragments  : {cl['n_confirmed']:>4}  (HIGH-spec site + motif evidence)")
            if cl['sp_pos_used'] > 0:
                print(f"  SP cleavage applied  : pos {cl['sp_pos_used']} (from stage 6)")
            # Top fragments by mass
            top = sorted(cl['fragments'], key=lambda x: x['mass_mono'], reverse=True)[:6]
            if top:
                print(f"  Top fragments by mass:")
                print(f"    {'Pos':<10}  {'Len':>4}  {'Mono (Da)':>12}  {'Avg (Da)':>10}  {'[M+H]+':>10}  Conf  Sequence")
                for f in top:
                    conf_mark = {'confirmed':'✓','supported':'~','predicted':'?'}[f['confidence']]
                    print(f"    {str(f['start'])+'-'+str(f['end']):<10}  {f['length']:>4}  "
                          f"{f['mass_mono']:>12.4f}  {f['mass_avg']:>10.2f}  "
                          f"{f['mz1']:>10.4f}  [{conf_mark}]   {f['seq'][:26]}{'…' if len(f['seq'])>26 else ''}")
            # Motif-evidenced HIGH sites only
            high_confirmed = [s for s in cl['sites'] if s['confidence'] == 'confirmed']
            if high_confirmed:
                print(f"  Confirmed sites (HIGH + motif evidence):")
                for s in high_confirmed[:8]:
                    print(f"    {s['site_name']:<28} pos {s['match_start']:>4}→{s['cut_pos']:<4}  "
                          f"match: {s['match_seq']:<10}  {s['description'][:45]}")



        if r.composition:
            c = r.composition
            print(f"\n{'COMPOSITION':-<42}")
            aas = sorted(AA_STANDARD)
            for i in range(0,20,5):
                row = aas[i:i+5]
                print("  " + "".join(f"{aa}:{c['monomer_freq'].get(aa,0)*100:4.1f}%  " for aa in row))
            print(f"  Hydrophobic  {_pct(c['hydrophobic_frac'])}  {_bar(c['hydrophobic_frac'])}")
            print(f"  Charged(+)   {_pct(c['charged_positive'])}  {_bar(c['charged_positive'],0.25)}")
            print(f"  Charged(-)   {_pct(c['charged_negative'])}  {_bar(c['charged_negative'],0.25)}")
            print(f"  Polar        {_pct(c['polar_frac'])}  {_bar(c['polar_frac'])}")

        if r.secondary_structure:
            ss = r.secondary_structure
            print(f"\n{'SECONDARY STRUCTURE  (Chou-Fasman w=7)':-<42}")
            print(f"  α-Helix  {_pct(ss['helix_fraction'])}  {_bar(ss['helix_fraction'])}")
            print(f"  β-Sheet  {_pct(ss['sheet_fraction'])}  {_bar(ss['sheet_fraction'])}")
            print(f"  Coil     {_pct(ss['coil_fraction'])}  {_bar(ss['coil_fraction'])}")
            print(f"  N-terminal helix score: {ss['n_term_helix_score']:.3f}")
            calls = ss['calls']
            print(f"  SS map: {calls[:72]}{'…' if len(calls)>72 else ''}")

        if r.hydropathy:
            h = r.hydropathy
            print(f"\n{'HYDROPATHY  (Kyte-Doolittle w=9)':-<42}")
            print(f"  Mean {h['mean_score']:+.4f}  Max {h['max_score']:+.4f}  "
                  f"Peak pos {h['peak_position']}")
            print(f"  TM helices (len≥18): {h['n_tm_helices']}")
            if h['hydrophobic_regions']:
                print(f"  Hydrophobic regions (>{h['threshold']}):")
                for reg in h['hydrophobic_regions'][:5]:
                    tm = "  ← TM HELIX" if reg.get('is_tm') else ""
                    print(f"    pos {reg['start']:>4}–{reg['end']:<4}  "
                          f"len={reg['length']:>3}  mean={reg['mean_score']:+.3f}{tm}")
            else:
                print("  No hydrophobic regions above threshold")

        if r.complexity:
            cx = r.complexity
            print(f"\n{'SEQUENCE COMPLEXITY  (Shannon entropy)':-<42}")
            flag = " ← LOW COMPLEXITY" if cx['is_low_complexity'] else ""
            print(f"  Mean {cx['mean_entropy']:.4f} bits  "
                  f"Min {cx['min_entropy']:.4f}{flag}")
            if cx['low_complexity_regions']:
                for reg in cx['low_complexity_regions'][:2]:
                    print(f"  LC region pos {reg['start']}–{reg['end']}:  "
                          f"'{reg['sequence'][:30]}'")

        if hasattr(r, 'disorder') and r.disorder:
            d = r.disorder
            print(f"\n{'DISORDER  (heuristic)':-<42}")
            flag2 = " ← DISORDERED" if d['is_disordered'] else ""
            print(f"  Disordered fraction : {d['disordered_frac']*100:.1f}%{flag2}")
            if d['disordered_regions']:
                for reg in d['disordered_regions'][:3]:
                    print(f"  IDR pos {reg['start']}–{reg['end']}  "
                          f"len={reg['length']}  score={reg['mean_score']:.3f}")

        if r.signal_peptide:
            sp = r.signal_peptide
            print(f"\n{'SIGNAL PEPTIDE':-<42}")
            if sp.get('detected'):
                print(f"  Detected  score={sp['score']:.3f}  "
                      f"cleavage pos={sp['predicted_cleavage_pos']}")
                print(f"  SP:  {sp['signal_peptide']}")
                print(f"  N:   {sp['n_region']}  (pos_density={sp['pos_density']:.2f})")
                print(f"  H:   {sp['h_region']}  (hydro_density={sp['hydro_density']:.2f})")
                print(f"  C:   {sp['c_region']}  (AXA={sp['cleavage_ok']})")
                print(f"  Mature start: {sp['mature_start']}…")
            else:
                print(f"  Not detected  ({sp.get('reason','')})")

        if r.motifs:
            m = r.motifs
            print(f"\n{'MOTIFS':-<42}")
            print(f"  {m['n_motif_types']} types  ·  {m['total_hits']} total hits")
            for mname, mdata in sorted(m['motifs'].items(),
                                       key=lambda x: -x[1]['count'])[:8]:
                hits_str = ", ".join(f"pos{h['start']}" for h in mdata['hits'][:3])
                if mdata['count']>3: hits_str += f" +{mdata['count']-3}"
                print(f"  {mname:<24} ×{mdata['count']:<3}  [{hits_str}]")

        if r.ml_prediction:
            ml = r.ml_prediction
            print(f"\n{'ML LOCALISATION PREDICTION':-<42}")
            if "error" in ml:
                print(f"  Error: {ml['error']}")
            else:
                col   = f"\033[{CLS_COLOUR.get(ml['prediction'],'37')}m"
                reset = "\033[0m"
                print(f"  Prediction : {col}{ml['prediction'].upper()}{reset}")
                print(f"  Confidence : {ml['confidence']:.4f}  [{ml['confidence_band']}]")
                print(f"  Per-class probabilities:")
                proba = ml['per_class_proba']
                mx    = max(proba.values())
                for cls, p in sorted(proba.items(), key=lambda x:-x[1]):
                    bar = "█"*int(p/mx*20) if mx else ""
                    marker = " ◄" if cls==ml['prediction'] else ""
                    print(f"    {cls:<16} {p:.4f}  {bar:<20}{marker}")

        if r.ghma:
            g = r.ghma
            print(f"\n{'GHMA  (Geometric Harmonic Manifold)':-<42}")
            if "error" in g:
                print(f"  Error: {g['error']}")
            else:
                warn_pc1 = "  ⚠ low — interpret PC1 with caution" if g["pc1_low_variance"] else ""
                print(f"  Channels retained  : {g['channels_retained']}"
                      + (f"  (dropped: {g['channels_dropped']})" if g['channels_dropped'] else ""))
                print(f"  PC1 variance frac  : {g['pc1_var_fraction']:.4f}{warn_pc1}")
                print(f"  Phase coherence    : {g['phase_coherence']:.4f}  (band-limited 0.05–0.45)")
                print(f"  Spectral entropy   : {g['spectral_entropy']:.4f}")
                print(f"  Resonance stability: {g['resonance_stability']:.4f}")
                print(f"  Higuchi FD         : {g['higuchi_fd']:.4f}")
                print(f"  Anisotropy         : {g['anisotropy']:.4f}")
                print(f"  Dominant period    : {g['dominant_period']:.2f} residues")
                if g.get('torsion_mean_abs') is not None:
                    print(f"  Torsion mean |τ|   : {g['torsion_mean_abs']:.4f}")
                    print(f"  Torsion max  |τ|   : {g['torsion_max_abs']:.4f}  @ residue {g['torsion_peak_pos']}")
                if g["copeak_positions"]:
                    pos_str = ", ".join(str(p) for p in g["copeak_positions"])
                    print(f"  Co-peak positions  : {pos_str}")
                    print(f"  (≥3 local metrics above 75th pctile — structural interest)")
                else:
                    print(f"  Co-peak positions  : none detected")

        print()

# ══════════════════════════════════════════════════════════════════════════════
# JSON Export
# ══════════════════════════════════════════════════════════════════════════════

def to_json(results: list[PipelineResult]) -> str:
    def _clean(obj):
        if isinstance(obj, dict):   return {k:_clean(v) for k,v in obj.items()}
        if isinstance(obj, list):   return [_clean(v) for v in obj]
        if isinstance(obj, (int,float,str,bool,type(None))): return obj
        return str(obj)
    out = []
    for r in results:
        d = {
            "accession":r.accession,"description":r.description,
            "sequence":r.sequence,
            "length":r.length,"timestamp":r.timestamp,
            "stages_run":r.stages_run,"warnings":r.warnings,
        }
        for attr in ["physicochemical","composition","hydropathy","secondary_structure",
                     "complexity","signal_peptide","motifs","mass","maldi","cleavage","ml_prediction","feature_vector","ghma"]:
            v = getattr(r,attr,None)
            if v is not None: d[attr] = _clean(v)
        out.append(d)
    return json.dumps(out if len(out)>1 else out[0], indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# HTML Report
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Sequence charge / hydropathy / pI profile SVG (per-residue, sliding window)
# ══════════════════════════════════════════════════════════════════════════════

# Per-residue reference: (side-chain pKa or None, pI, KD hydropathy, class)
_AA_PROPS = {
    'A':(None, 6.00, 1.8,'aliphatic'),'R':(12.48,10.76,-4.5,'basic'),
    'N':(None, 5.41,-3.5,'polar'),    'D':(3.65,  2.77,-3.5,'acidic'),
    'C':(8.18,  5.07, 2.5,'special'), 'E':(4.25,  3.22,-3.5,'acidic'),
    'Q':(None, 5.65,-3.5,'polar'),    'G':(None,  5.97,-0.4,'aliphatic'),
    'H':(6.04,  7.47,-3.2,'basic'),   'I':(None,  5.94, 4.5,'aliphatic'),
    'L':(None, 5.98, 3.8,'aliphatic'),'K':(10.53, 9.47,-3.9,'basic'),
    'M':(None, 5.74, 1.9,'aliphatic'),'F':(None,  5.48, 2.8,'aromatic'),
    'P':(None, 6.30,-1.6,'special'),  'S':(None,  5.68,-0.8,'polar'),
    'T':(None, 5.60,-0.7,'polar'),    'W':(None,  5.88,-0.9,'aromatic'),
    'Y':(10.07,5.63,-1.3,'aromatic'), 'V':(None,  5.96, 4.2,'aliphatic'),
}

def _res_charge(aa: str, pH: float) -> float:
    """Henderson-Hasselbalch side-chain charge for one residue."""
    props = _AA_PROPS.get(aa)
    if not props or props[0] is None:
        return 0.0
    pKa, _, _, cls = props
    if cls == 'acidic' or (cls == 'special' and aa == 'C') or (cls == 'aromatic' and aa == 'Y'):
        return -1.0 / (1.0 + 10 ** (pKa - pH))
    return  1.0 / (1.0 + 10 ** (pH - pKa))   # basic

def _window_avg(vals: list, win: int) -> list:
    half = win // 2
    out = []
    n = len(vals)
    for i in range(n):
        s, e = max(0, i - half), min(n, i + half + 1)
        seg = vals[s:e]
        out.append(sum(seg) / len(seg))
    return out

def _build_charge_profile_svg(seq: str, pH: float = 7.4, window: int = 9) -> str:
    """
    Three-track per-residue SVG profile:
      Track 1 — Net side-chain charge  (blue=positive, red=negative)
      Track 2 — Kyte-Doolittle hydropathy  (amber=hydrophobic)
      Track 3 — Side-chain isoelectric point  (purple gradient)

    Returns an inline SVG string. Empty string if seq is too short.
    """
    n = len(seq)
    if n < 4:
        return ''

    charges = [_res_charge(aa, pH) for aa in seq]
    hydros  = [_AA_PROPS.get(aa, (None,5.97,0,'aliphatic'))[2] for aa in seq]
    pis     = [_AA_PROPS.get(aa, (None,5.97,0,'aliphatic'))[1] for aa in seq]

    s_charge = _window_avg(charges, window)
    s_hydro  = _window_avg(hydros,  window)
    s_pi     = _window_avg(pis,     window)

    # SVG dimensions
    W, PAD_L, PAD_R = 460, 30, 8
    TRACK_H, TRACK_GAP = 60, 8
    TOP, BOT = 6, 18
    PLOT_W = W - PAD_L - PAD_R
    TOTAL_H = TOP + 3 * TRACK_H + 2 * TRACK_GAP + BOT

    def xp(i):
        return PAD_L + (i / max(n - 1, 1)) * PLOT_W

    def track_svg(vals, vmin, vmax, vzero, track_y,
                  pos_col, neg_col, line_col, label, label_col):
        vrange = vmax - vmin or 1.0
        def yp(v):
            return track_y + TRACK_H - ((v - vmin) / vrange) * TRACK_H

        parts = []

        # Background
        parts.append(
            f'<rect x="{PAD_L}" y="{track_y}" width="{PLOT_W}" height="{TRACK_H}" '
            f'rx="3" fill="#0c1520" opacity=".8"/>'
        )

        # Zero line
        y0 = yp(vzero)
        parts.append(
            f'<line x1="{PAD_L}" y1="{y0:.1f}" x2="{PAD_L+PLOT_W}" y2="{y0:.1f}" '
            f'stroke="#1e3048" stroke-width="1" stroke-dasharray="3 2"/>'
        )

        # Filled area (above zero = positive colour, below = negative colour)
        if n > 1:
            # Positive fill (values above vzero)
            pos_pts = [(xp(0), y0)]
            for i, v in enumerate(vals):
                pos_pts.append((xp(i), yp(v)))
            pos_pts.append((xp(n-1), y0))
            pos_path = 'M ' + ' L '.join(f'{x:.1f},{y:.1f}' for x, y in pos_pts) + ' Z'
            parts.append(f'<path d="{pos_path}" fill="{pos_col}" opacity=".2"/>')

            # Line
            line_pts = ' '.join(f'{xp(i):.1f},{yp(v):.1f}' for i, v in enumerate(vals))
            parts.append(
                f'<polyline points="{line_pts}" fill="none" '
                f'stroke="{line_col}" stroke-width="1.5" stroke-linejoin="round"/>'
            )

        # Y-axis ticks and labels
        for tick_v in [vmin, vzero, vmax]:
            ty = yp(tick_v)
            parts.append(
                f'<line x1="{PAD_L-3}" y1="{ty:.1f}" x2="{PAD_L}" y2="{ty:.1f}" '
                f'stroke="#2a3a4a" stroke-width="1"/>'
            )
            lbl = f'{tick_v:+.0f}' if tick_v != 0 else '0'
            parts.append(
                f'<text x="{PAD_L-5}" y="{ty+3:.1f}" text-anchor="end" '
                f'font-family="monospace" font-size="7" fill="#3a4a5a">{lbl}</text>'
            )

        # Track label
        parts.append(
            f'<text x="{PAD_L+4}" y="{track_y+10}" '
            f'font-family="monospace" font-size="8" fill="{label_col}" '
            f'opacity=".85">{label}</text>'
        )

        return ''.join(parts)

    # X-axis position labels
    x_labels = []
    step = max(1, n // 8)
    for i in range(0, n, step):
        x = xp(i)
        x_labels.append(
            f'<text x="{x:.1f}" y="{TOTAL_H - 4}" text-anchor="middle" '
            f'font-family="monospace" font-size="7" fill="#2a3a4a">{i+1}</text>'
        )
    # Always label the last position
    x_labels.append(
        f'<text x="{xp(n-1):.1f}" y="{TOTAL_H - 4}" text-anchor="middle" '
        f'font-family="monospace" font-size="7" fill="#2a3a4a">{n}</text>'
    )

    # Build three tracks
    ty1 = TOP
    ty2 = TOP + TRACK_H + TRACK_GAP
    ty3 = TOP + 2 * (TRACK_H + TRACK_GAP)

    ph_label = f'pH {pH:.1f}'
    t1 = track_svg(s_charge, -1.0, 1.0,   0.0, ty1, '#60a5fa', '#fb7185', '#93c5fd',
                   f'Charge ({ph_label})', '#93c5fd')
    t2 = track_svg(s_hydro,  -5.0, 5.0,   0.0, ty2, '#fbbf24', '#94a3b8', '#fcd34d',
                   'KD Hydropathy',        '#fcd34d')
    t3 = track_svg(s_pi,      0.0, 14.0,  7.0, ty3, '#c084fc', '#818cf8', '#d8b4fe',
                   'Side-chain pI',        '#d8b4fe')

    # Sequence strip (coloured by class, bottom row)
    CLS_COL = {
        'acidic':'#fb7185','basic':'#60a5fa','polar':'#34d399',
        'aliphatic':'#64748b','aromatic':'#c084fc','special':'#fbbf24'
    }
    strip_y = TOP + 3 * TRACK_H + 2 * TRACK_GAP - 2
    strip_h = 5
    strip_parts = []
    for i, aa in enumerate(seq):
        cls = _AA_PROPS.get(aa, (None,5.97,0,'aliphatic'))[3]
        col = CLS_COL.get(cls, '#64748b')
        x = xp(i)
        w_px = max(1.0, PLOT_W / n)
        strip_parts.append(
            f'<rect x="{x:.1f}" y="{strip_y}" width="{w_px:.1f}" height="{strip_h}" '
            f'fill="{col}" opacity=".7"/>'
        )

    # Window annotation
    win_note = f'window={window}'
    win_label = (
        f'<text x="{W - PAD_R}" y="{TOP + 8}" text-anchor="end" '
        f'font-family="monospace" font-size="7" fill="#2a3a4a">{win_note}</text>'
    )

    svg = (
        f'<svg viewBox="0 0 {W} {TOTAL_H}" class="charge-profile-svg" '
        f'xmlns="http://www.w3.org/2000/svg">'
        + t1 + t2 + t3
        + ''.join(strip_parts)
        + ''.join(x_labels)
        + win_label
        + '</svg>'
    )
    return svg


def _build_topology_map(r: PipelineResult) -> str:
    """
    SVG protein topology diagram: linear sequence map with annotated features.
    Shows signal peptide, TM helices, IDRs, motif clusters, and mature protein.
    Returns empty string if insufficient data.
    """
    if not r.hydropathy and not r.signal_peptide and not r.motifs:
        return ""

    n    = r.length or 1
    W, H = 460, 54        # SVG canvas
    PAD  = 24             # left/right padding inside track

    # Coordinate helper
    def x(pos): return PAD + (pos / n) * (W - 2*PAD)
    def w(start, end): return max(2, x(end) - x(start))

    COLOURS = {
        "sp":       "#38bdf8",   # signal peptide — blue
        "tm":       "#fb923c",   # TM helix — orange
        "idr":      "#a78bfa",   # disordered region — violet
        "mature":   "#1e293b",   # mature/background — dark
        "glyco":    "#4ade80",   # glycosylation — green
        "phospho":  "#fbbf24",   # phosphorylation — amber
        "nls":      "#e879f9",   # NLS — magenta
        "cleavage": "#f87171",   # cleavage sites — red
        "other":    "#94a3b8",   # other motifs — grey
    }

    MOTIF_COLOUR_MAP = {
        # Glycosylation
        "N_glycosylation": "glyco", "O_glycosylation": "glyco",
        "O_GalNAc_mucin":  "glyco", "O_GlcNAc_cytosolic": "glyco",
        "C_mannosylation": "glyco",
        # Phosphorylation
        "PKC_phospho":   "phospho", "CK2_phospho":   "phospho",
        "PKA_phospho":   "phospho", "Tyr_kinase":    "phospho",
        "CDK_phospho":   "phospho", "CDK_minimal":   "phospho",
        "ATM_ATR_phospho":"phospho","MAPK_phospho":  "phospho",
        "AURORA_phospho":"phospho", "DNAPK_phospho": "phospho",
        "CAMKII_phospho":"phospho", "NEK2_phospho":  "phospho",
        # Localisation
        "NLS_basic":       "nls",   "NLS_bipartite": "nls",
        "NES_leucine_rich":"nls",   "CRM1_NES":      "nls",
        "NoLS_nucleolar":  "nls",   "Peroxisome_PTS1":"nls",
        # Cleavage
        "Furin_cleavage":  "cleavage", "Caspase_3_6":   "cleavage",
        "Caspase_1":       "cleavage", "Caspase_8_9":   "cleavage",
        "Granzyme_B":      "cleavage", "MMP_cleavage":  "cleavage",
        "Zoonotic_furin":  "cleavage",
        # Other → default "other"
    }

    segments = []   # (x, width, colour, tooltip, zorder)
    labels   = []   # (x, y, text, colour)
    markers  = []   # (x, colour, label) — single-position events

    # Background track
    segments.append((PAD, W - 2*PAD, COLOURS["mature"], f"Full sequence ({n} aa)", 0))

    # Signal peptide
    sp_end = 0
    if r.signal_peptide and r.signal_peptide.get("detected"):
        sp_end = r.signal_peptide["predicted_cleavage_pos"]
        segments.append((x(0), w(0, sp_end), COLOURS["sp"],
                         f"Signal peptide 1–{sp_end}", 1))
        labels.append((x(0) + 2, 14, "SP", "white"))

    # TM helices
    if r.hydropathy:
        for i, tm in enumerate(r.hydropathy.get("tm_helices", [])):
            xs = x(tm["start"]); xe = w(tm["start"], tm["end"])
            segments.append((xs, xe, COLOURS["tm"],
                             f"TM helix {i+1}: {tm['start']}–{tm['end']} "
                             f"(len={tm['length']}, mean={tm['mean_score']:.2f})", 1))
            if xe > 12:
                labels.append((xs + 2, 14, f"TM{i+1}", "white"))

    # Disordered regions
    if r.disorder:
        for idr in r.disorder.get("disordered_regions", []):
            xs = x(idr["start"]); xe = w(idr["start"], idr["end"])
            segments.append((xs, xe, COLOURS["idr"],
                             f"IDR {idr['start']}–{idr['end']} "
                             f"(len={idr['length']}, score={idr['mean_score']:.2f})", 2))

    # Motif markers (single-position ticks above track)
    if r.motifs:
        for mname, mdata in r.motifs.get("motifs", {}).items():
            col_key = MOTIF_COLOUR_MAP.get(mname, "other")
            col     = COLOURS[col_key]
            for hit in mdata["hits"][:20]:   # cap at 20 per motif type
                markers.append((x(hit["start"]), col, mname))

    # ── Build SVG ─────────────────────────────────────────────────────────────
    svg_parts = [
        f'<svg viewBox="0 0 {W} {H+32}" class="topo-svg" '
        f'xmlns="http://www.w3.org/2000/svg">',
        # Scale bar
        f'<rect x="{PAD}" y="18" width="{W-2*PAD}" height="8" rx="3" '
        f'fill="#1e293b" opacity="0.08"/>',
    ]

    # Segments (sorted by zorder)
    for sx, sw, sc, tip, _ in sorted(segments, key=lambda s: s[4]):
        svg_parts.append(
            f'<rect x="{sx:.1f}" y="14" width="{sw:.1f}" height="16" rx="2" ' 
            f'fill="{sc}" opacity="0.88">' 
            f'<title>{tip}</title></rect>'
        )

    # Text labels on segments
    for lx, ly, lt, lc in labels:
        svg_parts.append(
            f'<text x="{lx:.1f}" y="{ly}" font-size="8" font-family="monospace" ' 
            f'fill="{lc}" dominant-baseline="hanging" pointer-events="none">{lt}</text>'
        )

    # Motif tick marks above the track
    seen_x: set = set()
    for mx2, mc, mlabel in markers:
        key = round(mx2)
        offset = 2 if key in seen_x else 0
        seen_x.add(key)
        svg_parts.append(
            f'<line x1="{mx2:.1f}" y1="{9-offset}" x2="{mx2:.1f}" y2="14" ' 
            f'stroke="{mc}" stroke-width="1.5" opacity="0.8">' 
            f'<title>{mlabel}</title></line>'
        )

    # Position ruler
    tick_step = max(50, (n // 5 // 50) * 50) if n > 100 else 25
    for pos in range(0, n+1, tick_step):
        tx = x(pos)
        svg_parts.append(
            f'<line x1="{tx:.1f}" y1="30" x2="{tx:.1f}" y2="34" ' 
            f'stroke="#94a3b8" stroke-width="1"/>' 
            f'<text x="{tx:.1f}" y="44" font-size="8" font-family="monospace" ' 
            f'fill="#94a3b8" text-anchor="middle">{pos}</text>'
        )

    # Legend
    legend_items = [
        ("SP", COLOURS["sp"]), ("TM", COLOURS["tm"]), ("IDR", COLOURS["idr"]),
        ("Glyco", COLOURS["glyco"]), ("Phospho", COLOURS["phospho"]),
        ("NLS", COLOURS["nls"]), ("Cleavage", COLOURS["cleavage"]),
    ]
    lx2 = PAD
    for lbl, lc2 in legend_items:
        svg_parts.append(
            f'<rect x="{lx2}" y="{H+14}" width="8" height="8" rx="1" fill="{lc2}" opacity="0.85"/>' 
            f'<text x="{lx2+10}" y="{H+21}" font-size="8" font-family="monospace" fill="#64748b">{lbl}</text>'
        )
        lx2 += len(lbl) * 5 + 24

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def to_html(results: list[PipelineResult], pipeline_version="1.0") -> str:
    CLS_HEX = {"secreted":"#818cf8","membrane":"#fb923c","cytosolic":"#4ade80",
               "nuclear":"#e879f9","mitochondrial":"#f87171","uncertain":"#94a3b8"}

    def _sec(title, content):
        return f'<div class="section"><div class="sec-title">{title}</div>{content}</div>'

    def _kv(k, v, hl=False):
        cls = " class=\"highlight\"" if hl else ""
        return f'<div class="kv"{cls}><span class="k">{k}</span><span class="v">{v}</span></div>'

    def _bar_html(v, mx=1.0, label=""):
        pct = min(100, int((v/mx)*100)) if mx else 0
        return (f'<div class="bar-row"><span class="bar-lbl">{label}</span>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>'
                f'<span class="bar-val">{v*100:.1f}%</span></div>')

    cards = []
    for r in results:
        inner = []

        if r.ml_prediction and "prediction" in r.ml_prediction:
            ml  = r.ml_prediction
            col = CLS_HEX.get(ml['prediction'],'#94a3b8')
            prob_bars = ""
            for cls,p in sorted(ml['per_class_proba'].items(),key=lambda x:-x[1]):
                c   = CLS_HEX.get(cls,'#94a3b8')
                pct = int(p*100)
                active = ' active' if cls==ml['prediction'] else ''
                prob_bars += (f'<div class="prob-row{active}">'
                              f'<span class="prob-cls">{cls}</span>'
                              f'<div class="prob-track">'
                              f'<div class="prob-fill" style="width:{pct}%;background:{c}"></div></div>'
                              f'<span class="prob-val">{p:.3f}</span></div>')
            inner.append(f'''<div class="pred-block" style="border-color:{col}">
  <div class="pred-label">ML Prediction</div>
  <div class="pred-cls" style="color:{col}">{ml['prediction'].upper()}</div>
  <div class="pred-conf">{ml['confidence']:.4f} · {ml['confidence_band']}</div>
  <div class="prob-chart">{prob_bars}</div>
</div>''')

        if r.physicochemical:
            p = r.physicochemical
            phys = (_kv("MW", f"{p['molecular_weight_da']:.2f} Da") +
                    _kv("pI", str(p['isoelectric_point'])) +
                    _kv("GRAVY", f"{p['gravy']:+.4f}", p['gravy']>0.5) +
                    _kv("Instability", f"{p['instability_index']:.2f} ({'unstable' if not p['stable'] else 'stable'})", not p['stable']) +
                    _kv("Net charge pH7.4", f"{p['net_charge_ph74']:+.2f}") +
                    _kv("Aliphatic idx", str(p['aliphatic_index'])))
            inner.append(_sec("Physicochemical", phys))

        if r.secondary_structure:
            ss = r.secondary_structure
            ss_html = (_bar_html(ss['helix_fraction'], label="α-Helix") +
                       _bar_html(ss['sheet_fraction'], label="β-Sheet") +
                       _bar_html(ss['coil_fraction'],  label="Coil") +
                       _kv("N-term helix score", f"{ss['n_term_helix_score']:.3f}") +
                       f'<div class="ss-map">{ss["calls"][:80]}{"…" if len(ss["calls"])>80 else ""}</div>')
            inner.append(_sec("Secondary Structure", ss_html))

        if r.hydropathy:
            h   = r.hydropathy
            pts = h['profile']
            if pts:
                mn,mx2 = min(pts), max(pts)
                span   = mx2-mn or 1
                W,H    = 460,80
                svg_pts = " ".join(
                    f"{int(i/(len(pts)-1)*W)},{int(H-(v-mn)/span*H)}"
                    for i,v in enumerate(pts))
                thresh_y = int(H-(1.6-mn)/span*H)
                svg = (f'<svg viewBox="0 0 {W} {H+20}" class="hydro-svg">'
                       f'<polyline points="{svg_pts}" fill="none" stroke="#60a5fa" stroke-width="1.5"/>'
                       f'<line x1="0" y1="{thresh_y}" x2="{W}" y2="{thresh_y}" '
                       f'stroke="#ef4444" stroke-width="1" stroke-dasharray="4 3"/>'
                       f'<text x="2" y="{thresh_y-3}" font-size="9" fill="#ef4444">TM 1.6</text>'
                       f'</svg>')
                h_html = (svg + _kv("Mean",f"{h['mean_score']:+.4f}") +
                          _kv("Max",f"{h['max_score']:+.4f} at pos {h['peak_position']}") +
                          _kv("Hydrophobic regions",str(h['n_hydrophobic_regions'])))
                for reg in h['hydrophobic_regions'][:3]:
                    tm = " ← TM?" if reg['length']>=18 else ""
                    h_html += _kv(f"  pos {reg['start']}–{reg['end']}",
                                  f"len={reg['length']} mean={reg['mean_score']:+.3f}{tm}")
                inner.append(_sec("Hydropathy", h_html))

        # PTM interpretation hint
        if hasattr(r, 'mass') and r.mass:
            delta = r.mass.get("ptm_delta_expected", 0)
            n_cys_m = r.mass.get("per_ptm", {}).get("Disulfide", {}).get("eligible_sites", 0)
            if abs(delta) > 500:
                ptm_badge = f'<div style="margin:6px 0;padding:5px 10px;background:#1a1040;border-left:3px solid #c084fc;border-radius:3px;font-size:11px;color:#d8b4fe">⚠ PTM shift +{delta:.0f} Da — substantial glycosylation or ubiquitination likely</div>'
            elif abs(delta) > 100:
                ptm_badge = f'<div style="margin:6px 0;padding:5px 10px;background:#1a2810;border-left:3px solid #4ade80;border-radius:3px;font-size:11px;color:#86efac">↑ PTM shift +{delta:.0f} Da — significant glycosylation or phosphorylation</div>'
            elif abs(delta) > 20:
                ptm_badge = f'<div style="margin:6px 0;padding:5px 10px;background:#1a1a0a;border-left:3px solid #fbbf24;border-radius:3px;font-size:11px;color:#fde68a">~ PTM shift +{delta:.1f} Da — moderate modification expected</div>'
            else:
                ptm_badge = ''
            if ptm_badge:
                inner.append(f'<div class="section" style="grid-column:1/-1">{ptm_badge}</div>')

        if r.signal_peptide:
            sp = r.signal_peptide
            if sp.get('detected'):
                sp_html = (_kv("Detected","YES ✓", True) +
                           _kv("Cleavage pos",str(sp['predicted_cleavage_pos'])) +
                           _kv("Score",f"{sp['score']:.3f}") +
                           _kv("Signal peptide",f'<code>{sp["signal_peptide"]}</code>') +
                           _kv("Mature start",f'<code>{sp["mature_start"]}…</code>') +
                           _kv("AXA rule",str(sp['cleavage_ok'])))
            else:
                sp_html = _kv("Detected","NO")
            inner.append(_sec("Signal Peptide", sp_html))

        if r.motifs:
            m = r.motifs
            m_html = _kv("Total hits",str(m['total_hits'])) + _kv("Motif types",str(m['n_motif_types']))
            for cat,cnt in sorted(m['by_category'].items(),key=lambda x:-x[1]):
                m_html += _kv(cat, str(cnt))
            m_html += '<div class="motif-list">'
            for mname,mdata in sorted(m['motifs'].items(),key=lambda x:-x[1]['count'])[:10]:
                hits_str = ", ".join(f"pos{h['start']}" for h in mdata['hits'][:3])
                if mdata['count']>3: hits_str+=f" +{mdata['count']-3}"
                m_html += (f'<div class="motif-item">'
                           f'<span class="motif-name">{mname}</span>'
                           f'<span class="motif-count">×{mdata["count"]}</span>'
                           f'<span class="motif-pos">{hits_str}</span></div>')
            m_html += '</div>'
            inner.append(_sec("Motifs", m_html))

        # ── Protein topology map ──────────────────────────────────────────
        topo_html = _build_topology_map(r)
        if topo_html:
            inner.insert(0, _sec("Protein Map", topo_html))

        # ── Per-residue charge / hydropathy / pI profile ──────────────────
        if r.sequence and len(r.sequence) >= 4:
            _ph = (r.physicochemical or {}).get('isoelectric_point', 7.4)
            profile_svg = _build_charge_profile_svg(r.sequence, pH=7.4, window=9)
            if profile_svg:
                inner.append(
                    f'<div class="section profile-section" style="grid-column:1/-1">' +
                    '<div class="sec-title">Sequence Profile — Charge · Hydropathy · pI</div>' +
                    profile_svg +
                    '</div>'
                )

        # ── MALDI spectrum SVG ─────────────────────────────────────────────
        if hasattr(r, 'maldi') and r.maldi and r.maldi.get('spectrum_svg'):
            md = r.maldi
            an = md['adducts_native']; ar = md['adducts_reduced']
            maldi_kv = (
                _kv('Cys / disulfides', f"{md['n_cys_total']} Cys · {md['n_disulfides']} SS") +
                _kv('Native (SS closed)', f"{md['native_avg_da']:.4f} Da") +
                _kv('Reduced (+DTT)', f"{md['reduced_avg_da']:.4f} Da") +
                _kv('Δ DTT', f"+{md['delta_dtt_da']:.4f} Da ({md['n_disulfides']} × 2.016)") +
                _kv('[M+H]+ native', str(an['[M+H]+'])) +
                _kv('[M+H]+ reduced', str(ar['[M+H]+'])) +
                _kv('[M+Na]+ native', str(an['[M+Na]+'])) +
                _kv('[M+Na]+ reduced', str(ar['[M+Na]+'])) +
                _kv('[M+K]+ native', str(an['[M+K]+'])) +
                _kv('[M+K]+ reduced', str(ar['[M+K]+'])) +
                _kv('σ native / reduced', f"{md['sigma_native_da']:.2f} / {md['sigma_reduced_da']:.2f} Da")
            )
            inner.append(
                f'<div class="section maldi-section" style="grid-column:1/-1">' +
                f'<div class="sec-title">MALDI-TOF Simulation — Native vs DTT-Reduced · {md["matrix"]} matrix · avg mass · Yergey envelope</div>' +
                md['spectrum_svg'] +
                '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:0">' +
                maldi_kv + '</div></div>'
            )

        # ── Natural cleavage section ────────────────────────────────────
        if hasattr(r, 'cleavage') and r.cleavage:
            cl = r.cleavage
            frags = cl['fragments']
            sites = cl['sites']
            SYS_COL = {
                'Furin/PCSK':'#f472b6','Signal peptidase':'#38bdf8','Caspase':'#fb7185',
                'Granzyme':'#f97316','MMP':'#a78bfa','ADAM':'#c084fc',
                'Coagulation':'#facc15','Complement':'#4ade80','Cathepsin':'#2dd4bf',
                'Hormone':'#60a5fa','Ubiquitin':'#94a3b8','Viral':'#f43f5e',
                'Neuropeptide':'#fb923c','Kallikrein':'#fbbf24','Mast cell':'#e879f9',
                'Neutrophil':'#f87171','Renin-Angiotensin':'#34d399',
                'Meprin':'#818cf8','Intramembrane':'#475569','Propeptide':'#84cc16',
            }
            # Summary kv
            cl_kv = (
                _kv('Systems found', ', '.join(cl['systems_found'][:4]) + ('…' if len(cl['systems_found'])>4 else '')) +
                _kv('Total sites', str(cl['n_sites'])) +
                _kv('Daughter fragments (≥8 aa)', str(cl['n_fragments']), True) +
                _kv('Confirmed fragments', str(cl['n_confirmed']), cl['n_confirmed']>0) +
                _kv('Min specificity used', cl['min_specificity']) +
                _kv('SP pos applied', str(cl['sp_pos_used']) if cl['sp_pos_used'] else 'none')
            )
            # Fragment table (top 12 by mass, descending)
            top_frags = sorted(frags, key=lambda x: -x['mass_mono'])[:12]
            frows = ''
            for fi, f in enumerate(top_frags):
                conf_col = {'confirmed':'#4ade80','supported':'#fbbf24','predicted':'#475569'}.get(f['confidence'],'#475569')
                sys_col  = SYS_COL.get(f['c_systems'][0] if f['c_systems'] else '', '#94a3b8')
                seq_html = ''.join(f'<span style="color:{AA_COLOUR.get(AA_CLASS.get(aa,"aliphatic"),"#94a3b8")};font-family:monospace;font-size:10px">{aa}</span>' for aa in f['seq'][:30])
                bg = '#fafffe' if fi%2==0 else '#ffffff'
                frows += (f'<tr style="background:{bg}">'
                    f'<td style="padding:3px 5px;font-family:monospace;font-size:10px">{f["start"]}–{f["end"]}</td>'
                    f'<td style="padding:3px 5px;font-family:monospace;font-size:10px;text-align:center">{f["length"]}</td>'
                    f'<td style="padding:3px 6px;font-family:monospace;font-size:11px;font-weight:600">{f["mass_mono"]:.4f}</td>'
                    f'<td style="padding:3px 5px;font-family:monospace;font-size:10px;color:#64748b">{f["mass_avg"]:.2f}</td>'
                    f'<td style="padding:3px 5px;font-family:monospace;font-size:10px">{f["mz1"]:.4f}</td>'
                    f'<td style="padding:3px 5px;font-family:monospace;font-size:10px;color:#64748b">{f["maldi_mh"]}  {f["maldi_mna"]}</td>'
                    f'<td style="padding:3px 5px"><span style="background:{conf_col}22;color:{conf_col};border-radius:3px;padding:1px 5px;font-family:monospace;font-size:9px">{f["confidence"]}</span></td>'
                    f'<td style="padding:3px 5px">{seq_html}{"…" if len(f["seq"])>30 else ""}</td>'
                    f'</tr>')
            frag_table = (
                '<div style="overflow-x:auto;margin-top:8px">'
                '<table style="width:100%;border-collapse:collapse">'
                '<thead><tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0">'
                + ''.join(f'<th style="padding:4px 5px;text-align:left;font-size:10px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.05em">{h}</th>'
                          for h in ['Position','Len','Mono (Da)','Avg (Da)','[M+H]+','MALDI [M+H]+  [M+Na]+','Conf','Sequence'])
                + '</tr></thead><tbody>' + frows + '</tbody></table></div>'
            )
            inner.append(
                '<div class="section cleavage-section" style="grid-column:1/-1">'
                '<div class="sec-title">Natural Cleavage — '
                + f'{cl["n_sites"]} sites · {cl["n_fragments"]} fragments · {cl["min_specificity"]}+ specificity'
                + '</div>'
                + '<div style="display:grid;grid-template-columns:280px 1fr;gap:0">'
                + '<div style="padding:8px 12px;border-right:1px solid #e2e8f0">'
                + cl_kv + '</div>'
                + '<div style="padding:8px 12px">'
                + frag_table + '</div></div></div>'
            )

        # ── GHMA section ────────────────────────────────────────────────
        if r.ghma and "error" not in r.ghma:
            g = r.ghma
            warn_pc1 = " <span style='color:#f59e0b'>⚠ low</span>" if g["pc1_low_variance"] else ""
            dropped_str = (", ".join(g["channels_dropped"])
                           if g["channels_dropped"] else "none")
            copeak_str  = (", ".join(str(p) for p in g["copeak_positions"])
                           if g["copeak_positions"] else "none detected")

            # Inline sparkline SVGs for local profiles
            def _spark(values, color="#2c7bb6", h=40, w=320):
                vals = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
                if len(vals) < 2:
                    return ""
                mn, mx2 = min(vals), max(vals)
                span = mx2 - mn or 1.0
                n    = len(vals)
                pts  = " ".join(
                    f"{int(i/(n-1)*w)},{int(h-(v-mn)/span*h)}"
                    for i, v in enumerate(vals)
                )
                return (f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:{h}px;display:block">'
                        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>'
                        f'</svg>')

            local = g["local"]
            sparks = (
                f'<div style="font-size:10px;color:#64748b;margin-top:2px">Spectral entropy</div>'
                + _spark(local.get("entropy", []), "#d7191c")
                + f'<div style="font-size:10px;color:#64748b;margin-top:6px">Phase coherence</div>'
                + _spark(local.get("coherence", []), "#fdae61")
                + f'<div style="font-size:10px;color:#64748b;margin-top:6px">Higuchi FD</div>'
                + _spark(local.get("fractal", []), "#404040")
                + f'<div style="font-size:10px;color:#64748b;margin-top:6px">Mean |torsion|</div>'
                + _spark(local.get("torsion", []), "#7b2d8b")
            )

            tau_str = (f"{g['torsion_mean_abs']:.4f} (mean |τ|) — peak @ res {g['torsion_peak_pos']}"
                       if g.get("torsion_mean_abs") is not None else "n/a")

            ghma_kv = (
                _kv("Channels retained", str(g["channels_retained"])) +
                _kv("Channels dropped", dropped_str) +
                _kv("PC1 variance fraction", f"{g['pc1_var_fraction']:.4f}" + warn_pc1) +
                _kv("Phase coherence", f"{g['phase_coherence']:.4f}") +
                _kv("Spectral entropy", f"{g['spectral_entropy']:.4f}") +
                _kv("Resonance stability", f"{g['resonance_stability']:.4f}") +
                _kv("Higuchi FD", f"{g['higuchi_fd']:.4f}") +
                _kv("Anisotropy", f"{g['anisotropy']:.4f}") +
                _kv("Dominant period", f"{g['dominant_period']:.2f} residues") +
                _kv("Manifold torsion", tau_str) +
                _kv("Co-peak positions", copeak_str, bool(g["copeak_positions"]))
            )

            inner.append(
                '<div class="section ghma-section" style="grid-column:1/-1">'
                '<div class="sec-title">GHMA — Geometric Harmonic Manifold Analysis</div>'
                '<div style="display:grid;grid-template-columns:300px 1fr;gap:0">'
                f'<div style="padding:8px 12px;border-right:1px solid #e2e8f0">{ghma_kv}</div>'
                f'<div style="padding:8px 12px">{sparks}</div>'
                '</div></div>'
            )

        warn_html = ""
        if r.warnings:
            warn_html = "".join(f'<div class="warn">⚠ {w}</div>' for w in r.warnings)

        cards.append(f'''<div class="card">
  <div class="card-header">
    <div class="card-acc">{r.accession}</div>
    <div class="card-desc">{r.description[:60]}</div>
    <div class="card-meta">{r.length} aa · stages {r.stages_run}</div>
  </div>
  {warn_html}
  <div class="card-body">{"".join(inner)}</div>
</div>''')

    body = "\n".join(cards)
    n    = len(results)
    sp_n = sum(1 for r in results if r.signal_peptide and r.signal_peptide.get('detected'))
    ml_n = sum(1 for r in results if r.ml_prediction and 'prediction' in r.ml_prediction)
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Proteomics Pipeline Report</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Outfit:wght@300;400;600;700&display=swap');
:root{{--bg:#f0f2f5;--card:#ffffff;--ink:#1e293b;--muted:#64748b;--rule:#e2e8f0;
  --acc:#0f172a;--mono:'JetBrains Mono',monospace}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--ink);font-family:'Outfit',sans-serif;font-size:14px;padding:32px 24px 80px}}
header{{background:var(--ink);color:#f1f5f9;padding:28px 32px;border-radius:8px;margin-bottom:24px;
  display:flex;justify-content:space-between;align-items:flex-end}}
header h1{{font-size:22px;font-weight:700;letter-spacing:-.02em}}
header .meta{{font-family:var(--mono);font-size:11px;color:#94a3b8;text-align:right;line-height:1.8}}
.stats-bar{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:24px}}
.stat{{background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:14px 18px}}
.stat .val{{font-size:24px;font-weight:700;font-family:var(--mono);letter-spacing:-.02em}}
.stat .lbl{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-top:2px}}
.card{{background:var(--card);border:1px solid var(--rule);border-radius:8px;margin-bottom:16px;overflow:hidden}}
.card-header{{padding:16px 20px;border-bottom:1px solid var(--rule);background:#f8fafc}}
.card-acc{{font-family:var(--mono);font-size:13px;font-weight:500;color:var(--acc)}}
.card-desc{{font-size:12px;color:var(--muted);margin-top:2px}}
.card-meta{{font-family:var(--mono);font-size:10px;color:#94a3b8;margin-top:4px}}
.card-body{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:0;}}
.warn{{background:#fff7ed;border-left:3px solid #f59e0b;padding:8px 16px;font-size:12px;color:#92400e}}
.section{{padding:16px 18px;border-right:1px solid var(--rule);border-bottom:1px solid var(--rule)}}
.sec-title{{font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--rule)}}
.kv{{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #f1f5f9;font-size:12px}}
.kv.highlight{{background:#f0fdf4;padding:3px 4px;border-radius:2px}}
.k{{color:var(--muted)}} .v{{font-family:var(--mono);font-size:11px;font-weight:500}}
.bar-row{{display:flex;align-items:center;gap:8px;margin:4px 0}}
.bar-lbl{{font-size:11px;color:var(--muted);width:60px;flex-shrink:0}}
.bar-track{{flex:1;height:10px;background:var(--rule);border-radius:3px;overflow:hidden}}
.bar-fill{{height:100%;background:#334155;border-radius:3px}}
.bar-val{{font-family:var(--mono);font-size:10px;color:var(--muted);min-width:36px;text-align:right}}
.ss-map{{font-family:var(--mono);font-size:9px;letter-spacing:.05em;color:var(--muted);
  margin-top:8px;word-break:break-all;background:#f8fafc;padding:6px;border-radius:3px}}
.hydro-svg{{width:100%;height:auto;display:block;margin:8px 0}}
.pred-block{{border:2px solid;border-radius:6px;padding:14px;margin-bottom:4px;
  grid-column:1/-1}}
.pred-label{{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}}
.pred-cls{{font-size:22px;font-weight:700;letter-spacing:-.02em;margin:4px 0 2px}}
.pred-conf{{font-family:var(--mono);font-size:11px;color:var(--muted);margin-bottom:10px}}
.prob-row{{display:flex;align-items:center;gap:8px;margin:3px 0}}
.prob-row.active .prob-cls{{font-weight:600;color:var(--ink)}}
.prob-cls{{font-family:var(--mono);font-size:10px;width:88px;color:var(--muted)}}
.prob-track{{flex:1;height:8px;background:var(--rule);border-radius:2px;overflow:hidden}}
.prob-fill{{height:100%;border-radius:2px}}
.prob-val{{font-family:var(--mono);font-size:10px;color:var(--muted);min-width:36px;text-align:right}}
.motif-list{{margin-top:8px}}
.motif-item{{display:flex;gap:8px;align-items:center;padding:3px 0;border-bottom:1px solid #f1f5f9;font-size:11px}}
.motif-name{{font-family:var(--mono);font-size:10px;width:140px;color:var(--ink)}}
.motif-count{{font-weight:600;min-width:24px}}
.motif-pos{{color:var(--muted);font-family:var(--mono);font-size:10px}}
code{{font-family:var(--mono);font-size:10px;background:#f1f5f9;padding:1px 4px;border-radius:2px;word-break:break-all}}
.topo-svg{{width:100%;height:auto;display:block;margin:4px 0 8px;overflow:visible}}
.charge-profile-svg{{width:100%;height:auto;display:block;margin:6px 0 2px;overflow:visible}}
.profile-section{{background:#0e1520;border-top:1px solid var(--rule)}}
.maldi-section{{background:#08111c;border-top:1px solid var(--rule)}}
.maldi-svg{{width:100%;height:auto;display:block;margin:6px 0 8px;overflow:visible}}
.cleavage-section{{border-top:1px solid var(--rule);background:#f8fafc}}
.ghma-section{{border-top:2px solid #0f172a;background:#f8fafc}}
</style>
</head>
<body>
<header>
  <div>
    <h1>Proteomics Pipeline Report</h1>
    <div style="font-size:13px;color:#94a3b8;margin-top:4px">
      Stages: FASTA → Physicochemical → SS → Hydropathy → Complexity → SP → Motifs → ML → Cleavage → GHMA
    </div>
  </div>
  <div class="meta">
    <div>Generated {ts}</div>
    <div>v{pipeline_version} · Jason Iles</div>
  </div>
</header>
<div class="stats-bar">
  <div class="stat"><span class="val">{n}</span><span class="lbl">Proteins analysed</span></div>
  <div class="stat"><span class="val">{sp_n}</span><span class="lbl">Signal peptides detected</span></div>
  <div class="stat"><span class="val">{ml_n}</span><span class="lbl">ML predictions made</span></div>
  <div class="stat"><span class="val">10</span><span class="lbl">Pipeline stages</span></div>
</div>
{body}
</body></html>"""

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def export_features(results: list[PipelineResult], path: str):
    """Write feature vectors as TSV — one row per protein, all 60 features."""
    have = [r for r in results if r.feature_vector]
    if not have:
        print("  No feature vectors to export (run stages 2-7 first)", file=sys.stderr)
        return
    keys = sorted(have[0].feature_vector.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="	")
        w.writerow(["accession"] + keys)
        for r in have:
            w.writerow([r.accession] + [r.feature_vector.get(k, "") for k in keys])
    print(f"  Features exported → {path}  ({len(have)} proteins × {len(keys)} features)",
          file=sys.stderr)


def benchmark(fasta: str, stages: list[int], workers: int = 1):
    """Time per-stage and per-protein throughput."""
    import time
    records = parse_fasta(fasta)
    n       = len(records)
    print(f"\n  Benchmark: {n} proteins, stages {stages}, workers={workers}")
    print(f"  {'Stage':<32} {'Time (s)':>10}  {'ms/protein':>12}")
    print(f"  {'─'*58}")

    stage_names = {
        2:"Physicochemical+Composition", 3:"Secondary structure",
        4:"Hydropathy (cumsum)", 5:"Complexity+Disorder",
        6:"Signal peptide", 7:"Motif scanner",
    }

    seq = clean_seq(records[0][2])          # benchmark on first protein
    timings = {}
    for s, fn in [
        (2, lambda q: (stage_physicochemical(q), stage_composition(q))),
        (3, lambda q: stage_secondary_structure(q)),
        (4, lambda q: stage_hydropathy(q)),
        (5, lambda q: (stage_complexity(q), stage_disorder(q))),
        (6, lambda q: stage_signal_peptide(q)),
        (7, lambda q: stage_motifs(q)),
    ]:
        if s not in stages: continue
        reps  = max(10, 500 // max(len(seq),1) * 10)
        t0    = time.perf_counter()
        for _ in range(reps): fn(seq)
        elapsed = time.perf_counter() - t0
        ms_per  = elapsed / reps * 1000
        timings[s] = ms_per
        print(f"  {stage_names[s]:<32} {elapsed:>10.3f}  {ms_per:>12.3f}")

    total_ms = sum(timings.values())
    print(f"  {'─'*58}")
    print(f"  {'Total (single protein)':<32} {'':>10}  {total_ms:>12.3f} ms")
    rate = 1000 / total_ms if total_ms else 0
    print(f"  Projected throughput: {rate:.0f} proteins/s single-core")
    print(f"  Human proteome (~20k): ~{20000/rate/60:.1f} min single-core")
    if workers > 1:
        print(f"  With {workers} workers:  ~{20000/rate/60/workers:.1f} min")
    print()


def main():
    p = argparse.ArgumentParser(
        description="Unified Proteomics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("fasta",                      help="Input FASTA file")
    p.add_argument("--model",    default=None,   help="Trained model .pkl for ML prediction")
    p.add_argument("--stages",   nargs="+", type=int, default=ALL_STAGES,
                   help="Stages to run (1=ingest 2=physico 3=ss 4=hydro "
                        "5=complexity+disorder 6=sp 7=motifs 8=ml 9=cleavage 10=ghma)")
    p.add_argument("--no-ml",    action="store_true", help="Skip ML prediction stage")
    p.add_argument("--workers",  type=int, default=1,
                   help="Parallel worker processes (0 = auto cpu_count)")
    p.add_argument("--json",     action="store_true", help="Output JSON to stdout")
    p.add_argument("--html",     default=None,   help="Write HTML report to file")
    p.add_argument("--export-features", default=None, metavar="FILE",
                   help="Export ML feature matrix as TSV")
    p.add_argument("--benchmark", action="store_true",
                   help="Print per-stage timing and throughput estimate")
    p.add_argument("--quiet",    action="store_true", help="Suppress terminal report")
    args = p.parse_args()

    stages = list(args.stages)
    if args.no_ml and 8 in stages:
        stages = [s for s in stages if s != 8]

    model_path = args.model
    if 8 in stages and not model_path:
        print("  Stage 8 (ML) requires --model. Skipping.", file=sys.stderr)
        stages = [s for s in stages if s != 8]

    if args.benchmark:
        benchmark(args.fasta, stages=[s for s in stages if s != 8],
                  workers=args.workers)

    results = run_pipeline(
        args.fasta, stages=stages, model_path=model_path,
        workers=args.workers, verbose=True,
    )
    print(f"  Analysed {len(results)} protein(s)", file=sys.stderr)

    if args.export_features:
        # Build feature vectors for all results if not already built
        for r in results:
            if r.feature_vector is None:
                r.feature_vector = build_feature_vector(r)
        export_features(results, args.export_features)

    if args.json:
        print(to_json(results))
    elif not args.quiet:
        print_terminal_report(results)

    if args.html:
        Path(args.html).write_text(to_html(results))
        print(f"  HTML report → {args.html}", file=sys.stderr)


if __name__ == "__main__":

    import os
    import sys

    # If launched without arguments (Spyder)
    if len(sys.argv) == 1:

        FASTA_FILE = r"C:\Users\jason\Downloads\uniprotkb_accession_P68871_2026_05_26.fasta\uniprotkb_accession_P68871_2026_05_26.fasta"

        print("Running in Spyder standalone mode")

        results = run_pipeline(
            FASTA_FILE,
            stages=[1,2,3,4,5,6,7,9,10],
            model_path=None,
            workers=1,
            verbose=True
        )

        print_terminal_report(results)

        html = to_html(results)

        with open("proteomics_report.html","w",encoding="utf-8") as f:
            f.write(html)

        print("HTML report saved")

    else:
        main()