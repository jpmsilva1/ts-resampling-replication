"""The paper's own published numbers, as data -- the ground truth this
project's replication is checked against.

Source: Moniz, Branco & Torgo (2017), "Resampling strategies for imbalanced
time series forecasting," Int J Data Sci Anal 3:161-181. Transcribed by hand
from `pdftotext -layout` on Papers/Resampling strategies for imbalanced time
series forecasting.pdf -- never from this project's own output, which would
make any comparison tautological.

Do NOT substitute `src/original/Results/paired_comparisons.csv` or
`results_table.csv` for any of this: those are a single-dataset demo bundled
with the original authors' code (15 rows, Win capped at 1), not the paper's
real 24-dataset benchmark result. See CLAUDE.md 2026-08-26 entry.

The paper's 24 datasets vs. this project's 20: DS21-DS24 (half-hourly,
17.4k-239.6k raw rows) are out of scope for this replication throughout, so
every count below is out of 24 while this project's own counts are out of 20
-- compare proportions, not raw counts.
"""

TOTAL_DATASETS_PAPER = 24

# Table 1 (p.170) -- %Rare per dataset, the phi-relevance oracle. Already
# reproduced by SERA Metric/test_sera_metric.py at 0.99pp MAE; kept here too
# so the paper-comparison audit carries the same regression guard.
PCT_RARE = {
    "DS01": 9.9, "DS02": 9.3, "DS03": 7.8, "DS04": 13.3,
    "DS05": 3.5, "DS06": 4.8, "DS07": 12.5, "DS08": 17.6,
    "DS09": 21.1, "DS10": 4.8, "DS11": 13.3, "DS12": 11.0,
    "DS13": 11.1, "DS14": 16.3, "DS15": 11.4, "DS16": 9.7,
    "DS17": 11.6, "DS18": 10.1, "DS19": 8.2, "DS20": 6.8,
    "DS21": 1.8, "DS22": 10.2, "DS23": 0.08, "DS24": 3.4,
}

# Table 3 (p.174) -- U_B/O_B/SM_B vs. the optimized regression-tool baseline
# (no resampling). Cell = (wins, sig_wins, losses, sig_losses), out of 24.
TABLE3 = {
    "U_B": {
        "lm": (19, 18, 5, 4), "svm": (8, 6, 16, 8), "mars": (15, 12, 9, 7),
        "rf": (18, 17, 6, 2), "rpart": (12, 8, 12, 3),
    },
    "O_B": {
        "lm": (18, 17, 6, 3), "svm": (7, 6, 17, 10), "mars": (17, 17, 7, 4),
        "rf": (20, 15, 4, 1), "rpart": (11, 9, 13, 8),
    },
    "SM_B": {
        "lm": (19, 18, 5, 3), "svm": (7, 6, 17, 10), "mars": (18, 17, 6, 4),
        "rf": (20, 20, 4, 1), "rpart": (10, 10, 14, 7),
    },
}

# Table 4 (p.174) -- temporal/relevance-biased variants vs. their own biased
# base strategy (U_T/U_TPhi vs. U_B, etc.), NOT vs. the plain baseline.
TABLE4 = {
    "U_T": {
        "lm": (14, 2, 10, 0), "svm": (10, 0, 14, 0), "mars": (11, 0, 13, 2),
        "rf": (12, 1, 12, 2), "rpart": (14, 4, 10, 0),
    },
    "U_TPhi": {
        "lm": (15, 10, 9, 3), "svm": (11, 5, 13, 4), "mars": (17, 6, 7, 1),
        "rf": (16, 6, 8, 3), "rpart": (16, 7, 8, 5),
    },
    "O_T": {
        "lm": (14, 8, 10, 9), "svm": (12, 5, 12, 6), "mars": (11, 3, 13, 4),
        "rf": (8, 4, 16, 4), "rpart": (12, 3, 12, 2),
    },
    "O_TPhi": {
        "lm": (14, 9, 10, 7), "svm": (12, 4, 12, 7), "mars": (11, 3, 13, 2),
        "rf": (8, 2, 16, 5), "rpart": (14, 3, 10, 2),
    },
    "SM_T": {
        "lm": (6, 5, 18, 13), "svm": (10, 5, 14, 10), "mars": (9, 6, 15, 10),
        "rf": (9, 3, 15, 10), "rpart": (8, 1, 15, 11),
    },
    "SM_TPhi": {
        "lm": (6, 4, 18, 11), "svm": (9, 6, 15, 12), "mars": (12, 4, 12, 6),
        "rf": (12, 5, 12, 10), "rpart": (6, 4, 17, 9),
    },
}
# Which biased-base strategy each Table 4 variant is compared against.
TABLE4_BASE = {
    "U_T": "U_B", "U_TPhi": "U_B",
    "O_T": "O_B", "O_TPhi": "O_B",
    "SM_T": "SM_B", "SM_TPhi": "SM_B",
}

# Table 5 (p.175) -- every resampling strategy vs. ARIMA and vs. BDES,
# per model family. Never checked against this project's own mc.arima /
# mc.BDES workflows before this audit.
TABLE5 = {
    "lm": {
        "U_B": {"ARIMA": (18, 18, 6, 3), "BDES": (22, 22, 2, 2)},
        "U_T": {"ARIMA": (18, 18, 6, 3), "BDES": (22, 22, 2, 2)},
        "U_TPhi": {"ARIMA": (18, 18, 6, 5), "BDES": (22, 22, 2, 2)},
        "O_B": {"ARIMA": (21, 18, 3, 2), "BDES": (22, 22, 2, 2)},
        "O_T": {"ARIMA": (18, 18, 6, 3), "BDES": (22, 22, 2, 2)},
        "O_TPhi": {"ARIMA": (18, 18, 6, 3), "BDES": (22, 22, 2, 2)},
        "SM_B": {"ARIMA": (20, 18, 4, 3), "BDES": (22, 22, 2, 2)},
        "SM_T": {"ARIMA": (18, 17, 6, 5), "BDES": (22, 20, 2, 2)},
        "SM_TPhi": {"ARIMA": (18, 18, 6, 5), "BDES": (22, 20, 2, 2)},
    },
    "svm": {
        "U_B": {"ARIMA": (21, 21, 3, 3), "BDES": (22, 22, 2, 1)},
        "U_T": {"ARIMA": (21, 21, 3, 3), "BDES": (22, 22, 2, 1)},
        "U_TPhi": {"ARIMA": (20, 20, 4, 4), "BDES": (22, 22, 2, 2)},
        "O_B": {"ARIMA": (21, 21, 3, 1), "BDES": (22, 22, 2, 2)},
        "O_T": {"ARIMA": (21, 21, 3, 3), "BDES": (22, 22, 2, 2)},
        "O_TPhi": {"ARIMA": (21, 21, 3, 3), "BDES": (22, 22, 2, 2)},
        "SM_B": {"ARIMA": (19, 19, 5, 1), "BDES": (22, 22, 2, 2)},
        "SM_T": {"ARIMA": (20, 20, 4, 3), "BDES": (20, 20, 4, 2)},
        "SM_TPhi": {"ARIMA": (19, 19, 5, 4), "BDES": (22, 20, 2, 2)},
    },
    "mars": {
        "U_B": {"ARIMA": (23, 18, 1, 1), "BDES": (21, 20, 3, 3)},
        "U_T": {"ARIMA": (20, 18, 4, 2), "BDES": (21, 19, 3, 2)},
        "U_TPhi": {"ARIMA": (22, 19, 2, 2), "BDES": (21, 21, 3, 3)},
        "O_B": {"ARIMA": (19, 18, 5, 1), "BDES": (22, 22, 2, 2)},
        "O_T": {"ARIMA": (18, 18, 6, 2), "BDES": (22, 22, 2, 2)},
        "O_TPhi": {"ARIMA": (18, 18, 6, 2), "BDES": (22, 22, 2, 2)},
        "SM_B": {"ARIMA": (19, 19, 5, 1), "BDES": (22, 22, 2, 2)},
        "SM_T": {"ARIMA": (19, 19, 5, 4), "BDES": (22, 22, 2, 2)},
        "SM_TPhi": {"ARIMA": (19, 19, 5, 4), "BDES": (22, 22, 2, 2)},
    },
    "rf": {
        "U_B": {"ARIMA": (19, 18, 5, 1), "BDES": (19, 18, 5, 2)},
        "U_T": {"ARIMA": (21, 18, 3, 2), "BDES": (19, 18, 5, 2)},
        "U_TPhi": {"ARIMA": (21, 17, 3, 2), "BDES": (18, 18, 6, 2)},
        "O_B": {"ARIMA": (20, 17, 4, 2), "BDES": (18, 16, 6, 2)},
        "O_T": {"ARIMA": (19, 17, 5, 2), "BDES": (15, 15, 9, 3)},
        "O_TPhi": {"ARIMA": (19, 16, 5, 2), "BDES": (15, 15, 9, 3)},
        "SM_B": {"ARIMA": (22, 22, 2, 1), "BDES": (22, 22, 2, 2)},
        "SM_T": {"ARIMA": (20, 20, 4, 2), "BDES": (22, 22, 2, 2)},
        "SM_TPhi": {"ARIMA": (20, 20, 4, 2), "BDES": (22, 22, 2, 2)},
    },
    "rpart": {
        "U_B": {"ARIMA": (22, 20, 2, 2), "BDES": (22, 22, 2, 1)},
        "U_T": {"ARIMA": (22, 20, 2, 2), "BDES": (22, 22, 2, 1)},
        "U_TPhi": {"ARIMA": (20, 18, 4, 1), "BDES": (23, 22, 1, 1)},
        "O_B": {"ARIMA": (20, 20, 4, 1), "BDES": (22, 22, 2, 2)},
        "O_T": {"ARIMA": (20, 20, 4, 1), "BDES": (22, 22, 2, 2)},
        "O_TPhi": {"ARIMA": (21, 20, 3, 1), "BDES": (22, 22, 2, 2)},
        "SM_B": {"ARIMA": (22, 18, 2, 1), "BDES": (22, 22, 2, 1)},
        "SM_T": {"ARIMA": (17, 17, 7, 4), "BDES": (22, 22, 2, 2)},
        "SM_TPhi": {"ARIMA": (19, 18, 5, 3), "BDES": (22, 22, 2, 2)},
    },
}

# Table 6 (p.176) -- the only place the paper prints absolute F1_phi values:
# SVM, 3 datasets, joint cost/gamma + resampling-percentage optimization,
# 10 MC reps (not 50). Anchors magnitude, not an exact replication target.
TABLE6_SVM_F1 = {
    "svm":     {"DS04": 0.584, "DS10": 0.638, "DS12": 0.554},
    "U_B":     {"DS04": 0.668, "DS10": 0.652, "DS12": 0.610},
    "U_T":     {"DS04": 0.659, "DS10": 0.643, "DS12": 0.614},
    "U_TPhi":  {"DS04": 0.651, "DS10": 0.647, "DS12": 0.630},
    "O_B":     {"DS04": 0.653, "DS10": 0.651, "DS12": 0.611},
    "O_T":     {"DS04": 0.650, "DS10": 0.652, "DS12": 0.615},
    "O_TPhi":  {"DS04": 0.651, "DS10": 0.652, "DS12": 0.611},
    "SM_B":    {"DS04": 0.662, "DS10": 0.675, "DS12": 0.609},
    "SM_T":    {"DS04": 0.656, "DS10": 0.698, "DS12": 0.600},
    "SM_TPhi": {"DS04": 0.649, "DS10": 0.721, "DS12": 0.620},
}

# Table 7 (Annex 1, p.180) -- optimal parametrization per dataset, DS1-DS24.
# Transcribed by hand from the paper; independently checked against the raw
# pdftotext extract in table7_pdf_extract.txt by
# test_opt_params_table7_matches_pdf_extract() in test_paired_comparisons.py,
# so a transcription error here can't silently pass just because
# src/adapted/Exps.R's OPT_PARAMS happens to carry the same error.
OPT_PARAMS_TABLE7 = {
    "DS01": dict(svm_cost=300, svm_gamma=0.01,  mars_nk=17, mars_degree=1, mars_thresh=0.001, rf_mtry=5, rf_ntree=1500, rpart_minsplit=10, rpart_cp=0.01),
    "DS02": dict(svm_cost=300, svm_gamma=0.01,  mars_nk=17, mars_degree=2, mars_thresh=0.001, rf_mtry=7, rf_ntree=750,  rpart_minsplit=10, rpart_cp=0.001),
    "DS03": dict(svm_cost=300, svm_gamma=0.01,  mars_nk=17, mars_degree=1, mars_thresh=0.001, rf_mtry=7, rf_ntree=500,  rpart_minsplit=10, rpart_cp=0.001),
    "DS04": dict(svm_cost=150, svm_gamma=0.01,  mars_nk=10, mars_degree=1, mars_thresh=0.001, rf_mtry=7, rf_ntree=750,  rpart_minsplit=10, rpart_cp=0.1),
    "DS05": dict(svm_cost=300, svm_gamma=0.001, mars_nk=10, mars_degree=2, mars_thresh=0.001, rf_mtry=7, rf_ntree=750,  rpart_minsplit=20, rpart_cp=0.001),
    "DS06": dict(svm_cost=300, svm_gamma=0.01,  mars_nk=17, mars_degree=2, mars_thresh=0.001, rf_mtry=5, rf_ntree=500,  rpart_minsplit=10, rpart_cp=0.001),
    "DS07": dict(svm_cost=300, svm_gamma=0.01,  mars_nk=10, mars_degree=1, mars_thresh=0.001, rf_mtry=7, rf_ntree=750,  rpart_minsplit=30, rpart_cp=0.001),
    "DS08": dict(svm_cost=300, svm_gamma=0.01,  mars_nk=17, mars_degree=2, mars_thresh=0.001, rf_mtry=7, rf_ntree=750,  rpart_minsplit=30, rpart_cp=0.001),
    "DS09": dict(svm_cost=10,  svm_gamma=0.01,  mars_nk=10, mars_degree=2, mars_thresh=0.001, rf_mtry=5, rf_ntree=750,  rpart_minsplit=30, rpart_cp=0.001),
    "DS10": dict(svm_cost=300, svm_gamma=0.01,  mars_nk=17, mars_degree=2, mars_thresh=0.001, rf_mtry=7, rf_ntree=500,  rpart_minsplit=10, rpart_cp=0.001),
    "DS11": dict(svm_cost=10,  svm_gamma=0.01,  mars_nk=17, mars_degree=1, mars_thresh=0.001, rf_mtry=7, rf_ntree=500,  rpart_minsplit=20, rpart_cp=0.001),
    "DS12": dict(svm_cost=300, svm_gamma=0.01,  mars_nk=17, mars_degree=1, mars_thresh=0.001, rf_mtry=7, rf_ntree=750,  rpart_minsplit=10, rpart_cp=0.001),
    "DS13": dict(svm_cost=150, svm_gamma=0.01,  mars_nk=17, mars_degree=2, mars_thresh=0.001, rf_mtry=7, rf_ntree=750,  rpart_minsplit=10, rpart_cp=0.001),
    "DS14": dict(svm_cost=150, svm_gamma=0.01,  mars_nk=17, mars_degree=2, mars_thresh=0.001, rf_mtry=7, rf_ntree=1500, rpart_minsplit=10, rpart_cp=0.001),
    "DS15": dict(svm_cost=300, svm_gamma=0.01,  mars_nk=17, mars_degree=2, mars_thresh=0.001, rf_mtry=5, rf_ntree=1500, rpart_minsplit=10, rpart_cp=0.001),
    "DS16": dict(svm_cost=300, svm_gamma=0.01,  mars_nk=17, mars_degree=2, mars_thresh=0.001, rf_mtry=7, rf_ntree=750,  rpart_minsplit=10, rpart_cp=0.001),
    "DS17": dict(svm_cost=300, svm_gamma=0.01,  mars_nk=17, mars_degree=2, mars_thresh=0.001, rf_mtry=7, rf_ntree=500,  rpart_minsplit=10, rpart_cp=0.001),
    "DS18": dict(svm_cost=300, svm_gamma=0.01,  mars_nk=17, mars_degree=2, mars_thresh=0.001, rf_mtry=5, rf_ntree=500,  rpart_minsplit=10, rpart_cp=0.001),
    "DS19": dict(svm_cost=150, svm_gamma=0.01,  mars_nk=17, mars_degree=1, mars_thresh=0.01,  rf_mtry=5, rf_ntree=500,  rpart_minsplit=10, rpart_cp=0.001),
    "DS20": dict(svm_cost=300, svm_gamma=0.01,  mars_nk=17, mars_degree=2, mars_thresh=0.001, rf_mtry=7, rf_ntree=500,  rpart_minsplit=10, rpart_cp=0.001),
    "DS21": dict(svm_cost=150, svm_gamma=0.001, mars_nk=17, mars_degree=2, mars_thresh=0.001, rf_mtry=7, rf_ntree=500,  rpart_minsplit=10, rpart_cp=0.001),
    "DS22": dict(svm_cost=150, svm_gamma=0.001, mars_nk=10, mars_degree=2, mars_thresh=0.001, rf_mtry=7, rf_ntree=500,  rpart_minsplit=10, rpart_cp=0.001),
    "DS23": dict(svm_cost=10,  svm_gamma=0.001, mars_nk=10, mars_degree=1, mars_thresh=0.001, rf_mtry=5, rf_ntree=500,  rpart_minsplit=10, rpart_cp=0.001),
    "DS24": dict(svm_cost=150, svm_gamma=0.01,  mars_nk=17, mars_degree=1, mars_thresh=0.001, rf_mtry=7, rf_ntree=750,  rpart_minsplit=10, rpart_cp=0.001),
}

# Paper Sec. 5 (p.169), verbatim: 50 Monte Carlo repetitions, 50% train / 25%
# test -- "Exceptionally, due to their size, in the case of the data sets DS21
# and DS22 we used 10% of the cases as training set and the following 5% as
# test set, and 20% of the cases as training set and the following 10% as test
# set for data sets DS23 and DS24."  Mirrors MC_SZ_TRAIN/MC_SZ_TEST in
# src/adapted/Exps.R.
MC_SPLIT = {f"DS{i:02d}": (0.5, 0.25) for i in range(1, 21)} | {
    "DS21": (0.10, 0.05), "DS22": (0.10, 0.05),
    "DS23": (0.20, 0.10), "DS24": (0.20, 0.10),
}

# Paper's published search grids (Annex 1) -- every OPT_PARAMS_TABLE7 value
# must fall inside these; used as the arithmetic sanity check on the
# transcription (test_paired_comparisons.py).
PARAM_GRIDS = {
    "svm_cost": {10, 150, 300}, "svm_gamma": {0.01, 0.001},
    "mars_nk": {10, 17}, "mars_degree": {1, 2}, "mars_thresh": {0.01, 0.001},
    "rf_mtry": {5, 7}, "rf_ntree": {500, 750, 1500},
    "rpart_minsplit": {10, 20, 30}, "rpart_cp": {0.1, 0.01, 0.001},
}

# Paper strategy code -> this project's raw_iterations_by_dataset_v2
# workflow suffix (mc.<family>_<SUFFIX>).
STRATEGY_CODE_MAP = {
    "U_B": "UNDERB", "U_T": "UNDERT", "U_TPhi": "UNDERTPhi",
    "O_B": "OVERB", "O_T": "OVERT", "O_TPhi": "OVERTPhi",
    "SM_B": "SMOTEB", "SM_T": "SMOTET", "SM_TPhi": "SMOTETPhi",
}

FAMILIES = ["lm", "svm", "mars", "rf", "rpart"]
