"""
================================================================================
09_tables_figures.py -- Table 1, Table 4 (information phenotypes), Fig. 3, Table S2
================================================================================
Inputs
  WORK_DIR: analysis_master_2h.parquet, edstays.parquet, patients.parquet,
            grid_search_results_v3/grid_search_summary_v3.csv
  OUT_DIR : Table_Subgroup_Performance_Evolution.csv (from 07_run_experiments.py)
Outputs (OUT_DIR, aggregate only)
  Table1_Baseline_Characteristics.csv
  Table4_Subgroup_Phenotypes.csv
  Figure3_Subgroup_Steps.png
  TableS2_Grid_Top15.csv

Information phenotypes (Methods, "Subgroup definitions and value of information"):
  anomalous trajectory  : AUROC not monotonically non-decreasing T0.5 -> T1.0 -> T2.0
  information dependent : dAUPRC >= 0.07 and dAUROC >= 0.04   (d = T2.0 - T0.5)
  precision dependent   : dAUPRC >= 0.07 and dAUROC <  0.04
  triage sufficient     : dAUPRC <  0.07 and dAUROC <  0.04
================================================================================
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import paths  # makes WORK_DIR the working directory

OUT = paths.OUT_DIR
TARGET = "early_critical_illness"
D_AUPRC_CUT, D_AUROC_CUT = 0.07, 0.04
WINDOWS = ["0.5h", "1h", "2h"]
PHENO_COLOR = {"Information dependent": "#0072B2", "Precision dependent": "#E69F00",
               "Triage sufficient": "#009E73", "Anomalous trajectory": "#D55E00"}


# -----------------------------------------------------------------------------
def table1():
    d = pd.read_parquet("analysis_master_2h.parquet")
    crit, non = d[d[TARGET] == 1], d[d[TARGET] == 0]
    rows = []

    def cont(label, col):
        a, b = crit[col].dropna(), non[col].dropna()
        q = lambda s: f"{s.median():.1f} [{s.quantile(.25):.1f}-{s.quantile(.75):.1f}]"
        rows.append([label, q(a), q(b), stats.mannwhitneyu(a, b).pvalue])

    def binary(label, m_crit, m_non, test=True):
        f = lambda s: f"{int(s.sum()):,} ({100 * s.mean():.1f}%)"
        p = stats.chi2_contingency([[m_crit.sum(), (~m_crit).sum()], [m_non.sum(), (~m_non).sum()]])[1] if test else np.nan
        rows.append([label, f(m_crit), f(m_non), p])

    cont("Age (years)", "anchor_age")
    cont("Maximum heart rate (bpm)", "heartrate_max")
    cont("Maximum respiratory rate (breaths/min)", "resprate_max")
    cont("Minimum oxygen saturation (%)", "o2sat_min")
    cont("Minimum systolic blood pressure (mmHg)", "sbp_min")
    cont("Initial white blood cell count (x10^3/uL)", "wbc")
    cont("Lactate (mmol/L)", "lactate")
    cont("Pain score", "triage_pain")
    binary("Sex: male", crit["gender"] == "M", non["gender"] == "M")
    rows.append(["ESI triage acuity (overall, all levels)", "", "",
                 stats.chi2_contingency(pd.crosstab(d["triage_acuity"], d[TARGET]))[1]])
    binary("ESI Level 1 (most urgent)", crit["triage_acuity"] == 1, non["triage_acuity"] == 1, test=False)
    binary("ESI Level 2", crit["triage_acuity"] == 2, non["triage_acuity"] == 2, test=False)
    binary("Arrival by ambulance", crit["arrival_transport"] == "AMBULANCE", non["arrival_transport"] == "AMBULANCE")
    binary("High-acuity triage (ESI 1-2)", crit["triage_acuity"] <= 2, non["triage_acuity"] <= 2)

    t = pd.DataFrame(rows, columns=["Variable", f"Critical (N={len(crit):,})",
                                    f"Noncritical (N={len(non):,})", "P_value"])
    t.to_csv(OUT / "Table1_Baseline_Characteristics.csv", index=False, encoding="utf-8-sig")
    print(f"Cohort N = {len(d):,}; early critical illness = {len(crit):,} ({100 * len(crit) / len(d):.1f}%)")
    print(t.to_string(index=False))


# -----------------------------------------------------------------------------
def phenotype(r):
    a = [r[f"AUROC_{w}"] for w in WINDOWS]
    if any(pd.isna(v) for v in a):
        return "Not estimable"
    if not (a[0] <= a[1] <= a[2]):
        return "Anomalous trajectory"
    if r["dAUPRC"] >= D_AUPRC_CUT and r["dAUROC"] >= D_AUROC_CUT:
        return "Information dependent"
    if r["dAUPRC"] >= D_AUPRC_CUT:
        return "Precision dependent"
    if r["dAUROC"] < D_AUROC_CUT:
        return "Triage sufficient"
    return "Unclassified (dAUROC >= 0.04, dAUPRC < 0.07)"


def table4_figure3():
    ev = pd.read_csv(OUT / "Table_Subgroup_Performance_Evolution.csv")
    w = ev.pivot_table(index="Subgroup", columns="Time_Window", values=["AUROC", "AUPRC"])
    t4 = pd.DataFrame(index=w.index)
    t4["N"] = ev[ev["Time_Window"] == "2h"].set_index("Subgroup")["N_Patients"]
    t4["Events"] = ev[ev["Time_Window"] == "2h"].set_index("Subgroup")["Events"]
    for m in ("AUROC", "AUPRC"):
        for win in WINDOWS:
            t4[f"{m}_{win}"] = w[(m, win)]
        t4[f"d{m}"] = t4[f"{m}_2h"] - t4[f"{m}_0.5h"]
        t4[f"d{m}_05_10"] = t4[f"{m}_1h"] - t4[f"{m}_0.5h"]
        t4[f"d{m}_10_20"] = t4[f"{m}_2h"] - t4[f"{m}_1h"]
    t4["Phenotype"] = t4.apply(phenotype, axis=1)
    order = {k: i for i, k in enumerate(PHENO_COLOR)}
    t4 = t4.reset_index().assign(_o=lambda x: x["Phenotype"].map(order).fillna(9)) \
        .sort_values(["_o", "dAUPRC"], ascending=[True, False]).drop(columns="_o")
    t4.to_csv(OUT / "Table4_Subgroup_Phenotypes.csv", index=False)
    print(t4[["Subgroup", "N", "Events", "AUROC_0.5h", "AUROC_1h", "AUROC_2h", "AUPRC_0.5h", "AUPRC_1h",
              "AUPRC_2h", "dAUROC", "dAUPRC", "Phenotype"]].round(4).to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    panels = [("dAUROC_05_10", "AUROC (T0.5 -> T1.0)"), ("dAUROC_10_20", "AUROC (T1.0 -> T2.0)"),
              ("dAUPRC_05_10", "AUPRC (T0.5 -> T1.0)"), ("dAUPRC_10_20", "AUPRC (T1.0 -> T2.0)")]
    for ax, (col, title) in zip(axes.ravel(), panels):
        d = t4.dropna(subset=[col]).sort_values(col)
        labels = [f"{s}  (N={int(n):,})" for s, n in zip(d["Subgroup"], d["N"])]
        ax.barh(labels, d[col], color=[PHENO_COLOR.get(p, "#777777") for p in d["Phenotype"]], height=0.6)
        for i, v in enumerate(d[col]):
            ax.text(v + (0.002 if v >= 0 else -0.002), i, f"{v:+.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=11)
        ax.axvline(0, color="#999999", lw=1)
        ax.margins(x=0.15)
        ax.set_title(title, fontweight="bold", fontsize=15)
        ax.set_xlabel("Change", fontsize=13)
        ax.grid(axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PHENO_COLOR.values()]
    fig.legend(handles, list(PHENO_COLOR), loc="lower center", ncol=4, fontsize=14, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "Figure3_Subgroup_Steps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
def table_s2():
    g = pd.read_csv("grid_search_results_v3/grid_search_summary_v3.csv")
    top = g.sort_values(["composite_score", "noise_ratio"], ascending=[False, True]).head(15)
    cols = ["combo_id", "min_cluster_size", "min_samples", "noise_ratio", "n_topics", "other_ratio",
            "size_cv", "coverage_ratio", "n_critical_covered", "missing_ccs", "mean_cohesion", "composite_score"]
    top[[c for c in cols if c in top.columns]].to_csv(OUT / "TableS2_Grid_Top15.csv", index=False)
    print(f"Grid search: {len(g)} configurations; top configuration {top.iloc[0]['combo_id']} "
          f"(composite score {top.iloc[0]['composite_score']})")


if __name__ == "__main__":
    table1()
    table4_figure3()
    table_s2()
