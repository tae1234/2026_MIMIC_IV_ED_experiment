"""
================================================================================
07_run_experiments.py -- model training and evaluation
================================================================================
Inputs (WORK_DIR): analysis_master_{0.5h,1h,2h}.parquet, edstays.parquet,
                   patients.parquet
Experiments
  1. XGBoost at each observation window (T0.5, T1.0, T2.0): overall and
     subgroup (13 CEDIS categories) performance; global SHAP at T2.0.
  2. LR, RF and XGBoost at T2.0.
  3. PyTorch residual MLP (ResMLP) at T2.0.
  4. Summary: Table 2 (with DeLong p-values vs XGBoost), Table 3, Fig. 2.

Split: anchor_year <= 2180 -> training, > 2180 -> independent test set.
Specificity at 90% sensitivity is read from the test-set ROC curve.

Outputs
  OUT_DIR  (aggregate): Table2_models_T2.csv, Table3_XGB_windows.csv,
           Table_Subgroup_Performance_Evolution.csv,
           Figure2_ROC_PR.png, Figure4_Global_SHAP_Beeswarm.png
  WORK_DIR (row-level, never share): Subgroup_Predictions_Raw.csv,
           model_predictions_2h.csv, pytorch_predictions_2h.csv,
           xgboost_t2_final.pkl (used by 08_fairness_audit.py)
================================================================================
"""
import copy
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import shap
from scipy import stats

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Scikit-learn & XGBoost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import paths  # makes WORK_DIR the working directory

warnings.filterwarnings("ignore")

# Fix seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

OUT = paths.OUT_DIR
SPLIT_YEAR = 2180
TIME_WINDOWS = ["0.5h", "1h", "2h"]
TARGET = "early_critical_illness"
FEATURES_CAT = ["gender", "race", "arrival_transport", "cc_system"]
FEATURES_NUM = [
    "anchor_age", "triage_acuity", "triage_pain",
    "sbp_min", "sbp_max", "heartrate_max", "resprate_max", "o2sat_min", "temperature_max",
    "wbc", "hb", "plt", "cr", "na", "k", "hco3", "lactate", "glucose",
    "wbc_measured", "hb_measured", "plt_measured", "cr_measured", "lactate_measured",
]


# =============================================================================
# Helpers
# =============================================================================
def load_and_split_data(window="2h", split_year=SPLIT_YEAR):
    """Loads the analysis master table and applies the anchor-year split."""
    lf_edstays = pl.scan_parquet("edstays.parquet").select(["stay_id", "subject_id"]).with_columns(
        pl.col("stay_id").cast(pl.Int64, strict=False))
    lf_patients = pl.scan_parquet("patients.parquet").select(["subject_id", "anchor_year"]).with_columns(
        pl.col("subject_id").cast(pl.Int64, strict=False))
    df_year_lookup = lf_edstays.join(lf_patients, on="subject_id", how="inner").select(
        ["stay_id", "anchor_year"]).collect().to_pandas()

    df = pl.scan_parquet(f"analysis_master_{window}.parquet").collect().to_pandas()
    if "anchor_year" in df.columns:
        df = df.drop(columns=["anchor_year"])
    df = pd.merge(df, df_year_lookup, on="stay_id", how="left")

    avail_num = [c for c in FEATURES_NUM if c in df.columns]
    avail_cat = [c for c in FEATURES_CAT if c in df.columns]
    for col in avail_num:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    train_mask = df["anchor_year"] <= split_year
    test_mask = df["anchor_year"] > split_year

    X_train = df.loc[train_mask, avail_num + avail_cat]
    y_train = df.loc[train_mask, TARGET]
    X_test = df.loc[test_mask, avail_num + avail_cat]
    y_test = df.loc[test_mask, TARGET]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), avail_num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), avail_cat),
    ])
    return X_train, X_test, y_train, y_test, df.loc[test_mask], preprocessor, avail_num, avail_cat


def make_xgb():
    return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=5,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss")


def spec_at_sens90(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return 1 - fpr[min(np.searchsorted(tpr, 0.90), len(fpr) - 1)]


def perf(y, p):
    return {"AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p), "Spec_at_Sens_90": spec_at_sens90(y, p)}


def delong_p(y, p1, p2):
    """Two-sided DeLong test for two correlated AUROCs (fast algorithm, Sun & Xu 2014)."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v01, v10 = [], [], []
    for p in (np.asarray(p1), np.asarray(p2)):
        tx = stats.rankdata(p[pos])
        ty = stats.rankdata(p[neg])
        tz = stats.rankdata(np.concatenate([p[pos], p[neg]]))
        aucs.append(tz[:m].sum() / m / n - (m + 1) / 2 / n)
        v01.append((tz[:m] - tx) / n)
        v10.append(1.0 - (tz[m:] - ty) / m)
    cov = np.cov(np.vstack(v01)) / m + np.cov(np.vstack(v10)) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = abs(aucs[0] - aucs[1]) / np.sqrt(var)
    return float(2 * stats.norm.sf(z))


# =============================================================================
# Experiment 1: XGBoost across observation windows, subgroups, SHAP
# =============================================================================
def evaluate_subgroup_time_trends():
    print("\n" + "=" * 80)
    print(" EXPERIMENT 1: Subgroup Performance Evolution & SHAP (0.5h, 1h, 2h)")
    print("=" * 80)

    all_results, overall, raw_preds = [], [], []
    xgb_2h_pipeline, X_test_2h, cols_num, cols_cat = None, None, None, None

    for w in TIME_WINDOWS:
        X_train, X_test, y_train, y_test, df_test, preprocessor, avail_num, avail_cat = load_and_split_data(window=w)
        model = Pipeline([("prep", preprocessor), ("clf", make_xgb())])

        print(f"Training [{w}] XGBoost model...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        overall.append({"Window": w, "N_train": len(y_train), "N_test": len(y_test),
                        "Events_test": int(y_test.sum()), **perf(y_test, y_prob)})
        raw_preds.append(pd.DataFrame({"stay_id": df_test["stay_id"].values, "cc_system": df_test["cc_system"].values,
                                       "Time_Window": w, "y_true": y_test.values, "y_prob": y_prob}))
        if w == "2h":
            xgb_2h_pipeline, X_test_2h = model, X_test
            cols_num, cols_cat = avail_num, avail_cat

        for system in df_test["cc_system"].dropna().unique():
            mask = df_test["cc_system"] == system
            if y_test[mask].nunique() < 2 or mask.sum() < 50:
                continue
            sub_y, sub_prob = y_test[mask], y_prob[mask]
            all_results.append({"Subgroup": system, "Time_Window": w, "N_Patients": mask.sum(),
                                "Events": int(sub_y.sum()), **perf(sub_y, sub_prob)})

    df_final = pd.DataFrame(all_results)
    df_final["Time_Window"] = pd.Categorical(df_final["Time_Window"], categories=TIME_WINDOWS, ordered=True)
    df_final.sort_values(by=["Subgroup", "Time_Window"]).to_csv(OUT / "Table_Subgroup_Performance_Evolution.csv", index=False)
    pd.DataFrame(overall).to_csv(OUT / "Table3_XGB_windows.csv", index=False)
    pd.concat(raw_preds, ignore_index=True).to_csv("Subgroup_Predictions_Raw.csv", index=False)
    joblib.dump(xgb_2h_pipeline, "xgboost_t2_final.pkl")
    print(pd.DataFrame(overall).round(4).to_string(index=False))

    # Global SHAP (Fig. 4): demographic/categorical and missingness indicators not displayed
    print("Generating Global SHAP Beeswarm Plot...")
    prep_2h, xgb_2h = xgb_2h_pipeline.named_steps["prep"], xgb_2h_pipeline.named_steps["clf"]
    X_test_proc = prep_2h.transform(X_test_2h)
    cat_feats = prep_2h.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cols_cat)
    all_feats = list(cols_num) + list(cat_feats)

    explainer = shap.TreeExplainer(xgb_2h)
    X_sample = pd.DataFrame(X_test_proc, columns=all_feats).sample(n=min(30000, len(X_test_proc)), random_state=42)
    shap_values = explainer(X_sample)
    keep_indices = [i for i, f in enumerate(all_feats)
                    if not f.startswith(("cc_system_", "race_", "gender_", "arrival_transport_"))
                    and not f.endswith("_measured")]

    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values[:, keep_indices], max_display=15, show=False, plot_size=(11, 8))
    plt.title("Global Feature Importance (SHAP)", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(OUT / "Figure4_Global_SHAP_Beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close("all")


# =============================================================================
# Experiment 2: LR, RF and XGBoost at T2.0
# =============================================================================
def compare_ml_baselines():
    print("\n" + "=" * 80)
    print(" EXPERIMENT 2: ML Baselines Comparison (LR vs RF vs XGB) for 2h Window")
    print("=" * 80)

    X_train, X_test, y_train, y_test, df_test, preprocessor, _, _ = load_and_split_data(window="2h")

    models = {
        "Logistic Regression": Pipeline([("prep", preprocessor), ("clf", CalibratedClassifierCV(
            LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
                               class_weight="balanced", random_state=42),
            method="sigmoid", cv=5))]),
        "Random Forest": Pipeline([("prep", preprocessor), ("clf", CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=300, max_depth=15, class_weight="balanced",
                                   n_jobs=-1, random_state=42),
            method="isotonic", cv=5))]),
        "XGBoost": Pipeline([("prep", preprocessor), ("clf", make_xgb())]),
    }

    predictions = {"stay_id": df_test["stay_id"].values, "target": y_test.values}
    for name, model in models.items():
        print(f"Training [{name}]...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        predictions[f"pred_{name.split()[0].lower()}"] = y_prob
        r = perf(y_test, y_prob)
        print(f" {name:<20} | AUROC: {r['AUROC']:.4f} | AUPRC: {r['AUPRC']:.4f} | Brier: {r['Brier']:.4f}")

    pd.DataFrame(predictions).to_csv("model_predictions_2h.csv", index=False)


# =============================================================================
# Experiment 3: PyTorch residual MLP at T2.0
# =============================================================================
class EHRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(drop))

    def forward(self, x):
        return self.net(x) + x


class TabularResMLP(nn.Module):
    """Input layer (128 units) followed by two residual blocks of width 128."""
    def __init__(self, in_dim, hid_dim=128, blocks=2, drop=0.3):
        super().__init__()
        self.in_layer = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.BatchNorm1d(hid_dim), nn.GELU(), nn.Dropout(drop))
        self.blocks = nn.ModuleList([ResidualBlock(hid_dim, drop) for _ in range(blocks)])
        self.out_layer = nn.Linear(hid_dim, 1)

    def forward(self, x):
        out = self.in_layer(x)
        for b in self.blocks:
            out = b(out)
        return self.out_layer(out)


def run_pytorch_dl():
    print("\n" + "=" * 80)
    print(" EXPERIMENT 3: PyTorch Deep Learning (Residual MLP) for 2h Window")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_full, X_test_raw, y_train_full, y_test, df_test, preprocessor, _, _ = load_and_split_data(window="2h")

    # 10% of the training set is held out for early stopping
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full)

    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    train_loader = DataLoader(EHRDataset(X_train, y_train), batch_size=1024, shuffle=True)
    val_loader = DataLoader(EHRDataset(X_val, y_val), batch_size=1024, shuffle=False)
    test_loader = DataLoader(EHRDataset(X_test, y_test), batch_size=1024, shuffle=False)

    model = TabularResMLP(in_dim=X_train.shape[1]).to(device)
    # Class weighting in the binary cross-entropy loss (pos_weight = negatives / positives)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_auprc, patience = 0, 0
    best_wts = copy.deepcopy(model.state_dict())

    print(f"Training on {device} (max 30 epochs, early stopping on validation AUPRC, patience 5)...")
    for epoch in range(30):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                val_preds.extend(torch.sigmoid(model(bx.to(device))).cpu().numpy())
                val_trues.extend(by.numpy())
        val_auprc = average_precision_score(val_trues, val_preds)
        print(f"  epoch {epoch + 1:>2} -- validation AUPRC {val_auprc:.4f}")
        if val_auprc > best_val_auprc:
            best_val_auprc, patience, best_wts = val_auprc, 0, copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= 5:
                print(f"  -> Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_wts)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for bx, _ in test_loader:
            test_preds.extend(torch.sigmoid(model(bx.to(device))).cpu().numpy())

    y_prob = np.array(test_preds).flatten()
    print(f" ResMLP | AUROC: {roc_auc_score(y_test, y_prob):.4f} | AUPRC: {average_precision_score(y_test, y_prob):.4f}")
    pd.DataFrame({"stay_id": df_test["stay_id"].values, "target": y_test.values,
                  "pred_resmlp": y_prob}).to_csv("pytorch_predictions_2h.csv", index=False)


# =============================================================================
# Summary: Table 2 and Figure 2
# =============================================================================
def summarize_t2():
    ml = pd.read_csv("model_predictions_2h.csv")
    dl = pd.read_csv("pytorch_predictions_2h.csv")
    d = ml.merge(dl[["stay_id", "pred_resmlp"]], on="stay_id", how="inner", validate="one_to_one")
    assert len(d) == len(ml) == len(dl), "prediction files are not aligned"
    y = d["target"].values

    cols = {"Logistic Regression": "pred_logistic", "Random Forest": "pred_random",
            "PyTorch (ResMLP)": "pred_resmlp", "XGBoost": "pred_xgboost"}
    rows = []
    for name, c in cols.items():
        p = "Reference" if name == "XGBoost" else delong_p(y, d[c].values, d["pred_xgboost"].values)
        rows.append({"Model": name, **perf(y, d[c].values), "DeLong_p_vs_XGBoost": p})
    t2 = pd.DataFrame(rows)
    t2.to_csv(OUT / "Table2_models_T2.csv", index=False)
    print("\nTable 2 (T2.0, test set)")
    print(t2.round(4).to_string(index=False))

    colors = {"Logistic Regression": "#999999", "Random Forest": "#2ca02c",
              "PyTorch (ResMLP)": "#1f77b4", "XGBoost": "#d62728"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for name, c in cols.items():
        fpr, tpr, _ = roc_curve(y, d[c].values)
        prec, rec, _ = precision_recall_curve(y, d[c].values)
        ax[0].plot(fpr, tpr, color=colors[name], lw=1.8,
                   label=f"{name} (AUROC = {roc_auc_score(y, d[c].values):.4f})")
        ax[1].plot(rec, prec, color=colors[name], lw=1.8,
                   label=f"{name} (AUPRC = {average_precision_score(y, d[c].values):.4f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance (0.5000)")
    ax[1].axhline(y.mean(), color="k", ls="--", lw=1.5, label=f"Baseline ({y.mean():.4f})")
    ax[0].set(xlabel="False Positive Rate (1 - Specificity)", ylabel="True Positive Rate (Sensitivity)",
              title="A. Receiver Operating Characteristic (ROC) Curve", xlim=(0, 1), ylim=(0, 1))
    ax[1].set(xlabel="Recall (Sensitivity)", ylabel="Precision (Positive Predictive Value)",
              title="B. Precision-Recall (PR) Curve", xlim=(0, 1), ylim=(0, 1))
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(loc="lower right", fontsize=8)
    ax[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "Figure2_ROC_PR.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    evaluate_subgroup_time_trends()
    compare_ml_baselines()
    run_pytorch_dl()
    summarize_t2()
