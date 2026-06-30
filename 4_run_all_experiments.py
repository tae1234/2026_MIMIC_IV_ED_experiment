"""
================================================================================
run_all_experiments.py
Description: Master script for training, evaluating, and extracting insights 
             from all Machine Learning and PyTorch Deep Learning models.
             Includes modules for:
             1. OOT Subgroup Time Trends (0.5h, 1h, 2h)
             2. Global SHAP Feature Importance
             3. Baseline ML Comparisons (LR vs RF vs XGB)
             4. PyTorch Residual MLP Deep Learning
             5. Uncertainty-based Selective Prediction (Risk-Coverage)
================================================================================
"""

import warnings
import copy
import polars as pl
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Scikit-Learn & XGBoost
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Fix seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# Helper: Data Loading & Out-Of-Time (OOT) Splitting
# =============================================================================
def load_and_split_data(window="2h", split_year=2180):
    """Loads the analysis master table and applies OOT split."""
    lf_edstays = pl.scan_parquet("edstays.parquet").select(["stay_id", "subject_id"]).with_columns(pl.col("stay_id").cast(pl.Int64, strict=False))
    lf_patients = pl.scan_parquet("patients.parquet").select(["subject_id", "anchor_year"]).with_columns(pl.col("subject_id").cast(pl.Int64, strict=False))
    df_year_lookup = lf_edstays.join(lf_patients, on="subject_id", how="inner").select(["stay_id", "anchor_year"]).collect().to_pandas()

    file_name = f"analysis_master_{window}.parquet"
    df = pl.scan_parquet(file_name).collect().to_pandas()

    if "anchor_year" in df.columns:
        df = df.drop(columns=["anchor_year"])
    df = pd.merge(df, df_year_lookup, on="stay_id", how="left")

    target = "early_critical_illness"
    features_cat = ["gender", "race", "arrival_transport", "cc_system"]
    features_num = [
        "anchor_age", "triage_acuity", "triage_pain",
        "sbp_min", "sbp_max", "heartrate_max", "resprate_max", "o2sat_min", "temperature_max",
        "wbc", "hb", "plt", "cr", "na", "k", "hco3", "lactate", "glucose",
        "wbc_measured", "hb_measured", "plt_measured", "cr_measured", "lactate_measured"
    ]

    avail_num = [c for c in features_num if c in df.columns]
    avail_cat = [c for c in features_cat if c in df.columns]

    for col in avail_num:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    train_mask = df["anchor_year"] <= split_year
    test_mask = df["anchor_year"] > split_year

    X_train = df.loc[train_mask, avail_num + avail_cat]
    y_train = df.loc[train_mask, target]
    X_test = df.loc[test_mask, avail_num + avail_cat]
    y_test = df.loc[test_mask, target]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), avail_num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), avail_cat),
    ])

    return X_train, X_test, y_train, y_test, df.loc[test_mask], preprocessor, avail_num, avail_cat

# =============================================================================
# Experiment 1: Subgroup Time Trends & Global SHAP (Step 13 & 14)
# =============================================================================
def evaluate_subgroup_time_trends():
    print("\n" + "="*80)
    print(" EXPERIMENT 1: Subgroup Performance Evolution & SHAP (0.5h, 1h, 2h)")
    print("="*80)

    time_windows = ["0.5h", "1h", "2h"]
    all_results = []

    xgb_2h_model = None
    X_test_2h = None
    prep_2h = None
    cols_num, cols_cat = None, None

    for w in time_windows:
        X_train, X_test, y_train, y_test, df_test, preprocessor, avail_num, avail_cat = load_and_split_data(window=w)

        model = Pipeline([
            ("prep", preprocessor),
            ("clf", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss"))
        ])

        print(f"Training [{w}] XGBoost model...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        if w == "2h":
            xgb_2h_model = model.named_steps["clf"]
            prep_2h = model.named_steps["prep"]
            X_test_2h = X_test
            cols_num, cols_cat = avail_num, avail_cat

        unique_systems = df_test['cc_system'].dropna().unique()
        for system in unique_systems:
            mask = df_test['cc_system'] == system
            if y_test[mask].nunique() < 2 or mask.sum() < 50:
                continue

            sub_y, sub_prob = y_test[mask], y_prob[mask]
            fpr, tpr, _ = roc_curve(sub_y, sub_prob)
            idx_90 = np.searchsorted(tpr, 0.90)

            all_results.append({
                "Subgroup": system, "Time_Window": w, "N_Patients": mask.sum(),
                "AUROC": roc_auc_score(sub_y, sub_prob),
                "AUPRC": average_precision_score(sub_y, sub_prob),
                "Brier": brier_score_loss(sub_y, sub_prob),
                "Spec_at_Sens_90": 1 - fpr[min(idx_90, len(fpr) - 1)]
            })

    # Save Trends
    df_final = pd.DataFrame(all_results)
    df_final['Time_Window'] = pd.Categorical(df_final['Time_Window'], categories=["0.5h", "1h", "2h"], ordered=True)
    df_final.sort_values(by=['Subgroup', 'Time_Window']).to_csv("Table_Subgroup_Performance_Evolution.csv", index=False)
    print(" -> Subgroup evolution exported to 'Table_Subgroup_Performance_Evolution.csv'")

    # Generate SHAP
    if xgb_2h_model is not None:
        print("Generating Global SHAP Beeswarm Plot...")
        X_test_proc = prep_2h.transform(X_test_2h)
        cat_feats = prep_2h.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cols_cat)
        all_feats = list(cols_num) + list(cat_feats)

        explainer = shap.TreeExplainer(xgb_2h_model)
        X_sample = pd.DataFrame(X_test_proc, columns=all_feats).sample(n=min(30000, len(X_test_proc)), random_state=42)
        shap_values = explainer(X_sample)

        keep_indices = [i for i, f in enumerate(all_feats) if not f.startswith(("cc_system_", "race_", "gender_", "arrival_transport_")) and not f.endswith("_measured")]

        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(shap_values[:, keep_indices], max_display=15, show=False, plot_size=(11, 8))
        plt.title('Global Feature Importance (SHAP)', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('Figure4_Global_SHAP_Beeswarm.png', dpi=300, bbox_inches='tight')
        print(" -> SHAP plot exported to 'Figure4_Global_SHAP_Beeswarm.png'")

# =============================================================================
# Experiment 2: ML Baselines Comparison (Step 16)
# =============================================================================
def compare_ml_baselines():
    print("\n" + "="*80)
    print(" EXPERIMENT 2: ML Baselines Comparison (LR vs RF vs XGB) for 2h Window")
    print("="*80)

    X_train, X_test, y_train, y_test, _, preprocessor, _, _ = load_and_split_data(window="2h")

    models = {
        "Logistic Regression": Pipeline([("prep", preprocessor), ("clf", CalibratedClassifierCV(LogisticRegression(penalty="l2", class_weight="balanced", random_state=42), method="sigmoid", cv=5))]),
        "Random Forest": Pipeline([("prep", preprocessor), ("clf", CalibratedClassifierCV(RandomForestClassifier(n_estimators=300, max_depth=15, class_weight="balanced", n_jobs=-1, random_state=42), method="isotonic", cv=5))]),
        "XGBoost": Pipeline([("prep", preprocessor), ("clf", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss"))])
    }

    results = []
    predictions = {'target': y_test.values}

    for name, model in models.items():
        print(f"Training [{name}]...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        predictions[f'pred_{name.split()[0].lower()}'] = y_prob

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results.append({
            "Model": name, "AUROC": roc_auc_score(y_test, y_prob), "AUPRC": average_precision_score(y_test, y_prob),
            "Brier": brier_score_loss(y_test, y_prob), "Spec@90": 1 - fpr[min(np.searchsorted(tpr, 0.90), len(fpr)-1)]
        })

    print("\n--- ML Baseline Results ---")
    for res in results:
        print(f" {res['Model']:<20} | AUROC: {res['AUROC']:.4f} | AUPRC: {res['AUPRC']:.4f}")

    pd.DataFrame(predictions).to_csv("model_predictions_2h.csv", index=False)
    print(" -> Predictions exported to 'model_predictions_2h.csv'")

# =============================================================================
# Experiment 3: PyTorch Deep Learning (Step 15)
# =============================================================================
class EHRDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(drop))
    def forward(self, x): return self.net(x) + x

class TabularResMLP(nn.Module):
    def __init__(self, in_dim, hid_dim=128, blocks=2, drop=0.3):
        super().__init__()
        self.in_layer = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.BatchNorm1d(hid_dim), nn.GELU(), nn.Dropout(drop))
        self.blocks = nn.ModuleList([ResidualBlock(hid_dim, drop) for _ in range(blocks)])
        self.out_layer = nn.Linear(hid_dim, 1)
    def forward(self, x):
        out = self.in_layer(x)
        for b in self.blocks: out = b(out)
        return self.out_layer(out)

def run_pytorch_dl():
    print("\n" + "="*80)
    print(" EXPERIMENT 3: PyTorch Deep Learning (Residual MLP) for 2h Window")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_full, X_test_raw, y_train_full, y_test, _, preprocessor, _, _ = load_and_split_data(window="2h")

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full)

    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    train_loader = DataLoader(EHRDataset(X_train, y_train), batch_size=1024, shuffle=True)
    val_loader = DataLoader(EHRDataset(X_val, y_val), batch_size=1024, shuffle=False)
    test_loader = DataLoader(EHRDataset(X_test, y_test), batch_size=1024, shuffle=False)

    model = TabularResMLP(in_dim=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_auprc, patience = 0, 0
    best_wts = copy.deepcopy(model.state_dict())

    print(f"Training on {device} (Max Epochs: 30)...")
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
        if val_auprc > best_val_auprc:
            best_val_auprc, patience, best_wts = val_auprc, 0, copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= 5:
                print(f"  -> Early Stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_wts)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for bx, _ in test_loader:
            test_preds.extend(torch.sigmoid(model(bx.to(device))).cpu().numpy())

    y_prob = np.array(test_preds).flatten()
    print(f"--- PyTorch Results ---")
    print(f" AUROC: {roc_auc_score(y_test, y_prob):.4f} | AUPRC: {average_precision_score(y_test, y_prob):.4f}")

    pd.DataFrame({'pred_resmlp': y_prob}).to_csv("pytorch_predictions_2h.csv", index=False)
    print(" -> Predictions exported to 'pytorch_predictions_2h.csv'")

# =============================================================================
# Experiment 4: Selective Prediction (Risk-Coverage) (Step 11)
# =============================================================================
def run_selective_prediction():
    print("\n" + "="*80)
    print(" EXPERIMENT 4: Uncertainty-Based Selective Prediction (Risk-Coverage)")
    print("="*80)

    # Note: Using T0 features only to simulate initial triage decision
    X_train, X_test, y_train, y_test, _, preprocessor, _, _ = load_and_split_data(window="2h")

    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), random_state=42)

    print("Training T0 XGBoost model...")
    # Fit the preprocessor first, then resolve T0 feature column indices.
    X_train_full = preprocessor.fit_transform(X_train)
    X_test_full = preprocessor.transform(X_test)

    # Extract only T0 features (Age, Acuity, Pain)
    t0_idx = [i for i, f in enumerate(preprocessor.get_feature_names_out()) if "anchor_age" in f or "triage_" in f]

    X_train_proc = X_train_full[:, t0_idx]
    X_test_proc = X_test_full[:, t0_idx]

    xgb.fit(X_train_proc, y_train)
    y_pred_proba = xgb.predict_proba(X_test_proc)[:, 1]

    p = y_pred_proba + 1e-9
    q = 1.0 - y_pred_proba + 1e-9
    entropy = -(p * np.log2(p) + q * np.log2(q))

    print(f"\n{'Coverage':<15} | {'Deferral Rate':<15} | {'Confident Subset AUROC'}")
    print("-" * 55)

    for cov in [1.0, 0.9, 0.8, 0.7, 0.6]:
        mask = entropy <= np.percentile(entropy, cov * 100)
        conf_auroc = roc_auc_score(y_test[mask], y_pred_proba[mask])
        cov_str = f"{(cov*100):.0f}% (Confident)"
        defer_str = f"{((1.0-cov)*100):.0f}% (Lab req)"
        print(f"{cov_str:<15} | {defer_str:<15} | {conf_auroc:.4f}")

# =============================================================================
# Master Execution
# =============================================================================
if __name__ == "__main__":
    evaluate_subgroup_time_trends()
    compare_ml_baselines()
    run_pytorch_dl()
    run_selective_prediction()
    print("\n🎉 All experiments successfully executed!")