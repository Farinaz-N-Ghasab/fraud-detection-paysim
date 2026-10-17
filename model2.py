import sqlite3
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score


conn = sqlite3.connect("fraud.db")
df = pd.read_sql("""
    SELECT step, type, amount, oldbalanceOrg, oldbalanceDest, isFraud
    FROM transactions
    WHERE type IN ('TRANSFER', 'CASH_OUT')
""", conn)
conn.close()


df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
df["amount_eq_orig"] = (df["amount"] == df["oldbalanceOrg"]).astype(int)
df["dest_old_zero"] = (df["oldbalanceDest"] == 0).astype(int)


cut = df["step"].quantile(0.8)
train, test = df[df["step"] <= cut], df[df["step"] > cut]
y_test = test["isFraud"]


print("=== 1) Simple rules (no ML) ===")
rules = {
    "R1: amount == oldbalanceOrg": test["amount_eq_orig"] == 1,
    "R2: R1 and dest_old_zero": (test["amount_eq_orig"] == 1) & (test["dest_old_zero"] == 1),
}
for name, mask in rules.items():
    pred = mask.astype(int)
    print(name, "| precision:", round(precision_score(y_test, pred), 4),
          "| recall:", round(recall_score(y_test, pred), 4))



fr = train[train["isFraud"] == 1]
ok = train[train["isFraud"] == 0].sample(frac=0.1, random_state=42)
small = pd.concat([fr, ok])
w = (small["isFraud"] == 0).sum() / small["isFraud"].sum()


def fit_eval(cols):
    m = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                      scale_pos_weight=w, tree_method="hist", n_jobs=-1)
    m.fit(small[cols], small["isFraud"])
    p = m.predict_proba(test[cols])[:, 1]
    return round(average_precision_score(y_test, p), 4), m

all_cols = ["is_transfer", "amount", "oldbalanceOrg", "oldbalanceDest",
            "amount_eq_orig", "dest_old_zero"]


print("\n=== 2) Ablation (PR-AUC on test) ===")
base, model = fit_eval(all_cols)
print("all features:", base)
for c in all_cols:
    score, _ = fit_eval([x for x in all_cols if x != c])
    print(f"without {c}:", score)
score, _ = fit_eval(["amount_eq_orig"])
print("ONLY amount_eq_orig:", score)


print("\n=== 3) Feature importance (gain) ===")
imp = model.get_booster().get_score(importance_type="gain")
for k, v in sorted(imp.items(), key=lambda kv: -kv[1]):
    print(k, round(v, 1))