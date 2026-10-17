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
df["dest_old_zero"] = (df["oldbalanceDest"] == 0).astype(int)


cut = df["step"].quantile(0.8)
train, test = df[df["step"] <= cut], df[df["step"] > cut]
y_test = test["isFraud"]
w = (train["isFraud"] == 0).sum() / train["isFraud"].sum()


def run(name, cols):
    m = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                      scale_pos_weight=w, tree_method="hist", n_jobs=-1)
    m.fit(train[cols], train["isFraud"])
    p = m.predict_proba(test[cols])[:, 1]
    pred = (p > 0.5).astype(int)
    print(f"\n{name}")
    print("PR-AUC   :", round(average_precision_score(y_test, p), 4))
    print("Precision:", round(precision_score(y_test, pred), 4))
    print("Recall   :", round(recall_score(y_test, pred), 4))


run("H1: no amount_eq_orig",
    ["is_transfer", "amount", "oldbalanceOrg", "oldbalanceDest", "dest_old_zero"])
run("H2: no amount_eq_orig, no oldbalanceOrg",
    ["is_transfer", "amount", "oldbalanceDest", "dest_old_zero"])