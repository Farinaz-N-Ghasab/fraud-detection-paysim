import sqlite3
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score


conn = sqlite3.connect("fraud.db")
df = pd.read_sql("""
    SELECT step, type, amount, oldbalanceOrg, newbalanceOrig,
           oldbalanceDest, newbalanceDest, isFraud
    FROM transactions
    WHERE type IN ('TRANSFER', 'CASH_OUT')
""", conn)
conn.close()


df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
df["amount_eq_orig"] = (df["amount"] == df["oldbalanceOrg"]).astype(int)
df["dest_old_zero"] = (df["oldbalanceDest"] == 0).astype(int)

safe = ["is_transfer", "amount", "oldbalanceOrg", "oldbalanceDest",
        "amount_eq_orig", "dest_old_zero"]
leaky = safe + ["newbalanceOrig", "newbalanceDest"]


cut = df["step"].quantile(0.8)
train, test = df[df["step"] <= cut], df[df["step"] > cut]
print("train:", len(train), "frauds:", train["isFraud"].sum())
print("test: ", len(test), "frauds:", test["isFraud"].sum())


def run(name, cols):
    w = (train["isFraud"] == 0).sum() / train["isFraud"].sum()
    m = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                      scale_pos_weight=w, tree_method="hist", n_jobs=-1)
    m.fit(train[cols], train["isFraud"])
    p = m.predict_proba(test[cols])[:, 1]
    pred = (p > 0.5).astype(int)
    print(f"\n{name}")
    print("PR-AUC   :", round(average_precision_score(test["isFraud"], p), 4))
    print("Precision:", round(precision_score(test["isFraud"], pred), 4))
    print("Recall   :", round(recall_score(test["isFraud"], pred), 4))


run("A) with newbalance columns (leaky)", leaky)
run("B) without newbalance columns (safe)", safe)