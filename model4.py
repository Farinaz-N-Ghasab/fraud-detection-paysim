import sqlite3
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score


FP_COST = 50 


conn = sqlite3.connect("fraud.db")
df = pd.read_sql("""
    SELECT step, type, amount, oldbalanceOrg, oldbalanceDest, isFraud
    FROM transactions
    WHERE type IN ('TRANSFER', 'CASH_OUT')
""", conn)
conn.close()


df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
df["dest_old_zero"] = (df["oldbalanceDest"] == 0).astype(int)
df["amount_eq_orig"] = (df["amount"] == df["oldbalanceOrg"]).astype(int)


cut_test = df["step"].quantile(0.8)
train_all = df[df["step"] <= cut_test]
test = df[df["step"] > cut_test]
cut_val = train_all["step"].quantile(0.75)
fit = train_all[train_all["step"] <= cut_val]
val = train_all[train_all["step"] > cut_val]
print("frauds -> fit:", fit["isFraud"].sum(), "| val:", val["isFraud"].sum(),
      "| test:", test["isFraud"].sum())


def total_cost(part, p, t):
    pred = p > t
    y = part["isFraud"].values
    amt = part["amount"].values
    missed = amt[(~pred) & (y == 1)].sum()      
    false_alarms = ((pred) & (y == 0)).sum()    
    return missed + FP_COST * false_alarms


def experiment(name, cols):
    w = (fit["isFraud"] == 0).sum() / fit["isFraud"].sum()
    m = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                      scale_pos_weight=w, tree_method="hist", n_jobs=-1)
    m.fit(fit[cols], fit["isFraud"])
    p_val = m.predict_proba(val[cols])[:, 1]
    p_test = m.predict_proba(test[cols])[:, 1]

    
    ts = np.arange(0.05, 1.0, 0.05)
    costs = [total_cost(val, p_val, t) for t in ts]
    best_t = ts[int(np.argmin(costs))]

    pred = (p_test > best_t).astype(int)
    print(f"\n{name}")
    print("best threshold (from validation):", round(best_t, 2))
    print("test cost at 0.5   :", round(total_cost(test, p_test, 0.5)))
    print("test cost at best  :", round(total_cost(test, p_test, best_t)))
    print("test precision/recall at best:",
          round(precision_score(test["isFraud"], pred), 4), "/",
          round(recall_score(test["isFraud"], pred), 4))


nothing = test.loc[test["isFraud"] == 1, "amount"].sum()
print("test cost with NO model:", round(nothing))


experiment("H1: no amount_eq_orig",
           ["is_transfer", "amount", "oldbalanceOrg", "oldbalanceDest", "dest_old_zero"])
experiment("H2: no amount_eq_orig, no oldbalanceOrg",
           ["is_transfer", "amount", "oldbalanceDest", "dest_old_zero"])