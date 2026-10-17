# Fraud Detection on 6.3M Transactions (PaySim)

Python + SQL (SQLite) + XGBoost project on the synthetic PaySim dataset.
The goal was not only to build a model, but to understand *why* it scores so high.


## Data
- PaySim (synthetic mobile-money transactions): 6,362,620 rows, 8,213 frauds (~0.13%).
- Not included in this repo (too large). Download paysim.csv, put it in the project folder, then run load_db.py.
- The data is simulated, so results do not represent a real bank.


## How to run
pip install -r requirements.txt
python load_db.py     # CSV -> SQLite (fraud.db)
python query1.py      # fraud count by transaction type
python query2.py      # fraud rate, amounts
python query3.py      # balance patterns
python model1.py      # XGBoost with / without newbalance columns
python model2.py      # simple rules, ablation, feature importance
python model3.py      # harder setting (shortcut feature removed)
python model4.py      # threshold chosen by cost


## Findings
1. Fraud only happens in TRANSFER (0.77% of them) and CASH_OUT (0.18%).
2. Fraud amounts are larger: mean ~1.47M vs ~314K for normal transactions.
3. Time-based split (first 80% of time = train, last 20% = test).
4. XGBoost reaches PR-AUC 0.9998, but a one-line rule
   (`amount == oldbalanceOrg`) already gets precision 1.0 and recall 0.975.
5. Ablation: removing amount_eq_orig drops PR-AUC to ~0.88. The model mostly
   relies on this one shortcut (the simulator's fraudsters always empty the account).
6. Without the shortcut:
   - keeping oldbalanceOrg: PR-AUC 0.877, precision 0.244, recall 0.994
   - also removing oldbalanceOrg: PR-AUC 0.390, precision 0.097, recall 0.757
7. Threshold by cost (blocking a good customer assumed to cost 50, an assumption):
   in the hardest setting, threshold 0.2 cut test cost from ~203M to ~81M.
   Cost with no model: ~6.7B.


## Limitations
- Synthetic data, so the 99% score is misleading.
- The cost of a false alarm is assumed, and results depend on it.
- Only XGBoost was tried; no logistic regression baseline or SHAP yet.


## Stack
Python, pandas, SQLite, scikit-learn, XGBoost