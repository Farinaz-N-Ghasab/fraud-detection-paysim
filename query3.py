import sqlite3
import pandas as pd

conn = sqlite3.connect("fraud.db")


q1 = """
SELECT isFraud, COUNT(*) AS n,
       SUM(CASE WHEN oldbalanceDest = 0 AND newbalanceDest = 0 THEN 1 ELSE 0 END) AS dest_zero,
       ROUND(100.0 * SUM(CASE WHEN oldbalanceDest = 0 AND newbalanceDest = 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS dest_zero_pct
FROM transactions
WHERE type IN ('TRANSFER', 'CASH_OUT') AND amount > 0
GROUP BY isFraud
"""


q2 = """
SELECT isFraud, COUNT(*) AS n,
       SUM(CASE WHEN ABS(oldbalanceDest + amount - newbalanceDest) > 0.01 THEN 1 ELSE 0 END) AS dest_mismatch,
       ROUND(100.0 * SUM(CASE WHEN ABS(oldbalanceDest + amount - newbalanceDest) > 0.01 THEN 1 ELSE 0 END) / COUNT(*), 2) AS mismatch_pct
FROM transactions
WHERE type IN ('TRANSFER', 'CASH_OUT')
GROUP BY isFraud
"""


q3 = """
SELECT isFraud, COUNT(*) AS n,
       SUM(CASE WHEN ABS(oldbalanceOrg - amount - newbalanceOrig) > 0.01 THEN 1 ELSE 0 END) AS orig_mismatch,
       ROUND(100.0 * SUM(CASE WHEN ABS(oldbalanceOrg - amount - newbalanceOrig) > 0.01 THEN 1 ELSE 0 END) / COUNT(*), 2) AS mismatch_pct
FROM transactions
WHERE type IN ('TRANSFER', 'CASH_OUT')
GROUP BY isFraud
"""


for q in [q1, q2, q3]:
    print(pd.read_sql(q, conn))
    print()

conn.close()