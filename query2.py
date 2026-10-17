import sqlite3 
import pandas as pd

conn = sqlite3.connect("fraud.db")

q1 = """
SELECT type, COUNT(*) AS n, SUM(isFraud) AS frauds,
       ROUND(100.0 * SUM(isFraud) / COUNT(*), 3) AS fraud_pct
FROM transactions
WHERE type IN ('TRANSFER', 'CASH_OUT')
GROUP BY type
"""


q2 = """
SELECT isFraud, COUNT(*) AS n,
       ROUND(AVG(amount), 0) AS avg_amount,
       ROUND(MAX(amount), 0) AS max_amount
FROM transactions
WHERE type IN ('TRANSFER', 'CASH_OUT')
GROUP BY isFraud
"""


q3 = """
SELECT isFraud, COUNT(*) AS n,
       SUM(CASE WHEN newbalanceOrig = 0 THEN 1 ELSE 0 END) AS emptied
FROM transactions
WHERE type IN ('TRANSFER', 'CASH_OUT')
GROUP BY isFraud
"""

for q in [q1, q2, q3]:
    print(pd.read_sql(q, conn))
    print()

conn.close()