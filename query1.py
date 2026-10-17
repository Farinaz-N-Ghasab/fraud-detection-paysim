import sqlite3
import pandas as pd

conn = sqlite3.connect("fraud.db")

q = """
SELECT type, COUNT(*) As n, SUM(isFraud) As frauds
FROM transactions
GROUP BY type
"""

print(pd.read_sql(q, conn))
conn.close()