import pandas as pd
import sqlite3

conn = sqlite3.connect("fraud.db")

for i, chunk in enumerate(pd.read_csv("paysim.csv", chunksize=500_000)):
    mode = "replace"if i == 0 else "append"
    chunk.to_sql("transactions", conn, if_exists=mode, index=False)
    print("chunk", i, "loaded")


conn.close()  