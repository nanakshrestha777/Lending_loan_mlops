# import pandas as pd
# from sqlalchemy import create_engine, text
# import os

# def ingest_data():
#     print("🚀 [Airflow] Starting Ingestion (Standard Mode)...")

#     # --- CONFIGURATION ---
#     DB_USER = 'admin'
#     DB_PASS = 'admin'
#     DB_HOST = '127.0.0.1'
#     DB_PORT = '3306'
#     TARGET_DB = 'lending_club'
#     csv_path = "/home/nanak/mlops/data/raw/loan.csv"

#     # 1. Create DB
#     system_conn = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/mysql"
#     try:
#         sys_eng = create_engine(system_conn)
#         with sys_eng.connect() as conn:
#             conn.execution_options(isolation_level="AUTOCOMMIT")
#             conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {TARGET_DB};"))
#     except Exception as e:
#         print(f"⚠️ Warning: {e}")

#     # 2. Connect
#     connection_str = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{TARGET_DB}"
#     chunk_size = 10000
    
#     cols_to_keep = [
#         'loan_status', 'annual_inc', 'dti', 'grade', 'int_rate', 
#         'home_ownership', 'purpose', 'term', 'loan_amnt',
#         'issue_d', 'addr_state', 'verification_status', 'emp_length', 'sub_grade'
#     ]

#     try:
#         # Create Engine
#         engine = create_engine(connection_str)
        
#         if not os.path.exists(csv_path):
#             raise FileNotFoundError(f"CSV Not Found at: {csv_path}")

#         print(f"📂 Reading CSV...")
        
#         for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)):
            
#             # Clean
#             existing_cols = [c for c in cols_to_keep if c in chunk.columns]
#             df_subset = chunk[existing_cols].copy()
#             df_subset.columns = [c.lower().replace(' ', '_') for c in df_subset.columns]

#             # Write to SQL (Standard Way)
#             # Since we downgraded SQLAlchemy, con=engine works perfectly again!
#             if i == 0:
#                 df_subset.to_sql('loans_raw', con=engine, if_exists='replace', index=False)
#                 print("   -> Created table 'loans_raw'")
#             else:
#                 df_subset.to_sql('loans_raw', con=engine, if_exists='append', index=False)
            
#             if i % 10 == 0:
#                 print(f"   -> Processed chunk {i}...")

#         print("🎉 [Airflow] Ingestion Complete.")

#     except Exception as e:
#         print(f"❌ Error: {e}")
#         raise e 

# if __name__ == "__main__":
#     ingest_data()
