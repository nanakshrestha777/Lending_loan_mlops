# import pandas as pd
# from sqlalchemy import create_engine, text
# import numpy as np

# CONN_STR = "mysql+pymysql://admin:admin@127.0.0.1:3306/lending_club"
# engine = create_engine(CONN_STR)

# query = """
# SELECT loan_status, annual_inc, dti, grade, int_rate, 
#        home_ownership, purpose, term, loan_amnt 
# FROM loans_raw
# """

# df = pd.read_sql(query, engine)
# df_clean = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
# df_clean['target'] = df_clean['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)

# if 'term' in df_clean.columns:
#     df_clean['term'] = df_clean['term'].astype(str).str.extract(r'(\d+)').astype(float)

# le = LabelEncoder()
# if 'grade' in df_clean.columns:
#     df_clean['grade'] = le.fit_transform(df_clean['grade'].astype(str))
# if 'home_ownership' in df_clean.columns:
#     df_clean['home_ownership'] = le.fit_transform(df_clean['home_ownership'].astype(str))
    



# try:

#     df_final.to_sql('loans_cleaned', engine, if_exists='replace', index=False, chunksize=10000)
#     print("   -> Data saved as Row-based table.")
    
#     try:
#         with engine.connect() as conn:
#             conn.execute(text("ALTER TABLE loans_cleaned ENGINE=Columnstore;"))

#     except Exception as cs_error:
#         print(f" Could not convert to Columnstore ({cs_error}).")

        
#     print(" PIPELINE FINISHED.")
    
# except Exception as e:
#     print(f"❌ Critical Error: {e}")
