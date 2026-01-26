# import great_expectations as gx
# import pandas as pd
# from sqlalchemy import create_engine



# CONN_STR = "mysql+pymysql://admin:admin@127.0.0.1:3306/lending_club"
# engine = create_engine(CONN_STR)


# df = pd.read_sql("SELECT * FROM loans_raw LIMIT 100000", engine)

# # --- 2. SETUP (The Fix: Use Datasource API instead of from_pandas) ---
# context = gx.get_context()

# # Create a temporary datasource to hold the dataframe
# datasource_name = "temp_pandas_datasource"
# # Check if it exists to avoid overwrite errors
# try:
#     datasource = context.sources.add_pandas(name=datasource_name)
# except:
#     datasource = context.datasources[datasource_name]

# # Add the dataframe as an asset
# asset_name = "loan_data_asset"
# asset = datasource.add_dataframe_asset(name=asset_name, dataframe=df)
# batch_request = asset.build_batch_request()

# # Create the suite
# suite_name = "lending_club_validation"
# context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

# # Get Validator
# validator = context.get_validator(
#     batch_request=batch_request,
#     expectation_suite_name=suite_name
# )


# validator.expect_table_row_count_to_be_between(min_value=1, max_value=3000000)
# validator.expect_column_values_to_not_be_null(column="loan_status")
# validator.expect_column_values_to_be_of_type(column="annual_inc", type_="float")


# validator.save_expectation_suite(discard_failed_expectations=False)

# checkpoint = context.add_or_update_checkpoint(
#     name="my_checkpoint",
#     validations=[
#         {
#             "batch_request": batch_request,
#             "expectation_suite_name": suite_name,
#         },
#     ],
# )

# results = checkpoint.run()

# context.open_data_docs()

