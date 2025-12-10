
import pandas as pd
from utils import load_and_process_data

print("Loading raw CSVs...")
u = pd.read_csv('user_data_dec.csv')
w = pd.read_csv('weight_tracked_data_dec.csv')

print(f"User Data Raw Unique IDs: {u['id'].nunique()}")
print(f"Weight Data Raw Unique IDs: {w['user_id'].nunique()}")

df_merged = load_and_process_data()
merged_ids = df_merged['user_id'].unique()
print(f"Merged Data Unique IDs: {len(merged_ids)}")

# Check logic used in Dashboard
valid_users = merged_ids
df_users_raw = u
df_users_active = df_users_raw[df_users_raw['id'].isin(valid_users)]
print(f"Dashboard Filtered Users Count: {len(df_users_active)}")

# Check if any ID in merged is NOT in user raw (impossible if merged inner?)
missing = set(merged_ids) - set(df_users_raw['id'])
print(f"IDs in Merged but not in User Raw: {missing}")


# Check duplicates in User Raw
dupes = df_users_raw[df_users_raw['id'].duplicated()]
print(f"Duplicate IDs in User Raw: {len(dupes)}")

# Check for NaN initial_product in active users
missing_product = df_users_active[df_users_active['initial_product'].isna()]
print(f"Users with NaN Initial Product: {len(missing_product)}")
if len(missing_product) > 0:
    print(missing_product['id'].tolist())

# Check total sum of counts
counts = df_users_active['initial_product'].value_counts()
print(f"Total Counts Sum: {counts.sum()}")

