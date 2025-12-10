import streamlit as st
import pandas as pd
from utils import load_and_process_data

st.set_page_config(page_title="Data Sources & Methodology", layout="wide")

st.title("Data Sources and Methodology")
st.write("This section details the raw data sources and the processing steps applied to generate the analytical dataset.")

# Section 1: User Data
st.header("1. User Data Source")
st.markdown("Contains demographic and initial product information for each patient.")
try:
    user_raw = pd.read_csv('user_data_dec.csv')
    st.markdown("### Table 1: Raw User Data Preview")
    st.dataframe(user_raw.head())
    st.caption(f"Shape: {user_raw.shape[0]} rows, {user_raw.shape[1]} columns")
    with st.expander("View Column Details"):
        st.write(user_raw.dtypes.astype(str))
except FileNotFoundError:
    st.error("User data file not found.")

# Section 2: Weight Tracked Data
st.header("2. Weight Tracked Data Source")
st.markdown("Contains longitudinal weight logs and their sources (e.g., Tracker, Consultation).")
try:
    weight_raw = pd.read_csv('weight_tracked_data_dec.csv')
    st.markdown("### Table 2: Raw Weight Data Preview")
    st.dataframe(weight_raw.head())
    st.caption(f"Shape: {weight_raw.shape[0]} rows, {weight_raw.shape[1]} columns")
    with st.expander("View Column Details"):
        st.write(weight_raw.dtypes.astype(str))
except FileNotFoundError:
    st.error("Weight data file not found.")

# Section 3: Merged Data
st.header("3. Merged and Processed Dataset")
st.markdown("""
This dataset is the result of merging the user and weight data. 
**Processing Steps:**
1.  **Date Parsing:** Converted `initial_order_datetime` and `tracked_datetime` to standard datetime objects.
2.  **Merging:** Inner join on User ID.
3.  **Feature Engineering:**
    *   `weeks_since_start`: Calculated as integer weeks between initial order and track date.
    *   `pct_weight_loss`: Percent change from initial weight.
        $$ \\frac{\\text{Weight}_{current} - \\text{Weight}_{initial}}{\\text{Weight}_{initial}} \\times 100 $$
""")

df_merged = load_and_process_data()
st.markdown("### Table 3: Merged Dataset Preview")
st.dataframe(df_merged.head())
st.caption(f"Shape: {df_merged.shape[0]} rows, {df_merged.shape[1]} columns")

# Section 4: Data Quality & Inclusion
st.header("4. Data Quality & Inclusion Analysis")
st.markdown("This section verifies the integrity of the data merger and explains any exclusions.")

# Load raw comparisons again for accurate counts
user_raw = pd.read_csv('user_data_dec.csv')
weight_raw = pd.read_csv('weight_tracked_data_dec.csv')

total_users_raw = user_raw['id'].nunique()
users_with_data = df_merged['user_id'].nunique()
excluded_users = total_users_raw - users_with_data

total_logs_raw = len(weight_raw)
merged_logs = len(df_merged)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Coverage")
    st.metric("Total Registered Users", total_users_raw)
    st.metric("Patients with Weight Data (Included)", users_with_data)
    st.metric("Patients without Data (Excluded)", excluded_users, delta=-excluded_users, delta_color="off")
    st.caption("Excluded users exist in the system but have 0 weight logs.")

with col2:
    st.subheader("Data Completeness")
    st.metric("Total Weight Logs", total_logs_raw)
    st.metric("Successfully Merged Logs", merged_logs)
    
    if total_logs_raw == merged_logs:
        st.success("✅ 100% of weight logs were successfully matched to a user.")
    else:
        st.error(f"❌ {total_logs_raw - merged_logs} weight logs were orphaned.")

st.subheader("Table 4: Excluded Users Detail")
st.markdown("The following users are present in the `User Data Source` but have **no corresponding entries** in the `Weight Tracked Data Source`. These observations are excluded from the main analysis.")

# Identify excluded IDs
excluded_ids = set(user_raw['id']) - set(df_merged['user_id'])

if excluded_ids:
    excluded_df = user_raw[user_raw['id'].isin(excluded_ids)]
    st.dataframe(excluded_df)
    st.caption(f"Showing {len(excluded_df)} excluded users.")
else:
    st.success("No users were excluded.")

st.subheader("Table 5: Orphaned Weight Logs Detail")
st.markdown("The following weight logs were excluded from the analysis. This occurs if the log itself has an invalid date, or if it belongs to a user with invalid data (e.g., missing start date).")

# 1. Identify logs with invalid tracked dates
weight_raw_check = weight_raw.copy()
weight_raw_check['tracked_datetime_parsed'] = pd.to_datetime(weight_raw_check['tracked_datetime'], errors='coerce')
invalid_date_logs = weight_raw_check[weight_raw_check['tracked_datetime_parsed'].isna()]

# 2. Identify logs belonging to users with invalid start dates
user_raw_check = user_raw.copy()
user_raw_check['initial_order_datetime_parsed'] = pd.to_datetime(user_raw_check['initial_order_datetime'], errors='coerce')
invalid_users = user_raw_check[user_raw_check['initial_order_datetime_parsed'].isna()]['id']
logs_from_invalid_users = weight_raw_check[weight_raw_check['user_id'].isin(invalid_users)]

# Combine and drop duplicates (in case a log is invalid AND belongs to an invalid user)
orphaned_logs = pd.concat([invalid_date_logs, logs_from_invalid_users]).drop_duplicates()

if not orphaned_logs.empty:
    # Cleanup for display
    display_orphans = orphaned_logs.drop(columns=['tracked_datetime_parsed'])
    
    # Add a 'Reason' column for clarity
    def get_reason(row):
        reasons = []
        if pd.isna(row['tracked_datetime_parsed']):
            reasons.append("Invalid Log Date")
        if row['user_id'] in invalid_users.values:
            reasons.append("Invalid User Data")
        return ", ".join(reasons)

    display_orphans['Exclusion Reason'] = orphaned_logs.apply(get_reason, axis=1)
    
    st.dataframe(display_orphans)
    st.caption(f"Showing {len(display_orphans)} orphaned weight logs.")
else:
    st.success("No orphaned weight logs found.")
