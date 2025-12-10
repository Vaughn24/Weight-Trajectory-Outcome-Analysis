import streamlit as st
import pandas as pd
from utils import load_and_process_data

# Page config (must be first)
st.set_page_config(page_title="Early Weight Loss Trajectories", layout="wide")

st.title("Worth the weight")
st.write("A Trajectory & Outcome Analysis of the 12-Week Cohort Performance.")

# Load data
df = load_and_process_data()

# Calculate metrics
total_patients = df['user_id'].nunique()
total_data_points = len(df)
most_common_med = df['initial_product'].mode()[0] if not df.empty else "N/A"

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Patients", total_patients)
col2.metric("Total Data Points", total_data_points)
col3.metric("Most Common Medication", most_common_med)

st.info("Use the sidebar navigation to view the Dashboard page.")
