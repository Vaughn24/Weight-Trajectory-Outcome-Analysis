import pandas as pd
import streamlit as st

# --- Global Constants ---
MED_COLORS = {
    'Saxenda': '#673094',      # Purple
    'Ozempic': '#F04F06',      # Orange
    'Semaglutide': '#4EC1CB',  # Teal (Compound)
    'Mounjaro': '#7E7E7E',     # Gray (Other)
    'PhenTop': '#7E7E7E',      # Gray (Other)
    'Contrave': '#7E7E7E',     # Gray (Other)
    'Orlistat': '#7E7E7E',     # Gray (Other)
    'Rybelsus': '#7E7E7E'      # Gray (Other)
}

MED_ORDER = ['Saxenda', 'Ozempic', 'Semaglutide', 'PhenTop', 'Mounjaro', 'Contrave', 'Orlistat', 'Rybelsus']

@st.cache_data
def load_and_process_data():
    """
    Loads and merges user and weight data.
    """
    # Load data
    user_data = pd.read_csv('user_data_dec.csv')
    weight_data = pd.read_csv('weight_tracked_data_dec.csv')

    # Convert to datetime with error handling
    user_data['initial_order_datetime'] = pd.to_datetime(user_data['initial_order_datetime'], errors='coerce')
    weight_data['tracked_datetime'] = pd.to_datetime(weight_data['tracked_datetime'], errors='coerce')

    # Drop rows with invalid dates if critical (optional, but good for stability)
    user_data = user_data.dropna(subset=['initial_order_datetime'])
    weight_data = weight_data.dropna(subset=['tracked_datetime'])

    # Merge data
    # 'id' in user_data matches 'user_id' in weight_data
    merged_data = pd.merge(weight_data, user_data, left_on='user_id', right_on='id', how='inner')

    # Calculate weeks since start
    merged_data['weeks_since_start'] = (merged_data['tracked_datetime'] - merged_data['initial_order_datetime']).dt.days // 7

    # Calculate percentage weight loss
    # ((weight_tracked - initial_weight) / initial_weight) * 100
    merged_data['pct_weight_loss'] = ((merged_data['weight_tracked'] - merged_data['initial_weight']) / merged_data['initial_weight']) * 100

    return merged_data

def filter_data(df, container=None):
    """
    Renders filters in the specified container (defaulting to sidebar) and returns the filtered dataframe.
    """
    if container is None:
        container = st.sidebar
        
    container.header("Filters")
    
    # --- Medication Filter ---
    # Ensure they are strings and drop NAs for the selector
    all_meds = sorted(df['initial_product'].dropna().astype(str).unique())
    selected_meds = container.multiselect(
        "Select Medications", 
        all_meds, 
        default=all_meds,
        key='med_filter_key' # Key ensures persistence across pages
    )
    
    if selected_meds:
        df = df[df['initial_product'].isin(selected_meds)]
    else:
        container.warning("No medication selected. Showing empty dataset.")
        df = df[df['initial_product'].isin([])] # Empty

    # --- Outlier Filter (IQR Method) ---
    remove_outliers = container.checkbox("Remove Outliers (IQR Method)", value=False, key='outlier_filter_key')
    
    if remove_outliers:
        Q1 = df['pct_weight_loss'].quantile(0.25)
        Q3 = df['pct_weight_loss'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds (standard 1.5x IQR)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter
        initial_count = len(df)
        df = df[(df['pct_weight_loss'] >= lower_bound) & (df['pct_weight_loss'] <= upper_bound)]
        filtered_count = len(df)
        
        if initial_count != filtered_count:
            container.caption(f"Removed {initial_count - filtered_count} outlier points.")
            
    return df

def filter_data_via_sidebar(df):
    """
    Backward compatibility wrapper for filter_data targeting the sidebar.
    """
    return filter_data(df, container=st.sidebar)
