import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import charts
from utils import load_and_process_data, filter_data_via_sidebar

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")

st.title("Researched Cohort - Exploratory Data Analysis")
st.write("This page provides a visual overview of the demographic and physical characteristics of the patient cohort included in the analysis.")

# Load the merged dataset (n=968)
df = load_and_process_data()

# Apply Global Filters
df = filter_data_via_sidebar(df)

# Ensure unique users for demographic plotting (one row per user)
df_unique_users = df.drop_duplicates(subset=['user_id'])

# --- Section 1: Demographics ---
st.header("1. Demographic Overview")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Age Distribution")
    fig_age = px.histogram(
        df_unique_users, 
        x="age_at_initial_order", 
        nbins=20, 
        title="Distribution of Age at Initial Order",
        labels={"age_at_initial_order": "Age (Years)"},
        color_discrete_sequence=['#4c78a8']
    )
    fig_age.update_layout(bargap=0.1)
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    st.subheader("Gender Distribution")
    fig_sex = px.pie(
        df_unique_users, 
        names="sex_at_birth", 
        title="Gender Split",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig_sex.update_traces(textinfo='value+percent')
    st.plotly_chart(fig_sex, use_container_width=True)

# New Chart: Age by Gender (Stacked)
st.subheader("Age Distribution by Gender")
fig_age_sex = px.histogram(
    df_unique_users, 
    x="age_at_initial_order", 
    color="sex_at_birth",
    nbins=20, 
    title="Age Distribution Segmented by Gender",
    labels={"age_at_initial_order": "Age (Years)"},
    barmode='stack'
)
fig_age_sex.update_layout(bargap=0.1)
st.plotly_chart(fig_age_sex, use_container_width=True)


# New Chart: Age by Medication (Stacked)
st.subheader("Age Distribution by Medication")
fig_age_med = px.histogram(
    df_unique_users, 
    x="age_at_initial_order", 
    color="initial_product",
    nbins=20, 
    title="Age Distribution Segmented by Medication",
    labels={"age_at_initial_order": "Age (Years)", "initial_product": "Medication"},
    barmode='stack'
)
fig_age_med.update_layout(bargap=0.1)
st.plotly_chart(fig_age_med, use_container_width=True)

# New Chart: Total Age Distribution (Unsegmented)
# New Chart: Total Age Distribution (100% Stacked by Age, Colored by Med) using Matplotlib
st.subheader("Medication Proportion by Age (100% Stacked)")

# Ensure Age Group exists
if 'age_group' not in df_unique_users.columns:
    df_unique_users['age_group'] = pd.cut(
        df_unique_users['age_at_initial_order'], 
        bins=[0, 29, 39, 49, 59, 100], 
        labels=['<30', '30-39', '40-49', '50-59', '60+']
    )

# 1. Prepare Data (Pivot and Normalize)
age_med_counts = pd.crosstab(df_unique_users['age_group'], df_unique_users['initial_product'])
# Ensure columns match the preferred color order if possible, or just let them auto-assign
# But user requested "Pink (dominant), Blue, Red, Green". 
# The dominant meds are typically Compound > Mounjaro > Ozempic > Wegovy (based on EDA)
# So we need to map colors to columns carefully if we want "Pink" to be "Dominant".
# For stability, let's just use the user's list and apply to the columns in alphabetical or count order.
# Let's check column order via sort_values to make Pink the biggest.
med_totals = age_med_counts.sum().sort_values(ascending=False)
age_med_counts = age_med_counts[med_totals.index] # Sort columns by volume

age_med_pct = age_med_counts.div(age_med_counts.sum(axis=1), axis=0)

# 2. Plot
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#FF69B4', '#1f77b4', '#d62728', '#2ca02c'] # Ordered: Pink, Blue, Red, Green
# If there are more than 4 meds, reuse or extend. Eucalyptus usually has ~4 key ones.
age_med_pct.plot(kind='bar', stacked=True, width=0.8, color=colors[:len(age_med_pct.columns)], ax=ax)

# 3. Styling
ax.set_ylim(0, 1)
ax.set_title('Medication Proportion by Age Group (100% Stacked)', fontsize=14)
ax.set_ylabel('Proportion (0-100%)', fontsize=12)
ax.set_xlabel('Age Group', fontsize=12)
ax.legend(title='Medication', bbox_to_anchor=(1.05, 1), loc='upper left')

# 4. Labels
for c in ax.containers:
    # Filter out labels for very small segments (< 5%) to keep it clean
    labels = [f'{v.get_height():.0%}' if v.get_height() > 0.05 else '' for v in c]
    ax.bar_label(c, labels=labels, label_type='center', fontsize=9, color='white', weight='bold')

plt.tight_layout()
st.pyplot(fig)


# --- Section 2: Physical Characteristics ---
st.header("2. Physical Characteristics")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Initial Weight Distribution")
    fig_weight = px.histogram(
        df_unique_users, 
        x="initial_weight", 
        nbins=30, 
        title="Distribution of Initial Weight",
        labels={"initial_weight": "Weight (kg)"},
        color_discrete_sequence=['#e45756']
    )
    fig_weight.update_layout(bargap=0.1)
    st.plotly_chart(fig_weight, use_container_width=True)

with col4:
    st.subheader("Height Distribution")
    fig_height = px.histogram(
        df_unique_users, 
        x="initial_height", 
        nbins=25, 
        title="Distribution of Height",
        labels={"initial_height": "Height (cm)"},
        color_discrete_sequence=['#72b7b2']
    )
    fig_height.update_layout(bargap=0.1)
    st.plotly_chart(fig_height, use_container_width=True)

# BMI Calculation
df_unique_users['bmi'] = df_unique_users['initial_weight'] / ((df_unique_users['initial_height'] / 100) ** 2)

st.subheader("Initial BMI Distribution")
fig_bmi = px.histogram(
    df_unique_users,
    x="bmi",
    nbins=30,
    title="Body Mass Index (BMI) Distribution",
    labels={"bmi": "BMI"},
    color_discrete_sequence=['#54a24b']
)
fig_bmi.add_vline(x=30, line_dash="dash", line_color="red", annotation_text="Obesity Threshold (30)")
fig_bmi.update_layout(bargap=0.1)
st.plotly_chart(fig_bmi, use_container_width=True)

st.subheader("BMI Distribution Over Time")
col_bmi1, col_bmi2 = st.columns(2)

with col_bmi1:
    df_wk4 = df[df['weeks_since_start'] == 4].copy()
    if not df_wk4.empty:
        df_wk4['bmi'] = df_wk4['weight_tracked'] / ((df_wk4['initial_height'] / 100) ** 2)
        fig_bmi4 = px.histogram(
            df_wk4, x="bmi", nbins=30, title="Week 4 BMI Distribution",
            labels={"bmi": "BMI"}, color_discrete_sequence=['#54a24b']
        )
        fig_bmi4.add_vline(x=30, line_dash="dash", line_color="red", annotation_text="Obesity (30)")
        fig_bmi4.update_layout(bargap=0.1)
        st.plotly_chart(fig_bmi4, use_container_width=True)
    else:
        st.info("Insufficient data for Week 4 BMI analysis.")

with col_bmi2:
    df_wk12 = df[df['weeks_since_start'] == 12].copy()
    if not df_wk12.empty:
        df_wk12['bmi'] = df_wk12['weight_tracked'] / ((df_wk12['initial_height'] / 100) ** 2)
        fig_bmi12 = px.histogram(
            df_wk12, x="bmi", nbins=30, title="Week 12 BMI Distribution",
            labels={"bmi": "BMI"}, color_discrete_sequence=['#54a24b']
        )
        fig_bmi12.add_vline(x=30, line_dash="dash", line_color="red", annotation_text="Obesity (30)")
        fig_bmi12.update_layout(bargap=0.1)
        st.plotly_chart(fig_bmi12, use_container_width=True)
    else:
        st.info("Insufficient data for Week 12 BMI analysis.")


# New Chart: Initial Weight by Medication (Stacked Matplotlib)
st.subheader("Initial Weight Distribution by Medication")

# 1. Create Bins for Weight
# Common range: 60kg to 160kg. Let's make 10kg bins.
weight_bins = range(50, 160, 10)
weight_labels = [f"{i}-{i+9}" for i in weight_bins[:-1]]
df_unique_users['weight_group'] = pd.cut(df_unique_users['initial_weight'], bins=weight_bins, labels=weight_labels)

# 2. Prepare Data
weight_med_counts = pd.crosstab(df_unique_users['weight_group'], df_unique_users['initial_product'])
# Reuse variable 'med_totals' index from Section 1 to ensure same column order/colors
if 'med_totals' in locals():
    valid_cols = [c for c in med_totals.index if c in weight_med_counts.columns]
    weight_med_counts = weight_med_counts[valid_cols]
else:
    # Fallback if med_totals not defined (though it should be)
    weight_med_counts = weight_med_counts.sort_index(axis=1)

# 3. Plot
fig_w, ax_w = plt.subplots(figsize=(10, 6))
# Reuse 'colors' from Section 1
plot_colors = colors[:len(weight_med_counts.columns)] if 'colors' in locals() else None

weight_med_counts.plot(kind='bar', stacked=True, width=0.8, color=plot_colors, ax=ax_w)

# 4. Styling
ax_w.set_title('Initial Weight Distribution by Medication', fontsize=14)
ax_w.set_ylabel('Count of Patients', fontsize=12)
ax_w.set_xlabel('Weight Group (kg)', fontsize=12)
if len(weight_med_counts.columns) > 0:
    ax_w.legend(title='Medication')
plt.xticks(rotation=45)
ax_w.grid(axis='y', alpha=0.3)

plt.tight_layout()
st.pyplot(fig_w)


# --- Section 3: Program Overview ---
st.header("3. Program & Engagement")
col5, col6 = st.columns(2)

with col5:
    st.subheader("Medication Prescribed")
    
    # 1. Donut Chart (Summary)
    fig_med = charts.create_market_share_donut(df_unique_users)
    st.plotly_chart(fig_med, use_container_width=True)
    
    # 2. Detailed Table
    med_counts = df_unique_users['initial_product'].value_counts().reset_index()
    med_counts.columns = ['Medication', 'Count']
    med_counts['Share'] = (med_counts['Count'] / med_counts['Count'].sum() * 100).map('{:.1f}%'.format)
    
    # Display table with formatting
    st.dataframe(
        med_counts,
        column_config={
            "Medication": st.column_config.TextColumn("Medication", width="medium"),
            "Count": st.column_config.NumberColumn("Patients", format="%d"),
            "Share": st.column_config.TextColumn("Market Share", width="small"),
        },
        hide_index=True,
        use_container_width=True
    )

with col6:
    st.subheader("App Interactions")
    fig_app = px.histogram(
        df_unique_users, 
        x="app_interactions_count", 
        nbins=20, 
        title="Distribution of App Interactions (Prior to Program)",
        labels={"app_interactions_count": "Interaction Count"},
        color_discrete_sequence=['#eea429']
    )
    fig_app.update_layout(bargap=0.1)
    st.plotly_chart(fig_app, use_container_width=True)


    
    st.subheader("Engagement Overview: Population & Activity")
    st.caption("Butterfly chart comparison: Number of users (Left) vs. Average Interactions (Right) by Age Group.")
    
    fig_butt = charts.create_engagement_butterfly_chart(df_unique_users)
    st.plotly_chart(fig_butt, use_container_width=True)


    # --- New Analysis: Interactions vs Retention ---
    # Aggregate retention per user (max week recorded)
    retention_df = df.groupby('user_id').agg({
        'weeks_since_start': 'max',
        'app_interactions_count': 'first', # Constant per user
        'initial_product': 'first'
    }).reset_index()
    
    st.markdown("#### Interactions vs. Retention")
    fig_corr = px.scatter(
        retention_df,
        x='app_interactions_count',
        y='weeks_since_start',
        color='initial_product',
        title="Do High Interactions Lead to Longer Retention?",
        labels={'app_interactions_count': 'Prior Interactions', 'weeks_since_start': 'Weeks Retained'},
        trendline="ols" # Add trendline to see correlation
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# --- New Section: Multivariate Analysis ---
st.markdown("---")
st.header("4. Multivariate Analysis: Impact of Interaction on Weight Loss")
st.write("This chart analyzes how **App Interaction Levels** (Low, Medium, High) correlate with **Weight Loss Trajectories** (Weeks 0-12) for each medication.")

# 1. Prepare Data for Multivariate Plot
# Filter for first 12 weeks, excluding negative weeks (pre-program data)
df_multi = df[(df['weeks_since_start'] >= 0) & (df['weeks_since_start'] <= 12)].copy()

# Bin Interactions into Low/Medium/High using Quantiles
# Calculate quantiles based on unique users to avoid skew from long-retention users
interaction_quantiles = df_unique_users['app_interactions_count'].quantile([0.33, 0.66])
low_thresh = interaction_quantiles[0.33]
high_thresh = interaction_quantiles[0.66]

def categorize_interaction(count):
    if count <= low_thresh:
        return 'Low'
    elif count <= high_thresh:
        return 'Medium'
    else:
        return 'High'

df_multi['interaction_level'] = df_multi['app_interactions_count'].apply(categorize_interaction)

# Aggregate Mean Weight Loss by Week, Med, and Interaction Level
df_traj = df_multi.groupby(['weeks_since_start', 'initial_product', 'interaction_level'])['pct_weight_loss'].mean().reset_index()

# 2. Plot Faceted Line Chart
fig_multi = px.line(
    df_traj,
    x="weeks_since_start",
    y="pct_weight_loss",
    color="initial_product",
    facet_col="interaction_level",
    category_orders={"interaction_level": ["Low", "Medium", "High"]}, # Force logical order
    title="Weight Loss Trajectories (0-12 Weeks) by Interaction Level",
    labels={
        "weeks_since_start": "Weeks", 
        "pct_weight_loss": "Avg % Weight Loss", 
        "initial_product": "Medication",
        "interaction_level": "Interaction Level"
    },
    markers=True
)
# Reverse Y-axis to show weight loss going "down" visually (optional, but standard for loss)
# Or keep standard negative values. Let's keep standard negative values but clarify axis.
fig_multi.update_yaxes(autorange="reversed") # Down is "more loss" (more negative)
fig_multi.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # Clean facet titles
st.plotly_chart(fig_multi, use_container_width=True)
