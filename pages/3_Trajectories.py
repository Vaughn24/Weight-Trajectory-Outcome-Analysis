import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import load_and_process_data, filter_data_via_sidebar, MED_COLORS, MED_ORDER
import charts

st.set_page_config(page_title="Weight Loss Trajectories", layout="wide")

st.title("Weight Loss Trajectories (0-12 Weeks)")
st.write("This analysis focuses on the early weight loss patterns of patients, tracking their percentage weight loss from baseline over the first 12 weeks.")

# Load data
df = load_and_process_data()

# Apply Global Filters
df = filter_data_via_sidebar(df)

# Filter for Weeks 0-12
df_filtered = df[(df['weeks_since_start'] >= 0) & (df['weeks_since_start'] <= 12)]

if df_filtered.empty:
    st.error("No data available for the 0-12 week period.")
else:
    # --- 1. Average Trajectory by Medication ---
    st.header("1. Comparative Trajectories by Medication")
    st.write("Average percentage weight loss over time, grouped by the prescribed medication.")

    # Group by Medication and Week to get the mean
    traj_data = df_filtered.groupby(['initial_product', 'weeks_since_start'])['pct_weight_loss'].mean().reset_index()
    
    # Calculate patient counts per group for tooltip (optional but helpful context)
    # Using a simplified approach effectively just mapping mean for the line chart
    
    fig_line = px.line(
        traj_data, 
        x='weeks_since_start', 
        y='pct_weight_loss', 
        color='initial_product',
        title="Average % Weight Loss Trajectory (Weeks 0-12)",
        labels={
            "weeks_since_start": "Weeks Since Start",
            "pct_weight_loss": "Average % Weight Loss",
            "initial_product": "Medication"
        },
        markers=True,
        color_discrete_map=MED_COLORS,
        category_orders={'initial_product': MED_ORDER}
    )
    # Reverse Y-axis usually makes sense for weight loss (down is good), 
    # but standard "Percentage Weight Loss" is usually negative. 
    # If the calcs in utils are ((Cur - Init)/Init)*100, they are negative.
    # So a downward slope is visually intuitive.
    fig_line.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_line, use_container_width=True)

    # Insight 1
    w12_avg = df_filtered[df_filtered['weeks_since_start'] == 12].groupby('initial_product')['pct_weight_loss'].mean().sort_values()
    if not w12_avg.empty:
        best_med = w12_avg.index[0] # Ascending sort, so most negative is first
        best_val = w12_avg.iloc[0]
        st.info(f"💡 **Finding:** At Week 12, **{best_med}** demonstrates the strongest response with an average weight loss of **{best_val:.1f}%**.")


    # --- 2. Distribution at Key Milestones ---
    st.header("2. Distribution of Outcomes at Key Milestones")
    st.write("Variability in patient outcomes at Weeks 4, 8, and 12.")

    milestone_weeks = [4, 8, 12]
    df_milestones = df_filtered[df_filtered['weeks_since_start'].isin(milestone_weeks)]

    if not df_milestones.empty:
        fig_box = px.box(
            df_milestones, 
            x='weeks_since_start', 
            y='pct_weight_loss',
            color='initial_product', # Breaking down by med is informative
            title="Distribution of % Weight Loss at Weeks 4, 8, 12",
            labels={
                "weeks_since_start": "Week Consideration",
                "pct_weight_loss": "% Weight Loss"
            },
            color_discrete_map=MED_COLORS,
            category_orders={'initial_product': MED_ORDER}
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Insight 2
        w12_stats = df_filtered[df_filtered['weeks_since_start'] == 12]['pct_weight_loss'].describe()
        if not w12_stats.empty:
            p75 = w12_stats['25%'] # In negative numbers, 25th percentile is the "larger" loss number (e.g. -8 is smaller than -2)
            # Actually describe() gives 25%, 50%, 75%. 
            # If values are -10, -5, -2 => 25% is -10 (most loss), 75% is -2 (least loss).
            # Let's verify: pd.Series([-10, -5, -2]).describe() -> 25% is -7.5. 
            # So 25% quantile is the "Super Responder" boundary.
            median_loss = w12_stats['50%']
            st.info(f"💡 **Finding:** By Week 12, there is significant variability. The median patient has lost **{median_loss:.1f}%**, but the top 25% of responders have lost more than **{p75:.1f}%**.")
    else:
        st.info("Insufficient data for milestone weeks (4, 8, 12).")

    # --- 3. Retention Analysis ---
    st.header("3. Retention Overview")
    st.write("Deep-dive into patient loyalty and churn. Analyzing not just raw numbers, but survival rates and drop-off points.")

    # Data Calculation: Total Patients per Week
    weekly_counts = df_filtered.groupby('weeks_since_start')['user_id'].nunique().reset_index()
    weekly_counts.columns = ['Week', 'Active Patients']
    
    # Display as a table (using columns for layout)
    st.subheader("Active Patient Count by Week")
    col_table, col_empty = st.columns([1, 2]) # Keep table narrow
    with col_table:
        st.dataframe(weekly_counts, hide_index=True)

    # Data Calculation: Patients per Week by Medication
    weekly_med_counts = df_filtered.pivot_table(
        index='weeks_since_start', 
        columns='initial_product', 
        values='user_id', 
        aggfunc='nunique',
        fill_value=0
    ).reset_index()
    weekly_med_counts.rename(columns={'weeks_since_start': 'Week'}, inplace=True)

    st.subheader("Active Patient Count by Medication")
    st.dataframe(weekly_med_counts, hide_index=True)

    # Heatmap Visualization
    st.subheader("Heatmap: Patient Density by Week & Medication")
    st.caption("Color intensity indicates the **% of patients retained** relative to Week 0 for that specific medication (Row-Relative Grading).")
    
    # Using Shared Chart Function
    fig_heatmap = charts.create_patient_density_heatmap(df_filtered)
    fig_heatmap.update_layout(title="Volume Density Heatmap (Row-Normalized)")
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # --- Insight: Why do counts increase? ---
    with st.expander("Why does the patient count sometimes go UP in later weeks?"):
        st.write("""
        You might notice (e.g., Week 2 having more patients than Week 0). This occurs because some users are **"Late Starters"**. 
        They may receive their medication or start tracking a week or two after their "Order Date" (Start Date).
        The table below shows **when users logged their first weight entry**:
        """)
        
        # Calculate First Log Week for each user
        first_logs = df_filtered.groupby('user_id')['weeks_since_start'].min().reset_index()
        first_logs.columns = ['user_id', 'First Log Week']
        
        # Count users by their Entry Week
        entry_week_counts = first_logs['First Log Week'].value_counts().sort_index().reset_index()
        entry_week_counts.columns = ['Week', 'New Active Loggers']
        
        st.dataframe(entry_week_counts, hide_index=True)

        st.markdown("---")
        st.write("**Source Data (User-Level Detail):**")
        st.write("This list shows the calculated 'Start Week' for every single patient.")
        st.dataframe(first_logs, hide_index=True)

    # --- Heatmap 2: Weight Loss Progression ---
    st.subheader("Heatmap: Weight Loss Progression (Row-Normalized)")
    st.write("Color intensity indicates **% of Peak Weight Loss achieved** for that medication. This standardizes the view to show *speed of effect* rather than raw magnitude.")

    # 1. Pivot for Average Weight Loss
    # Using Shared Chart Function
    fig_weight_heat = charts.create_weight_loss_heatmap(df_filtered)
    fig_weight_heat.update_layout(title="Weight Loss Intensity Heatmap")
    st.plotly_chart(fig_weight_heat, use_container_width=True)
    
    # --- Restoration of Missing Data ---
    retention_data = df_filtered.groupby(['weeks_since_start', 'initial_product'])['user_id'].nunique().reset_index()
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Patient Volume (Raw)", "Survival Curve (Loyalty)", "Share of Volume (Market Share)"])
    
    with tab1:
        st.caption("Raw count of active patients providing data per week.")
        fig_retention = px.bar(
            retention_data,
            x='weeks_since_start',
            y='user_id',
            color='initial_product',
            title="Weekly Patient Count",
            labels={"weeks_since_start": "Weeks Since Start", "user_id": "Active Patients"},
            color_discrete_map=MED_COLORS,
            category_orders={'initial_product': MED_ORDER}
        )
        st.plotly_chart(fig_retention, use_container_width=True)

    with tab2:
        st.caption("Survival Chart: Percentage of the ORIGINAL starting group remaining at each week (Week 0 = 100%).")
        
        # Calculate Survival %
        # 1. Get baseline (Week 0) counts for each med
        baseline_counts = retention_data[retention_data['weeks_since_start'] == 0][['initial_product', 'user_id']].rename(columns={'user_id': 'baseline_count'})
        
        # 2. Merge baseline back
        survival_df = pd.merge(retention_data, baseline_counts, on='initial_product', how='left')
        
        # 3. Calculate %
        survival_df['survival_rate'] = (survival_df['user_id'] / survival_df['baseline_count']) * 100
        
        fig_survival = px.line(
            survival_df,
            x='weeks_since_start',
            y='survival_rate',
            color='initial_product',
            title="Survival Curve (Retention Rate)",
            labels={"weeks_since_start": "Weeks Since Start", "survival_rate": "% of Week 0 Cohort Remaining"},
            markers=True,
            color_discrete_map=MED_COLORS,
            category_orders={'initial_product': MED_ORDER}
        )
        fig_survival.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_survival, use_container_width=True)

    with tab3:
        st.caption("Market Share: Which medication makes up the bulk of the patient population over time?")
        # Stacked Area (Normalized to 100%)
        fig_share = px.area(
            retention_data,
            x='weeks_since_start',
            y='user_id',
            color='initial_product',
            groupnorm='percent', # This makes it a 100% stacked chart
            title="Share of Volume (100% Stacked)",
            labels={"weeks_since_start": "Weeks Since Start", "user_id": "Share of Volume"},
            color_discrete_map=MED_COLORS,
            category_orders={'initial_product': MED_ORDER}
        )
        st.plotly_chart(fig_share, use_container_width=True)

    # --- Drill Down: Churn Analysis (Drop-off from Peak) ---
    st.subheader("📉 Churn Analysis: Drop-off from Peak")
    st.write("Calculating the 'Drop-off Rate': The % of patients lost from the medication's *Peak* engagement week to Week 12.")
    
    churn_metrics = []
    
    for med in retention_data['initial_product'].unique():
        med_data = retention_data[retention_data['initial_product'] == med]
        
        # Find Peak
        peak_count = med_data['user_id'].max()
        peak_week = med_data.loc[med_data['user_id'].idxmax(), 'weeks_since_start']
        
        # Find End (Week 12 or last available)
        end_data = med_data[med_data['weeks_since_start'] == 12]
        if not end_data.empty:
            end_count = end_data['user_id'].values[0]
        else:
            end_count = med_data.iloc[-1]['user_id'] # Fallback to last data point
            
        drop_off_pct = ((peak_count - end_count) / peak_count) * 100 if peak_count > 0 else 0
        
        churn_metrics.append({
            "Medication": med,
            "Peak Count": peak_count,
            "Peak Week": f"Week {peak_week}",
            "End Count": end_count,
            "Drop-off Rate": f"{drop_off_pct:.1f}%",
            "drop_off_numeric": drop_off_pct # Store numeric for plotting
        })
        
    churn_df = pd.DataFrame(churn_metrics)
    # Display table (exclude numeric helper col for cleaner UI)
    st.dataframe(churn_df.drop(columns=['drop_off_numeric']).set_index("Medication"), use_container_width=True)

    # Visualization (Requested)
    fig_churn = px.bar(
        churn_df,
        x='Medication',
        y='drop_off_numeric',
        text='Drop-off Rate', # Show the formatted string on bars
        title="Drop-off Rate from Peak (Visualized)",
        labels={'drop_off_numeric': 'Drop-off %'},
        color='drop_off_numeric',
        color_continuous_scale='Reds'
    )
    fig_churn.update_layout(height=400)
    st.plotly_chart(fig_churn, use_container_width=True)


    # --- 4. Clinical Benchmarks ---
    st.header("4. Clinical Success Rate (>5% Weight Loss)")
    st.write("Proportion of active patients who have achieved greater than 5% weight loss at each week.")
    
    # Calculate % success per week/medication
    # Success = pct_weight_loss <= -5 (assuming negative is loss)
    # Be careful: if pct_weight_loss is e.g. -5.5, that is > 5% loss. So we check <= -5.
    df_filtered['achieved_5pct'] = df_filtered['pct_weight_loss'] <= -5
    
    success_rate = df_filtered.groupby(['weeks_since_start', 'initial_product'])['achieved_5pct'].mean().reset_index()
    success_rate['percent_success'] = success_rate['achieved_5pct'] * 100
    
    fig_success = px.line(
        success_rate,
        x='weeks_since_start',
        y='percent_success',
        color='initial_product',
        title="Percentage of Patients Achieving >5% Weight Loss",
        labels={"weeks_since_start": "Weeks Since Start", "percent_success": "% of Patients"},
        markers=True,
        color_discrete_map=MED_COLORS,
        category_orders={'initial_product': MED_ORDER}
    )
    fig_success.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% Threshold")
    st.plotly_chart(fig_success, use_container_width=True)

    # Insight 4
    success_w12 = df_filtered[df_filtered['weeks_since_start'] == 12]['achieved_5pct'].mean()
    if not pd.isna(success_w12):
        st.info(f"💡 **Finding:** By Week 12, **{success_w12*100:.1f}%** of the active cohort has achieved clinically significant weight loss (>5%).")


    # --- 5. Waterfall Plot (Week 12 Outcomes) ---
    st.header("5. Week 12 Outcome Distribution (Waterfall)")
    st.write("Ranked distribution of percentage weight loss for all patients reaching Week 12. This visualizes the full range of responder types.")

    df_week12 = df_filtered[df_filtered['weeks_since_start'] == 12].copy()

    if not df_week12.empty:
        # Sort by weight loss (descending order of loss, so most negative first for waterfall left-to-right)
        # Actually standard is usually: Best result (most neg) on left -> Worst result (pos) on right.
        df_week12 = df_week12.sort_values(by='pct_weight_loss', ascending=True)
        
        # Create a rank for X-axis
        df_week12['Patient Rank'] = range(1, len(df_week12) + 1)
        
        fig_waterfall = px.bar(
            df_week12,
            x='Patient Rank',
            y='pct_weight_loss',
            color='initial_product',
            title="Week 12 Weight Loss Waterfall (n={})".format(len(df_week12)),
            labels={"pct_weight_loss": "% Weight Loss (Negative = Loss)", "Patient Rank": "Patient (Ranked)"},
            color_discrete_map=MED_COLORS,
            category_orders={'initial_product': MED_ORDER}
        )
        fig_waterfall.update_layout(bargap=0.0) # Remove gap to make it look like a continuous curve
        st.plotly_chart(fig_waterfall, use_container_width=True)
    else:
        st.info("No data available specifically for Week 12.")

    # --- 6. Behavioral Hypothesis (Adherence vs Success) ---
    st.header("6. Behavioral Hypothesis: Adherence Effect")
    st.write("Does frequent logging in the first 4 weeks predict better clinical outcomes at Week 12?")
    st.markdown("> **Hypothesis**: Patients with **'High'** logging frequency (>= 20 logs in first 4 weeks) achieve Clinical Success (>5% loss) at a higher rate than those with **'Low'** frequency.")

    # 1. Determine Adherence Level (Weeks 0-4)
    # We use 'df' (full dataset) but restrict to first 4 weeks for the count
    df_early_logs = df[(df['weeks_since_start'] >= 0) & (df['weeks_since_start'] <= 4)]
    log_counts = df_early_logs.groupby('user_id').size().reset_index(name='log_count')
    
    # Define Tiers: High >= 20 (approx daily M-F), Low < 20
    log_counts['adherence_tier'] = log_counts['log_count'].apply(lambda x: 'High (>=20 logs)' if x >= 20 else 'Low (<20 logs)')

    # 2. Determine Week 12 Outcome (Success)
    df_w12_outcomes = df[df['weeks_since_start'] == 12].copy()
    df_w12_outcomes['is_success'] = df_w12_outcomes['pct_weight_loss'] <= -5

    # 3. Merge
    analysis_df = pd.merge(df_w12_outcomes, log_counts[['user_id', 'adherence_tier', 'log_count']], on='user_id', how='inner')

    if not analysis_df.empty:
        # Convert weight loss to positive magnitude for intuitive "Higher is Better" visualization
        analysis_df['loss_magnitude'] = analysis_df['pct_weight_loss'] * -1
        
        # Define Colors: Neon Blue (Low) / Neon Purple (High)
        # Using specific hex codes for that "Neon" look
        dataset_colors = {'High (>=20 logs)': '#A855F7', 'Low (<20 logs)': '#0EA5E9'} # Purple-500, Sky-500

        fig_box = px.box(
            analysis_df,
            x='adherence_tier',
            y='loss_magnitude',
            color='adherence_tier',
            title="Impact of Logging Frequency on Weight Loss (Week 12)",
            labels={
                'loss_magnitude': '% Total Weight Loss (Positive = Loss)',
                'adherence_tier': 'Adherence Group'
            },
            color_discrete_map=dataset_colors,
            points='all' # Show all points for distribution density
        )
        
        fig_box.update_layout(
            height=500,
            showlegend=False, # Legend is redundant with X-axis labels
            yaxis=dict(title="% Weight Loss", zeroline=True),
            xaxis=dict(title=None)
        )
        
        st.plotly_chart(fig_box, use_container_width=True)

        # 5. Insight / Recommendation
        # Calculate median loss to quantify the shift
        medians = analysis_df.groupby('adherence_tier')['loss_magnitude'].median()
        
        if 'High (>=20 logs)' in medians and 'Low (<20 logs)' in medians:
            med_high = medians['High (>=20 logs)']
            med_low = medians['Low (<20 logs)']
            uplift = med_high - med_low
            
            st.info(f"""
            **Result**: The Median High Adherence user lost **{med_high:.1f}%** of their body weight, compared to **{med_low:.1f}%** for Low Adherence users.
            (Uplift: **+{uplift:.1f}%** extra weight loss).
            """)
            
            if med_high > med_low * 1.2:
                 st.success("✅ **Hypothesis Supported**: Consistent logging shifts the entire performance distribution upwards. **Recommendation**: Implement 'Week 2 Check-in' automation for low loggers.")
            else:
                 st.warning("⚠️ **Hypothesis Neutral**: The distribution shift is minor.")
    else:
        st.warning("Insufficient overlap between patients with early logs and Week 12 outcomes to perform analysis.")

    st.write("Explore different predictive models by adjusting the timeframe and success thresholds.")
    
    # Controls
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Early Milestone (Predictor)")
        early_week = st.selectbox("Select Early Week", options=[2, 4, 6, 8], index=1) # Default Week 4
        early_target = st.slider("Early Weight Loss Target (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)

    with col_b:
        st.subheader("Late Outcome (Target)")
        late_week = st.selectbox("Select Outcome Week", options=[8, 12], index=1) # Default Week 12
        late_target = st.slider("Late Weight Loss Goal (%)", min_value=5.0, max_value=20.0, value=10.0, step=0.5)
        
    if early_week >= late_week:
        st.error("Error: 'Early Week' must be before 'Outcome Week'.")
    else:
        # Data Processing for Dynamic Selection
        df_early = df[df['weeks_since_start'] == early_week][['user_id', 'pct_weight_loss']].rename(columns={'pct_weight_loss': 'early_loss'})
        df_late = df[df['weeks_since_start'] == late_week][['user_id', 'pct_weight_loss', 'initial_product']].rename(columns={'pct_weight_loss': 'late_loss'})
        
        dynamic_df = pd.merge(df_early, df_late, on='user_id', how='inner')
        
        if not dynamic_df.empty:
            dynamic_df['early_mag'] = dynamic_df['early_loss'] * -1
            dynamic_df['late_mag'] = dynamic_df['late_loss'] * -1
            
            # Dynamic Quadrant Calculations
            # Failure = < early_target
            # Success later = >= late_target
            
            non_responders = dynamic_df[(dynamic_df['early_mag'] < early_target)]
            recovered_responders = non_responders[non_responders['late_mag'] >= late_target]
            
            count_non_resp = len(non_responders)
            count_recovered = len(recovered_responders)
            
            recovery_rate = (count_recovered / count_non_resp * 100) if count_non_resp > 0 else 0
            
            # Visual
            fig_dyn = px.scatter(
                dynamic_df,
                x='early_mag',
                y='late_mag',
                color='initial_product',
                title=f"Predictive Model: Week {early_week} vs Week {late_week}",
                labels={
                    'early_mag': f'Week {early_week} Weight Loss (%)',
                    'late_mag': f'Week {late_week} Weight Loss (%)',
                    'initial_product': 'Medication'
                },
                hover_data=['user_id'],
                color_discrete_map=MED_COLORS,
                category_orders={'initial_product': MED_ORDER}
            )
            
            # Dynamic Lines
            fig_dyn.add_vline(x=early_target, line_dash="dash", line_color="red", annotation_text=f"Cut-off {early_target}%")
            fig_dyn.add_hline(y=late_target, line_dash="dash", line_color="green", annotation_text=f"Goal {late_target}%")
            
            fig_dyn.update_layout(height=500)
            st.plotly_chart(fig_dyn, use_container_width=True)
            
            
            # Dynamic Insight
            st.info(f"""
            **Model Result**: Using a cut-off of **{early_target}%** at **Week {early_week}**:
            - **{count_non_resp}** patients were flagged as "At Risk" (Slow Starters).
            - Only **{count_recovered}** ({recovery_rate:.1f}%) of them recovered to hit the **{late_target}%** goal by **Week {late_week}**.
            - **Predictive Power**: If a patient fails this early check, they have a **{100-recovery_rate:.1f}%** probability of missing the long-term goal.
            """)

            # --- Quadrant Distribution Bar Chart (New Request) ---
            st.subheader("Quadrant Distribution by Medication")
            st.write("Breakdown of patient outcomes for each medication based on the selected criteria.")
            
            # classify quadrants
            def classify_quadrant(row):
                early_pass = row['early_mag'] >= early_target
                late_pass = row['late_mag'] >= late_target
                
                if early_pass and late_pass:
                    return "Consistent Responder" # Top Right
                elif early_pass and not late_pass:
                    return "Early Plateau" # Bottom Right
                elif not early_pass and late_pass:
                    return "Late Bloomer" # Top Left
                else:
                    return "Non-Responder" # Bottom Left
            
            dynamic_df['Quadrant'] = dynamic_df.apply(classify_quadrant, axis=1)
            
            # Pre-calculate Total Counts per Medication for the Denominator
            med_total_counts = dynamic_df['initial_product'].value_counts()

            # --- 2x2 Grid Layout Redesign ---
            # Helper to display a quadrant
            def show_quadrant_panel(quadrant_name, color, description):
                subset = dynamic_df[dynamic_df['Quadrant'] == quadrant_name]
                count = len(subset)
                pct = (count / len(dynamic_df) * 100) if len(dynamic_df) > 0 else 0
                
                # Header with Color
                st.markdown(f"<h4 style='color: {color}; margin-bottom: 0;'>{quadrant_name}</h4>", unsafe_allow_html=True)
                st.caption(description)
                
                # Big Number
                st.metric(label="Total Patients", value=f"{count}", delta=f"{pct:.1f}% of Cohort")
                
                # Table Breakdown
                if not subset.empty:
                    breakdown = subset['initial_product'].value_counts().reset_index()
                    breakdown.columns = ['Medication', 'Count']
                    
                    # Add Percentage Column (Relative to THAT Medication's Total Volume)
                    # "What % of Ozempic users are in this bucket?"
                    breakdown['% of Med'] = breakdown.apply(
                        lambda x: (x['Count'] / med_total_counts.get(x['Medication'], 1) * 100), 
                        axis=1
                    ).map('{:.1f}%'.format)
                    
                    st.dataframe(breakdown, hide_index=True, use_container_width=True)
                else:
                    st.write("No patients in this group.")

            # Row 1: Top Quadrants (Late Bloomers | Consistent Responders)
            st.markdown("---")
            row1_col1, row1_col2 = st.columns(2)
            
            with row1_col1:
                # Top Left: Late Bloomers
                show_quadrant_panel(
                    "Late Bloomer", 
                    "#A855F7", # Purple
                    "Started Slow (< Target), Finished Strong (>= Goal). The 'Comeback' group."
                )
                
            with row1_col2:
                # Top Right: Consistent Responder
                show_quadrant_panel(
                    "Consistent Responder", 
                    "#10B981", # Green
                    "Started Strong (>= Target) and Finished Strong (>= Goal). Ideal trajectory."
                )

            # Row 2: Bottom Quadrants (Non-Responder | Early Plateau)
            st.markdown("---")
            row2_col1, row2_col2 = st.columns(2)
            
            with row2_col1:
                # Bottom Left: Non-Responder
                show_quadrant_panel(
                    "Non-Responder", 
                    "#EF4444", # Red
                    "Started Slow (< Target) and missed Goal. Needs intervention."
                )
                
            with row2_col2:
                # Bottom Right: Early Plateau
                show_quadrant_panel(
                    "Early Plateau", 
                    "#F59E0B", # Orange
                    "Started Strong (>= Target) but fell off (missed Goal). Retention issue?"
                )

        else:
             st.warning(f"No sufficient data overlap between Week {early_week} and Week {late_week}.")


    # --- 9. Medication Efficacy Hypothesis (Dual-Axis) ---
    st.header("9. Medication Efficacy Hypothesis")
    st.write("Separating the effect of the **Drug** (Outcome) vs the **Patient Effort** (Behavior).")
    st.markdown("> **Hypothesis**: Some medications drive high weight loss even with average adherence (High Potency), while others require high adherence to achieve moderate results.")

    # 1. Data Prep
    # Outcome: Avg Weight Loss at Week 12 per Med
    df_outcome = df[df['weeks_since_start'] == 12].groupby('initial_product')['pct_weight_loss'].mean().reset_index()
    df_outcome['loss_magnitude'] = df_outcome['pct_weight_loss'] * -1
    
    # Behavior: Avg Total Logs (approx adherence)
    # We count total logs per user in the whole dataset, then avg per med
    user_log_counts = df.groupby(['user_id', 'initial_product']).size().reset_index(name='total_logs')
    df_behavior = user_log_counts.groupby('initial_product')['total_logs'].mean().reset_index()
    
    # Merge
    efficacy_df = pd.merge(df_outcome, df_behavior, on='initial_product')
    
    if not efficacy_df.empty:
        # Sort by Outcome for visual clarity (High to Low bars)
        efficacy_df = efficacy_df.sort_values('loss_magnitude', ascending=False)
        
        # 2. Dual-Axis Chart
        fig_dual = go.Figure()
        
        # Bar Chart (Outcome - Left Axis)
        fig_dual.add_trace(go.Bar(
            x=efficacy_df['initial_product'],
            y=efficacy_df['loss_magnitude'],
            name='Avg Weight Loss (%)',
            marker_color=px.colors.qualitative.Prism, # Use consistent palette logic if possible, or just teal
            text=efficacy_df['loss_magnitude'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto',
            yaxis='y'
        ))
        
        # Line Chart (Behavior - Right Axis)
        fig_dual.add_trace(go.Scatter(
            x=efficacy_df['initial_product'],
            y=efficacy_df['total_logs'],
            name='Avg Total Logs (Effort)',
            mode='lines+markers',
            marker=dict(color='#1E293B', size=10), # Dark Slate
            line=dict(width=3, color='#1E293B'),
            yaxis='y2'
        ))
        
        # Layout
        fig_dual.update_layout(
            title="Efficacy (Bars) vs Adherence (Line) by Medication",
            yaxis=dict(
                title="Avg Weight Loss (%)",
                titlefont=dict(color="#1f77b4"),
                tickfont=dict(color="#1f77b4"),
                side="left",
                nticks=6,
                rangemode="tozero"
            ),
            yaxis2=dict(
                title="Avg Log Likelihood (Adherence)",
                titlefont=dict(color="#1E293B"),
                tickfont=dict(color="#1E293B"),
                anchor="x",
                overlaying="y",
                side="right",
                showgrid=False, # Grid handled by primary
                nticks=6,
                rangemode="tozero"
            ),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            height=500,
            xaxis=dict(title="Medication")
        )
        
        st.plotly_chart(fig_dual, use_container_width=True)
        
        # Insight
        # Find max discrepancy?
        # Just generalized info for now as per design
        st.info("""
        **Interpreting the Graph**:
        - **High Bar + Low Line**: **High Potency**. The drug works well even with lower patient effort.
        - **Medium Bar + High Line**: **High Effort**. Patient is working hard for the results.
        """)
    else:
        st.warning("Insufficient data to calculate efficacy metrics.")

