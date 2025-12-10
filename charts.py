import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import MED_COLORS, MED_ORDER

def create_patient_density_heatmap(df_filtered):
    """
    Generates the 'Patient Density by Week & Medication' heatmap.
    Color intensity indicates retention relative to Week 0.
    """
    # 1. Pivot Data
    heatmap_data = df_filtered.pivot_table(
        index='initial_product', 
        columns='weeks_since_start', 
        values='user_id', 
        aggfunc='nunique', 
        fill_value=0
    )
    
    # Reindex
    valid_order = [m for m in MED_ORDER if m in heatmap_data.index]
    heatmap_data = heatmap_data.reindex(valid_order)
    
    # 2. Normalize relative to Week 0
    # If Week 0 count is 0, retention is undefined (NaN)
    week_0_counts = heatmap_data[0] if 0 in heatmap_data.columns else heatmap_data.max(axis=1)
    heatmap_data_norm = heatmap_data.div(week_0_counts.replace(0, np.nan), axis=0)
    
    # 3. Plot (Using go.Heatmap for explicit control)
    # Ensure index is treated as list of strings
    y_labels = heatmap_data.index.astype(str).tolist()
    x_labels = heatmap_data.columns.tolist()
    z_values = heatmap_data_norm.values
    text_values = heatmap_data.values

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        text=text_values,
        texttemplate="%{text}",
        hovertemplate="Medication: %{y}<br>Week: %{x}<br>Active Patients: %{text}<br>Retention (vs Wk0): %{z:.0%}<extra></extra>",
        colorscale="Blues",
        showscale=False
    ))
    
    fig.update_layout(
        title="", # Title is handled by card wrapper
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(side="top", title="Week"),
        yaxis=dict(title="Medication", side="left", autorange="reversed", type='category') # Reversed to match table order (Top-Down)
    )
    
    return fig

def create_weight_loss_heatmap(df_filtered):
    """
    Generates the 'Weight Loss Intensity Heatmap' (Row-Normalized).
    Color intensity indicates % of Peak Weight Loss achieved.
    """
    # 1. Pivot for Average Weight Loss
    weight_heatmap_raw = df_filtered.pivot_table(
        index='initial_product', 
        columns='weeks_since_start', 
        values='pct_weight_loss', 
        aggfunc='mean', 
        fill_value=0
    )

    # Reindex to enforce custom order
    valid_order_weight = [m for m in MED_ORDER if m in weight_heatmap_raw.index]
    weight_heatmap_raw = weight_heatmap_raw.reindex(valid_order_weight)

    # 2. Normalize Row-wise (Relative to PEAK LOSS, which is the minimum negative number)
    # Avoid div by zero if max loss is 0.
    row_mins = weight_heatmap_raw.min(axis=1)
    # Handle edge case where min is 0 (no weight loss yet) to avoid NaN
    weight_heatmap_norm = weight_heatmap_raw.div(row_mins.replace(0, -0.0001), axis=0)
    
    # 3. Plot
    # Explicitly cast index to list to ensure Plotly uses strings as labels
    y_labels = weight_heatmap_raw.index.astype(str).tolist()
    x_labels = weight_heatmap_raw.columns.tolist()
    z_values = weight_heatmap_norm.values
    text_values = weight_heatmap_raw.values

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        text=text_values,
        texttemplate="%{text:.1f}%",
        hovertemplate="Medication: %{y}<br>Week: %{x}<br>Avg Loss: %{text:.2f}%<br>Progress: %{z:.0%}<extra></extra>",
        colorscale="Greens",
        showscale=False
    ))
    
    fig.update_layout(
        title="", # Handled by card
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(side="top", title=None),
        yaxis=dict(title=None, side="left", autorange="reversed", type='category')
    )
    
    return fig

def create_engagement_butterfly_chart(df_unique_users):
    """
    Generates a Butterfly Chart: User Count vs Avg Interactions by Age Group.
    Left Side (Grey): User Count (Volume)
    Right Side (Orange): Avg Interactions (Engagement)
    Center: Age Groups (<=25 to 56+)
    """
    # Filter for all (no age threshold needed if bins handle it, but catch errors)
    df_age_filtered = df_unique_users.copy() # No filter, keep all ages

    # Create Bins: <=25, 26-35, 36-45, 46-55, 56+
    df_age_filtered['age_group_20'] = pd.cut(
        pd.to_numeric(df_age_filtered['age_at_initial_order'], errors='coerce'), 
        bins=[0, 25, 35, 45, 55, 100], 
        labels=['<=25', '26-35', '36-45', '46-55', '56+'],
        include_lowest=True
    )
    
    # Determine ID column for counting (Dashboard passes 'id', EDA passes 'user_id')
    id_col = 'user_id' if 'user_id' in df_age_filtered.columns else 'id'

    # Aggregate Data
    df_age_agg = df_age_filtered.groupby('age_group_20', observed=True).agg(
        user_count=(id_col, 'count'),
        avg_interaction=('app_interactions_count', 'mean')
    ).reset_index()
    
    # Imports for Subplots
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Create Subplots: 1 Row, 2 Cols, Shared Y
    fig_butt = make_subplots(
        rows=1, cols=2, 
        shared_yaxes=True, 
        horizontal_spacing=0.15, # Space for center labels
        subplot_titles=("User Count (Volume)", "Avg App Interactions (Engagement)")
    )
    
    # Trace 1: User Count (Left)
    fig_butt.add_trace(
        go.Bar(
            y=df_age_agg['age_group_20'], 
            x=df_age_agg['user_count'], 
            orientation='h',
            name='User Count',
            text=df_age_agg['user_count'],
            textposition='auto',
            marker_color='#94a3b8' # Slate-400 (Neutral)
        ), 
        row=1, col=1
    )
    
    # Trace 2: Avg Interactions (Right)
    fig_butt.add_trace(
        go.Bar(
            y=df_age_agg['age_group_20'], 
            x=df_age_agg['avg_interaction'], 
            orientation='h',
            name='Avg Interactions',
            text=df_age_agg['avg_interaction'].round(1),
            textposition='auto',
            marker_color='#ff9f4b' # Orange (Highlight)
        ), 
        row=1, col=2
    )
    
    # Layout Config for Butterfly Effect
    fig_butt.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=50, r=50, t=60, b=50), # Increased margins for titles
        yaxis=dict(
            autorange='reversed', # Youngest Top
            showticklabels=False,  # Hide default labels
        )
    )

    # Add Age Group Header
    fig_butt.add_annotation(
        x=0.5, y=1.1, 
        xref="paper", yref="paper",
        text="Age Group",
        showarrow=False,
        font=dict(size=12, color="gray")
    )
    
    # Add Center Age Labels
    for age in df_age_agg['age_group_20']:
        fig_butt.add_annotation(
            x=0.5, y=age,
            xref="paper", yref="y",
            text=str(age),
            showarrow=False,
            font=dict(size=12, color="black"),
            xanchor="center"
        )

    # Specific X-Axis Configs
    fig_butt.update_xaxes(
        row=1, col=1, 
        autorange='reversed', # Left side reversed
    )
    fig_butt.update_xaxes(
        row=1, col=2,
    )
    
    return fig_butt

def create_market_share_donut(df):
    """
    Generates a Donut Chart for Medication Market Share.
    Includes center text for Total Users.
    """
    # 1. Aggregate
    share_counts = df['initial_product'].value_counts().reset_index()
    share_counts.columns = ['Medication', 'Count']
    total_users = share_counts['Count'].sum()
    
    # 2. Plot Donut
    fig_pie = px.pie(
        share_counts, 
        values='Count', 
        names='Medication', 
        title="Market Share Distribution",
        color='Medication',
        color_discrete_map=MED_COLORS,
        hole=0.5, # Donut
        labels={'Medication': 'Medication', 'initial_product': 'Medication'}
    )
    
    # 3. Style
    fig_pie.update_traces(
        textinfo='percent+label', 
        textposition='inside',
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"
    )
    
    # Center Text
    fig_pie.update_layout(
        annotations=[dict(text=f"Total<br>{total_users}", x=0.5, y=0.5, font_size=20, showarrow=False)],
        showlegend=False, # Legend is redundant if labels are inside/callout, but for many slices legend is better?
        # User wants "more details". Legend + Table below is good.
        # Let's keep legend but maybe bottom?
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=40, b=50, l=20, r=20)
    )
    
    return fig_pie

def create_landmark_scatter(df):
    """
    Generates the 'Landmark Analysis: Week 4 vs Week 12' scatter plot with Quadrants.
    Requires dataframe with 'weeks_since_start', 'user_id', 'pct_weight_loss', 'initial_product'.
    """
    # 1. Prepare Data: Join Week 4 and Week 12
    df_w4 = df[df['weeks_since_start'] == 4][['user_id', 'pct_weight_loss']].rename(columns={'pct_weight_loss': 'w4_loss'})
    df_w12 = df[df['weeks_since_start'] == 12][['user_id', 'pct_weight_loss', 'initial_product']].rename(columns={'pct_weight_loss': 'w12_loss'})
    
    landmark_df = pd.merge(df_w4, df_w12, on='user_id', how='inner')
    
    if landmark_df.empty:
        # Return empty fig with message or None
        fig_empty = go.Figure()
        fig_empty.update_layout(title="No Overlapping Data (Week 4 & 12)")
        return fig_empty

    # Convert to Positive Magnitude for Plotting if desired (Trajectories.py does this)
    # The user request asks to "copy" it.
    landmark_df['w4_mag'] = landmark_df['w4_loss'] * -1
    landmark_df['w12_mag'] = landmark_df['w12_loss'] * -1

    fig_land = px.scatter(
        landmark_df,
        x='w4_mag',
        y='w12_mag',
        color='initial_product',
        labels={
            'w4_mag': 'Week 4 Weight Loss (%)',
            'w12_mag': 'Week 12 Weight Loss (%)',
            'initial_product': 'Medication'
        },
        hover_data=['user_id'],
        color_discrete_map=MED_COLORS,
        category_orders={'initial_product': MED_ORDER}
    )
    
    # Add Quadrant Lines
    fig_land.add_vline(x=3, line_dash="dash", line_color="gray", annotation_text="W4 Cut-off (3%)")
    fig_land.add_hline(y=5, line_dash="dash", line_color="gray", annotation_text="W12 Target (5%)")
    
    # Add Quadrant Labels (Annotations)
    # Positions (x,y) might need tuning if data range is small, but using Trajectories defaults.
    fig_land.add_annotation(x=1, y=14, text="Late Bloomers", showarrow=False, font=dict(color="#A855F7", size=10))
    fig_land.add_annotation(x=6, y=14, text="Consistent Responders", showarrow=False, font=dict(color="green", size=10))
    fig_land.add_annotation(x=1, y=2, text="Non-Responders", showarrow=False, font=dict(color="red", size=10))
    fig_land.add_annotation(x=6, y=2, text="Early Plateau", showarrow=False, font=dict(color="orange", size=10))

    fig_land.update_layout(height=500)
    
    # User Request: Remove solid axis lines, keep broken quadrant lines
    fig_land.update_xaxes(showline=False, zeroline=False)
    fig_land.update_yaxes(showline=False, zeroline=False)
    
    return fig_land

def create_clinical_response_chart(df_filtered):
    """
    Generates a Multi-Line Chart showing % of patients hitting clinical targets over time.
    Targets: >5% Loss, >10% Loss, >15% Loss.
    """
    # 1. Define Targets
    targets = {
        '>5% Loss': -5, 
        '>10% Loss': -10, 
        '>15% Loss': -15
    }
    
    # 2. Process Data Per Week
    # We need to calculate % of active users per week who met the criteria
    # active users = unique users with a log in that week? Or cohort based?
    # Usually logged users in that week.
    
    weeks = sorted(df_filtered['weeks_since_start'].unique())
    # User Request: Focus on Week 1 to Week 12
    weeks = [w for w in weeks if 1 <= w <= 12]
    results = []

    for w in weeks:
        week_data = df_filtered[df_filtered['weeks_since_start'] == w]
        total_tracked = week_data['user_id'].nunique()
        
        if total_tracked > 0:
            row = {'Week': w}
            for label, threshold in targets.items():
                # Success count: pct_weight_loss <= threshold (e.g. -5)
                success_count = len(week_data[week_data['pct_weight_loss'] <= threshold])
                row[label] = (success_count / total_tracked) * 100
            results.append(row)
            
    df_trends = pd.DataFrame(results)
    
    if df_trends.empty:
        return px.line(title="No Data for Clinical Response")

    # 3. Create Multi-Line Plot
    # Melt for Plotly
    df_melt = df_trends.melt('Week', var_name='Target', value_name='Percentage')
    
    # Custom Colors for Targets
    target_colors = {
        '>5% Loss': '#3B82F6',   # Blue
        '>10% Loss': '#8B5CF6',  # Purple
        '>15% Loss': '#10B981'   # Green
    }
    
    fig = px.line(
        df_melt, 
        x='Week', 
        y='Percentage', 
        color='Target',
        title='Clinical Response Rates Over Time',
        color_discrete_map=target_colors,
        markers=True
    )
    
    fig.update_layout(
        yaxis_title="% Patients",
        xaxis_title="Weeks Since Start",
        yaxis=dict(range=[0, 100]), # 0-100% scale
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_success_rate_by_med_chart(df_filtered):
    """
    Generates a Multi-Line Chart showing % of patients hitting >5% Weight Loss
    grouped by Medication (Initial Product) over time.
    """
    # 1. Calculate Success Metric (pct_weight_loss <= -5)
    # Use a copy to avoid SettingWithCopy warnings if df is a slice
    df_calc = df_filtered.copy()
    df_calc['achieved_5pct'] = df_calc['pct_weight_loss'] <= -5
    
    # 2. Group by Week and Medication
    success_rate = df_calc.groupby(['weeks_since_start', 'initial_product'])['achieved_5pct'].mean().reset_index()
    success_rate['percent_success'] = success_rate['achieved_5pct'] * 100
    
    # 3. Create Line Chart
    fig = px.line(
        success_rate,
        x='weeks_since_start',
        y='percent_success',
        color='initial_product',
        markers=True,
        color_discrete_map=MED_COLORS,
        category_orders={'initial_product': MED_ORDER},
        labels={'percent_success': '% Patients achieving >5% Loss', 'weeks_since_start': 'Week', 'initial_product': 'Medication'}
    )
    
    # User Request: Add 50% threshold line
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% Threshold", annotation_position="bottom right")

    fig.update_layout(
        yaxis_title="% Success (>5% Loss)",
        xaxis_title="Week",
        yaxis=dict(range=[0, 100]), # 0-100% scale
        margin=dict(l=0, r=0, t=30, b=0),
        height=300,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig
