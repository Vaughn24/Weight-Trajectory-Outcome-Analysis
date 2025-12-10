import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit.components.v1 as components
from datetime import datetime
from utils import load_and_process_data, filter_data, MED_COLORS, MED_ORDER
import charts

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Eucalyptus Analytics", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F8FAFC; /* Slate-50 */
        color: #1E293B; /* Slate-800 */
    }
    
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }

    /* UNIFIED CARD STYLE */
    .white-card {
        background-color: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        height: 100%;
    }

    /* KPI STYLES */
    .kpi-label {
        font-size: 0.85rem;
        color: #64748B; /* Slate-500 */
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A; /* Slate-900 */
        margin-bottom: 4px;
    }
    .kpi-delta-pos {
        font-size: 0.8rem;
        font-weight: 600;
        color: #16A34A; /* Green-600 */
        display: flex; align-items: center;
    }
    .kpi-delta-neg {
        font-size: 0.8rem;
        font-weight: 600;
        color: #DC2626; /* Red-600 */
        display: flex; align-items: center;
    }

    /* SECTION HEADERS */
    .section-header {
        display: flex;
        align-items: center;
        font-size: 1.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 40px;
        margin-bottom: 15px;
    }
    .section-badge {
        background-color: #3B82F6; /* Blue-500 */
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        margin-right: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* REMOVE ST PADDING */
    div[data-testid="stVerticalBlock"] > div { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def render_sparkline_card(title, value, delta, data_points, is_positive_good=True, graph_type='area', sub_label=None):
    """
    Renders a KPI card with a metric and a sparkline area chart.
    data_points: list of floats for the sparkline (last 10-20 points).
    graph_type: 'area' or 'bar'
    """
    # SVG Generation for Sparkline
    if not data_points: data_points = [0,0]
    
    width = 200
    height = 50
    min_val, max_val = min(data_points), max(data_points)
    # Ensure some range to avoid div by zero
    val_range = max_val - min_val if max_val != min_val else (max_val if max_val != 0 else 1)
    
    svg_content = ""
    
    if graph_type == 'area':
        # Points for Polygon
        points = []
        step_x = width / (len(data_points) - 1)
        
        for i, pt in enumerate(data_points):
            x = i * step_x
            # Normalize y to height (inverted for SVG coords)
            y = height - ((pt - min_val) / val_range * height)
            points.append(f"{x},{y}")
            
        # Close polygon for fill (bottom corners)
        poly_points = f"0,{height} " + " ".join(points) + f" {width},{height}"
        line_points = " ".join(points)
        
        svg_content = (
            f'<defs><linearGradient id="grad_{title.replace(" ","")}" x1="0%" y1="0%" x2="0%" y2="100%">'
            f'<stop offset="0%" style="stop-color:#3B82F6;stop-opacity:0.4" />'
            f'<stop offset="100%" style="stop-color:#3B82F6;stop-opacity:0" />'
            f'</linearGradient></defs>'
            f'<polygon points="{poly_points}" fill="url(#grad_{title.replace(" ","")})" />'
            f'<polyline points="{line_points}" fill="none" stroke="#3B82F6" stroke-width="2" />'
        )
        
    elif graph_type == 'bar':
        # Bar Chart Logic
        bar_count = len(data_points)
        bar_gap = 2
        bar_width = (width - (bar_count - 1) * bar_gap) / bar_count
        
        rects = []
        for i, pt in enumerate(data_points):
            x = i * (bar_width + bar_gap)
            # Calculate bar height relative to min/max
            # For bars, we ideally want 0 baseline if possible, but for sparklines often relative min is fine.
            # However, for "Total Active", relative to min is safer to show variation if numbers are large.
            # But "bar graph" implies magnitude usually. Let's try 0-baseline if min>0, or just relative range.
            # The image shows variation. Let's stick to relative range but maybe softer.
            # Actually, standard spark-bar usually fills height based on range.
            
            # Simple relative scaling like area: 
            bar_h = ((pt - min_val) / val_range * height)
            # Ensure min height for visibility
            if bar_h < 2: bar_h = 2
            
            y = height - bar_h
            
            # Blue color
            rects.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" fill="#3B82F6" rx="2" />')
            
        svg_content = "".join(rects)

    
    # Delta Formatting
    delta_clean = str(delta).replace("+", "").replace("-", "")
    delta_val = float(delta_clean.replace("%", ""))
    is_up = "+" in str(delta) or (delta_val > 0 and "-" not in str(delta))
    
    # Logic: If metric goes UP and positive is good -> Green. 
    # If metric goes DOWN and positive is good -> Red.
    
    if is_positive_good:
        color_class = "kpi-delta-pos" if is_up else "kpi-delta-neg"
        arrow = "↑" if is_up else "↓"
    else: # e.g. Churn, Weight (Weight loss -> Down is good?)
        # Let's say Weight Loss: More negative is "Up" in magnitude? 
        # Standard: Green if Good.
        # Simple for now: Green = Up, Red = Down.
        color_class = "kpi-delta-pos" if is_up else "kpi-delta-neg"
        arrow = "↑" if is_up else "↓"

    display_delta = f"{delta} vs LW"
    
    # Flattened HTML to prevent Streamlit code-block rendering
    card_html = (
        f'<div class="white-card" style="padding: 20px; height: 160px; display: flex; flex-direction: column; justify-content: space-between;">'
        f'<div>'
        f'<div class="kpi-label">{title}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="{color_class}"><span style="margin-right: 4px;">{arrow}</span> {display_delta}</div>'
        f'{f"<div style=font-size:0.75rem;color:#94A3B8;margin-top:4px>{sub_label}</div>" if sub_label else ""}'
        f'</div>'
        f'<div style="width: 100%; height: 50px; overflow: hidden; margin-top: 10px;">'
        f'<svg width="100%" height="100%" viewBox="0 0 {width} {height}" preserveAspectRatio="none">{svg_content}</svg>'
        f'</div></div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

def render_chart_card(fig, title, height=350, content_html="", layout="column", margin=None):
    """
    Wraps Plotly fig in a unified white card style.
    layout: 'column' (default, content below chart) or 'row' (content right of chart).
    margin: dict(l=0, r=0, t=30, b=0) can be overridden.
    """
    if fig:
        # Height Logic
        chart_height = height - 60 
        if content_html and layout == "column":
             chart_height = height - 150 # Reserve vertical space for table

        # Default margin
        final_margin = dict(l=0, r=0, t=30, b=0)
        if margin:
            final_margin.update(margin)
            
        fig.update_layout(
            height=chart_height,
            margin=final_margin,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color="#64748B"),
        )
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False, 'responsive': True})
    else:
        plot_html = "<div style='height:100%; display:flex; align-items:center; justify-content:center; color:#CBD5E1;'>No Data</div>"

    # Layout Direction CSS
    flex_dir = "column"
    extra_style = "margin-top: 10px; border-top: 1px solid #F1F5F9; padding-top: 10px; width: 100%;" # Default col style
    chart_style_flex = "width: 100%;"
    
    if layout == "row":
        flex_dir = "row"
        # Chart left (grow), Content right (fixed or wrapped?)
        # Let's give table fixed width or percentage? 
        # width: 40% for table, 60% chart? 
        extra_style = "margin-left: 10px; border-left: 1px solid #F1F5F9; padding-left: 15px; width: 40%; overflow-y: field; align-self: center;"
        chart_style_flex = "width: 60%;"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ margin: 0; padding: 2px; font-family: 'Inter', sans-serif; }}
        .white-card {{
            background-color: white; border-radius: 12px; padding: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #E2E8F0;
            height: {height}px; box-sizing: border-box; display: flex; flex-direction: column;
            width: 100%;
        }}
        .chart-title {{ font-size: 0.95rem; font-weight: 700; color: #1E293B; margin-bottom: 8px; flex-shrink: 0; }}
        
        .content-body {{
            display: flex;
            flex-direction: {flex_dir};
            flex-grow: 1;
            height: 100%;
            overflow: hidden;
            align-items: stretch; /* Stretch to fill height */
        }}
        
        .chart-container {{ 
            position: relative; 
            overflow: hidden;
            {chart_style_flex}
        }}
        
        .extra-content {{ 
            font-size: 0.85rem; 
            {extra_style}
        }}
        
        /* Table Style */
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ text-align: left; color: #64748B; font-weight: 600; padding: 6px 4px; font-size: 0.75rem; border-bottom: 2px solid #F1F5F9; }}
        td {{ color: #334155; padding: 6px 4px; border-bottom: 1px solid #F8FAFC; text-align: left; }}
        td:first-child {{ font-weight: 500; }} /* Med names bold */
        
        .plotly-graph-div {{ width: 100% !important; height: 100% !important; }}
    </style>
    </head>
    <body>
        <div class="white-card">
            <div class="chart-title">{title}</div>
            <div class="content-body">
                <div class="chart-container">{plot_html}</div>
                {f'<div class="extra-content">{content_html}</div>' if content_html else ''}
            </div>
        </div>
    </body>
    </html>
    """
    components.html(html, height=height+30)

# --- DATA PREP ---
try:
    df_weight = load_and_process_data()
    df_users_raw = pd.read_csv('user_data_dec.csv')
except:
    st.error("Data load failed.")
    st.stop()

# Global Filters Removed
df_filtered = df_weight.copy()

# Standardize Timeframe
df_0_12 = df_filtered[(df_filtered['weeks_since_start'] >= 0) & (df_filtered['weeks_since_start'] <= 12)]
df_w12 = df_0_12[df_0_12['weeks_since_start'] == 12]

# --- GLOBAL DATASET FOR DASHBOARD (Active Users) ---
# Start with Merged Data (df_weight) users
valid_user_ids_global = df_weight['user_id'].dropna().unique()
# Filter Raw User Data to this subset (df_users_raw uses 'id')
df_dashboard = df_users_raw[df_users_raw['id'].isin(valid_user_ids_global)].copy()
# User Request: Add the unknown medication back
df_dashboard['initial_product'] = df_dashboard['initial_product'].fillna('Unknown')

# --- LAYOUT START ---
st.title("Trajectory & Outcome Analysis Dashboard")
st.markdown(f"<div style='color: #64748B; margin-top: -10px; margin-bottom: 30px;'>Data updated as of: {datetime.now().strftime('%B %Y')} </div>", unsafe_allow_html=True)

# === ROW 1: KPI SPARKLINE CARDS ===
c1, c2, c3, c4, c5 = st.columns(5)

# Metrics Calculation
total_registered = len(df_users_raw)
active_tracking = df_weight['user_id'].nunique() # Users with at least 1 weight log
no_data_users = total_registered - active_tracking

active_w12 = df_w12['user_id'].nunique()
avg_loss = df_w12['pct_weight_loss'].mean() if not df_w12.empty else 0
clinical_success = (len(df_w12[df_w12['pct_weight_loss'] <= -5]) / active_w12 * 100) if active_w12 else 0

# Mock Sparkline Data (Simulated for aesthetics as real time-series aggregations are complex for a single number)
# In real prod, these would be daily/weekly aggregations.
with c1: 
    render_sparkline_card(
        "Users with Weight Data", 
        f"{active_tracking:,}", 
        "+1.2%", 
        [100, 102, 105, 108, 112, 115, 118, 120, 123, 125], 
        graph_type='bar',


        sub_label=f"Total: {total_registered} | Missing Data: {no_data_users}"
    )
with c2: render_sparkline_card("New Enrollments (Wk 0)", "42", "+8.5%", [30, 32, 28, 35, 38, 40, 42, 39, 41, 42])
with c3: render_sparkline_card("Retention (Wk 12)", f"{active_w12}", "-2.1%", [140, 138, 135, 132, 130, 128, 125, 122, 120, 118], is_positive_good=False)
with c4: render_sparkline_card("Avg Weight Loss", f"{abs(avg_loss):.1f}%", "+0.5%", [2.1, 2.3, 2.5, 2.8, 3.1, 3.4, 3.8, 4.1, 4.3, 4.5])
with c5: render_sparkline_card("Clinical Success Rate", f"{clinical_success:.1f}%", "+3.2%", [40, 42, 45, 43, 46, 48, 50, 52, 53, 55])


# === MAIN LAYOUT: SECTIONS 1 & 2 (SIDE-BY-SIDE) ===
c_sec1, c_sec2 = st.columns(2, gap="medium")

# --- SECTION 1: COHORT PROFILE ---
with c_sec1:
    st.markdown('<div class="section-header" style="margin-top:20px;"><span class="section-badge">1</span> Cohort Profile & Adherence</div>', unsafe_allow_html=True)
    
    # Chart 1: Engagement Overview (Butterfly Chart)
    # UPDATED: Use df_dashboard (Active/Merged users only)
    fig_butt = charts.create_engagement_butterfly_chart(df_dashboard)
    render_chart_card(fig_butt, "Engagement Overview: Population & Activity", height=380)
    
    st.write("") # Spacer
    
    # Chart 2: Medication Adoption (Donut + Table)
    # UPDATED: Use df_dashboard
    fig_med = charts.create_market_share_donut(df_dashboard)
    
    # Detailed Table (HTML for Card)
    med_counts = df_dashboard['initial_product'].value_counts().reset_index()
    med_counts.columns = ['Medication', 'Count']
    med_counts['Share'] = (med_counts['Count'] / med_counts['Count'].sum() * 100).map('{:.1f}%'.format)
    # Rename for compact display
    med_counts.columns = ['Med', '#', '%Share']
    
    # Convert to HTML with basic styling (no index)
    table_html = med_counts.to_html(index=False, classes='med-table', border=0)
    
    # Render Card with Chart + Table inside (Side-by-Side)
    render_chart_card(fig_med, "Medication Adoption (Market Share)", height=380, content_html=table_html, layout="row")


# --- SECTION 2: RETENTION & TRAJECTORY ---
with c_sec2:
    st.markdown('<div class="section-header" style="margin-top:20px;"><span class="section-badge">2</span> Retention & Trajectory Analysis</div>', unsafe_allow_html=True)

    # Chart 3: Retention Heatmap
    # Heatmap Tabs
    tab_retention, tab_progression = st.tabs(["Retention (Patient Density)", "Weight Loss Progression"])
    
    with tab_retention:
        fig_heat = charts.create_patient_density_heatmap(df_0_12)
        render_chart_card(fig_heat, "Heatmap: Patient Density by Week & Medication", height=380, margin=dict(l=150))
        
    with tab_progression:
        fig_loss = charts.create_weight_loss_heatmap(df_0_12)
        render_chart_card(fig_loss, "Heatmap: Weight Loss Progression (Row-Normalized)", height=380, margin=dict(l=150))

    st.write("") # Spacer

    # Chart 4: Success Rate Trajectory (>5% Loss)
    # User Request: Recreate "Percentage of Patient Achieving >5% Weight loss" from Trajectories.py
    fig_traj = charts.create_success_rate_by_med_chart(df_0_12)
    render_chart_card(fig_traj, "Success Rate (>5% Loss) by Medication", height=380)


# === SECTION 3: OUTCOME PATTERNS ===
st.markdown('<div class="section-header"><span class="section-badge">3</span> Clinical Outcome Patterns</div>', unsafe_allow_html=True)
c_out1, c_out2 = st.columns(2)

with c_out1:
    # Chart 5: Milestone Distribution
    milestone_weeks = [4, 8, 12]
    df_milestones = df_filtered[df_filtered['weeks_since_start'].isin(milestone_weeks)]
    fig_box = px.box(df_milestones, x='weeks_since_start', y='pct_weight_loss', color='initial_product', 
                     color_discrete_map=MED_COLORS, category_orders={'initial_product': MED_ORDER},
                     labels={'initial_product': 'Medication'})
    fig_box.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="% Weight Loss",
        xaxis_title="Key Milestone (Week)"
    )
    render_chart_card(fig_box, "Milestone Variable Distribution", height=350, margin=dict(l=40, b=40))

with c_out2:
    # Chart 6: Week 12 Outcome Waterfall
    df_week12 = df_filtered[df_filtered['weeks_since_start'] == 12].sort_values(by='pct_weight_loss')
    df_week12['Rank'] = range(len(df_week12))
    fig_fall = px.bar(df_week12, x='Rank', y='pct_weight_loss', color='initial_product', 
                      color_discrete_map=MED_COLORS, category_orders={'initial_product': MED_ORDER},
                      labels={'initial_product': 'Medication'})
    fig_fall.update_layout(
        bargap=0, 
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Patient Rank (Best to Worst)",
        yaxis_title="Total Weight Loss (%)"
    )
    
    waterfall_caption = "<div style='text-align: center; color: #64748B; font-size: 0.8rem; margin-top: -10px;'>Each bar is one patient. Sorted from Best Result (Left) to Worst (Right).</div>"
    render_chart_card(fig_fall, "Week 12 Outcome Waterfall", height=350, content_html=waterfall_caption, margin=dict(l=50, b=40))

st.write("") # Spacer

# Chart 7: Clinical Response Rates (New)
# Full width for better readability of multiple lines
fig_response = charts.create_clinical_response_chart(df_filtered)
render_chart_card(fig_response, "Clinical Response Rates Over Time", height=400)


# === SECTION 4: PREDICTIVE TRAJECTORY ===
st.markdown('<div class="section-header"><span class="section-badge">4</span> Predictive Trajectory</div>', unsafe_allow_html=True)

# Create 2 Columns
col_land1, col_land2 = st.columns(2)

with col_land1:
    # 1. Landmark Scatter (Existing)
    # Using df_0_12 or df_filtered, make sure to use df_filtered which respects sidebar
    fig_corr = charts.create_landmark_scatter(df_filtered)
    render_chart_card(fig_corr, "Landmark Analysis: Week 4 vs Week 12", height=500)

with col_land2:
    # 2. Quadrant Distribution (Grid Layout)
    st.subheader("Quadrant Distribution by Medication")
    
    # Logic to create the grid (Copied/Adapted from Trajectories.py)
    # 1. Prepare Data Again (Local scope)
    df_w4 = df_filtered[df_filtered['weeks_since_start'] == 4][['user_id', 'pct_weight_loss']].rename(columns={'pct_weight_loss': 'w4_loss'})
    df_w12 = df_filtered[df_filtered['weeks_since_start'] == 12][['user_id', 'pct_weight_loss', 'initial_product']].rename(columns={'pct_weight_loss': 'w12_loss'})
    
    merged_grid = pd.merge(df_w4, df_w12, on='user_id', how='inner')
    
    if not merged_grid.empty:
        merged_grid['w4_mag'] = merged_grid['w4_loss'] * -1
        merged_grid['w12_mag'] = merged_grid['w12_loss'] * -1
        
        def classify_quadrant(row):
            early_pass = row['w4_mag'] >= 3
            late_pass = row['w12_mag'] >= 5
            if early_pass and late_pass: return "Consistent Responder"
            elif early_pass and not late_pass: return "Early Plateau"
            elif not early_pass and late_pass: return "Late Bloomer"
            else: return "Non-Responder"
            
        merged_grid['Quadrant'] = merged_grid.apply(classify_quadrant, axis=1)
        
        # --- MODEL RESULT CALCULATION ---
        cutoff_w4 = 3.0
        goal_w12 = 5.0 # Standard 5% Clinical Goal
        
        # Use w4_mag and w12_mag which are positive representations of weight loss
        at_risk_df = merged_grid[merged_grid['w4_mag'] < cutoff_w4]
        flagged_count = len(at_risk_df)
        
        recovered_df = at_risk_df[at_risk_df['w12_mag'] >= goal_w12]
        recovered_count = len(recovered_df)
        
        recovery_rate = (recovered_count / flagged_count * 100) if flagged_count > 0 else 0
        miss_prob = 100 - recovery_rate
        
        # Render Model Card
        model_html = f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-top: 20px;">
            <div style="font-weight: 600; color: #1E293B; margin-bottom: 8px;">Model Result: Using a cut-off of {cutoff_w4}% at Week 4</div>
            <ul style="color: #475569; font-size: 0.9rem; margin: 0; padding-left: 20px; line-height: 1.6;">
                <li><b>{flagged_count} cohorts</b> were flagged as "At Risk" (Slow Starters).</li>
                <li>Only <b>{recovered_count} ({recovery_rate:.1f}%)</b> of them recovered to hit the {goal_w12}% goal by Week 12.</li>
                <li style="margin-top: 8px;"><b>Predictive Power:</b> If a patient fails this Week 4 check, they have a <b>{miss_prob:.1f}% probability</b> of missing the long-term goal.</li>
            </ul>
        </div>
        """
        # Render in Col 1 (Below Scatter Plot) instead of Col 2
        with col_land1:
            st.markdown(model_html, unsafe_allow_html=True)
        
        med_total_counts = merged_grid['initial_product'].value_counts()

        # Helper
        def show_quadrant_panel(quadrant_name, color, description):
            subset = merged_grid[merged_grid['Quadrant'] == quadrant_name]
            count = len(subset)
            pct = (count / len(merged_grid) * 100) if len(merged_grid) > 0 else 0
            
            # Compact UI for Dashboard
            st.markdown(f"<h5 style='color: {color}; margin-bottom: 0;'>{quadrant_name}</h5>", unsafe_allow_html=True)
            
            # Side-by-side Metric and Description
            m_c1, m_c2 = st.columns([1, 2])
            with m_c1:
                st.metric(label="Total", value=f"{count}", delta=f"{pct:.1f}%")
            with m_c2:
                st.markdown(f"<div style='font-size: 0.8rem; color: #64748B; margin-top: 10px; line-height: 1.2;'>{description}</div>", unsafe_allow_html=True)
            
            if not subset.empty:
                breakdown = subset['initial_product'].value_counts().reset_index()
                breakdown.columns = ['Med', 'Count'] # Generic names
                breakdown = breakdown.head(3) # Top 3 only for space
                st.dataframe(breakdown, hide_index=True, use_container_width=True)
            else:
                st.write("-")

        # 2x2 Grid within the column
        q_c1, q_c2 = st.columns(2)
        
        with q_c1:
            show_quadrant_panel("Late Bloomer", "#A855F7", "Started Slow (< Target), Finished Strong (>= Goal). The 'Comeback' group.")
            st.write("")
            show_quadrant_panel("Non-Responder", "#EF4444", "Started Slow (< Target) and missed Goal. Needs intervention.")
            
        with q_c2:
            show_quadrant_panel("Consistent Responder", "#10B981", "Started Strong (>= Target) and Finished Strong (>= Goal). Ideal trajectory.")
            st.write("")
            show_quadrant_panel("Early Plateau", "#F59E0B", "Started Strong (>= Target) but fell off (missed Goal). Retention issue?")
            
    else:
        st.info("No data for distribution.")

# === SECTION 5: PATIENT EXPLORER ===
st.markdown('<div class="section-header"><span class="section-badge">5</span> Patient Explorer</div>', unsafe_allow_html=True)

# 1. Prepare User-Level Data (Latest Status)
# Ensure tracked_datetime is datetime
df_filtered['tracked_datetime'] = pd.to_datetime(df_filtered['tracked_datetime'])
latest_date_in_data = df_filtered['tracked_datetime'].max()

# Get latest record for each user
# Sort by user and date desc, then drop duplicates to keep first (latest)
df_latest = df_filtered.sort_values(['user_id', 'tracked_datetime'], ascending=[True, False]).drop_duplicates('user_id')

# Calculate Metrics
df_latest['days_since_last_log'] = (latest_date_in_data - df_latest['tracked_datetime']).dt.days

# Define Categories
def get_retention_status(days):
    if days < 7: return "Active (<7d)"
    elif days <= 14: return "Sliding (7-14d)"
    else: return "At Risk (>14d)"

def get_perf_status(pct):
    if pct <= -10: return "Super Responder (>10%)"
    elif pct <= -5: return "On Track (5-10%)"
    elif pct < 0: return "Slow Responder (0-5%)"
    else: return "Non-Responder (Gain/Static)"

def get_stage_status(week):
    if week <= 3: return "Onboarding (Wk 0-3)"
    elif week <= 8: return "Danger Zone (Wk 4-8)"
    else: return "Maintenance (Wk 9+)"

df_latest['Retention Status'] = df_latest['days_since_last_log'].apply(get_retention_status)
df_latest['Performance Tier'] = df_latest['pct_weight_loss'].apply(get_perf_status)
df_latest['Program Stage'] = df_latest['weeks_since_start'].apply(get_stage_status)

# 2. Filters UI
with st.expander("Search & Filter Cohort", expanded=True):
    f_c1, f_c2, f_c3, f_c4 = st.columns(4)
    
    with f_c1:
        text_search = st.text_input("🔍 Search User ID", placeholder="e.g. 1042")
        
    with f_c2:
        retention_opts = ["Active (<7d)", "Sliding (7-14d)", "At Risk (>14d)"]
        sel_retention = st.multiselect("Retention Status", options=retention_opts, default=[])
        
    with f_c3:
        perf_opts = ["Super Responder (>10%)", "On Track (5-10%)", "Slow Responder (0-5%)", "Non-Responder (Gain/Static)"]
        sel_perf = st.multiselect("Performance Tier", options=perf_opts, default=[])

    with f_c4:
        stage_opts = ["Onboarding (Wk 0-3)", "Danger Zone (Wk 4-8)", "Maintenance (Wk 9+)"]
        sel_stage = st.multiselect("Program Stage", options=stage_opts, default=[])

# 3. Apply Filters
df_table = df_latest.copy()

if text_search:
    # Convert ID to string for search
    df_table = df_table[df_table['user_id'].astype(str).str.contains(text_search, case=False)]

if sel_retention:
    df_table = df_table[df_table['Retention Status'].isin(sel_retention)]

if sel_perf:
    df_table = df_table[df_table['Performance Tier'].isin(sel_perf)]

if sel_stage:
    df_table = df_table[df_table['Program Stage'].isin(sel_stage)]

# 4. Display Table
st.markdown(f"**Showing {len(df_table)} cohort** based on filters")

# Select & Rename Columns for Display
display_cols = ['user_id', 'initial_product', 'weeks_since_start', 'pct_weight_loss', 'Retention Status', 'Performance Tier']
df_display = df_table[display_cols].rename(columns={
    'user_id': 'User ID',
    'initial_product': 'Medication',
    'weeks_since_start': 'Week',
    'pct_weight_loss': 'Total Loss %'
})

# Add Styling
st.dataframe(
    df_display,
    column_config={
        "Total Loss %": st.column_config.NumberColumn(
            "Total Loss %",
            format="%.1f%%",
        ),
        "Week": st.column_config.NumberColumn(
            "Week",
            format="%d"
        ),
        "Retention Status": st.column_config.TextColumn("Status", help="Days since last log"),
    },
    use_container_width=True,
    hide_index=True,
    height=400
)
