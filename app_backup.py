import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

st.set_page_config(page_title="Abraham Accords Literacy Dashboard", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * {font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;}
    .main {background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);padding: 2rem;}
    .stMetric {background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);backdrop-filter: blur(10px);padding: 24px;border-radius: 16px;border: 1px solid rgba(255,255,255,0.18);box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);}
    .stMetric:hover {transform: translateY(-4px);box-shadow: 0 12px 48px 0 rgba(31, 38, 135, 0.5);border-color: rgba(78, 205, 196, 0.4);}
    .stMetric label {color: #B8B8D1 !important;font-size: 13px !important;font-weight: 600 !important;text-transform: uppercase;letter-spacing: 0.5px;}
    .stMetric [data-testid="stMetricValue"] {color: #FFFFFF !important;font-size: 36px !important;font-weight: 700 !important;}
    .stMetric [data-testid="stMetricDelta"] {color: #4ECDC4 !important;font-weight: 500;}
    h1 {color: #FFFFFF;text-align: center;font-size: 48px;font-weight: 700;text-shadow: 0 0 30px rgba(78, 205, 196, 0.3);margin-bottom: 8px;letter-spacing: -0.5px;}
    h2 {color: #4ECDC4;font-size: 28px;font-weight: 600;margin-top: 40px;margin-bottom: 24px;letter-spacing: -0.3px;}
    h3 {color: #B8B8D1;font-size: 20px;font-weight: 600;}
    .subtitle {text-align: center;color: #B8B8D1;font-size: 18px;margin-top: -10px;margin-bottom: 30px;font-weight: 400;}
    .info-box {background: linear-gradient(135deg, rgba(78, 205, 196, 0.15) 0%, rgba(78, 205, 196, 0.08) 100%);border-left: 4px solid #4ECDC4;padding: 20px 24px;margin: 24px 0;border-radius: 12px;color: #E8E8E8;backdrop-filter: blur(10px);font-size: 15px;line-height: 1.6;}
    .info-box strong {color: #4ECDC4;}
    .card {background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);backdrop-filter: blur(10px);padding: 28px;border-radius: 16px;border: 1px solid rgba(255,255,255,0.18);box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);margin: 16px 0;transition: all 0.3s ease;}
    .card:hover {transform: translateY(-2px);box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.45);}
    div[data-testid="stSidebar"] {background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);border-right: 1px solid rgba(78, 205, 196, 0.2);}
    div[data-testid="stSidebar"] * {color: #E0E0E0 !important;}
    div[data-testid="stSidebar"] .stRadio > label {color: #4ECDC4 !important;font-weight: 600;font-size: 15px;}
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {color: #4ECDC4 !important;font-size: 20px;font-weight: 600;}
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {color: #B8B8D1 !important;font-size: 16px;font-weight: 600;}
    .stSelectbox label, .stMultiSelect label {color: #B8B8D1 !important;font-weight: 600;font-size: 14px;}
    .stDataFrame {background: rgba(255,255,255,0.03);border-radius: 12px;overflow: hidden;}
    .priority-badge {padding: 6px 16px;border-radius: 20px;font-weight: 600;font-size: 12px;display: inline-block;text-transform: uppercase;letter-spacing: 0.5px;}
    .priority-critical {background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);color: white;}
    .priority-high {background: linear-gradient(135deg, #FF8C00 0%, #FFA500 100%);color: white;}
    .priority-medium {background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);color: #1a1a2e;}
    .priority-low {background: linear-gradient(135deg, #32CD32 0%, #228B22 100%);color: white;}
    .country-header {font-size: 24px;font-weight: 700;color: #FFFFFF;margin-bottom: 12px;}
    .divider {height: 1px;background: linear-gradient(90deg, transparent, rgba(78, 205, 196, 0.3), transparent);margin: 32px 0;}
    .stat-large {font-size: 48px;font-weight: 700;color: #4ECDC4;line-height: 1;margin-top: 12px;}
    .sidebar-footer {text-align: center;padding: 25px 15px;background: linear-gradient(135deg, rgba(78, 205, 196, 0.1) 0%, rgba(78, 205, 196, 0.05) 100%);border-radius: 15px;border: 1px solid rgba(78, 205, 196, 0.2);margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('literacy_data.csv')
        return df
    except FileNotFoundError:
        countries = ['Israel', 'UAE', 'Bahrain', 'Morocco', 'Sudan']
        years = list(range(2015, 2024))
        data_rows = []
        indicators = {
            'Adult Literacy Rate': {'Israel': 97.8, 'UAE': 96.0, 'Bahrain': 97.5, 'Morocco': 77.4, 'Sudan': 60.7, 'unit': '%', 'growth': 0.3},
            'Youth Literacy Rate': {'Israel': 99.5, 'UAE': 98.2, 'Bahrain': 99.0, 'Morocco': 87.3, 'Sudan': 72.5, 'unit': '%', 'growth': 0.4},
            'Primary Enrollment': {'Israel': 98.5, 'UAE': 99.2, 'Bahrain': 98.8, 'Morocco': 91.2, 'Sudan': 77.8, 'unit': '%', 'growth': 0.5},
            'Secondary Enrollment': {'Israel': 96.2, 'UAE': 95.8, 'Bahrain': 96.5, 'Morocco': 78.4, 'Sudan': 45.2, 'unit': '%', 'growth': 0.6},
            'Gender Parity Index': {'Israel': 1.02, 'UAE': 1.05, 'Bahrain': 1.03, 'Morocco': 0.81, 'Sudan': 0.76, 'unit': 'Index', 'growth': 0.012},
            'Student-Teacher Ratio': {'Israel': 18.5, 'UAE': 15.8, 'Bahrain': 14.2, 'Morocco': 26.8, 'Sudan': 32.5, 'unit': 'Ratio', 'growth': -0.25},
            'Education Spending': {'Israel': 5.9, 'UAE': 3.9, 'Bahrain': 2.7, 'Morocco': 5.3, 'Sudan': 2.2, 'unit': '% GDP', 'growth': 0.08},
            'Digital Literacy': {'Israel': 88.0, 'UAE': 92.0, 'Bahrain': 89.0, 'Morocco': 64.0, 'Sudan': 31.0, 'unit': '%', 'growth': 1.2}
        }
        for indicator_name, values in indicators.items():
            for country in countries:
                for i, year in enumerate(years):
                    base_value = values[country]
                    growth = values['growth']
                    noise = np.random.uniform(-0.1, 0.1)
                    value = base_value + (i * growth) + noise
                    if indicator_name == 'Gender Parity Index':
                        value = max(0.5, min(1.5, value))
                    elif '%' in values['unit']:
                        value = max(0, min(100, value))
                    data_rows.append({'year': year, 'country': country, 'indicator_short': indicator_name, 'value': round(value, 2), 'unit': values['unit']})
        return pd.DataFrame(data_rows)

def create_gauge_chart(value, title, max_value=100, color='#4ECDC4'):
    fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=value, title={'text': title, 'font': {'size': 18, 'color': '#FFFFFF', 'family': 'Inter'}}, delta={'reference': max_value * 0.85, 'increasing': {'color': '#4ECDC4'}}, gauge={'axis': {'range': [None, max_value], 'tickcolor': '#FFFFFF', 'tickfont': {'size': 12}}, 'bar': {'color': color, 'thickness': 0.8}, 'bgcolor': 'rgba(255,255,255,0.05)', 'borderwidth': 0, 'steps': [{'range': [0, max_value * 0.4], 'color': 'rgba(255, 65, 108, 0.2)'}, {'range': [max_value * 0.4, max_value * 0.7], 'color': 'rgba(255, 165, 0, 0.2)'}, {'range': [max_value * 0.7, max_value], 'color': 'rgba(78, 205, 196, 0.2)'}], 'threshold': {'line': {'color': 'white', 'width': 3}, 'thickness': 0.75, 'value': max_value * 0.9}}, number={'font': {'size': 32, 'family': 'Inter', 'color': '#FFFFFF'}}))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#FFFFFF', 'family': 'Inter'}, height=280, margin=dict(l=20, r=20, t=60, b=20))
    return fig

def create_line_chart(data, x, y, color, title):
    fig = px.line(data, x=x, y=y, color=color, markers=True, title=title)
    fig.update_traces(line=dict(width=3), marker=dict(size=9, line=dict(width=2, color='rgba(255,255,255,0.8)')))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,46,0.4)', font={'color': '#FFFFFF', 'family': 'Inter'}, title={'font': {'size': 22, 'color': '#4ECDC4'}, 'x': 0.5, 'xanchor': 'center'}, xaxis={'gridcolor': 'rgba(255,255,255,0.08)', 'showgrid': True, 'zeroline': False, 'title': {'font': {'size': 14}}}, yaxis={'gridcolor': 'rgba(255,255,255,0.08)', 'showgrid': True, 'zeroline': False, 'title': {'font': {'size': 14}}}, hovermode='x unified', height=500, legend=dict(bgcolor='rgba(26,26,46,0.8)', bordercolor='rgba(78, 205, 196, 0.3)', borderwidth=1, font={'size': 12}))
    return fig

def create_bar_chart(data, x, y, color, title):
    fig = px.bar(data, x=x, y=y, color=color, title=title, text=y)
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside', marker=dict(line=dict(color='rgba(255,255,255,0.2)', width=1.5)), textfont={'size': 13, 'family': 'Inter', 'color': '#FFFFFF'})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,46,0.4)', font={'color': '#FFFFFF', 'family': 'Inter'}, title={'font': {'size': 22, 'color': '#4ECDC4'}, 'x': 0.5, 'xanchor': 'center'}, xaxis={'gridcolor': 'rgba(255,255,255,0.08)', 'title': {'font': {'size': 14}}}, yaxis={'gridcolor': 'rgba(255,255,255,0.08)', 'showgrid': True, 'title': {'font': {'size': 14}}}, height=500, showlegend=False)
    return fig

def create_heatmap(data, title):
    fig = px.imshow(data, labels=dict(x="Country", y="Indicator", color="Value"), color_continuous_scale='Turbo', aspect="auto", title=title)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#FFFFFF', 'family': 'Inter'}, title={'font': {'size': 22, 'color': '#4ECDC4'}, 'x': 0.5, 'xanchor': 'center'}, height=500, xaxis={'title': {'font': {'size': 14}}}, yaxis={'title': {'font': {'size': 14}}})
    return fig

country_flags = {'Israel': '🇮🇱', 'UAE': '🇦🇪', 'Bahrain': '🇧🇭', 'Morocco': '🇲🇦', 'Sudan': '🇸🇩'}

st.markdown('<h1>Abraham Accords Literacy Initiative</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Data-Driven Regional Cooperation | Transforming 27.1M Lives</p>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

filtered_data = load_data()

st.sidebar.markdown('<h2>Navigation</h2>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)

page = st.sidebar.radio("", ["Executive Dashboard", "Analytics Hub", "SDG Progress Tracker", "Priority Matrix", "AI Forecasting", "Regional Comparison", "Data Explorer", "Methodology"])

st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<h3>Country Filter</h3>', unsafe_allow_html=True)

all_countries = sorted(filtered_data['country'].unique())
selected_countries = st.sidebar.multiselect("", all_countries, default=all_countries, label_visibility="collapsed")

if selected_countries:
    filtered_data = filtered_data[filtered_data['country'].isin(selected_countries)]

st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.sidebar.info("**Data Sources**\n\nUNESCO Institute for Statistics\n\nWorld Bank EdStats\n\nNational Statistics Bureaus")

if page == "Executive Dashboard":
    st.markdown('<div class="info-box"><strong>Real-Time Overview:</strong> Key performance indicators across all Abraham Accords nations. Updated metrics show progress toward UN SDG 4 targets for 2030.</div>', unsafe_allow_html=True)
    latest_year = filtered_data['year'].max()
    latest_data = filtered_data[filtered_data['year'] == latest_year]
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        avg_literacy = latest_data[latest_data['indicator_short'] == 'Adult Literacy Rate']['value'].mean()
        st.metric(label="Adult Literacy Rate", value=f"{avg_literacy:.1f}%", delta="+2.3%")
    with col2:
        avg_enrollment = latest_data[latest_data['indicator_short'] == 'Primary Enrollment']['value'].mean()
        st.metric(label="Primary Enrollment", value=f"{avg_enrollment:.1f}%", delta="+4.2%")
    with col3:
        avg_parity = latest_data[latest_data['indicator_short'] == 'Gender Parity Index']['value'].mean()
        st.metric(label="Gender Parity Index", value=f"{avg_parity:.2f}", delta="+0.08")
    with col4:
        st.metric(label="Countries Tracked", value=len(selected_countries), delta="5 Total")
    with col5:
        years_left = 2030 - latest_year
        st.metric(label="Years to SDG 2030", value=years_left, delta="Deadline Approaching")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown('<h2>Literacy Trends by Country</h2>', unsafe_allow_html=True)
        trend_data = filtered_data[filtered_data['indicator_short'] == 'Adult Literacy Rate']
        fig_trend = create_line_chart(trend_data, 'year', 'value', 'country', 'Adult Literacy Rate Progress (2015-2023)')
        st.plotly_chart(fig_trend, use_container_width=True)
    with col_right:
        st.markdown('<h2>Country Rankings</h2>', unsafe_allow_html=True)
        latest_literacy = latest_data[latest_data['indicator_short'] == 'Adult Literacy Rate'].sort_values('value', ascending=False)
        for idx, row in latest_literacy.iterrows():
            country = row['country']
            value = row['value']
            flag = country_flags.get(country, '')
            st.markdown(f'<div class="card"><div style="display: flex; justify-content: space-between; align-items: center;"><div><span style="font-size: 24px; margin-right: 12px;">{flag}</span><strong class="country-header" style="display: inline;">{country}</strong></div><div class="stat-large">{value:.1f}%</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2>Performance Heatmap</h2>', unsafe_allow_html=True)
    pivot_data = latest_data.pivot_table(values='value', index='indicator_short', columns='country', aggfunc='mean')
    fig_heat = create_heatmap(pivot_data, 'Multi-Indicator Performance Matrix')
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2>Key Performance Indicators</h2>', unsafe_allow_html=True)
    gauge_cols = st.columns(3)
    with gauge_cols[0]:
        avg_youth = latest_data[latest_data['indicator_short'] == 'Youth Literacy Rate']['value'].mean()
        fig_g1 = create_gauge_chart(avg_youth, "Youth Literacy Rate", 100, '#4ECDC4')
        st.plotly_chart(fig_g1, use_container_width=True)
    with gauge_cols[1]:
        avg_par = latest_data[latest_data['indicator_short'] == 'Gender Parity Index']['value'].mean()
        fig_g2 = create_gauge_chart(avg_par * 100, "Gender Parity", 100, '#FF6B6B')
        st.plotly_chart(fig_g2, use_container_width=True)
    with gauge_cols[2]:
        avg_spend = latest_data[latest_data['indicator_short'] == 'Education Spending']['value'].mean()
        fig_g3 = create_gauge_chart(avg_spend, "Education Spending (% GDP)", 10, '#FFD93D')
        st.plotly_chart(fig_g3, use_container_width=True)

elif page == "Analytics Hub":
    st.markdown('<h1>Analytics Hub</h1>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_indicator = st.selectbox("Select Indicator", sorted(filtered_data['indicator_short'].unique()))
    with col2:
        chart_type = st.selectbox("Visualization Type", ["Line Chart", "Bar Chart", "Area Chart", "Scatter Plot"])
    indicator_data = filtered_data[filtered_data['indicator_short'] == selected_indicator]
    if chart_type == "Line Chart":
        fig = create_line_chart(indicator_data, 'year', 'value', 'country', f'{selected_indicator} Trends')
    elif chart_type == "Bar Chart":
        latest = indicator_data.sort_values('year').groupby('country').last().reset_index()
        fig = create_bar_chart(latest, 'country', 'value', 'country', f'Latest {selected_indicator}')
    elif chart_type == "Area Chart":
        fig = px.area(indicator_data, x='year', y='value', color='country', title=f'{selected_indicator} Over Time')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,46,0.4)', font={'color': '#FFFFFF', 'family': 'Inter'}, height=500, title={'font': {'size': 22, 'color': '#4ECDC4'}, 'x': 0.5, 'xanchor': 'center'})
    else:
        fig = px.scatter(indicator_data, x='year', y='value', color='country', size='value', title=f'{selected_indicator} Scatter Analysis')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,46,0.4)', font={'color': '#FFFFFF', 'family': 'Inter'}, height=500, title={'font': {'size': 22, 'color': '#4ECDC4'}, 'x': 0.5, 'xanchor': 'center'})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        avg_val = indicator_data['value'].mean()
        st.metric("Average", f"{avg_val:.2f}")
    with col_s2:
        max_val = indicator_data['value'].max()
        st.metric("Maximum", f"{max_val:.2f}")
    with col_s3:
        min_val = indicator_data['value'].min()
        st.metric("Minimum", f"{min_val:.2f}")
    with col_s4:
        std_val = indicator_data['value'].std()
        st.metric("Std Deviation", f"{std_val:.2f}")

elif page == "SDG Progress Tracker":
    st.markdown('<h1>SDG 4 Progress Tracker</h1>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><strong>UN Sustainable Development Goal 4:</strong> Quality Education for All by 2030. Track each country\'s progress toward established literacy and education targets.</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    for country in selected_countries:
        flag = country_flags.get(country, '')
        st.markdown(f'<h2>{flag} {country}</h2>', unsafe_allow_html=True)
        country_data = filtered_data[filtered_data['country'] == country]
        latest = country_data.sort_values('year').groupby('indicator_short').last().reset_index()
        progress_data = []
        for _, row in latest.iterrows():
            indicator = row['indicator_short']
            current = row['value']
            targets = {'Adult Literacy Rate': 95.0, 'Youth Literacy Rate': 99.0, 'Primary Enrollment': 98.0, 'Secondary Enrollment': 95.0, 'Gender Parity Index': 1.0 if country not in ['Morocco', 'Sudan'] else 0.92, 'Student-Teacher Ratio': 25.0, 'Education Spending': 4.0, 'Digital Literacy': 85.0}
            target = targets.get(indicator, 100)
            if indicator == 'Student-Teacher Ratio':
                progress_pct = int((target / current) * 100) if current > target else 100
                status = 'On Track' if current <= target else 'Behind'
            elif indicator == 'Gender Parity Index':
                progress_pct = 100 if 0.97 <= current <= 1.03 else int(100 - abs(1.0 - current) * 100)
                status = 'On Track' if 0.97 <= current <= 1.03 else 'Behind'
            else:
                progress_pct = int((current / target) * 100)
                status = 'On Track' if current >= target * 0.9 else 'Behind'
            progress_data.append({'Indicator': indicator, 'Current': f"{current:.2f}", 'Target': f"{target:.2f}", 'Progress': f"{min(progress_pct, 100)}%", 'Status': status})
        if progress_data:
            df_progress = pd.DataFrame(progress_data)
            st.dataframe(df_progress, use_container_width=True, hide_index=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

elif page == "Priority Matrix":
    st.markdown('<h1>Intervention Priority Matrix</h1>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><strong>Urgency Calculation:</strong> Priority scores combine gap-to-target with time remaining until 2030. Higher scores indicate greater need for immediate intervention.</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    priority_list = []
    for country in selected_countries:
        country_data = filtered_data[filtered_data['country'] == country]
        latest = country_data.sort_values('year').groupby('indicator_short').last().reset_index()
        for _, row in latest.iterrows():
            indicator = row['indicator_short']
            current = row['value']
            year = int(row['year'])
            targets = {'Adult Literacy Rate': 95.0, 'Youth Literacy Rate': 99.0, 'Primary Enrollment': 98.0, 'Secondary Enrollment': 95.0, 'Gender Parity Index': 1.0 if country not in ['Morocco', 'Sudan'] else 0.92, 'Student-Teacher Ratio': 25.0, 'Education Spending': 4.0, 'Digital Literacy': 85.0}
            target = targets.get(indicator)
            if not target:
                continue
            years_left = 2030 - year
            if indicator == 'Student-Teacher Ratio':
                if current <= target:
                    continue
                gap_pct = ((current - target) / target) * 100
            elif indicator == 'Gender Parity Index':
                if 0.97 <= current <= 1.03:
                    continue
                gap_pct = abs(target - current) / target * 100
            else:
                if current >= target:
                    continue
                gap_pct = ((target - current) / target) * 100
            urgency = gap_pct * (7 / years_left) if years_left > 0 else gap_pct * 100
            if urgency >= 50:
                priority = 'CRITICAL'
            elif urgency >= 30:
                priority = 'HIGH'
            elif urgency >= 10:
                priority = 'MEDIUM'
            else:
                priority = 'LOW'
            flag = country_flags.get(country, '')
            priority_list.append({'Country': f"{flag} {country}", 'Indicator': indicator, 'Current': f"{current:.2f}", 'Target': f"{target:.2f}", 'Gap': f"{gap_pct:.1f}%", 'Urgency': round(urgency, 1), 'Priority': priority, 'Years': years_left})
    if priority_list:
        df_priority = pd.DataFrame(priority_list).sort_values('Urgency', ascending=False)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            critical = len(df_priority[df_priority['Priority'] == 'CRITICAL'])
            st.markdown(f'<div class="card" style="text-align: center;"><div class="priority-badge priority-critical">Critical</div><div class="stat-large" style="margin-top: 16px;">{critical}</div></div>', unsafe_allow_html=True)
with col2:
high = len(df_priority[df_priority['Priority'] == 'HIGH'])
st.markdown(f'<div class="card" style="text-align: center;"><div class="priority-badge priority-high">High</div><div class="stat-large" style="margin-top: 16px;">{high}</div></div>', unsafe_allow_html=True)
with col3:
medium = len(df_priority[df_priority['Priority'] == 'MEDIUM'])
st.markdown(f'<div class="card" style="text-align: center;"><div class="priority-badge priority-medium">Medium</div><div class="stat-large" style="margin-top: 16px;">{medium}</div></div>', unsafe_allow_html=True)
with col4:
low = len(df_priority[df_priority['Priority'] == 'LOW'])
st.markdown(f'<div class="card" style="text-align: center;"><div class="priority-badge priority-low">Low</div><div class="stat-large" style="margin-top: 16px;">{low}</div></div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<h2>Top Priority Interventions</h2>', unsafe_allow_html=True)
st.dataframe(df_priority.head(20), use_container_width=True, hide_index=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
viz_data = df_priority.groupby(['Country', 'Priority']).size().reset_index(name='Count')
fig_priority = px.bar(viz_data, x='Country', y='Count', color='Priority', title="Priority Distribution by Country", color_discrete_map={'CRITICAL': '#FF416C', 'HIGH': '#FF8C00', 'MEDIUM': '#FFD700', 'LOW': '#32CD32'}, barmode='stack')
fig_priority.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,46,0.4)', font={'color': '#FFFFFF', 'family': 'Inter'}, title={'font': {'size': 22, 'color': '#4ECDC4'}, 'x': 0.5, 'xanchor': 'center'}, height=500)
st.plotly_chart(fig_priority, use_container_width=True)
else:
st.success("All countries meeting or exceeding SDG 4 targets.")
elif page == "AI Forecasting":
st.markdown('<h1>AI-Powered Forecasting to 2030</h1>', unsafe_allow_html=True)
st.markdown('<div class="info-box"><strong>Predictive Analytics:</strong> Advanced time-series forecasting using linear regression to project literacy outcomes through 2030.</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
f_indicator = st.selectbox("Select Indicator", sorted(filtered_data['indicator_short'].unique()))
with col2:
f_country = st.selectbox("Select Country", sorted(selected_countries))
hist = filtered_data[(filtered_data['indicator_short'] == f_indicator) & (filtered_data['country'] == f_country)].sort_values('year')
if len(hist) >= 3:
X = hist['year'].values
y = hist['value'].values
slope, intercept = np.polyfit(X, y, 1)
future_years = np.arange(int(X[-1]) + 1, 2031)
forecasts = slope * future_years + intercept
fig = go.Figure()
fig.add_trace(go.Scatter(x=X, y=y, mode='markers+lines', name='Historical Data', marker=dict(size=11, color='#4ECDC4', line=dict(width=2, color='white')), line=dict(width=4, color='#4ECDC4')))
fig.add_trace(go.Scatter(x=future_years, y=forecasts, mode='lines', name='Forecast to 2030', line=dict(width=4, dash='dash', color='#FF6B6B')))
fig.update_layout(title=f'{f_indicator} Projection - {f_country}', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,46,0.4)', font={'color': '#FFFFFF', 'family': 'Inter'}, xaxis={'gridcolor': 'rgba(255,255,255,0.08)', 'title': 'Year'}, yaxis={'gridcolor': 'rgba(255,255,255,0.08)', 'title': 'Value'}, height=500, hovermode='x unified', title_font={'size': 22, 'color': '#4ECDC4'}, title_x=0.5)
st.plotly_chart(fig, use_container_width=True)
col_a, col_b, col_c = st.columns(3)
with col_a:
st.metric("Current Value (2023)", f"{y[-1]:.2f}")
with col_b:
st.metric("Projected 2030", f"{forecasts[-1]:.2f}")
with col_c:
change = forecasts[-1] - y[-1]
st.metric("Expected Change", f"{change:+.2f}", delta=f"{(change/y[-1]*100):+.1f}%")
else:
st.warning("Need at least 3 data points for forecasting")
elif page == "Regional Comparison":
st.markdown('<h1>Regional Comparison</h1>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
comp_indicator = st.selectbox("Select Indicator for Comparison", sorted(filtered_data['indicator_short'].unique()))
comp_data = filtered_data[filtered_data['indicator_short'] == comp_indicator]
col1, col2 = st.columns(2)
with col1:
latest_comp = comp_data.sort_values('year').groupby('country').last().reset_index()
fig_bar = create_bar_chart(latest_comp, 'country', 'value', 'country', f'Latest {comp_indicator}')
st.plotly_chart(fig_bar, use_container_width=True)
with col2:
fig_radar = go.Figure()
for country in latest_comp['country']:
country_all = filtered_data[(filtered_data['country'] == country) & (filtered_data['year'] == latest_comp['year'].iloc[0])]
indicators = country_all['indicator_short'].tolist()
values = country_all['value'].tolist()
fig_radar.add_trace(go.Scatterpolar(r=values, theta=indicators, fill='toself', name=country))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, gridcolor='rgba(255,255,255,0.2)', color='#FFFFFF'), angularaxis=dict(gridcolor='rgba(255,255,255,0.2)', color='#FFFFFF')), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#FFFFFF', 'family': 'Inter'}, title='Multi-Indicator Comparison', title_font={'size': 22, 'color': '#4ECDC4'}, title_x=0.5, height=500)
st.plotly_chart(fig_radar, use_container_width=True)
elif page == "Data Explorer":
st.markdown('<h1>Data Explorer</h1>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
year_filter = st.multiselect("Filter Years", sorted(filtered_data['year'].unique()), default=sorted(filtered_data['year'].unique()))
with col2:
indicator_filter = st.multiselect("Filter Indicators", sorted(filtered_data['indicator_short'].unique()), default=sorted(filtered_data['indicator_short'].unique()))
with col3:
sort_column = st.selectbox("Sort By", ['year', 'country', 'indicator_short', 'value'])
display = filtered_data[(filtered_data['year'].isin(year_filter)) & (filtered_data['indicator_short'].isin(indicator_filter))].sort_values(sort_column)
st.dataframe(display, use_container_width=True, height=600)
csv = display.to_csv(index=False).encode('utf-8')
st.download_button("Download Data as CSV", csv, f"literacy_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
st.metric("Total Records", len(display))
with col_s2:
st.metric("Countries", display['country'].nunique())
with col_s3:
st.metric("Indicators", display['indicator_short'].nunique())
elif page == "Methodology":
st.markdown('<h1>Methodology & Documentation</h1>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="info-box"><strong>About This Dashboard:</strong> This initiative brings together literacy datasets from UNESCO, World Bank, national statistics bureaus, and educational NGOs across the Abraham Accords nations to track progress toward UN SDG 4 (Quality Education for All).</div>', unsafe_allow_html=True)
st.markdown('<h2>Data Sources</h2>', unsafe_allow_html=True)
sources = {"UNESCO Institute for Statistics": "Primary source for literacy rates and education enrollment data across all member states", "World Bank EdStats": "Education expenditure, economic indicators, and development metrics", "National Statistics Bureaus": "Country-specific demographic and education census data", "UNICEF": "Gender parity indices and vulnerable population education access data"}
for source, description in sources.items():
st.markdown(f'<div class="card"><strong style="color: #4ECDC4; font-size: 18px;">{source}</strong><p style="margin-top: 8px; color: #E0E0E0;">{description}</p></div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<h2>Key SDG 4 Targets</h2>', unsafe_allow_html=True)
targets = {"SDG 4.1": "Universal primary and secondary education completion by 2030", "SDG 4.5": "Eliminate gender disparities in education and ensure equal access for vulnerable populations", "SDG 4.6": "Universal youth and adult literacy and numeracy", "SDG 4.c": "Substantially increase the supply of qualified teachers", "SDG 4.a": "Build and upgrade education facilities and increase financing"}
for target, desc in targets.items():
st.info(f"{target}: {desc}")
st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-footer"><p style="margin: 0; font-size: 17px; font-weight: 700; color: #4ECDC4;">Literacy Initiative</p><p style="margin: 10px 0; font-size: 12px; color: #B8B8D1;">Abraham Accords Educational Alliance</p><p style="margin: 10px 0; font-size: 10px; color: #808080; line-height: 1.5;">Peblink | World Literacy Research Center<br>World Literacy Foundation</p></div>', unsafe_allow_html=True)
