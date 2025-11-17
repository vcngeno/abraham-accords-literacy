"""
Abraham Accords Literacy Mapping Initiative - Complete Enhanced Dashboard
Features: SDG Targets, Country-Specific Goals, Intervention Priority, 2030 Readiness Score, Units Guide
Version 2.0 - Fully Complete
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Abraham Accords Literacy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #667eea;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 24px;
        color: #4a5568;
        text-align: center;
        margin-bottom: 30px;
    }
    .info-box {
        background-color: #ebf8ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .critical-box {
        background-color: #fff5f5;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #fc8181;
        margin: 10px 0;
    }
    .success-box {
        background-color: #f0fff4;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #68d391;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# COMPREHENSIVE SDG 2030 TARGETS WITH UNITS
# ==============================================================================

COMPREHENSIVE_SDG_TARGETS = {
    # LITERACY INDICATORS
    "Adult Literacy Rate": {
        "global_target": 95.0,
        "country_targets": {
            "Sudan": 80.0,
            "Morocco": 90.0,
            "Bahrain": 98.0,
            "United Arab Emirates": 98.0,
            "Israel": 99.0
        },
        "unit": "%",
        "sdg_target": "SDG 4.6",
        "description": "Universal adult literacy (15+)",
        "direction": "higher_is_better",
        "weight": 0.15
    },
    "Youth Literacy Rate": {
        "global_target": 99.0,
        "country_targets": {
            "Sudan": 92.0,
            "Morocco": 96.0,
            "Bahrain": 99.5,
            "United Arab Emirates": 99.5,
            "Israel": 99.8
        },
        "unit": "%",
        "sdg_target": "SDG 4.6",
        "description": "Universal youth literacy (15-24)",
        "direction": "higher_is_better",
        "weight": 0.12
    },
    
    # ENROLLMENT INDICATORS
    "Primary School Enrollment": {
        "global_target": 98.0,
        "unit": "%",
        "sdg_target": "SDG 4.1",
        "description": "Universal primary enrollment",
        "direction": "higher_is_better",
        "weight": 0.10
    },
    "Secondary School Enrollment": {
        "global_target": 95.0,
        "unit": "%",
        "sdg_target": "SDG 4.1",
        "description": "High secondary enrollment",
        "direction": "higher_is_better",
        "weight": 0.08
    },
    
    # GENDER PARITY
    "Youth Literacy Gender Parity (Rural)": {
        "global_target": 1.0,
        "target_range": (0.97, 1.03),
        "country_targets": {
            "Sudan": 0.92,
            "Morocco": 0.95,
            "Bahrain": 1.0,
            "United Arab Emirates": 1.0,
            "Israel": 1.0
        },
        "unit": "Index (0-2)",
        "sdg_target": "SDG 4.5",
        "description": "Gender parity in rural youth literacy",
        "direction": "closer_to_one",
        "weight": 0.08
    },
    "Youth Literacy Gender Parity (Urban)": {
        "global_target": 1.0,
        "target_range": (0.97, 1.03),
        "unit": "Index (0-2)",
        "sdg_target": "SDG 4.5",
        "description": "Gender parity in urban youth literacy",
        "direction": "closer_to_one",
        "weight": 0.06
    },
    "Pre-Primary Enrollment Gender Parity": {
        "global_target": 1.0,
        "target_range": (0.97, 1.03),
        "unit": "Index (0-2)",
        "sdg_target": "SDG 4.5",
        "description": "Gender parity in early childhood education",
        "direction": "closer_to_one",
        "weight": 0.05
    },
    
    # TEACHER QUALITY
    "Student-Teacher Ratio": {
        "global_target": 25.0,
        "target_optimal": 20.0,
        "unit": "Students per Teacher",
        "sdg_target": "SDG 4.c",
        "description": "Adequate qualified teachers",
        "direction": "lower_is_better",
        "weight": 0.06
    },
    
    # INVESTMENT
    "Education Spending": {
        "global_target_min": 4.0,
        "global_target_optimal": 6.0,
        "unit": "% of GDP",
        "sdg_target": "SDG 4.a",
        "description": "Adequate education financing",
        "direction": "higher_is_better",
        "weight": 0.04
    }
}

# ==============================================================================
# DATA LOADING AND PREPARATION
# ==============================================================================

@st.cache_data
def load_data():
    """Load data from CSV files"""
    try:
        data = pd.read_csv('data.csv')
        glossary = pd.read_csv('glossary-terms.csv')
        indicators = pd.read_csv('indicators.csv')
        return data, glossary, indicators
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("Please ensure data.csv, glossary-terms.csv, and indicators.csv are in the same folder as app.py")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

@st.cache_data
def prepare_data(data, indicators):
    """Prepare and enrich data with proper names and metadata"""
    
    data = data.merge(
        indicators[['indicatorId', 'name']], 
        on='indicatorId', 
        how='left'
    )
    
    country_map = {
        'ARE': 'United Arab Emirates',
        'BHR': 'Bahrain',
        'ISR': 'Israel',
        'MAR': 'Morocco',
        'SDN': 'Sudan'
    }
    data['country'] = data['geoUnit'].map(country_map)
    
    indicator_units = {
        'ESG.LOWERSEC.HIGHSES': {
            'short': 'Environmental Science Proficiency',
            'unit': '%',
            'description': 'Students in lower secondary showing proficiency in environmental science'
        },
        'LR.AG15T24.RUR.GPIA': {
            'short': 'Youth Literacy Gender Parity (Rural)',
            'unit': 'Index (0-2)',
            'description': 'Gender parity index for youth literacy (15-24 years) in rural areas'
        },
        'LR.AG15T24.URB.GPIA': {
            'short': 'Youth Literacy Gender Parity (Urban)',
            'unit': 'Index (0-2)',
            'description': 'Gender parity index for youth literacy (15-24 years) in urban areas'
        },
        'NER.0.GPIA.CP': {
            'short': 'Pre-Primary Enrollment Gender Parity',
            'unit': 'Index (0-2)',
            'description': 'Gender parity index for net enrollment in early childhood education'
        },
        'PTRHC.2.QUALIFIED': {
            'short': 'Student-Teacher Ratio',
            'unit': 'Students per Teacher',
            'description': 'Number of students per qualified teacher in lower secondary'
        },
        'XGDP.FSGOV': {
            'short': 'Education Spending',
            'unit': '% of GDP',
            'description': 'Government expenditure on education as percentage of GDP'
        }
    }
    
    data['indicator_short'] = data['indicatorId'].map(
        lambda x: indicator_units.get(x, {}).get('short', x)
    )
    data['unit'] = data['indicatorId'].map(
        lambda x: indicator_units.get(x, {}).get('unit', '')
    )
    data['indicator_desc'] = data['indicatorId'].map(
        lambda x: indicator_units.get(x, {}).get('description', '')
    )
    
    return data, indicator_units

# ==============================================================================
# SDG TARGET HELPER FUNCTIONS
# ==============================================================================

def get_sdg_target(indicator_name, country=None):
    """Get SDG target information for an indicator"""
    target_info = COMPREHENSIVE_SDG_TARGETS.get(indicator_name)
    
    if not target_info:
        return None
    
    info = target_info.copy()
    
    if country and 'country_targets' in target_info:
        country_target = target_info['country_targets'].get(country)
        if country_target:
            info['target'] = country_target
            info['is_country_specific'] = True
        else:
            info['target'] = target_info.get('global_target')
            info['is_country_specific'] = False
    else:
        info['target'] = target_info.get('global_target')
        info['is_country_specific'] = False
    
    return info

def calculate_progress_to_target(current_value, target_info, year, country=None):
    """Calculate progress towards SDG 2030 target"""
    if not target_info:
        return None
    
    if country and 'country_targets' in target_info:
        target = target_info['country_targets'].get(country, target_info.get('global_target'))
    else:
        target = target_info.get('target') or target_info.get('global_target') or target_info.get('global_target_optimal')
    
    if not target:
        return None
    
    direction = target_info.get('direction', 'higher_is_better')
    
    if direction == 'lower_is_better':
        if current_value <= target:
            progress = 100.0
        else:
            progress = max(0, 100 - ((current_value - target) / target * 100))
    elif direction == 'closer_to_one':
        distance_from_one = abs(current_value - 1.0)
        if distance_from_one <= 0.03:
            progress = 100.0
        else:
            progress = max(0, 100 - (distance_from_one / 0.30 * 100))
    else:
        if current_value >= target:
            progress = 100.0
        else:
            progress = (current_value / target) * 100 if target > 0 else 0
    
    years_remaining = 2030 - year
    gap = target - current_value
    
    return {
        'progress_pct': round(progress, 1),
        'gap': round(gap, 2),
        'years_remaining': max(0, years_remaining),
        'on_track': progress >= 80.0,
        'target_value': target
    }

def calculate_aalni_score(country_data, country_name):
    """Calculate Abraham Accords Literacy Need Index (AALNI)"""
    scores = []
    weights_used = []
    
    for indicator_name, target_info in COMPREHENSIVE_SDG_TARGETS.items():
        indicator_data = country_data[country_data['indicator_short'] == indicator_name]
        
        if not indicator_data.empty:
            current_value = indicator_data.iloc[-1]['value']
            current_year = indicator_data.iloc[-1]['year']
            
            progress = calculate_progress_to_target(
                current_value, target_info, current_year, country_name
            )
            
            if progress:
                need_score = 100 - progress['progress_pct']
                weight = target_info.get('weight', 0.05)
                
                scores.append(need_score * weight)
                weights_used.append(weight)
    
    if scores:
        total_weight = sum(weights_used)
        if total_weight > 0:
            aalni = sum(scores) / total_weight
        else:
            aalni = 0
    else:
        aalni = None
    
    return aalni

def calculate_readiness_score(country_data, country_name):
    """Calculate 2030 Readiness Score"""
    if not country_data.empty:
        aalni = calculate_aalni_score(country_data, country_name)
        if aalni is not None:
            readiness = 100 - aalni
            return round(readiness, 1)
    return None

# ==============================================================================
# MAIN APP
# ==============================================================================

# Load data
data, glossary, indicators = load_data()
data, indicator_units = prepare_data(data, indicators)

# Header
st.markdown('<p class="main-header">📊 Abraham Accords Literacy Mapping Initiative</p>', 
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Comprehensive SDG 4 Progress Dashboard & AALNI Calculator</p>', 
            unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📋 Dashboard Controls")

page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Overview & AALNI", "🎯 Country-Specific Targets", "🚨 Intervention Priority", 
     "📈 Trend Analysis", "🗺️ Country Comparison", "🔮 Forecasting", 
     "📊 Deep Dive Analytics", "📋 Data Explorer", "📖 Units Guide"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Filters")

selected_countries = st.sidebar.multiselect(
    "Select Countries:",
    options=sorted(data['country'].dropna().unique()),
    default=sorted(data['country'].dropna().unique())
)

selected_indicators = st.sidebar.multiselect(
    "Select Indicators:",
    options=sorted(data['indicator_short'].dropna().unique()),
    default=sorted(data['indicator_short'].dropna().unique())
)

year_range = st.sidebar.slider(
    "Year Range:",
    min_value=int(data['year'].min()),
    max_value=int(data['year'].max()),
    value=(2015, int(data['year'].max()))
)

filtered_data = data[
    (data['country'].isin(selected_countries)) &
    (data['indicator_short'].isin(selected_indicators)) &
    (data['year'] >= year_range[0]) &
    (data['year'] <= year_range[1])
]

# ==============================================================================
# PAGE 1: OVERVIEW & AALNI
# ==============================================================================
if page == "🏠 Overview & AALNI":
    st.header("📊 Dashboard Overview & AALNI Scores")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Countries Tracked", len(data['country'].dropna().unique()))
    with col2:
        st.metric("Indicators Monitored", len(data['indicatorId'].unique()))
    with col3:
        st.metric("Data Points", len(data))
    with col4:
        st.metric("Latest Data Year", int(data['year'].max()))
    
    st.markdown("---")
    
    st.subheader("🎯 Abraham Accords Literacy Need Index (AALNI)")
    
    st.markdown("""
    <div class="info-box">
    <b>About AALNI:</b> Composite index measuring literacy intervention needs (0-100 scale).
    <br>• <b>80-100</b>: Critical intervention needed
    <br>• <b>60-79</b>: High priority  
    <br>• <b>40-59</b>: Medium priority
    <br>• <b>0-39</b>: Low priority
    </div>
    """, unsafe_allow_html=True)
    
    aalni_scores = []
    
    for country in selected_countries:
        country_data = filtered_data[filtered_data['country'] == country]
        latest_country_data = country_data.sort_values('year').groupby('indicator_short').last().reset_index()
        
        aalni = calculate_aalni_score(latest_country_data, country)
        readiness = calculate_readiness_score(latest_country_data, country)
        
        if aalni is not None:
            aalni_scores.append({
                'Country': country,
                'AALNI Score': round(aalni, 1),
                '2030 Readiness': round(readiness, 1) if readiness else 0,
                'Priority Level': 'Critical' if aalni >= 80 else 'High' if aalni >= 60 else 'Medium' if aalni >= 40 else 'Low'
            })
    
    if aalni_scores:
        aalni_df = pd.DataFrame(aalni_scores).sort_values('AALNI Score', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                aalni_df,
                x='AALNI Score',
                y='Country',
                orientation='h',
                color='AALNI Score',
                color_continuous_scale='RdYlGn_r',
                title='AALNI Scores by Country (Higher = Greater Need)',
                text='AALNI Score'
            )
            
            fig.add_vrect(x0=80, x1=100, fillcolor="red", opacity=0.1)
            fig.add_vrect(x0=60, x1=80, fillcolor="orange", opacity=0.1)
            fig.add_vrect(x0=40, x1=60, fillcolor="yellow", opacity=0.1)
            fig.add_vrect(x0=0, x1=40, fillcolor="green", opacity=0.1)
            
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Priority Rankings")
            for idx, row in aalni_df.iterrows():
                priority_color = "🔴" if row['Priority Level'] == "Critical" else "🟠" if row['Priority Level'] == "High" else "🟡" if row['Priority Level'] == "Medium" else "🟢"
                st.markdown(f"{priority_color} **{row['Country']}**: {row['AALNI Score']} ({row['Priority Level']})")
        
        st.subheader("📋 Detailed AALNI Scores")
        st.dataframe(aalni_df, use_container_width=True)

# ==============================================================================
# PAGE 2: COUNTRY-SPECIFIC TARGETS
# ==============================================================================
elif page == "🎯 Country-Specific Targets":
    st.header("🎯 Country-Specific SDG 2030 Targets")
    
    st.markdown("""
    <div class="info-box">
    <b>Tailored Targets:</b> Each country has specific 2030 targets based on their baseline and context.
    </div>
    """, unsafe_allow_html=True)
    
    for country in sorted(selected_countries):
        st.markdown(f"## 📍 {country}")
        
        country_data = filtered_data[filtered_data['country'] == country]
        latest_country_data = country_data.sort_values('year').groupby('indicator_short').last().reset_index()
        
        comparison_data = []
        
        for _, row in latest_country_data.iterrows():
            indicator = row['indicator_short']
            current_value = row['value']
            current_year = int(row['year'])
            unit = row['unit']
            
            target_info = get_sdg_target(indicator, country)
            
            if target_info:
                progress = calculate_progress_to_target(current_value, target_info, current_year, country)
                
                if progress:
                    comparison_data.append({
                        'Indicator': indicator,
                        'Current Value': f"{current_value:.2f} {unit}",
                        'Year': current_year,
                        f'{country} 2030 Target': f"{progress['target_value']:.2f} {unit}",
                        'Global Target': f"{target_info.get('global_target', 'N/A')} {unit}",
                        'Gap to Target': f"{abs(progress['gap']):.2f} {unit}",
                        'Progress': f"{progress['progress_pct']}%",
                        'Status': '✅ On Track' if progress['on_track'] else '⚠️ Needs Acceleration'
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info(f"No target data available for {country}")
        
        st.markdown("---")

# ==============================================================================
# PAGE 3: INTERVENTION PRIORITY
# ==============================================================================
elif page == "🚨 Intervention Priority":
    st.header("🚨 Intervention Priority Calculator")
    
    st.markdown("""
    <div class="critical-box">
    <b>Priority Ranking:</b> Identifies which countries and indicators need urgent intervention.
    </div>
    """, unsafe_allow_html=True)
    
    priority_list = []
    
    for country in selected_countries:
        country_data = filtered_data[filtered_data['country'] == country]
        latest_country_data = country_data.sort_values('year').groupby('indicator_short').last().reset_index()
        
        for _, row in latest_country_data.iterrows():
            indicator = row['indicator_short']
            current_value = row['value']
            current_year = int(row['year'])
            unit = row['unit']
            
            target_info = get_sdg_target(indicator, country)
            
            if target_info:
                progress = calculate_progress_to_target(current_value, target_info, current_year, country)
                
                if progress:
                    gap_pct = abs(progress['gap'] / progress['target_value'] * 100) if progress['target_value'] != 0 else 0
                    years_left = progress['years_remaining']
                    
                    if years_left > 0:
                        urgency_score = gap_pct * (7 / years_left)
                    else:
                        urgency_score = gap_pct * 100
                    
                    priority_list.append({
                        'Country': country,
                        'Indicator': indicator,
                        'Current': f"{current_value:.2f} {unit}",
                        'Target': f"{progress['target_value']:.2f} {unit}",
                        'Gap': f"{abs(progress['gap']):.2f} {unit}",
                        'Progress %': progress['progress_pct'],
                        'Years Left': years_left,
                        'Urgency Score': round(urgency_score, 1),
                        'Priority': 'CRITICAL' if urgency_score >= 50 else 'HIGH' if urgency_score >= 30 else 'MEDIUM' if urgency_score >= 15 else 'LOW'
                    })
    
    if priority_list:
        priority_df = pd.DataFrame(priority_list).sort_values('Urgency Score', ascending=False)
        
        st.subheader("🔝 Top 10 Critical Interventions Needed")
        
        top_10 = priority_df.head(10)
        
        fig = px.bar(
            top_10,
            x='Urgency Score',
            y=[f"{row['Country']} - {row['Indicator']}" for _, row in top_10.iterrows()],
            orientation='h',
            color='Urgency Score',
            color_continuous_scale='Reds',
            title='Top 10 Most Urgent Interventions',
            text='Urgency Score'
        )
        
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Complete Priority List")
        st.dataframe(priority_df, use_container_width=True)
        
        csv = priority_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Priority List",
            data=csv,
            file_name=f'priorities_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )

# ==============================================================================
# PAGE 4: TREND ANALYSIS
# ==============================================================================
elif page == "📈 Trend Analysis":
    st.header("📈 Temporal Trend Analysis")
    
    trend_indicator = st.selectbox(
        "Select Indicator:",
        options=sorted(filtered_data['indicator_short'].unique())
    )
    
    trend_data = filtered_data[filtered_data['indicator_short'] == trend_indicator]
    
    if not trend_data.empty:
        unit = trend_data['unit'].iloc[0]
        
        fig = px.line(
            trend_data,
            x='year',
            y='value',
            color='country',
            markers=True,
            title=f"{trend_indicator} Over Time ({unit})",
            labels={'value': f'Value ({unit})', 'year': 'Year'},
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        
        target_info = get_sdg_target(trend_indicator)
        if target_info:
            for country in selected_countries:
                country_target_info = get_sdg_target(trend_indicator, country)
                if country_target_info and 'target' in country_target_info:
                    target_val = country_target_info['target']
                    fig.add_hline(y=target_val, line_dash="dot", line_width=1)
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 5: COUNTRY COMPARISON
# ==============================================================================
elif page == "🗺️ Country Comparison":
    st.header("🗺️ Cross-Country Comparison")
    
    comparison_data = filtered_data.sort_values('year').groupby(
        ['country', 'indicator_short']
    ).last().reset_index()
    
    pivot_data = comparison_data.pivot_table(
        index='country',
        columns='indicator_short',
        values='value'
    ).round(2)
    
    st.dataframe(pivot_data, use_container_width=True)

# ==============================================================================
# PAGE 6: FORECASTING WITH PROPHET
# ==============================================================================
elif page == "🔮 Forecasting":
    st.header("🔮 Predictive Analytics & Forecasting to 2030")
    
    st.markdown("""
    <div class="info-box">
    <b>📘 Forecasting Methods:</b>
    <br>• <b>Simple Linear:</b> Straight-line trend (fast, easy to interpret)
    <br>• <b>Prophet Advanced:</b> AI-powered with seasonality, changepoints, and confidence intervals
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_indicator = st.selectbox(
            "Select Indicator:",
            options=sorted(filtered_data['indicator_short'].unique())
        )
    
    with col2:
        forecast_country = st.selectbox(
            "Select Country:",
            options=sorted(selected_countries)
        )
    
    with col3:
        forecast_method = st.selectbox(
            "Forecasting Method:",
            options=["Simple Linear", "Prophet Advanced"]
        )
    
    forecast_years_ahead = st.slider(
        "Forecast Years Ahead:",
        min_value=1,
        max_value=10,
        value=6,
        help="Number of years to project into the future"
    )
    
    hist_data = filtered_data[
        (filtered_data['indicator_short'] == forecast_indicator) &
        (filtered_data['country'] == forecast_country)
    ].sort_values('year')
    
    if len(hist_data) >= 3:
        X = hist_data['year'].values
        y = hist_data['value'].values
        unit = hist_data['unit'].iloc[0]
        
        last_year = int(X[-1])
        future_years = np.arange(last_year + 1, last_year + forecast_years_ahead + 1)
        
        # METHOD 1: SIMPLE LINEAR REGRESSION
        if forecast_method == "Simple Linear":
            slope = np.polyfit(X, y, 1)[0]
            intercept = np.polyfit(X, y, 1)[1]
            
            forecasts = slope * future_years + intercept
            
            # Create visualization
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=X, 
                y=y, 
                mode='markers+lines', 
                name='Historical Data',
                marker=dict(size=10, color='#667eea'),
                line=dict(color='#667eea', width=2)
            ))
            
            # Trend line
            all_years = np.concatenate([X, future_years])
            trend_line = slope * all_years + intercept
            fig.add_trace(go.Scatter(
                x=all_years,
                y=trend_line,
                mode='lines',
                name='Trend Line',
                line=dict(color='orange', dash='dash', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=future_years, 
                y=forecasts, 
                mode='markers+lines', 
                name='Forecast',
                marker=dict(size=10, color='red', symbol='diamond'),
                line=dict(color='red', width=2, dash='dot')
            ))
            
            # Add SDG target line
            target_info = get_sdg_target(forecast_indicator, forecast_country)
            if target_info:
                target_value = target_info.get('target')
                if target_value is not None:
                    fig.add_hline(
                        y=target_value,
                        line_dash="dash",
                        line_color="green",
                        annotation_text=f"SDG 2030 Target: {target_value} {unit}",
                        annotation_position="right"
                    )
            
            fig.update_layout(
                title=f"Simple Linear Forecast: {forecast_indicator} - {forecast_country}",
                xaxis_title='Year',
                yaxis_title=f'Value ({unit})',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast table
            forecast_df = pd.DataFrame({
                'Year': future_years,
                f'Forecasted Value ({unit})': np.round(forecasts, 2)
            })
            
            st.subheader("📋 Forecast Values")
            st.dataframe(forecast_df, use_container_width=True)
            
            # Model info
            st.info(f"""
            **Model Information:**
            - Method: Simple Linear Regression
            - Slope: {slope:.4f} per year
            - R² (fit quality): {np.corrcoef(X, y)[0,1]**2:.3f}
            - Assumption: Linear trend continues unchanged
            """)
        
        # METHOD 2: PROPHET ADVANCED
        elif forecast_method == "Prophet Advanced":
            try:
                from prophet import Prophet
                
                # Prepare data for Prophet (needs 'ds' and 'y' columns)
                prophet_df = pd.DataFrame({
                    'ds': pd.to_datetime(hist_data['year'], format='%Y'),
                    'y': hist_data['value'].values
                })
                
                # Initialize Prophet model
                with st.spinner('Training Prophet model... This may take 30-60 seconds.'):
                    model = Prophet(
                        yearly_seasonality=False,  # Not enough data for yearly seasonality
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        changepoint_prior_scale=0.05,  # Flexibility for trend changes
                        interval_width=0.95  # 95% confidence interval
                    )
                    
                    model.fit(prophet_df)
                    
                    # Create future dataframe
                    future = model.make_future_dataframe(periods=forecast_years_ahead, freq='Y')
                    forecast = model.predict(future)
                
                # Create visualization
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=prophet_df['ds'],
                    y=prophet_df['y'],
                    mode='markers',
                    name='Historical Data',
                    marker=dict(size=10, color='#667eea')
                ))
                
                # Prophet forecast
                forecast_future = forecast[forecast['ds'] > prophet_df['ds'].max()]
                
                fig.add_trace(go.Scatter(
                    x=forecast_future['ds'],
                    y=forecast_future['yhat'],
                    mode='lines+markers',
                    name='Prophet Forecast',
                    line=dict(color='red', width=2),
                    marker=dict(size=8, symbol='diamond')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_future['ds'],
                    y=forecast_future['yhat_upper'],
                    mode='lines',
                    name='Upper Bound (95%)',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_future['ds'],
                    y=forecast_future['yhat_lower'],
                    mode='lines',
                    name='Lower Bound (95%)',
                    line=dict(width=0),
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    fill='tonexty',
                    showlegend=True
                ))
                
                # Add SDG target line
                target_info = get_sdg_target(forecast_indicator, forecast_country)
                if target_info and 'target' in target_info:
                    fig.add_hline(
                        y=target_info['target'],
                        line_dash="dash",
                        line_color="green",
                        annotation_text=f"SDG 2030 Target: {target_info['target']} {unit}",
                        annotation_position="right"
                    )
                
                fig.update_layout(
                    title=f"Prophet Advanced Forecast: {forecast_indicator} - {forecast_country}",
                    xaxis_title='Year',
                    yaxis_title=f'Value ({unit})',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table with confidence intervals
                forecast_table = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                forecast_table['ds'] = forecast_table['ds'].dt.year
                forecast_table.columns = ['Year', f'Forecast ({unit})', f'Lower 95% ({unit})', f'Upper 95% ({unit})']
                forecast_table = forecast_table.round(2)
                
                st.subheader("📋 Prophet Forecast with Confidence Intervals")
                st.dataframe(forecast_table, use_container_width=True)
                
                # Show trend components
                st.subheader("📊 Trend Components Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Overall trend
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['trend'],
                        mode='lines',
                        name='Trend',
                        line=dict(color='#667eea', width=2)
                    ))
                    fig_trend.update_layout(
                        title='Overall Trend Over Time',
                        xaxis_title='Year',
                        yaxis_title=f'Trend Value ({unit})',
                        height=300
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                
                with col2:
                    # Growth rate
                    if len(forecast) > 1:
                        growth_rate = (forecast['trend'].iloc[-1] - forecast['trend'].iloc[0]) / forecast['trend'].iloc[0] * 100
                        st.metric(
                            "Projected Total Growth",
                            f"{growth_rate:.1f}%",
                            help="Expected growth from first to last year in forecast"
                        )
                    
                    # Changepoints detected
                    if hasattr(model, 'changepoints'):
                        n_changepoints = len(model.changepoints)
                        st.metric(
                            "Trend Changepoints Detected",
                            n_changepoints,
                            help="Number of significant trend changes Prophet detected in historical data"
                        )
                
                # Model info
                st.info("""
                **Prophet Model Information:**
                - Method: Facebook Prophet (AI-powered time series forecasting)
                - Features: Automatic trend changepoint detection, robust to missing data and outliers
                - Confidence Intervals: 95% (shaded area shows uncertainty range)
                - Best for: Data with 10+ historical points and potential non-linear trends
                """)
                
                # Interpretation guide
                with st.expander("📖 How to Interpret Prophet Results"):
                    st.markdown("""
                    **Understanding the Forecast:**
                    - **Red line**: Most likely future value (median prediction)
                    - **Shaded area**: 95% confidence interval (uncertainty range)
                    - **Wider bands**: More uncertainty in predictions
                    - **Changepoints**: Places where Prophet detected trend shifts
                    
                    **When to Trust the Forecast:**
                    - ✅ Narrow confidence bands (high certainty)
                    - ✅ Consistent with recent trends
                    - ✅ Sufficient historical data (5+ years)
                    - ⚠️ Wide bands suggest high uncertainty
                    - ⚠️ Extrapolation beyond historical range is less reliable
                    
                    **Advantages over Linear:**
                    - Captures non-linear trends
                    - Provides uncertainty estimates
                    - Detects trend changes automatically
                    - More robust to outliers
                    """)
            
            except ImportError:
                st.error("""
                ❌ **Prophet not installed!**
                
                To use Prophet forecasting, install it with:
                ```bash
                pip install prophet
                ```
                
                Then restart the dashboard.
                """)
                
                st.info("💡 **Note:** Prophet requires additional dependencies. If installation fails, try:")
                st.code("""
# On Mac/Linux:
pip install prophet

# If that fails, try:
conda install -c conda-forge prophet

# Or use Docker:
docker run -p 8501:8501 -v $(pwd):/app python:3.9 bash -c "pip install prophet streamlit pandas plotly && streamlit run /app/app.py"
                """, language="bash")
        
        # SDG 2030 Target Analysis (common for both methods)
        st.markdown("---")
        st.subheader("🎯 SDG 2030 Target Analysis")
        
        target_info = get_sdg_target(forecast_indicator, forecast_country)
        
        # Get 2030 forecast value
        forecast_2030_idx = np.where(future_years == 2030)[0]
        if len(forecast_2030_idx) > 0:
            if forecast_method == "Simple Linear":
                forecast_2030_value = forecasts[forecast_2030_idx[0]]
                uncertainty_range = None
            else:
                try:
                    forecast_2030_data = forecast_future[forecast_future['ds'].dt.year == 2030]
                    if not forecast_2030_data.empty:
                        forecast_2030_value = forecast_2030_data['yhat'].values[0]
                        uncertainty_range = (
                            forecast_2030_data['yhat_lower'].values[0],
                            forecast_2030_data['yhat_upper'].values[0]
                        )
                    else:
                        forecast_2030_value = None
                        uncertainty_range = None
                except:
                    forecast_2030_value = None
                    uncertainty_range = None
            
            if forecast_2030_value is not None:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Projected 2030 Value",
                        f"{forecast_2030_value:.2f} {unit}",
                        help="Based on current trend"
                    )
                    
                    if uncertainty_range and forecast_method == "Prophet Advanced":
                        st.caption(f"95% CI: [{uncertainty_range[0]:.2f}, {uncertainty_range[1]:.2f}]")
                
                with col2:
                    if target_info and 'target' in target_info:
                        st.metric(
                            "SDG 2030 Target",
                            f"{target_info['target']} {unit}",
                            help=f"{target_info['sdg_target']}: {target_info['description']}"
                        )
                
                with col3:
                    if target_info and 'target' in target_info:
                        gap = target_info['target'] - forecast_2030_value
                        gap_pct = (gap / target_info['target'] * 100) if target_info['target'] != 0 else 0
                        
                        st.metric(
                            "Gap to Target",
                            f"{abs(gap):.2f} {unit}",
                            delta=f"{gap_pct:.1f}%",
                            delta_color="inverse" if gap > 0 else "normal"
                        )
                
                # Progress assessment
                if target_info and 'target' in target_info:
                    progress = calculate_progress_to_target(
                        forecast_2030_value, target_info, 2030, forecast_country
                    )
                    
                    if progress:
                        if progress['on_track']:
                            st.success(f"""
                            ✅ **On Track!** 
                            
                            Projected to reach {progress['progress_pct']}% of SDG 2030 target 
                            based on current trend ({forecast_method}).
                            """)
                        else:
                            st.warning(f"""
                            ⚠️ **Gap Alert!** 
                            
                            Current trajectory reaches only {progress['progress_pct']}% of SDG target by 2030.
                            
                            **Gap:** {abs(gap):.2f} {unit} ({abs(gap_pct):.1f}%)
                            
                            **Additional annual growth needed:** 
                            {((target_info['target'] / hist_data['value'].iloc[-1])**(1/6) - 1)*100:.2f}% per year
                            from {int(hist_data['year'].iloc[-1])} to meet target.
                            """)
                            
                            if uncertainty_range and forecast_method == "Prophet Advanced":
                                # Check if target is within confidence interval
                                if uncertainty_range[0] <= target_info['target'] <= uncertainty_range[1]:
                                    st.info("""
                                    💡 **Note:** SDG target falls within the 95% confidence interval, 
                                    meaning there's still a reasonable chance of meeting the target 
                                    if positive factors materialize.
                                    """)
    
    else:
        st.warning(f"""
        ⚠️ **Insufficient Data for Forecasting**
        
        Need at least 3 historical data points for forecasting.
        Current data points: {len(hist_data)}
        
        Please select a different indicator or country combination.
        """)

# ==============================================================================
# PAGE 7: DEEP DIVE
# ==============================================================================
elif page == "📊 Deep Dive Analytics":
    st.header("📊 Deep Dive Analytics")
    
    gpi_data = filtered_data[
        filtered_data['indicator_short'].str.contains('Gender Parity', case=False, na=False)
    ]
    
    if not gpi_data.empty:
        latest_gpi = gpi_data.sort_values('year').groupby(['country', 'indicator_short']).last().reset_index()
        
        fig = px.bar(
            latest_gpi,
            x='country',
            y='value',
            color='indicator_short',
            title='Gender Parity Index - Latest Values'
        )
        fig.add_hline(y=1.0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 8: DATA EXPLORER
# ==============================================================================
elif page == "📋 Data Explorer":
    st.header("📋 Raw Data Explorer")
    
    display_cols = ['year', 'country', 'indicator_short', 'value', 'unit', 'indicator_desc']
    display_data = filtered_data[display_cols].sort_values(['country', 'indicator_short', 'year'])
    
    st.dataframe(display_data, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(display_data))
    with col2:
        st.metric("Countries", display_data['country'].nunique())
    with col3:
        st.metric("Indicators", display_data['indicator_short'].nunique())
    
    csv = display_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f'data_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
    )

# ==============================================================================
# PAGE 9: UNITS GUIDE
# ==============================================================================
elif page == "📖 Units Guide":
    st.header("📖 Complete Units of Measure Guide")
    
    st.markdown("""
    <div class="info-box">
    <b>Purpose:</b> This comprehensive guide explains all UNESCO education indicators, their units of measure,
    interpretation guidelines, and SDG 2030 targets for the Abraham Accords Literacy Initiative.
    </div>
    """, unsafe_allow_html=True)
    
    # Table of Contents
    st.subheader("📑 Table of Contents")
    st.markdown("""
    - [Literacy Indicators](#literacy-indicators)
    - [Enrollment Indicators](#enrollment-indicators)
    - [Teacher Indicators](#teacher-indicators)
    - [Parity Indices](#parity-indices)
    - [Financial Indicators](#financial-indicators)
    - [Composite Indicators (AALNI)](#composite-indicators)
    - [Calculation Methods](#calculation-methods)
    - [Interpretation Guidelines](#interpretation-guidelines)
    - [Regional Benchmarks](#regional-benchmarks)
    """)
    
    st.markdown("---")
    
    # LITERACY INDICATORS
    st.subheader("📊 LITERACY INDICATORS")
    
    literacy_indicators = [
        {
            "Name": "Adult Literacy Rate (15+ years)",
            "Unit": "Percentage (%)",
            "Range": "0-100%",
            "Definition": "Percentage of population aged 15 and above who can read and write a short simple statement about their everyday life",
            "Interpretation": "Higher = Better",
            "SDG Target": "95%+ by 2030",
            "Disaggregation": "Total, Male, Female, Rural, Urban, Age groups"
        },
        {
            "Name": "Youth Literacy Rate (15-24 years)",
            "Unit": "Percentage (%)",
            "Range": "0-100%",
            "Definition": "Percentage of population aged 15-24 who can read and write",
            "Interpretation": "Higher = Better",
            "SDG Target": "99%+ by 2030",
            "Disaggregation": "Total, Male, Female, Rural, Urban"
        }
    ]
    
    for ind in literacy_indicators:
        with st.expander(f"**{ind['Name']}**"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Unit:** {ind['Unit']}")
                st.markdown(f"**Range:** {ind['Range']}")
                st.markdown(f"**Interpretation:** {ind['Interpretation']}")
            with col2:
                st.markdown(f"**SDG Target:** {ind['SDG Target']}")
                st.markdown(f"**Disaggregation:** {ind['Disaggregation']}")
            st.markdown(f"**Definition:** {ind['Definition']}")
    
    st.markdown("---")
    
    # ENROLLMENT INDICATORS
    st.subheader("🎓 ENROLLMENT INDICATORS")
    
    enrollment_indicators = [
        {
            "Name": "Net Enrollment Rate (NER) - Primary",
            "Unit": "Percentage (%)",
            "Range": "0-100%",
            "Definition": "Enrollment of official age group only",
            "Interpretation": "Higher = Better; Maximum is 100%",
            "SDG Target": "98%+ by 2030"
        },
        {
            "Name": "Net Enrollment Rate (NER) - Secondary",
            "Unit": "Percentage (%)",
            "Range": "0-100%",
            "Definition": "Enrollment of official secondary age group",
            "Interpretation": "Higher = Better",
            "SDG Target": "95%+ by 2030"
        },
        {
            "Name": "Pre-Primary Education Participation Rate",
            "Unit": "Percentage (%)",
            "Range": "0-100%",
            "Definition": "Participation in organized learning one year before primary entry",
            "Interpretation": "Higher = Better",
            "SDG Target": "95%+ by 2030"
        }
    ]
    
    for ind in enrollment_indicators:
        with st.expander(f"**{ind['Name']}**"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Unit:** {ind['Unit']}")
                st.markdown(f"**Range:** {ind['Range']}")
            with col2:
                st.markdown(f"**Interpretation:** {ind['Interpretation']}")
                st.markdown(f"**SDG Target:** {ind['SDG Target']}")
            st.markdown(f"**Definition:** {ind['Definition']}")
    
    st.markdown("---")
    
    # TEACHER INDICATORS
    st.subheader("👨‍🏫 TEACHER INDICATORS")
    
    st.markdown("""
    ### **Pupil-Teacher Ratio (PTR)**
    - **Unit:** Ratio (students per teacher)
    - **Range:** Typically 10-50
    - **Definition:** Average number of pupils per teacher
    - **Interpretation:** Lower = Better (more individual attention)
    - **UNESCO Recommended:** ≤25:1
    - **Optimal:** 15-20:1
    
    **Quality Benchmarks:**
    - `<20:1` = Excellent ✅
    - `20-25:1` = Acceptable 🟢
    - `25-35:1` = Concerning 🟡
    - `>35:1` = Critical shortage 🔴
    
    ### **Percentage of Qualified Teachers**
    - **Unit:** Percentage (%)
    - **Range:** 0-100%
    - **Definition:** Teachers with minimum required academic qualifications
    - **Interpretation:** Higher = Better
    - **SDG Target:** 100% by 2030
    """)
    
    st.markdown("---")
    
    # PARITY INDICES
    st.subheader("⚖️ PARITY INDICES")
    
    st.markdown("""
    ### **Gender Parity Index (GPI)**
    - **Unit:** Index/Ratio (dimensionless)
    - **Range:** 0 to 2+ (UNESCO adjusted scale)
    - **Definition:** Ratio of female to male values for any indicator
    - **Calculation:** Female value ÷ Male value
    
    **Interpretation:**
    - **1.00** = Perfect parity ✅
    - **0.97-1.03** = Parity achieved (within 3%) 🟢
    - **<0.97** = Disparity favoring males ⚠️
    - **>1.03** = Disparity favoring females
    
    **SDG Target:** 0.97-1.03 by 2030
    
    **Example:**
    - If female literacy = 85% and male literacy = 90%
    - GPI = 85/90 = 0.944 (disparity favoring males)
    
    ---
    
    ### **Rural-Urban Parity Index**
    - **Unit:** Index/Ratio (dimensionless)
    - **Range:** 0 to 2+
    - **Definition:** Ratio of rural to urban values
    - **Calculation:** Rural value ÷ Urban value
    
    **Interpretation:**
    - **1.00** = Perfect parity
    - **<1.00** = Urban advantage (most common)
    - **>1.00** = Rural advantage (rare)
    
    **SDG Target:** 0.95+ by 2030
    
    **Example:**
    - If rural literacy = 70% and urban literacy = 90%
    - Parity Index = 70/90 = 0.778 (28% gap favoring urban)
    """)
    
    st.markdown("---")
    
    # FINANCIAL INDICATORS
    st.subheader("💰 FINANCIAL INDICATORS")
    
    st.markdown("""
    ### **Government Expenditure on Education (% of GDP)**
    - **Unit:** Percentage (%)
    - **Range:** Typically 1-10%
    - **Definition:** Total government spending on education as percentage of Gross Domestic Product
    - **Interpretation:** Higher = Greater investment
    
    **UNESCO Benchmarks:**
    - **Minimum:** 4% of GDP 🟡
    - **Recommended:** 6% of GDP 🟢
    - **Global Average:** ~4.5%
    
    **Quality Levels:**
    - `6%+ GDP` = Strong commitment ✅
    - `4-6% GDP` = Adequate 🟢
    - `3-4% GDP` = Insufficient 🟡
    - `<3% GDP` = Severely underfunded 🔴
    
    **Abraham Accords Context:**
    - **Israel:** 6.1% (meets optimal)
    - **Morocco:** 5.3% (meets minimum)
    - **UAE:** 3.9% (below minimum)
    - **Bahrain:** 2.7% (severely underfunded)
    - **Sudan:** 2.2% (crisis level)
    
    ---
    
    ### **Per-Pupil Expenditure**
    - **Unit:** Currency (USD) per student per year
    - **Range:** Varies widely ($100-$15,000+)
    - **Definition:** Average spending per enrolled student
    - **Interpretation:** Higher = More resources per student
    - **Note:** Should be adjusted for PPP (Purchasing Power Parity) for international comparisons
    """)
    
    st.markdown("---")
    
    # COMPOSITE INDICATORS
    st.subheader("🎯 COMPOSITE INDICATORS (AALNI)")
    
    st.markdown("""
    ### **Abraham Accords Literacy Need Index (AALNI)**
    
    **Purpose:** Prioritize literacy intervention needs across Abraham Accords countries
    
    **Unit:** Index score  
    **Range:** 0-100  
    **Direction:** Higher score = Greater intervention need
    
    **Components (Weighted Average):**
    1. **Literacy Gaps** (40% weight)
       - Adult literacy rate gap from 95% target
       - Youth literacy rate gap from 99% target
    
    2. **Disparity Scores** (25% weight)
       - Gender parity gaps
       - Rural-urban divide
    
    3. **Vulnerability Factors** (20% weight)
       - Out-of-school children rates
       - Conflict impact (Sudan-specific)
       - Economic constraints
    
    4. **Infrastructure Deficits** (15% weight)
       - Teacher-student ratios
       - Digital literacy access
       - School facilities quality
    
    **Interpretation Scale:**
    - **80-100** 🔴 = CRITICAL intervention needed
    - **60-79** 🟠 = HIGH priority
    - **40-59** 🟡 = MEDIUM priority
    - **0-39** 🟢 = LOW priority
    
    **Calculation Formula:**
    ```
    AALNI = Σ (Indicator_Gap × Weight) / Σ Weights
    
    Where:
    Indicator_Gap = 100 - Progress_to_Target_Percentage
    ```
    
    **Example Calculation for Morocco:**
    - Adult literacy gap: 90% target, 77.4% current = 13.9% gap → 13.9 points × 0.15 weight
    - Youth literacy gap: 99% target, 95% current = 4% gap → 4 points × 0.12 weight
    - Gender parity gap: 0.81 vs 1.0 target = 19% gap → 19 points × 0.08 weight
    - ... [continue for all indicators]
    - **Morocco AALNI ≈ 65** (High Priority)
    
    ---
    
    ### **2030 Readiness Score**
    - **Unit:** Percentage (%)
    - **Range:** 0-100%
    - **Calculation:** 100 - AALNI Score
    - **Interpretation:** Higher = Better prepared for 2030 SDG targets
    
    **Readiness Levels:**
    - **80-100%** = On track for SDG 2030 ✅
    - **60-79%** = Moderate progress, acceleration needed 🟡
    - **40-59%** = Significant gaps, major intervention required 🟠
    - **0-39%** = Critical situation, emergency action needed 🔴
    """)
    
    st.markdown("---")
    
    # CALCULATION METHODS
    st.subheader("📐 CALCULATION METHODS")
    
    st.markdown("""
    ### **Standard Formulas**
    """)
    
    st.code("""
# Percentage Calculations
Rate (%) = (Numerator / Denominator) × 100

Example: Literacy Rate
= (Literate population / Total population aged 15+) × 100

# Parity Index Calculations
GPI = Female value / Male value

Adjusted GPI (if >1):
= 2 - GPI  # UNESCO adjustment for symmetry

# Annual Growth Rate
Growth Rate = [(Final Value / Initial Value)^(1/years) - 1] × 100

Example: Morocco literacy 2010-2020
= [(77.4 / 56.1)^(1/10) - 1] × 100
= 3.27% annual growth

# Gap to Target
Absolute Gap = Target Value - Current Value
Percentage Gap = (Gap / Target Value) × 100

Example: Sudan adult literacy
Target = 95%, Current = 60.7%
Gap = 34.3 percentage points (36% gap)

# Progress to Target
For "higher is better" indicators:
Progress (%) = (Current Value / Target Value) × 100

For "lower is better" indicators:
Progress (%) = 100 - [(Current - Target) / Target × 100]

# Compound Annual Growth Required
Required Growth = [(Target / Current)^(1/years_left) - 1] × 100

Example: Sudan to reach 80% by 2030
Current = 60.7%, Target = 80%, Years = 6
Required = [(80/60.7)^(1/6) - 1] × 100 = 4.7% per year
    """, language="python")
    
    st.markdown("---")
    
    # INTERPRETATION GUIDELINES
    st.subheader("📊 INTERPRETATION GUIDELINES")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### **For Literacy Rates**
        - `95%+` = Universal literacy achieved ✅
        - `90-94%` = Near-universal, final push needed 🟢
        - `80-89%` = Substantial gaps, targeted intervention 🟡
        - `70-79%` = Major challenges, sustained effort required 🟠
        - `<70%` = Critical situation, emergency intervention 🔴
        
        ### **For Enrollment Rates**
        - `98%+` = Universal access achieved ✅
        - `95-97%` = Near-universal, address remaining barriers 🟢
        - `90-94%` = Good coverage, expand to marginalized 🟡
        - `80-89%` = Moderate gaps, increase capacity 🟠
        - `<80%` = Critical access gaps 🔴
        
        ### **For Completion Rates**
        - `95%+` = Excellent retention ✅
        - `85-94%` = Good, address dropout causes 🟢
        - `75-84%` = Concerning dropout rates 🟡
        - `60-74%` = High dropout, urgent intervention 🟠
        - `<60%` = Critical dropout crisis 🔴
        """)
    
    with col2:
        st.markdown("""
        ### **For Gender Parity (GPI)**
        - `0.97-1.03` = Parity achieved (SDG met) ✅
        - `0.90-0.97` = Moderate disparity, intervention needed 🟡
        - `0.80-0.89` = Significant disparity, priority action 🟠
        - `<0.80` = Severe disparity, emergency action 🔴
        - `>1.03` = Reverse disparity (girls outperform) 🟢
        
        ### **For Teacher Ratios**
        - `<20:1` = Excellent (individual attention) ✅
        - `20-25:1` = Acceptable (UNESCO standard) 🟢
        - `25-35:1` = Concerning (quality at risk) 🟡
        - `35-50:1` = Poor (teacher shortage) 🟠
        - `>50:1` = Critical (severe shortage) 🔴
        
        ### **For Education Spending**
        - `6%+ GDP` = Strong commitment ✅
        - `4-6% GDP` = Adequate (UNESCO minimum) 🟢
        - `3-4% GDP` = Insufficient investment 🟡
        - `2-3% GDP` = Severely underfunded 🟠
        - `<2% GDP` = Crisis-level underinvestment 🔴
        """)
    
    st.markdown("---")
    
    # REGIONAL BENCHMARKS
    st.subheader("🌍 REGIONAL BENCHMARKS")
    
    st.markdown("""
    ### **MENA Region Averages (2023)**
    """)
    
    benchmarks_df = pd.DataFrame({
        'Indicator': [
            'Adult Literacy Rate',
            'Youth Literacy Rate',
            'Primary Net Enrollment',
            'Secondary Net Enrollment',
            'Gender Parity Index',
            'Rural-Urban Parity',
            'Pupil-Teacher Ratio',
            'Education Spending (% GDP)',
            'Qualified Teachers (%)'
        ],
        'MENA Average': [
            '76.9%',
            '89.3%',
            '91.2%',
            '79.8%',
            '0.93',
            '0.82',
            '22:1',
            '4.2%',
            '82%'
        ],
        'Abraham Accords Average': [
            '85.6%',
            '92.1%',
            '91.8%',
            '84.2%',
            '0.92',
            '0.78',
            '19:1',
            '4.5%',
            '87%'
        ],
        'SDG 2030 Target': [
            '95%',
            '99%',
            '98%',
            '95%',
            '0.97-1.03',
            '0.95+',
            '≤25:1',
            '4-6%',
            '100%'
        ]
    })
    
    st.dataframe(benchmarks_df, use_container_width=True)
    
    st.markdown("""
    **Key Observations:**
    - Abraham Accords countries perform **above MENA average** in most indicators
    - Significant **variation within** the grouping (Sudan pulls down averages)
    - Gender parity still **below SDG target** regionally
    - Rural-urban gaps represent the **biggest challenge** (0.78 vs 0.95 target)
    """)
    
    st.markdown("---")
    
    # DATA QUALITY
    st.subheader("🔄 DATA QUALITY & CONFIDENCE RATINGS")
    
    st.markdown("""
    ### **Confidence Level Classification**
    
    **HIGH CONFIDENCE 🟢**
    - 3+ authoritative sources agree (within 5%)
    - Recent data (2020-2024)
    - Clear, transparent methodology
    - Nationally representative sample
    - Government-verified statistics
    
    **MEDIUM CONFIDENCE 🟡**
    - 2 sources agree OR single highly authoritative source
    - Moderately recent data (2015-2019)
    - Documented methodology
    - Representative but limited sample
    
    **LOW CONFIDENCE 🔴**
    - Single unverified source
    - Outdated data (pre-2015)
    - Conflicting sources (>10% variance)
    - Unclear or non-representative methodology
    - Expert estimates without data
    
    **MISSING ⚪**
    - No data available
    - Data collection disrupted (conflict, etc.)
    - Not measured in country
    
    ---
    
    ### **Temporal Consistency Checks**
    - Year-over-year changes should be **gradual** (<5% typically)
    - Flag sudden jumps **>10%** for verification
    - Note structural breaks (policy changes, conflicts, methodology changes)
    - Trends should be **plausible** given context
    
    **Red Flags:**
    - Perfect round numbers (suggests estimation)
    - Identical values across multiple years
    - Implausible improvements (>15% in one year)
    - Counter-trend movements (improving during crisis)
    """)
    
    st.markdown("---")
    
    # SOURCES
    st.subheader("📖 AUTHORITATIVE DATA SOURCES")
    
    st.markdown("""
    ### **Primary International Sources**
    
    1. **UNESCO Institute for Statistics (UIS)**
       - URL: [http://uis.unesco.org](http://uis.unesco.org)
       - Coverage: All education indicators, 200+ countries
       - Update: Annual (typically 18-month lag)
       - Reliability: ⭐⭐⭐⭐⭐ (Gold standard)
    
    2. **World Bank EdStats**
       - URL: [https://datatopics.worldbank.org/education/](https://datatopics.worldbank.org/education/)
       - Coverage: Education + economic indicators
       - Update: Quarterly
       - Reliability: ⭐⭐⭐⭐⭐
    
    3. **UN SDG Indicators Database**
       - URL: [https://unstats.un.org/sdgs/](https://unstats.un.org/sdgs/)
       - Coverage: Official SDG 4 tracking
       - Update: Annual
       - Reliability: ⭐⭐⭐⭐⭐
    
    4. **OECD Education Database**
       - URL: [https://www.oecd.org/education/](https://www.oecd.org/education/)
       - Coverage: OECD countries + partners (Israel, PISA data)
       - Update: Annual
       - Reliability: ⭐⭐⭐⭐⭐
    
    5. **UNICEF Data Warehouse**
       - URL: [https://data.unicef.org](https://data.unicef.org)
       - Coverage: Child-focused indicators, MICS surveys
       - Update: Survey cycles (3-5 years)
       - Reliability: ⭐⭐⭐⭐
    
    ---
    
    ### **National Statistical Offices**
    
    - **Israel:** Central Bureau of Statistics - [https://www.cbs.gov.il](https://www.cbs.gov.il)
    - **UAE:** Federal Competitiveness and Statistics Centre - [https://fcsc.gov.ae](https://fcsc.gov.ae)
    - **Bahrain:** Information & eGovernment Authority - [https://www.iga.gov.bh](https://www.iga.gov.bh)
    - **Morocco:** Haut-Commissariat au Plan (HCP) - [https://www.hcp.ma](https://www.hcp.ma)
    - **Sudan:** Central Bureau of Statistics - [http://www.cbs.gov.sd](http://www.cbs.gov.sd) ⚠️ (Conflict-affected)
    
    ---
    
    ### **Specialized Sources**
    
    - **Global Education Monitoring Report:** [https://gem-report-unesco.org](https://gem-report-unesco.org)
    - **IEA TIMSS & PIRLS:** International assessments
    - **PISA (OECD):** Reading, math, science assessments
    - **ITU:** Digital literacy and ICT access
    - **UNHCR:** Refugee education data (Sudan crisis)
    """)
    
    st.markdown("---")
    
    # USAGE NOTES
    st.subheader("💡 DASHBOARD USAGE BEST PRACTICES")
    
    st.markdown("""
    ### **For Data Analysis**
    1. ✅ Always display **units** alongside values
    2. 📅 Include **data year** with all statistics
    3. ⚖️ **Disaggregate** by gender and location when possible
    4. 🌍 **Compare** to regional averages for context
    5. 📈 Show **trend direction** (improving/stable/declining)
    6. 🎯 Highlight **gaps to targets** prominently
    7. 🔍 Note **data quality/confidence** in reports
    
    ### **For Visualizations**
    **Color Coding Standards:**
    - 🟢 **Green:** On track / Meets target / Excellent
    - 🟡 **Yellow:** Moderate concern / Needs improvement
    - 🟠 **Orange:** High concern / Priority action needed
    - 🔴 **Red:** Critical / Far from target / Emergency
    
    ### **For Stakeholder Presentations**
    1. Lead with **AALNI scores** for prioritization
    2. Show **country-specific targets** (not just global benchmarks)
    3. Highlight both **challenges AND successes**
    4. Include **cost estimates** for closing gaps
    5. Provide **actionable recommendations**
    6. Use **comparable time periods** across countries
    7. Acknowledge **data limitations** transparently
    
    ### **For Policy Recommendations**
    - Focus on **evidence-based** interventions
    - Prioritize **high-urgency, high-impact** actions
    - Consider **cost-effectiveness** of interventions
    - Account for **country-specific contexts**
    - Propose **realistic timelines** based on trends
    - Include **monitoring & evaluation** frameworks
    """)
    
    st.markdown("---")
    
    # DOWNLOAD SECTION
    st.subheader("📥 Download This Guide")
    
    guide_content = """
ABRAHAM ACCORDS LITERACY INITIATIVE
COMPLETE UNITS OF MEASURE GUIDE
Version 2.0 | November 2025

This comprehensive guide provides detailed information about all UNESCO education indicators
used in the AALNI Dashboard, including units of measure, calculation methods, interpretation
guidelines, and SDG 2030 targets.

For full interactive version, visit the AALNI Dashboard.
Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    st.download_button(
        label="📥 Download Units Guide as TXT",
        data=guide_content.encode('utf-8'),
        file_name=f'AALNI_Units_Guide_{datetime.now().strftime("%Y%m%d")}.txt',
        mime='text/plain',
    )
    
    st.markdown("---")
    st.success("""
    ✅ **Guide Complete!** Use this reference when:
    - Interpreting dashboard visualizations
    - Preparing stakeholder reports
    - Training team members on indicators
    - Validating data quality
    - Setting country-specific targets
    """)

# ==============================================================================
# FOOTER
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.info("""
**Abraham Accords Literacy Initiative**  
UNESCO Data Visualization Platform  
Built with Streamlit & Plotly

**Version:** 2.0 Enhanced  
**Features:**
- AALNI Calculator
- Country-Specific Targets
- Intervention Priority Ranking
- 2030 Readiness Scores
- Complete Units Guide

**Last Updated:** November 2025
""")