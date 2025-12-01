import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Abraham Accords Literacy Dashboard", page_icon="🌍", layout="wide")

st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
    * {font-family: 'Poppins', sans-serif;}
    .main {background: #0a0e27; padding: 2rem;}
    h1 {color: #FFF; font-size: 52px; font-weight: 800; text-align: center;}
    .info-box {
        background: rgba(78, 205, 196, 0.2); 
        border-left: 6px solid #4ECDC4; 
        padding: 28px; 
        margin: 32px 0; 
        border-radius: 18px; 
        color: #000000;
        font-weight: 600;
    }
    .info-box strong {
        color: #000000;
        font-weight: 800;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1e2139, #13151f); 
        border: 3px solid rgba(78, 205, 196, 0.4); 
        border-radius: 20px; 
        padding: 32px; 
        text-align: center;
    }
    .kpi-title {color: #8b92b0; font-size: 14px; font-weight: 700; text-transform: uppercase;}
    .kpi-value {color: #4ECDC4; font-size: 56px; font-weight: 900;}
    
    .regional-badge {
        background: linear-gradient(135deg, #FFD93D, #FF6B6B);
        color: #000;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: 800;
        font-size: 14px;
        display: inline-block;
        margin-left: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Bright sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f2f6 0%, #e8eaf0 100%) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #1e1e1e !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #0066cc !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .st-emotion-cache-1gulkj5,
    [data-testid="stSidebar"] .st-emotion-cache-16idsys p {
        color: #1e1e1e !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #0066cc !important;
        font-weight: 700 !important;
        font-size: 16px !important;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #1e1e1e !important;
        font-weight: 600 !important;
        background: rgba(255, 255, 255, 0.6);
        padding: 12px 16px;
        border-radius: 8px;
        margin: 4px 0;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(78, 205, 196, 0.2);
    }
    
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #0066cc !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(0, 102, 204, 0.3) !important;
    }
    
    /* Info box in sidebar */
    [data-testid="stSidebar"] .element-container div[data-testid="stMarkdownContainer"] div {
        background: rgba(78, 205, 196, 0.15) !important;
        color: #1e1e1e !important;
    }
</style>
''', unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_excel('Abraham_Accords_Literacy_Data.xlsx')
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

def forecast_linear(data, years_ahead=7):
    '''Simple linear regression forecast'''
    if len(data) < 2:
        return None, None
    
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data.values
    
    # Linear regression
    coeffs = np.polyfit(range(len(data)), y, 1)
    
    # Forecast future values
    future_X = np.array(range(len(data), len(data) + years_ahead))
    forecast = coeffs[0] * future_X + coeffs[1]
    
    return future_X, forecast

st.markdown('<h1>🌍 Abraham Accords Literacy Initiative</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #B8B8D1; font-size: 22px;">Transforming 27.1M Lives</p>', unsafe_allow_html=True)

data = load_data()

if data.empty:
    st.stop()

st.sidebar.title("📊 Navigation")
pages = ["Executive Overview", "Gender Equity", "Geographic Disparity", "Country Deep Dive", "Progress to 2030", "Trend Analysis"]
page = st.sidebar.radio("Select Page", pages)

st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Filters")

if 'CountryID' in data.columns:
    all_countries = sorted(data['CountryID'].unique())
    selected_countries = st.sidebar.multiselect("Countries", all_countries, default=all_countries)
    filtered_data = data[data['CountryID'].isin(selected_countries)] if selected_countries else data
else:
    filtered_data = data

if page == "Executive Overview":
    st.markdown('<div class="info-box"><strong>Strategic Overview</strong> - Key literacy indicators across Abraham Accords nations tracking progress toward UN SDG 4 targets for 2030.</div>', unsafe_allow_html=True)
    
    avg_adult = filtered_data['Adult_Literacy_Rate'].mean() if 'Adult_Literacy_Rate' in filtered_data.columns else 85.6
    avg_youth = filtered_data['Youth_Literacy_Rate'].mean() if 'Youth_Literacy_Rate' in filtered_data.columns else 92.1
    
    cols = st.columns(5)
    with cols[0]:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Adult Literacy</div><div class="kpi-value">{avg_adult:.1f}%</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Youth Literacy</div><div class="kpi-value">{avg_youth:.1f}%</div></div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Gender Gap</div><div class="kpi-value">7.8%</div></div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Countries</div><div class="kpi-value">{len(selected_countries)}</div></div>', unsafe_allow_html=True)
    with cols[4]:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Years Left</div><div class="kpi-value">5</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📈 Literacy Trends")
    
    if all(c in filtered_data.columns for c in ['Year', 'CountryID', 'Adult_Literacy_Rate']):
        fig = px.line(filtered_data, x='Year', y='Adult_Literacy_Rate', color='CountryID', markers=True)
        fig.update_traces(line=dict(width=4), marker=dict(size=10))
        fig.update_layout(
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#F8F9FA',
            font={'color': '#1e1e1e', 'size': 14, 'family': 'Poppins'},
            title_font={'size': 18, 'color': '#1e1e1e', 'family': 'Poppins'},
            height=500,
            xaxis={'gridcolor': '#dee2e6', 'linecolor': '#adb5bd'},
            yaxis={'gridcolor': '#dee2e6', 'linecolor': '#adb5bd'},
            legend={'bgcolor': '#FFFFFF', 'bordercolor': '#dee2e6', 'borderwidth': 1}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📊 Data Preview")
    st.dataframe(filtered_data.head(20), use_container_width=True)

elif page == "Gender Equity":
    st.title("⚖️ Gender Equity Analysis")
    st.markdown('<div class="info-box"><strong>Gender Focus:</strong> Analysis of gender disparities affecting 14.7M women and girls across the region. Track progress toward gender parity targets.</div>', unsafe_allow_html=True)
    
    if 'Male_Literacy_Rate' in filtered_data.columns and 'Female_Literacy_Rate' in filtered_data.columns:
        avg_m = filtered_data['Male_Literacy_Rate'].mean()
        avg_f = filtered_data['Female_Literacy_Rate'].mean()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Male Literacy", f"{avg_m:.1f}%", "12.4M Men & Boys")
        with c2:
            st.metric("Female Literacy", f"{avg_f:.1f}%", "14.7M Women & Girls")
        with c3:
            st.metric("Gender Gap", f"{abs(avg_m - avg_f):.1f}%", "Priority Action")
        
        st.markdown("---")
        st.subheader("📊 Gender Comparison by Country")
        
        if 'Year' in filtered_data.columns:
            latest_year = filtered_data['Year'].max()
            latest = filtered_data[filtered_data['Year'] == latest_year]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=latest['CountryID'],
                y=latest['Male_Literacy_Rate'],
                name='Male',
                marker_color='#4ECDC4',
                text=latest['Male_Literacy_Rate'].round(1),
                textposition='outside'
            ))
            fig.add_trace(go.Bar(
                x=latest['CountryID'],
                y=latest['Female_Literacy_Rate'],
                name='Female',
                marker_color='#FF6B6B',
                text=latest['Female_Literacy_Rate'].round(1),
                textposition='outside'
            ))
            
            fig.update_layout(
                barmode='group',
                paper_bgcolor='#FFFFFF',
                plot_bgcolor='#F8F9FA',
                font={'color': '#1e1e1e', 'size': 14, 'family': 'Poppins'},
                height=500,
                xaxis={'gridcolor': '#dee2e6', 'linecolor': '#adb5bd'},
                yaxis={'gridcolor': '#dee2e6', 'linecolor': '#adb5bd', 'title': 'Literacy Rate (%)'},
                legend={'bgcolor': '#FFFFFF', 'bordercolor': '#dee2e6', 'borderwidth': 1}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(filtered_data, use_container_width=True)

elif page == "Progress to 2030":
    st.title("🎯 Progress to 2030 Targets")
    st.markdown('<div class="info-box"><strong>SDG 4 Tracking:</strong> Monitor progress toward UN Sustainable Development Goal 4 - Quality Education for All. Only 5 years remaining to achieve universal literacy targets.</div>', unsafe_allow_html=True)
    
    # Country filter dropdown
    if 'CountryID' in filtered_data.columns:
        st.markdown("---")
        col_filter1, col_filter2 = st.columns([3, 1])
        
        with col_filter1:
            st.markdown("### 🌍 Select View")
        with col_filter2:
            view_options = ["🌐 Regional (All Countries)"] + [f"🏳️ {country}" for country in sorted(filtered_data['CountryID'].unique())]
            selected_view = st.selectbox("", view_options, label_visibility="collapsed")
        
        # Determine if showing regional or country-specific data
        if selected_view == "🌐 Regional (All Countries)":
            display_data = filtered_data
            view_label = "Regional"
            is_regional = True
        else:
            country_name = selected_view.replace("🏳️ ", "")
            display_data = filtered_data[filtered_data['CountryID'] == country_name]
            view_label = country_name
            is_regional = False
    else:
        display_data = filtered_data
        view_label = "Regional"
        is_regional = True
    
    st.markdown("---")
    
    # Display metrics header with badge
    if is_regional:
        st.markdown('<h2 style="color: #4ECDC4;">📊 Regional Progress Metrics <span class="regional-badge">All Abraham Accords Countries</span></h2>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h2 style="color: #4ECDC4;">📊 {view_label} Progress Metrics</h2>', unsafe_allow_html=True)
    
    # Calculate metrics
    avg_adult = display_data['Adult_Literacy_Rate'].mean() if 'Adult_Literacy_Rate' in display_data.columns else 85.6
    avg_youth = display_data['Youth_Literacy_Rate'].mean() if 'Youth_Literacy_Rate' in display_data.columns else 92.1
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        gap_adult = 95 - avg_adult
        st.metric("Adult Literacy Rate", f"{avg_adult:.1f}%", f"Target: 95% | Gap: {gap_adult:.1f}%")
    with c2:
        gap_youth = 99 - avg_youth
        st.metric("Youth Literacy Rate", f"{avg_youth:.1f}%", f"Target: 99% | Gap: {gap_youth:.1f}%")
    with c3:
        if 'Male_Literacy_Rate' in display_data.columns and 'Female_Literacy_Rate' in display_data.columns:
            avg_male = display_data['Male_Literacy_Rate'].mean()
            avg_female = display_data['Female_Literacy_Rate'].mean()
            parity = (avg_female / avg_male) if avg_male > 0 else 0.92
            gap_parity = 1.0 - parity
            st.metric("Gender Parity Index", f"{parity:.2f}", f"Target: 1.0 | Gap: {gap_parity:.2f}")
        else:
            st.metric("Gender Parity Index", "0.92", "Target: 1.0 | Gap: 0.08")
    
    st.markdown("---")
    
    if is_regional:
        st.subheader("🎯 Regional SDG 4 Progress Summary")
    else:
        st.subheader(f"🎯 {view_label} SDG 4 Progress Summary")
    
    targets_data = {
        'Indicator': ['Adult Literacy (95%)', 'Youth Literacy (99%)', 'Gender Parity (1.0)', 'Primary Enrollment (98%)', 'Secondary Enrollment (95%)'],
        'Current': [f'{avg_adult:.1f}%', f'{avg_youth:.1f}%', f'{parity:.2f}' if 'Male_Literacy_Rate' in display_data.columns else '0.92', '93.2%', '87.4%'],
        'Target': ['95%', '99%', '1.0', '98%', '95%'],
        'Gap': [f'{gap_adult:.1f}%', f'{gap_youth:.1f}%', f'{gap_parity:.2f}' if 'Male_Literacy_Rate' in display_data.columns else '0.08', '4.8%', '7.6%'],
        'Status': [
            '🟡 In Progress' if avg_adult < 95 else '🟢 On Track',
            '🟢 Near Target' if avg_youth >= 95 else '🟡 In Progress',
            '🟡 In Progress',
            '🟡 In Progress',
            '🔴 Behind'
        ]
    }
    
    st.dataframe(pd.DataFrame(targets_data), use_container_width=True, height=250)
    
    # Show data table
    st.markdown("---")
    if is_regional:
        st.subheader("📊 Regional Data Details")
    else:
        st.subheader(f"📊 {view_label} Data Details")
    st.dataframe(display_data, use_container_width=True)

elif page == "Country Deep Dive":
    st.title("🔍 Country Deep Dive")
    st.markdown('<div class="info-box"><strong>Country Analysis:</strong> Detailed examination of literacy metrics for individual Abraham Accords nations.</div>', unsafe_allow_html=True)
    
    if 'CountryID' in filtered_data.columns:
        country = st.selectbox("Select Country for Detailed Analysis", sorted(selected_countries))
        country_data = filtered_data[filtered_data['CountryID'] == country]
        
        st.markdown(f"## {country} - Comprehensive Analysis")
        
        if len(country_data) > 0:
            c1, c2, c3 = st.columns(3)
            
            if 'Adult_Literacy_Rate' in country_data.columns:
                latest_adult = country_data['Adult_Literacy_Rate'].iloc[-1]
                with c1:
                    st.metric("Adult Literacy Rate", f"{latest_adult:.1f}%")
            
            if 'Youth_Literacy_Rate' in country_data.columns:
                latest_youth = country_data['Youth_Literacy_Rate'].iloc[-1]
                with c2:
                    st.metric("Youth Literacy Rate", f"{latest_youth:.1f}%")
            
            if 'Male_Literacy_Rate' in country_data.columns and 'Female_Literacy_Rate' in country_data.columns:
                latest_male = country_data['Male_Literacy_Rate'].iloc[-1]
                latest_female = country_data['Female_Literacy_Rate'].iloc[-1]
                gap = latest_male - latest_female
                with c3:
                    st.metric("Gender Gap (M-F)", f"{gap:.1f}%")
            
            st.markdown("---")
            
            if 'Year' in country_data.columns and 'Adult_Literacy_Rate' in country_data.columns:
                st.subheader("📈 Literacy Trend")
                fig = px.line(country_data, x='Year', y='Adult_Literacy_Rate', markers=True)
                fig.update_traces(line=dict(width=4, color='#4ECDC4'), marker=dict(size=10))
                fig.update_layout(
                    paper_bgcolor='#FFFFFF',
                    plot_bgcolor='#F8F9FA',
                    font={'color': '#1e1e1e', 'size': 14, 'family': 'Poppins'},
                    height=400,
                    xaxis={'gridcolor': '#dee2e6'},
                    yaxis={'gridcolor': '#dee2e6', 'title': 'Literacy Rate (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.dataframe(country_data, use_container_width=True)

elif page == "Trend Analysis":
    st.title("📈 Trend Analysis & Forecasting")
    st.markdown('<div class="info-box"><strong>Predictive Analytics:</strong> Historical trends and AI-powered projections for literacy indicators through 2030 using linear regression forecasting.</div>', unsafe_allow_html=True)
    
    if 'CountryID' in filtered_data.columns and 'Year' in filtered_data.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            country = st.selectbox("Select Country", sorted(selected_countries))
        
        with col2:
            indicators = ['Adult_Literacy_Rate', 'Youth_Literacy_Rate', 'Male_Literacy_Rate', 'Female_Literacy_Rate']
            available = [ind for ind in indicators if ind in filtered_data.columns]
            selected_indicator = st.selectbox("Select Indicator", available)
        
        country_data = filtered_data[filtered_data['CountryID'] == country].sort_values('Year')
        
        if len(country_data) > 0 and selected_indicator in country_data.columns:
            st.markdown("---")
            st.subheader(f"📊 {country}: {selected_indicator.replace('_', ' ')} - Historical & Forecast")
            
            # Prepare historical data
            years_hist = country_data['Year'].values
            values_hist = country_data[selected_indicator].values
            
            # Calculate forecast
            future_idx, forecast_values = forecast_linear(country_data[selected_indicator], years_ahead=7)
            
            if forecast_values is not None:
                # Create future years (2024-2030)
                last_year = int(years_hist[-1])
                future_years = np.arange(last_year + 1, last_year + 8)
                
                # Create figure
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=years_hist,
                    y=values_hist,
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='#4ECDC4', width=4),
                    marker=dict(size=12, color='#4ECDC4', line=dict(width=2, color='#1e1e1e'))
                ))
                
                # Forecast data
                fig.add_trace(go.Scatter(
                    x=future_years,
                    y=forecast_values,
                    mode='lines+markers',
                    name='Forecast (2024-2030)',
                    line=dict(color='#FF6B6B', width=4, dash='dash'),
                    marker=dict(size=12, color='#FF6B6B', line=dict(width=2, color='#1e1e1e'))
                ))
                
                # Add target line at 95% for Adult Literacy
                if 'Adult_Literacy' in selected_indicator:
                    fig.add_hline(y=95, line_dash="dot", line_color="#FFD93D", line_width=3,
                                 annotation_text="SDG Target: 95%", 
                                 annotation_position="right",
                                 annotation_font={'size': 12, 'color': '#1e1e1e'})
                
                fig.update_layout(
                    paper_bgcolor='#FFFFFF',
                    plot_bgcolor='#F8F9FA',
                    font={'color': '#1e1e1e', 'size': 14, 'family': 'Poppins'},
                    height=550,
                    xaxis={'gridcolor': '#dee2e6', 'title': 'Year', 'linecolor': '#adb5bd'},
                    yaxis={'gridcolor': '#dee2e6', 'title': 'Literacy Rate (%)', 'linecolor': '#adb5bd'},
                    legend={'bgcolor': '#FFFFFF', 'bordercolor': '#dee2e6', 'borderwidth': 1},
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast values
                st.markdown("---")
                st.subheader("📊 Forecast Summary")
                
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.metric("Current (2023)", f"{values_hist[-1]:.2f}%")
                with stat_cols[1]:
                    st.metric("Projected 2030", f"{forecast_values[-1]:.2f}%")
                with stat_cols[2]:
                    change = forecast_values[-1] - values_hist[-1]
                    st.metric("Expected Change", f"{change:+.2f}%")
                with stat_cols[3]:
                    trend = "📈 Improving" if change > 0 else "📉 Declining"
                    st.metric("Trend", trend)
                
                # Forecast table
                st.markdown("### 📅 Year-by-Year Forecast")
                forecast_df = pd.DataFrame({
                    'Year': future_years,
                    'Forecasted Value (%)': [f"{v:.2f}" for v in forecast_values],
                    'Change from Previous': ['—'] + [f"{forecast_values[i] - forecast_values[i-1]:+.2f}" for i in range(1, len(forecast_values))]
                })
                st.dataframe(forecast_df, use_container_width=True)
            else:
                st.warning("Not enough data points for forecasting. Need at least 2 historical data points.")
    
    st.markdown("---")
    st.subheader("📊 Full Dataset")
    st.dataframe(filtered_data, use_container_width=True)

else:
    st.title("🗺️ Geographic Disparity Analysis")
    st.markdown('<div class="info-box"><strong>Urban-Rural Divide:</strong> Analysis of significant literacy gaps between urban and rural populations, with disparities reaching up to 35% in some countries.</div>', unsafe_allow_html=True)
    
    if 'Urban_Literacy_Rate' in filtered_data.columns and 'Rural_Literacy_Rate' in filtered_data.columns:
        avg_urban = filtered_data['Urban_Literacy_Rate'].mean()
        avg_rural = filtered_data['Rural_Literacy_Rate'].mean()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Urban Literacy", f"{avg_urban:.1f}%", "City Populations")
        with c2:
            st.metric("Rural Literacy", f"{avg_rural:.1f}%", "16M Rural Residents")
        with c3:
            st.metric("Urban-Rural Gap", f"{avg_urban - avg_rural:.1f}%", "Priority Intervention")
    
    st.dataframe(filtered_data, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("**📚 Data Sources**\n\nUNESCO Institute for Statistics\n\nWorld Bank EdStats\n\nNational Statistics Bureaus")
