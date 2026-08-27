import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st

st.set_page_config(page_title="Victoria 3 Macroeconomics", layout="wide")
st.title("🌍 Victoria 3 Macroeconomic Dashboard")

# 1. FILE DISCOVERY
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
if not csv_files:
    st.error("No CSV files found in the directory.")
    st.stop()

@st.cache_data
def load_data(file_path):
    df_raw = pd.read_csv(file_path)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw['gdpc'] = df_raw['gdpc'] / 100000
    df_raw['population'] = df_raw['gdp'] / df_raw['gdpc']
    df_raw['year'] = df_raw['date'].dt.year
    
    df = df_raw.sort_values('date').groupby(['country', 'year']).last().reset_index()
    df['gdpGrowth'] = df.groupby('country')['gdp'].pct_change(fill_method=None)
    df['gdpcGrowth'] = df.groupby('country')['gdpc'].pct_change(fill_method=None)
    df['popGrowth'] = df.groupby('country')['population'].pct_change(fill_method=None)
    df['solGrowth'] = df.groupby('country')['sol'].pct_change(fill_method=None)
    return df

vic3_tags = {
    "GBR": "Great Britain", "FRA": "France", "USA": "United States", "GER": "Germany", "PRU": "Prussia", "RUS": "Russia", 
    "ITA": "Italy", "SPA": "Spain", "POR": "Portugal", "AUS": "Austria", "TUR": "Ottoman Empire", "SWE": "Sweden", 
    "NOR": "Norway", "DEN": "Denmark", "FIN": "Finland", "BEL": "Belgium", "NET": "Netherlands", "SWI": "Switzerland",
    "JAP": "Japan", "CHI": "China", "QIN": "Qing", "EIC": "East India Company", "KOR": "Korea", "DAI": "Dai Nam",
    "SIA": "Siam", "SIK": "Sikh Empire", "PER": "Persia", "AFG": "Afghanistan",
    "BRZ": "Brazil", "MEX": "Mexico", "CAN": "Canada", "TEX": "Texas", "CUB": "Cuba", "HAI": "Haiti", 
    "ARG": "Argentina", "CHL": "Chile", "PEU": "Peru", "CLM": "Colombia", "COL": "Colombia", "NGR": "New Granada",
    "VNZ": "Venezuela", "ECU": "Ecuador", "BOL": "Bolivia", "PAR": "Paraguay", "URU": "Uruguay",
    "SAR": "Sardinia-Piedmont", "SIC": "Two Sicilies", "PAP": "Papal States", "TUS": "Tuscany", 
    "BAV": "Bavaria", "SAX": "Saxony", "WUR": "Württemberg", "HAN": "Hanover",
    "GRE": "Greece", "SER": "Serbia", "ROM": "Romania", "BUL": "Bulgaria", "HUN": "Hungary", "POL": "Poland",
    "EGY": "Egypt", "MOR": "Morocco",
    "DEI": "Dutch East Indies", "AST": "Australia", "PHI": "Philippines", 
    "SAF": "South Africa", "ETH": "Ethiopia", "SOK": "Sokoto", "ZUL": "Zulu", "MAD": "Madagascar"
}

def get_scale_menu():
    return [
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.0,
            xanchor="left",
            y=1.18,
            yanchor="top",
            bgcolor="rgba(128, 128, 128, 0.15)",
            bordercolor="rgba(128, 128, 128, 0.4)",
            font=dict(size=12),
            buttons=[
                dict(label="Linear", method="relayout", args=[{"yaxis.type": "linear"}]),
                dict(label="Log", method="relayout", args=[{"yaxis.type": "log"}])
            ]
        )
    ]

# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 1. Global Landscape", 
    "🔍 2. Single Economy Dossier", 
    "⚔️ 3. Bilateral & Convergence",
    "📊 4. Cross-Campaign Meta-Analysis"
])

# ==========================================
# TAB 1: GLOBAL LANDSCAPE
# ==========================================
with tab1:
    st.header("The Great Power Race")
    selected_file_t1 = st.selectbox("Select Campaign Data:", csv_files, key="t1_file")
    played_sigla_t1 = selected_file_t1.split('-')[0].upper() if '-' in selected_file_t1 else "UNKNOWN"
    df_t1 = load_data(selected_file_t1)

    latest_rank = df_t1.dropna(subset=['gdp']).groupby('country').last().reset_index().sort_values(by='gdp', ascending=False)
    default_tags = latest_rank['country'].head(8).tolist()
    if played_sigla_t1 in latest_rank['country'].values and played_sigla_t1 not in default_tags:
        default_tags.append(played_sigla_t1)

    selected_tags = st.multiselect(
        "Select Nations to Compare:", 
        options=latest_rank['country'].tolist(), 
        default=default_tags, 
        format_func=lambda x: f"{x} - {vic3_tags.get(x, x)}"
    )

    if selected_tags:
        filtered_df = df_t1[df_t1['country'].isin(selected_tags)].copy()
        palette = ['#2E86AB', '#E84855', '#2A9D8F', '#F4A261', '#9B5DE5', '#E76F51', '#00BBF9', '#00F5D4', '#D62828', '#003049']
        g_start, g_end = filtered_df['year'].min(), filtered_df['year'].max()

        def plot_multitrace(col, title, y_title, is_indexed=False):
            fig = go.Figure()
            for i, tag in enumerate(selected_tags):
                c_data = filtered_df[filtered_df['country'] == tag].sort_values('year')
                if c_data.empty: 
                    continue
                c_name = vic3_tags.get(tag, tag)
                y_vals = (c_data[col] / c_data[col].iloc[0] * 100) if is_indexed else c_data[col]
                color = '#2E86AB' if tag == played_sigla_t1 else palette[i % len(palette)]
                width = 3.5 if tag == played_sigla_t1 else 2.0
                fmt = '.2f' if not is_indexed else '.1f'
                trace_name = f"{'⭐ ' if tag == played_sigla_t1 else ''}{c_name} ({tag})"
                
                fig.add_trace(go.Scatter(
                    x=c_data['year'], 
                    y=y_vals, 
                    mode='lines', 
                    name=trace_name, 
                    line=dict(color=color, width=width),
                    hovertemplate=f"<b>{trace_name}</b>: %{{y:{fmt}}}<extra></extra>"
                ))

            fig.update_layout(
                title=dict(text=f"{title} ({g_start}-{g_end})", font=dict(size=18), x=0.5, xanchor="center"),
                xaxis=dict(title='Year', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                yaxis=dict(title=y_title, showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                height=450,
                hovermode='x unified',
                updatemenus=get_scale_menu(),
                margin=dict(t=85, b=30, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            plot_multitrace('gdp', 'Total GDP', 'Gross Domestic Product (£ Millions)')
            plot_multitrace('gdpc', 'GDP per Capita', 'GDP per Capita (£)')
        with col2:
            plot_multitrace('gdp', 'GDP Indexed', 'Indexed GDP (100 = Base Year)', is_indexed=True)
            plot_multitrace('sol', 'Standard of Living', 'Standard of Living Index')

        st.divider()
        st.subheader("Global Dataset Master Summary")
        
        master_records = []
        for tag in df_t1['country'].unique():
            c_df = df_t1[(df_t1['country'] == tag) & (df_t1['gdp'].notnull())].sort_values('year')
            if len(c_df) < 2: continue
            
            span = f"{c_df['year'].min()}-{c_df['year'].max()}"
            yrs = len(c_df)
            f_gdp, i_gdp = c_df['gdp'].iloc[-1], c_df['gdp'].iloc[0]
            f_gdpc, i_gdpc = c_df['gdpc'].iloc[-1], c_df['gdpc'].iloc[0]
            f_sol, i_sol = c_df['sol'].iloc[-1], c_df['sol'].iloc[0]
            f_pop, i_pop = c_df['population'].iloc[-1], c_df['population'].iloc[0]
            
            def safe_cagr_pct(start_val, end_val, periods):
                if start_val > 0 and periods > 1: return ((end_val/start_val)**(1/(periods-1)) - 1) * 100
                return np.nan
                
            master_records.append({
                'Tag': tag,
                'Country': f"⭐ {vic3_tags.get(tag, tag)}" if tag == played_sigla_t1 else vic3_tags.get(tag, tag),
                'Span': span,
                'Yrs': yrs,
                'Final GDP': f_gdp,
                'GDP CAGR': safe_cagr_pct(i_gdp, f_gdp, yrs),
                'Final GDP/c': f_gdpc,
                'GDP/c CAGR': safe_cagr_pct(i_gdpc, f_gdpc, yrs),
                'Final SoL': f_sol,
                'SoL CAGR': safe_cagr_pct(i_sol, f_sol, yrs),
                'Final Pop': f_pop,
                'Pop CAGR': safe_cagr_pct(i_pop, f_pop, yrs)
            })
            
        if master_records:
            master_df = pd.DataFrame(master_records).sort_values('Final GDP', ascending=False)
            st.dataframe(
                master_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Final GDP": st.column_config.NumberColumn("Final GDP", format="£%.2fM"),
                    "GDP CAGR": st.column_config.NumberColumn("GDP CAGR", format="%.2f%%"),
                    "Final GDP/c": st.column_config.NumberColumn("Final GDP/c", format="£%.2f"),
                    "GDP/c CAGR": st.column_config.NumberColumn("GDP/c CAGR", format="%.2f%%"),
                    "Final SoL": st.column_config.NumberColumn("Final SoL", format="%.2f"),
                    "SoL CAGR": st.column_config.NumberColumn("SoL CAGR", format="%.2f%%"),
                    "Final Pop": st.column_config.NumberColumn("Final Pop", format="%.2fM"),
                    "Pop CAGR": st.column_config.NumberColumn("Pop CAGR", format="%.2f%%")
                }
            )

# ==========================================
# TAB 2: SINGLE ECONOMY DOSSIER
# ==========================================
with tab2:
    st.header("Single Economy Dossier")
    selected_file_t2 = st.selectbox("Select Campaign Data:", csv_files, key="t2_file")
    played_sigla_t2 = selected_file_t2.split('-')[0].upper() if '-' in selected_file_t2 else "UNKNOWN"
    df_t2 = load_data(selected_file_t2)

    col_t2_1, col_t2_2, col_t2_3 = st.columns([1, 2, 1])
    available_tags_t2 = sorted(df_t2.dropna(subset=['gdp'])['country'].unique().tolist())
    
    with col_t2_1:
        default_idx = available_tags_t2.index(played_sigla_t2) if played_sigla_t2 in available_tags_t2 else 0
        target_tag = st.selectbox(
            "Target Economy:", 
            options=available_tags_t2, 
            index=default_idx, 
            format_func=lambda x: f"{x} - {vic3_tags.get(x, x)}",
            key="t2_target"
        )
        
    country_df = df_t2[df_t2['country'] == target_tag].copy().reset_index(drop=True)
    min_yr, max_yr = int(country_df['year'].min()), int(country_df['year'].max())
    
    with col_t2_2:
        start_year, end_year = st.slider("Timeline Range:", min_value=min_yr, max_value=max_yr, value=(min_yr, max_yr), key="t2_slider")
        
    with col_t2_3:
        ma_window = st.number_input("Trend Window (Years):", min_value=1, max_value=50, value=10, key="t2_ma")
        vol_window = st.number_input("Volatility Window (Years):", min_value=2, max_value=50, value=10, key="t2_vol")

    p_df = country_df[(country_df['year'] >= start_year) & (country_df['year'] <= end_year)].copy().reset_index(drop=True)
    country_name = vic3_tags.get(target_tag, target_tag)

    if p_df.empty:
        st.warning("No data available for the selected timeframe.")
    else:
        def plot_metric(y_col, title, y_title, color, is_pct=False):
            fig = go.Figure()
            if is_pct:
                roll = p_df[y_col].rolling(window=ma_window, min_periods=1).apply(
                    lambda x: np.prod(1 + x)**(1.0/len(x)) - 1 if (1 + x >= 0).all() else np.nan, 
                    raw=True
                )
                t_lbl = f'{ma_window}-Yr CAGR'
            else:
                roll = p_df[y_col].rolling(window=ma_window, min_periods=1).mean()
                t_lbl = f'{ma_window}-Yr Trend'

            val_fmt = '.2%' if is_pct else '.2f'

            fig.add_trace(go.Scatter(
                x=p_df['year'], y=roll, mode='lines', name=t_lbl, 
                line=dict(color='rgba(150, 150, 150, 0.7)', width=2, dash='dash'),
                hovertemplate=f"<b>{t_lbl}</b>: %{{y:{val_fmt}}}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=p_df['year'], y=p_df[y_col], mode='lines+markers', name=title, 
                line=dict(color=color, width=2.5), marker=dict(size=4),
                hovertemplate=f"<b>{title}</b>: %{{y:{val_fmt}}}<extra></extra>"
            ))

            fig.update_layout(
                title=dict(text=f"{country_name} — {title} ({start_year}-{end_year})", font=dict(size=18), x=0.5, xanchor="center"),
                xaxis=dict(title='Year', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                yaxis=dict(
                    title=y_title, tickformat='.1%' if is_pct else None, 
                    zeroline=True, zerolinecolor='#888888', zerolinewidth=1.5,
                    showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'
                ),
                hovermode='x unified',
                height=360,
                updatemenus=[] if is_pct else get_scale_menu(),
                margin=dict(t=85, b=30, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Macroeconomic Trajectories")
        c1, c2 = st.columns(2)
        with c1:
            plot_metric('gdp', 'Total GDP', '£ Millions', '#2E86AB')
            plot_metric('gdpc', 'GDP per Capita', '£', '#E84855')
            plot_metric('population', 'Total Population', 'Millions', '#2A9D8F')
            plot_metric('sol', 'Standard of Living', 'Index', '#9B5DE5')
        with c2:
            plot_metric('gdpGrowth', 'Annual GDP Growth', 'Rate (%)', '#F4A261', True)
            plot_metric('gdpcGrowth', 'Annual GDP/c Growth', 'Rate (%)', '#E76F51', True)
            plot_metric('popGrowth', 'Annual Population Growth', 'Rate (%)', '#264653', True)
            plot_metric('solGrowth', 'Annual SoL Growth', 'Rate (%)', '#8338EC', True)

        st.divider()
        st.subheader("Structural Growth Accounting")
        c3, c4 = st.columns(2)
        with c3:
            fig_decomp = go.Figure()
            fig_decomp.add_trace(go.Bar(x=p_df['year'], y=p_df['gdpcGrowth'], name='Intensive (Productivity)', marker_color='#E76F51', hovertemplate='<b>Intensive</b>: %{y:.2%}<extra></extra>'))
            fig_decomp.add_trace(go.Bar(x=p_df['year'], y=p_df['popGrowth'], name='Extensive (Demographics)', marker_color='#2A9D8F', hovertemplate='<b>Extensive</b>: %{y:.2%}<extra></extra>'))
            fig_decomp.add_trace(go.Scatter(x=p_df['year'], y=p_df['gdpGrowth'], mode='lines+markers', name='Total GDP Growth', line=dict(color='#1D3557', width=2.5), hovertemplate='<b>Total Growth</b>: %{y:.2%}<extra></extra>'))
            fig_decomp.update_layout(
                barmode='relative', 
                title=dict(text=f"{country_name} — Annual Growth Decomposition ({start_year}-{end_year})", font=dict(size=18), x=0.5, xanchor="center"), 
                yaxis=dict(tickformat='.1%', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', zeroline=True, zerolinecolor='#888888', zerolinewidth=1.5), 
                xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                hovermode='x unified', height=400, margin=dict(t=60, b=20, l=10, r=10)
            )
            st.plotly_chart(fig_decomp, use_container_width=True)

        with c4:
            roll_gdpc = p_df['gdpcGrowth'].rolling(window=ma_window, min_periods=1).apply(lambda x: np.prod(1 + x)**(1.0/len(x)) - 1 if (1 + x >= 0).all() else np.nan, raw=True)
            roll_pop = p_df['popGrowth'].rolling(window=ma_window, min_periods=1).apply(lambda x: np.prod(1 + x)**(1.0/len(x)) - 1 if (1 + x >= 0).all() else np.nan, raw=True)
            roll_gdp = p_df['gdpGrowth'].rolling(window=ma_window, min_periods=1).apply(lambda x: np.prod(1 + x)**(1.0/len(x)) - 1 if (1 + x >= 0).all() else np.nan, raw=True)
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Bar(x=p_df['year'], y=roll_gdpc, name='Trend Intensive', marker_color='#E76F51', hovertemplate='<b>Trend Intensive</b>: %{y:.2%}<extra></extra>'))
            fig_trend.add_trace(go.Bar(x=p_df['year'], y=roll_pop, name='Trend Extensive', marker_color='#2A9D8F', hovertemplate='<b>Trend Extensive</b>: %{y:.2%}<extra></extra>'))
            fig_trend.add_trace(go.Scatter(x=p_df['year'], y=roll_gdp, mode='lines+markers', name='Trend GDP', line=dict(color='#1D3557', width=2.5), hovertemplate='<b>Trend GDP</b>: %{y:.2%}<extra></extra>'))
            fig_trend.update_layout(
                barmode='relative', 
                title=dict(text=f"{country_name} — Structural Decomposition ({ma_window}-Yr CAGR) ({start_year}-{end_year})", font=dict(size=18), x=0.5, xanchor="center"), 
                yaxis=dict(tickformat='.1%', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', zeroline=True, zerolinecolor='#888888', zerolinewidth=1.5), 
                xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                hovermode='x unified', height=400, margin=dict(t=60, b=20, l=10, r=10)
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        st.divider()
        st.subheader("Volatility & Crises")
        c5, c6 = st.columns(2)
        with c5:
            colors = np.where(p_df['gdpGrowth'] >= 0, '#2A9D8F', '#E76F51')
            fig_boom = go.Figure(go.Bar(x=p_df['year'], y=p_df['gdpGrowth'], marker_color=colors, name='GDP Growth', hovertemplate='<b>Growth</b>: %{y:.2%}<extra></extra>'))
            fig_boom.update_layout(
                title=dict(text=f"{country_name} — Economic Expansion vs Contraction ({start_year}-{end_year})", font=dict(size=18), x=0.5, xanchor="center"), 
                yaxis=dict(tickformat='.1%', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', zeroline=True, zerolinecolor='#888888', zerolinewidth=1.5), 
                xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                hovermode='x unified', height=350, margin=dict(t=60, b=20, l=10, r=10)
            )
            st.plotly_chart(fig_boom, use_container_width=True)
        with c6:
            roll_vol = p_df['gdpGrowth'].rolling(window=vol_window, min_periods=2).std()
            fig_vol = go.Figure(go.Scatter(x=p_df['year'], y=roll_vol, mode='lines+markers', name='Volatility', line=dict(color='#9B5DE5', width=2.5), hovertemplate='<b>Volatility</b>: %{y:.2%}<extra></extra>'))
            fig_vol.update_layout(
                title=dict(text=f"{country_name} — Macroeconomic Volatility ({vol_window}-Yr) ({start_year}-{end_year})", font=dict(size=18), x=0.5, xanchor="center"), 
                yaxis=dict(tickformat='.1%', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', zeroline=True, zerolinecolor='#888888', zerolinewidth=1.5), 
                xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                hovermode='x unified', height=350, margin=dict(t=60, b=20, l=10, r=10)
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        st.divider()
        st.subheader("Dossier Master Tables")
        
        def get_stats(g):
            g = g.dropna()
            gm = stats.gmean(1+g)-1 if not g.empty and (1+g>0).all() else np.nan
            return [f"{gm:.2%}" if pd.notnull(gm) else "N/A", f"{g.median():.2%}", f"{g.max():.2%}", f"{g.min():.2%}", f"{g.std():.2%}"]

        stats_df = pd.DataFrame({
            'Metric': ['CAGR', 'Median', 'Max', 'Min', 'Std. Dev'],
            'GDP Growth': get_stats(p_df['gdpGrowth']),
            'GDPpC Growth': get_stats(p_df['gdpcGrowth']),
            'SoL Growth': get_stats(p_df['solGrowth']),
            'Pop Growth': get_stats(p_df['popGrowth'])
        })
        
        i_gdp, f_gdp = p_df['gdp'].iloc[0], p_df['gdp'].iloc[-1]
        i_gdpc, f_gdpc = p_df['gdpc'].iloc[0], p_df['gdpc'].iloc[-1]
        i_pop, f_pop = p_df['population'].iloc[0], p_df['population'].iloc[-1]
        ln_gdp = np.log(f_gdp / i_gdp)
        s_int = (np.log(f_gdpc / i_gdpc) / ln_gdp) if ln_gdp != 0 else 0
        s_ext = (np.log(f_pop / i_pop) / ln_gdp) if ln_gdp != 0 else 0

        decomp_df = pd.DataFrame({
            'Component': ['Total Expansion', 'Intensive (Productivity)', 'Extensive (Demographics)'],
            'Initial': [f"£{i_gdp:,.2f}M", f"£{i_gdpc:,.2f}", f"{i_pop:,.2f}M"],
            'Final': [f"£{f_gdp:,.2f}M", f"£{f_gdpc:,.2f}", f"{f_pop:,.2f}M"],
            'Share': ['100.0%', f"{s_int:.1%}", f"{s_ext:.1%}"]
        })

        p_df['peak_gdp'] = p_df['gdp'].cummax()
        p_df['drawdown'] = (p_df['gdp'] - p_df['peak_gdp']) / p_df['peak_gdp']
        crises, in_crisis, s_yr, min_dd, t_yr = [], False, None, 0, None

        for i, row in p_df.iterrows():
            yr, dd = int(row['year']), row['drawdown']
            if dd < 0:
                if not in_crisis:
                    in_crisis, s_yr, min_dd, t_yr = True, p_df.loc[i-1, 'year'] if i > 0 else yr, dd, yr
                elif dd < min_dd:
                    min_dd, t_yr = dd, yr
            elif dd == 0 and in_crisis:
                in_crisis = False
                crises.append({'Peak Year': s_yr, 'Trough Year': t_yr, 'Recovery Year': yr, 'Drawdown': min_dd, 'Duration': f"{yr - s_yr} yrs"})

        if in_crisis:
            crises.append({'Peak Year': s_yr, 'Trough Year': t_yr, 'Recovery Year': 'Unrecovered', 'Drawdown': min_dd, 'Duration': f">{p_df['year'].max() - s_yr} yrs"})

        crisis_df = pd.DataFrame(crises).sort_values(by='Drawdown').head(5) if crises else pd.DataFrame([{'Message': 'No major economic contractions detected.'}])
        if 'Drawdown' in crisis_df.columns: 
            crisis_df['Drawdown'] = crisis_df['Drawdown'].apply(lambda x: f"{x:.2%}")

        c7, c8 = st.columns(2)
        with c7:
            st.write("**General Performance Stats**")
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            st.write("**Growth Accounting Breakdown**")
            st.dataframe(decomp_df, use_container_width=True, hide_index=True)
        with c8:
            st.write("**Top Economic Crises Leaderboard**")
            st.dataframe(crisis_df, use_container_width=True, hide_index=True)

# ==========================================
# TAB 3: BILATERAL & CONVERGENCE
# ==========================================
with tab3:
    st.header("Bilateral & Convergence Analysis")
    selected_file_t3 = st.selectbox("Select Campaign Data:", csv_files, key="t3_file")
    played_sigla_t3 = selected_file_t3.split('-')[0].upper() if '-' in selected_file_t3 else "UNKNOWN"
    df_t3 = load_data(selected_file_t3)

    available_tags_t3 = sorted(df_t3.dropna(subset=['gdp'])['country'].unique().tolist())
    col_t3_1, col_t3_2, col_t3_3, col_t3_4 = st.columns([1, 1, 2, 1])

    with col_t3_1:
        default_target_idx = available_tags_t3.index(played_sigla_t3) if played_sigla_t3 in available_tags_t3 else 0
        tag1 = st.selectbox(
            "Target Nation:", 
            options=available_tags_t3, 
            index=default_target_idx, 
            format_func=lambda x: f"{x} - {vic3_tags.get(x, x)}",
            key="t3_tag1"
        )
        
    with col_t3_2:
        default_bm_idx = available_tags_t3.index("GBR") if "GBR" in available_tags_t3 else (1 if len(available_tags_t3) > 1 else 0)
        tag2 = st.selectbox(
            "Benchmark / Rival:", 
            options=available_tags_t3, 
            index=default_bm_idx, 
            format_func=lambda x: f"{x} - {vic3_tags.get(x, x)}",
            key="t3_tag2"
        )

    df_t = df_t3[df_t3['country'] == tag1].copy().reset_index(drop=True)
    df_b = df_t3[df_t3['country'] == tag2].copy().reset_index(drop=True)

    if df_t.empty or df_b.empty:
        st.warning("Insufficient data for the selected pair.")
    else:
        min_b_yr = min(int(df_t['year'].min()), int(df_b['year'].min()))
        max_b_yr = max(int(df_t['year'].max()), int(df_b['year'].max()))

        with col_t3_3:
            b_start, b_end = st.slider(
                "Common Historical Range:", 
                min_value=min_b_yr, 
                max_value=max_b_yr, 
                value=(min_b_yr, max_b_yr),
                key="t3_slider"
            )
        with col_t3_4:
            b_ma = st.number_input("Trend Window (Years):", min_value=1, max_value=50, value=10, key="t3_ma")
            b_vol = st.number_input("Volatility Window (Years):", min_value=2, max_value=50, value=10, key="t3_vol")

        d1 = df_t[(df_t['year'] >= b_start) & (df_t['year'] <= b_end)].copy().sort_values('year').reset_index(drop=True)
        d2 = df_b[(df_b['year'] >= b_start) & (df_b['year'] <= b_end)].copy().sort_values('year').reset_index(drop=True)

        n1, n2 = f"{vic3_tags.get(tag1, tag1)} ({tag1})", f"{vic3_tags.get(tag2, tag2)} ({tag2})"
        col1_c, col2_c = '#2E86AB', '#E84855'

        st.subheader("Head-to-Head Trajectories")

        def plot_bilateral(col, title, y_title, is_pct=False):
            fig = go.Figure()
            if is_pct:
                roll1 = d1[col].rolling(window=b_ma, min_periods=1).apply(
                    lambda x: np.prod(1 + x)**(1.0/len(x)) - 1 if (1 + x >= 0).all() else np.nan, raw=True
                )
                roll2 = d2[col].rolling(window=b_ma, min_periods=1).apply(
                    lambda x: np.prod(1 + x)**(1.0/len(x)) - 1 if (1 + x >= 0).all() else np.nan, raw=True
                )
                t_lbl = f'{b_ma}-Yr CAGR'
            else:
                roll1 = d1[col].rolling(window=b_ma, min_periods=1).mean()
                roll2 = d2[col].rolling(window=b_ma, min_periods=1).mean()
                t_lbl = f'{b_ma}-Yr Trend'

            val_fmt = '.2%' if is_pct else '.2f'

            fig.add_trace(go.Scatter(
                x=d1['year'], y=roll1, mode='lines', name=f'{n1} {t_lbl}', 
                line=dict(color=col1_c, width=1.5, dash='dash'), 
                hovertemplate=f"<b>{n1} {t_lbl}</b>: %{{y:{val_fmt}}}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=d2['year'], y=roll2, mode='lines', name=f'{n2} {t_lbl}', 
                line=dict(color=col2_c, width=1.5, dash='dash'), 
                hovertemplate=f"<b>{n2} {t_lbl}</b>: %{{y:{val_fmt}}}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=d1['year'], y=d1[col], mode='lines', name=f'{n1} Data', 
                line=dict(color=col1_c, width=2.5), 
                hovertemplate=f"<b>{n1} Data</b>: %{{y:{val_fmt}}}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=d2['year'], y=d2[col], mode='lines', name=f'{n2} Data', 
                line=dict(color=col2_c, width=2.5), 
                hovertemplate=f"<b>{n2} Data</b>: %{{y:{val_fmt}}}<extra></extra>"
            ))

            fig.update_layout(
                title=dict(text=f"{title} ({b_start}-{b_end})", font=dict(size=18), x=0.5, xanchor="center"),
                xaxis=dict(title='Year', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                yaxis=dict(
                    title=y_title, 
                    tickformat='.1%' if is_pct else None, 
                    showgrid=True, 
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    zeroline=True, 
                    zerolinecolor='#888888', 
                    zerolinewidth=1.5
                ),
                hovermode='x unified',
                height=380,
                updatemenus=[] if is_pct else get_scale_menu(),
                margin=dict(t=85, b=30, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        bc1, bc2 = st.columns(2)
        with bc1:
            plot_bilateral('gdp', f'Total GDP: {n1} vs {n2}', '£ Millions')
            plot_bilateral('gdpc', f'GDP per Capita: {n1} vs {n2}', '£')
            plot_bilateral('population', f'Population: {n1} vs {n2}', 'Millions')
            plot_bilateral('sol', f'Standard of Living: {n1} vs {n2}', 'Index')
        with bc2:
            plot_bilateral('gdpGrowth', f'GDP Growth Rate: {n1} vs {n2}', 'Rate (%)', is_pct=True)
            plot_bilateral('gdpcGrowth', f'GDP/c Growth Rate: {n1} vs {n2}', 'Rate (%)', is_pct=True)
            plot_bilateral('popGrowth', f'Population Growth Rate: {n1} vs {n2}', 'Rate (%)', is_pct=True)
            plot_bilateral('solGrowth', f'SoL Growth Rate: {n1} vs {n2}', 'Rate (%)', is_pct=True)

        st.divider()
        st.subheader("Bilateral Growth Accounting")

        def plot_bilateral_decomp(df_run, run_name):
            r_start, r_end = (int(df_run['year'].min()), int(df_run['year'].max())) if not df_run.empty else ("N/A", "N/A")
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(x=df_run['year'], y=df_run['gdpcGrowth'], name='Intensive (Productivity)', marker_color='#E76F51', hovertemplate='<b>Intensive</b>: %{y:.2%}<extra></extra>'))
            fig1.add_trace(go.Bar(x=df_run['year'], y=df_run['popGrowth'], name='Extensive (Demographics)', marker_color='#2A9D8F', hovertemplate='<b>Extensive</b>: %{y:.2%}<extra></extra>'))
            fig1.add_trace(go.Scatter(x=df_run['year'], y=df_run['gdpGrowth'], mode='lines+markers', name='Total GDP Growth', line=dict(color='#1D3557', width=2.5), hovertemplate='<b>Total Growth</b>: %{y:.2%}<extra></extra>'))
            fig1.update_layout(
                barmode='relative', 
                title=dict(text=f"{run_name} — Annual Decomposition ({r_start}-{r_end})", font=dict(size=17), x=0.5, xanchor="center"),
                yaxis=dict(title='Annual Contribution', tickformat='.1%', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', zeroline=True, zerolinecolor='#888888', zerolinewidth=1.5),
                xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', range=[b_start - 0.5, b_end + 0.5]),
                hovermode='x unified', height=400, margin=dict(t=60, b=20, l=10, r=10)
            )
            st.plotly_chart(fig1, use_container_width=True)

            roll_gdpc = df_run['gdpcGrowth'].rolling(window=b_ma, min_periods=1).apply(lambda x: np.prod(1 + x)**(1.0/len(x)) - 1 if (1 + x >= 0).all() else np.nan, raw=True)
            roll_pop = df_run['popGrowth'].rolling(window=b_ma, min_periods=1).apply(lambda x: np.prod(1 + x)**(1.0/len(x)) - 1 if (1 + x >= 0).all() else np.nan, raw=True)
            roll_gdp = df_run['gdpGrowth'].rolling(window=b_ma, min_periods=1).apply(lambda x: np.prod(1 + x)**(1.0/len(x)) - 1 if (1 + x >= 0).all() else np.nan, raw=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df_run['year'], y=roll_gdpc, name='Trend Intensive', marker_color='#E76F51', hovertemplate='<b>Trend Intensive</b>: %{y:.2%}<extra></extra>'))
            fig2.add_trace(go.Bar(x=df_run['year'], y=roll_pop, name='Trend Extensive', marker_color='#2A9D8F', hovertemplate='<b>Trend Extensive</b>: %{y:.2%}<extra></extra>'))
            fig2.add_trace(go.Scatter(x=df_run['year'], y=roll_gdp, mode='lines+markers', name='Trend GDP', line=dict(color='#1D3557', width=2.5), hovertemplate='<b>Trend GDP</b>: %{y:.2%}<extra></extra>'))
            fig2.update_layout(
                barmode='relative',
                title=dict(text=f"{run_name} — Structural Decomposition ({b_ma}-Yr CAGR) ({r_start}-{r_end})", font=dict(size=17), x=0.5, xanchor="center"),
                yaxis=dict(title='Trend Contribution', tickformat='.1%', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', zeroline=True, zerolinecolor='#888888', zerolinewidth=1.5),
                xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', range=[b_start - 0.5, b_end + 0.5]),
                hovermode='x unified', height=400, margin=dict(t=60, b=20, l=10, r=10)
            )
            st.plotly_chart(fig2, use_container_width=True)

        bd_col1, bd_col2 = st.columns(2)
        with bd_col1:
            plot_bilateral_decomp(d1, n1)
        with bd_col2:
            plot_bilateral_decomp(d2, n2)

        st.divider()
        st.subheader("Productivity Convergence & Catch-Up Dynamics")

        d1_idx = d1.set_index('year')
        d2_idx = d2.set_index('year')

        combined = pd.DataFrame({
            'target_gdpc': d1_idx['gdpc'],
            'hegemon_gdpc': d2_idx['gdpc'],
            'target_g': d1_idx['gdpcGrowth'],
            'hegemon_g': d2_idx['gdpcGrowth']
        }).dropna()

        if not combined.empty:
            c_start, c_end = int(combined.index.min()), int(combined.index.max())
            combined['catchup_ratio'] = combined['target_gdpc'] / combined['hegemon_gdpc']
            combined['velocity'] = combined['target_g'] - combined['hegemon_g']
            years = combined.index

            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=years, y=combined['catchup_ratio'], mode='lines+markers', 
                name=f'{n1} / {n2} Ratio', line=dict(color='#2E86AB', width=2.5), 
                hovertemplate=f"<b>Productivity Ratio</b>: %{{y:.2%}}<extra></extra>"
            ))
            fig_conv.add_hline(
                y=1.0, line_dash="dash", line_color="#D62828", 
                annotation_text="Productivity Parity (100%)", annotation_position="top right"
            )
            fig_conv.update_layout(
                title=dict(text=f"Productivity Convergence: {n1} relative to {n2} ({c_start}-{c_end})", font=dict(size=18), x=0.5, xanchor="center"),
                xaxis=dict(title='Year', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                yaxis=dict(title=f'GDP/c as % of {n2}', tickformat='.1%', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                hovermode='x unified', height=420, margin=dict(t=60, b=20, l=10, r=10)
            )
            st.plotly_chart(fig_conv, use_container_width=True)

            colors = np.where(combined['velocity'] >= 0, '#2A9D8F', '#E84855')
            fig_vel = go.Figure(go.Bar(
                x=years, y=combined['velocity'], marker_color=colors, 
                name='Catch-up Spread', 
                hovertemplate="<b>Growth Rate Spread</b>: %{y:+.2%}<extra></extra>"
            ))
            fig_vel.update_layout(
                title=dict(text=f"Convergence Velocity ({tag1} GDP/c Growth minus {tag2} GDP/c Growth) ({c_start}-{c_end})", font=dict(size=18), x=0.5, xanchor="center"),
                xaxis=dict(title='Year', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                yaxis=dict(
                    title='Annual Spread (%)', tickformat='+.1%', 
                    showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', 
                    zeroline=True, zerolinecolor='#888888', zerolinewidth=1.5
                ),
                hovermode='x unified', height=380, margin=dict(t=60, b=20, l=10, r=10)
            )
            st.plotly_chart(fig_vel, use_container_width=True)

        st.divider()
        st.subheader("Comparative Macroeconomic Volatility")

        bc5, bc6 = st.columns(2)
        with bc5:
            a1_start, a1_end = (int(d1['year'].min()), int(d1['year'].max())) if not d1.empty else ("N/A", "N/A")
            colors1 = np.where(d1['gdpGrowth'] >= 0, '#2A9D8F', '#E76F51')
            fig_b_boom1 = go.Figure(go.Bar(x=d1['year'], y=d1['gdpGrowth'], marker_color=colors1, name=n1, hovertemplate=f"<b>{n1}</b>: %{{y:.2%}}<extra></extra>"))
            fig_b_boom1.update_layout(
                title=dict(text=f"{n1} — Expansion vs Contraction ({a1_start}-{a1_end})", font=dict(size=15), x=0.5, xanchor="center"),
                xaxis=dict(title='Year', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', range=[b_start - 0.5, b_end + 0.5]),
                yaxis=dict(tickformat='.1%', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', zeroline=True, zerolinecolor='#888888', zerolinewidth=1.5),
                hovermode='x unified', height=280, margin=dict(t=50, b=20, l=10, r=10)
            )
            st.plotly_chart(fig_b_boom1, use_container_width=True)

            a2_start, a2_end = (int(d2['year'].min()), int(d2['year'].max())) if not d2.empty else ("N/A", "N/A")
            colors2 = np.where(d2['gdpGrowth'] >= 0, '#2A9D8F', '#E76F51')
            fig_b_boom2 = go.Figure(go.Bar(x=d2['year'], y=d2['gdpGrowth'], marker_color=colors2, name=n2, hovertemplate=f"<b>{n2}</b>: %{{y:.2%}}<extra></extra>"))
            fig_b_boom2.update_layout(
                title=dict(text=f"{n2} — Expansion vs Contraction ({a2_start}-{a2_end})", font=dict(size=15), x=0.5, xanchor="center"),
                xaxis=dict(title='Year', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', range=[b_start - 0.5, b_end + 0.5]),
                yaxis=dict(tickformat='.1%', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', zeroline=True, zerolinecolor='#888888', zerolinewidth=1.5),
                hovermode='x unified', height=280, margin=dict(t=50, b=20, l=10, r=10)
            )
            st.plotly_chart(fig_b_boom2, use_container_width=True)

        with bc6:
            roll_vol1 = d1['gdpGrowth'].rolling(window=b_vol, min_periods=2).std()
            roll_vol2 = d2['gdpGrowth'].rolling(window=b_vol, min_periods=2).std()

            fig_b_vol = go.Figure()
            fig_b_vol.add_trace(go.Scatter(
                x=d1['year'], y=roll_vol1, mode='lines+markers', name=f'{n1} Volatility', 
                line=dict(color=col1_c, width=2.5), 
                hovertemplate=f"<b>{n1} Volatility</b>: %{{y:.2%}}<extra></extra>"
            ))
            fig_b_vol.add_trace(go.Scatter(
                x=d2['year'], y=roll_vol2, mode='lines+markers', name=f'{n2} Volatility', 
                line=dict(color=col2_c, width=2.5), 
                hovertemplate=f"<b>{n2} Volatility</b>: %{{y:.2%}}<extra></extra>"
            ))
            fig_b_vol.update_layout(
                title=dict(text=f"Comparative Macroeconomic Volatility ({b_vol}-Yr) ({b_start}-{b_end})", font=dict(size=17), x=0.5, xanchor="center"),
                xaxis=dict(title='Year', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                yaxis=dict(title='Standard Deviation', tickformat='.1%', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                hovermode='x unified', height=580, margin=dict(t=60, b=20, l=10, r=10)
            )
            st.plotly_chart(fig_b_vol, use_container_width=True)

        st.divider()
        st.subheader("Bilateral Performance & Convergence Tables")

        def get_bilateral_stats(g):
            g = g.dropna()
            gm = stats.gmean(1 + g) - 1 if not g.empty and (1 + g > 0).all() else np.nan
            return [f"{gm:.2%}" if pd.notnull(gm) else "N/A", f"{g.median():.2%}", f"{g.max():.2%}", f"{g.min():.2%}", f"{g.std():.2%}"]

        b_stats_df = pd.DataFrame({
            'Metric': ['CAGR', 'Median', 'Max', 'Min', 'Std Dev'],
            f'{tag1} GDP Gr': get_bilateral_stats(d1['gdpGrowth']),
            f'{tag2} GDP Gr': get_bilateral_stats(d2['gdpGrowth']),
            f'{tag1} GDP/c Gr': get_bilateral_stats(d1['gdpcGrowth']),
            f'{tag2} GDP/c Gr': get_bilateral_stats(d2['gdpcGrowth']),
            f'{tag1} SoL Gr': get_bilateral_stats(d1['solGrowth']),
            f'{tag2} SoL Gr': get_bilateral_stats(d2['solGrowth']),
            f'{tag1} Pop Gr': get_bilateral_stats(d1['popGrowth']),
            f'{tag2} Pop Gr': get_bilateral_stats(d2['popGrowth'])
        })

        def get_b_decomp(df_run):
            i_gdp, f_gdp = df_run['gdp'].iloc[0], df_run['gdp'].iloc[-1]
            i_gdpc, f_gdpc = df_run['gdpc'].iloc[0], df_run['gdpc'].iloc[-1]
            i_pop, f_pop = df_run['population'].iloc[0], df_run['population'].iloc[-1]
            ln_gdp = np.log(f_gdp / i_gdp)
            s_int = (np.log(f_gdpc / i_gdpc) / ln_gdp) if ln_gdp != 0 else 0
            s_ext = (np.log(f_pop / i_pop) / ln_gdp) if ln_gdp != 0 else 0
            return [f"£{i_gdp:,.2f}M -> £{f_gdp:,.2f}M", f"{s_int:.1%}", f"{s_ext:.1%}"]

        b_decomp_df = pd.DataFrame({
            'Component': ['Total GDP Span', 'Intensive Share (Productivity)', 'Extensive Share (Demographics)'],
            f'{n1}': get_b_decomp(d1),
            f'{n2}': get_b_decomp(d2)
        })

        t3_c1, t3_c2 = st.columns(2)
        with t3_c1:
            st.write(f"**Comparative Performance Statistics: {tag1} vs {tag2}**")
            st.dataframe(b_stats_df, use_container_width=True, hide_index=True)
            st.write("**Growth Accounting Decomposition Comparison**")
            st.dataframe(b_decomp_df, use_container_width=True, hide_index=True)

        with t3_c2:
            if not combined.empty:
                init_r, final_r = combined['catchup_ratio'].iloc[0], combined['catchup_ratio'].iloc[-1]
                net_spread = (final_r - init_r) * 100
                
                conv_summary_df = pd.DataFrame({
                    'Convergence Metric': [
                        'Initial Productivity Ratio', 
                        'Final Productivity Ratio', 
                        'Net Convergence Spread', 
                        f'{n1} GDP/c CAGR', 
                        f'{n2} GDP/c CAGR'
                    ],
                    'Result': [
                        f"{init_r:.2%}", 
                        f"{final_r:.2%}", 
                        f"{net_spread:+.2f} percentage points",
                        f"{(stats.gmean(1 + combined['target_g'].dropna()) - 1):.2%}",
                        f"{(stats.gmean(1 + combined['hegemon_g'].dropna()) - 1):.2%}"
                    ]
                })
                st.write(f"**Convergence Dynamics Summary: {tag1} vs {tag2}**")
                st.dataframe(conv_summary_df, use_container_width=True, hide_index=True)

# ==========================================
# TAB 4: CROSS-CAMPAIGN META-ANALYSIS
# ==========================================
with tab4:
    st.header("Cross-Campaign Meta-Analysis")
    if len(csv_files) < 2:
        st.info("Cross-campaign analysis requires at least 2 CSV files in your workspace folder.")
    else:
        col_t4_f1, col_t4_f2 = st.columns(2)
        with col_t4_f1:
            file_meta_1 = st.selectbox("First Campaign File (Run 1):", csv_files, index=0, key="t4_f1")
            played_t4_1 = file_meta_1.split('-')[0].upper() if '-' in file_meta_1 else "UNKNOWN"
        with col_t4_f2:
            file_meta_2 = st.selectbox("Second Campaign File (Run 2):", csv_files, index=1 if len(csv_files) > 1 else 0, key="t4_f2")
            played_t4_2 = file_meta_2.split('-')[0].upper() if '-' in file_meta_2 else "UNKNOWN"

        df_meta_1 = load_data(file_meta_1)
        df_meta_2 = load_data(file_meta_2)

        tags_meta_1 = sorted(df_meta_1.dropna(subset=['gdp'])['country'].unique().tolist())
        tags_meta_2 = sorted(df_meta_2.dropna(subset=['gdp'])['country'].unique().tolist())

        col_t4_1, col_t4_2, col_t4_3, col_t4_4 = st.columns([1, 1, 2, 1])
        with col_t4_1:
            idx1 = tags_meta_1.index(played_t4_1) if played_t4_1 in tags_meta_1 else 0
            meta_tag1 = st.selectbox("Nation in Run 1:", tags_meta_1, index=idx1, format_func=lambda x: f"{x} - {vic3_tags.get(x, x)}", key="t4_tag1")
        with col_t4_2:
            idx2 = tags_meta_2.index(played_t4_2) if played_t4_2 in tags_meta_2 else 0
            meta_tag2 = st.selectbox("Nation in Run 2:", tags_meta_2, index=idx2, format_func=lambda x: f"{x} - {vic3_tags.get(x, x)}", key="t4_tag2")

        d1_m_temp = df_meta_1[df_meta_1['country'] == meta_tag1].copy().reset_index(drop=True)
        d2_m_temp = df_meta_2[df_meta_2['country'] == meta_tag2].copy().reset_index(drop=True)

        if d1_m_temp.empty or d2_m_temp.empty:
            st.warning("Insufficient data for the selected countries.")
        else:
            min_m_yr = min(int(d1_m_temp['year'].min()), int(d2_m_temp['year'].min()))
            max_m_yr = max(int(d1_m_temp['year'].max()), int(d2_m_temp['year'].max()))

            with col_t4_3:
                m_start, m_end = st.slider(
                    "Cross-Campaign Timeline Range:", 
                    min_value=min_m_yr, 
                    max_value=max_m_yr, 
                    value=(min_m_yr, max_m_yr),
                    key="t4_slider"
                )
            with col_t4_4:
                m_ma = st.number_input("Trend Window (Years):", min_value=1, max_value=50, value=10, key="t4_ma")
                m_vol = st.number_input("Volatility Window (Years):", min_value=2, max_value=50, value=10, key="t4_vol")

            d1_m = d1_m_temp[(d1_m_temp['year'] >= m_start) & (d1_m_temp['year'] <= m_end)].copy().sort_values('year').reset_index(drop=True)
            d2_m = d2_m_temp[(d2_m_temp['year'] >= m_start) & (d2_m_temp['year'] <= m_end)].copy().sort_values('year').reset_index(drop=True)

            n1_m = f"Run 1: {vic3_tags.get(meta_tag1, meta_tag1)} ({meta_tag1})"
            n2_m = f"Run 2: {vic3_tags.get(meta_tag2, meta_tag2)} ({meta_tag2})"
            col1_m, col2_m = '#2E86AB', '#E84855'

            st.subheader("Cross-Campaign Trajectories")

            def plot_cross_meta(col, title, y_title, is_pct=False):
                fig = go.Figure()
                if is_pct:
                    roll1 = d1_m[col].rolling(window=m_ma, min_periods=1).apply(
                        lambda x: np.prod(1 + x)**(1.0/len(x)) - 1 if (1 + x >= 0).all() else np.nan, raw=True
                    )
                    roll2 = d2_m[col].rolling(window=m_ma, min_periods=1).apply(
                        lambda x: np.prod(1 + x)**(1.0/len(x)) - 1 if (1 + x >= 0).all() else np.nan, raw=True
                    )
                    t_lbl = f'{m_ma}-Yr CAGR'
                else:
                    roll1 = d1_m[col].rolling(window=m_ma, min_periods=1).mean()
                    roll2 = d2_m[col].rolling(window=m_ma, min_periods=1).mean()
                    t_lbl = f'{m_ma}-Yr Trend'

                val_fmt = '.2%' if is_pct else '.2f'

                fig.add_trace(go.Scatter(
                    x=d1_m['year'], y=roll1, mode='lines', name=f'{n1_m} {t_lbl}', 
                    line=dict(color=col1_m, width=1.5, dash='dash'), 
                    hovertemplate=f"<b>{n1_m} {t_lbl}</b>: %{{y:{val_fmt}}}<extra></extra>"
                ))
                fig.add_trace(go.Scatter(
                    x=d2_m['year'], y=roll2, mode='lines', name=f'{n2_m} {t_lbl}', 
                    line=dict(color=col2_m, width=1.5, dash='dash'), 
                    hovertemplate=f"<b>{n2_m} {t_lbl}</b>: %{{y:{val_fmt}}}<extra></extra>"
                ))
                fig.add_trace(go.Scatter(
                    x=d1_m['year'], y=d1_m[col], mode='lines', name=f'{n1_m} Data', 
                    line=dict(color=col1_m, width=2.5), 
                    hovertemplate=f"<b>{n1_m} Data</b>: %{{y:{val_fmt}}}<extra></extra>"
                ))
                fig.add_trace(go.Scatter(
                    x=d2_m['year'], y=d2_m[col], mode='lines', name=f'{n2_m} Data', 
                    line=dict(color=col2_m, width=2.5), 
                    hovertemplate=f"<b>{n2_m} Data</b>: %{{y:{val_fmt}}}<extra></extra>"
                ))

                fig.update_layout(
                    title=dict(text=f"{title} ({m_start}-{m_end})", font=dict(size=18), x=0.5, xanchor="center"),
                    xaxis=dict(title='Year', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                    yaxis=dict(
                        title=y_title, 
                        tickformat='.1%' if is_pct else None, 
                        showgrid=True, 
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        zeroline=True, 
                        zerolinecolor='#888888', 
                        zerolinewidth=1.5
                    ),
                    hovermode='x unified',
                    height=380,
                    updatemenus=[] if is_pct else get_scale_menu(),
                    margin=dict(t=85, b=30, l=10, r=10)
                )
                st.plotly_chart(fig, use_container_width=True)

            mc1, mc2 = st.columns(2)
            with mc1:
                plot_cross_meta('gdp', f'Total GDP: {n1_m} vs {n2_m}', '£ Millions')
                plot_cross_meta('gdpc', f'GDP per Capita: {n1_m} vs {n2_m}', '£')
                plot_cross_meta('population', f'Population: {n1_m} vs {n2_m}', 'Millions')
                plot_cross_meta('sol', f'Standard of Living: {n1_m} vs {n2_m}', 'Index')
            with mc2:
                plot_cross_meta('gdpGrowth', f'GDP Growth Rate: {n1_m} vs {n2_m}', 'Rate (%)', is_pct=True)
                plot_cross_meta('gdpcGrowth', f'GDP/c Growth Rate: {n1_m} vs {n2_m}', 'Rate (%)', is_pct=True)
                plot_cross_meta('popGrowth', f'Population Growth Rate: {n1_m} vs {n2_m}', 'Rate (%)', is_pct=True)
                plot_cross_meta('solGrowth', f'SoL Growth Rate: {n1_m} vs {n2_m}', 'Rate (%)', is_pct=True)

            st.divider()
            st.subheader("Cross-Campaign Growth Accounting")

            def plot_meta_decomp(df_run, run_name):
                r_start, r_end = (int(df_run['year'].min()), int(df_run['year'].max())) if not df_run.empty else ("N/A", "N/A")
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(x=df_run['year'], y=df_run['gdpcGrowth'], name='Intensive (Productivity)', marker_color='#E76F51', hovertemplate='<b>Intensive</b>: %{y:.2%}<extra></extra>'))
                fig1.add_trace(go.Bar(x=df_run['year'], y=df_run['popGrowth'], name='Extensive (Demographics)', marker_color='#2A9D8F', hovertemplate='<b>Extensive</b>: %{y:.2%}<extra></extra>'))
                fig1.add_trace(go.Scatter(x=df_run['year'], y=df_run['gdpGrowth'], mode='lines+markers', name='Total GDP Growth', line=dict(color='#1D3557', width=2.5), hovertemplate='<b>Total Growth</b>: %{y:.2%}<extra></extra>'))
                fig1.update_layout(
                    barmode='relative', 
                    title=dict(text=f"{run_name} — Annual Decomposition ({r_start}-{r_end})", font=dict(size=17), x=0.5, xanchor="center"),
                    yaxis=dict(title='Annual Contribution', tickformat='.1%', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', zeroline=True, zerolinecolor='#888888', zerolinewidth=1.5),
                    xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', range=[m_start - 0.5, m_end + 0.5]),
                    hovermode='x unified', height=400, margin=dict(t=60, b=20, l=10, r=10)
                )
                st.plotly_chart(fig1, use_container_width=True)

                roll_gdpc = df_run['gdpcGrowth'].rolling(window=m_ma, min_periods=1).apply(lambda x: np.prod(1 + x)**(1.0/len(x)) - 1 if (1 + x >= 0).all() else np.nan, raw=True)
                roll_pop = df_run['popGrowth'].rolling(window=m_ma, min_periods=1).apply(lambda x: np.prod(1 + x)**(1.0/len(x)) - 1 if (1 + x >= 0).all() else np.nan, raw=True)
                roll_gdp = df_run['gdpGrowth'].rolling(window=m_ma, min_periods=1).apply(lambda x: np.prod(1 + x)**(1.0/len(x)) - 1 if (1 + x >= 0).all() else np.nan, raw=True)

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=df_run['year'], y=roll_gdpc, name='Trend Intensive', marker_color='#E76F51', hovertemplate='<b>Trend Intensive</b>: %{y:.2%}<extra></extra>'))
                fig2.add_trace(go.Bar(x=df_run['year'], y=roll_pop, name='Trend Extensive', marker_color='#2A9D8F', hovertemplate='<b>Trend Extensive</b>: %{y:.2%}<extra></extra>'))
                fig2.add_trace(go.Scatter(x=df_run['year'], y=roll_gdp, mode='lines+markers', name='Trend GDP', line=dict(color='#1D3557', width=2.5), hovertemplate='<b>Trend GDP</b>: %{y:.2%}<extra></extra>'))
                fig2.update_layout(
                    barmode='relative',
                    title=dict(text=f"{run_name} — Structural Decomposition ({m_ma}-Yr CAGR) ({r_start}-{r_end})", font=dict(size=17), x=0.5, xanchor="center"),
                    yaxis=dict(title='Trend Contribution', tickformat='.1%', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', zeroline=True, zerolinecolor='#888888', zerolinewidth=1.5),
                    xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', range=[m_start - 0.5, m_end + 0.5]),
                    hovermode='x unified', height=400, margin=dict(t=60, b=20, l=10, r=10)
                )
                st.plotly_chart(fig2, use_container_width=True)

            md_col1, md_col2 = st.columns(2)
            with md_col1:
                plot_meta_decomp(d1_m, n1_m)
            with md_col2:
                plot_meta_decomp(d2_m, n2_m)

            st.divider()
            st.subheader("Cross-Campaign Convergence & Catch-Up Dynamics")

            d1_m_idx = d1_m.set_index('year')
            d2_m_idx = d2_m.set_index('year')

            combined_m = pd.DataFrame({
                'target_gdpc': d1_m_idx['gdpc'],
                'hegemon_gdpc': d2_m_idx['gdpc'],
                'target_g': d1_m_idx['gdpcGrowth'],
                'hegemon_g': d2_m_idx['gdpcGrowth']
            }).dropna()

            if not combined_m.empty:
                c_start_m, c_end_m = int(combined_m.index.min()), int(combined_m.index.max())
                combined_m['catchup_ratio'] = combined_m['target_gdpc'] / combined_m['hegemon_gdpc']
                combined_m['velocity'] = combined_m['target_g'] - combined_m['hegemon_g']
                years_m = combined_m.index

                fig_m_conv = go.Figure()
                fig_m_conv.add_trace(go.Scatter(
                    x=years_m, y=combined_m['catchup_ratio'], mode='lines+markers', 
                    name=f'{n1_m} / {n2_m} Ratio', line=dict(color='#2E86AB', width=2.5), 
                    hovertemplate=f"<b>Productivity Ratio</b>: %{{y:.2%}}<extra></extra>"
                ))
                fig_m_conv.add_hline(
                    y=1.0, line_dash="dash", line_color="#D62828", 
                    annotation_text="Productivity Parity (100%)", annotation_position="top right"
                )
                fig_m_conv.update_layout(
                    title=dict(text=f"Productivity Convergence: {n1_m} relative to {n2_m} ({c_start_m}-{c_end_m})", font=dict(size=18), x=0.5, xanchor="center"),
                    xaxis=dict(title='Year', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                    yaxis=dict(title=f'GDP/c Ratio', tickformat='.1%', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                    hovermode='x unified', height=420, margin=dict(t=60, b=20, l=10, r=10)
                )
                st.plotly_chart(fig_m_conv, use_container_width=True)

                colors_m = np.where(combined_m['velocity'] >= 0, '#2A9D8F', '#E84855')
                fig_m_vel = go.Figure(go.Bar(
                    x=years_m, y=combined_m['velocity'], marker_color=colors_m, 
                    name='Catch-up Spread', 
                    hovertemplate="<b>Growth Rate Spread</b>: %{y:+.2%}<extra></extra>"
                ))
                fig_m_vel.update_layout(
                    title=dict(text=f"Convergence Velocity (Run 1 GDP/c Growth minus Run 2 GDP/c Growth) ({c_start_m}-{c_end_m})", font=dict(size=18), x=0.5, xanchor="center"),
                    xaxis=dict(title='Year', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                    yaxis=dict(
                        title='Annual Spread (%)', tickformat='+.1%', 
                        showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', 
                        zeroline=True, zerolinecolor='#888888', zerolinewidth=1.5
                    ),
                    hovermode='x unified', height=380, margin=dict(t=60, b=20, l=10, r=10)
                )
                st.plotly_chart(fig_m_vel, use_container_width=True)

            st.divider()
            st.subheader("Cross-Campaign Macroeconomic Volatility")

            mc5, mc6 = st.columns(2)
            with mc5:
                # Chart 1: Run 1 Expansion vs Contraction
                a1_start_m, a1_end_m = (int(d1_m['year'].min()), int(d1_m['year'].max())) if not d1_m.empty else ("N/A", "N/A")
                colors1_m = np.where(d1_m['gdpGrowth'] >= 0, '#2A9D8F', '#E76F51')
                fig_m_boom1 = go.Figure(go.Bar(x=d1_m['year'], y=d1_m['gdpGrowth'], marker_color=colors1_m, name=n1_m, hovertemplate=f"<b>{n1_m}</b>: %{{y:.2%}}<extra></extra>"))
                fig_m_boom1.update_layout(
                    title=dict(text=f"{n1_m} — Expansion vs Contraction ({a1_start_m}-{a1_end_m})", font=dict(size=15), x=0.5, xanchor="center"),
                    xaxis=dict(title='Year', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', range=[m_start - 0.5, m_end + 0.5]),
                    yaxis=dict(tickformat='.1%', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', zeroline=True, zerolinecolor='#888888', zerolinewidth=1.5),
                    hovermode='x unified', height=280, margin=dict(t=50, b=20, l=10, r=10)
                )
                st.plotly_chart(fig_m_boom1, use_container_width=True)

                # Chart 2: Run 2 Expansion vs Contraction
                a2_start_m, a2_end_m = (int(d2_m['year'].min()), int(d2_m['year'].max())) if not d2_m.empty else ("N/A", "N/A")
                colors2_m = np.where(d2_m['gdpGrowth'] >= 0, '#2A9D8F', '#E76F51')
                fig_m_boom2 = go.Figure(go.Bar(x=d2_m['year'], y=d2_m['gdpGrowth'], marker_color=colors2_m, name=n2_m, hovertemplate=f"<b>{n2_m}</b>: %{{y:.2%}}<extra></extra>"))
                fig_m_boom2.update_layout(
                    title=dict(text=f"{n2_m} — Expansion vs Contraction ({a2_start_m}-{a2_end_m})", font=dict(size=15), x=0.5, xanchor="center"),
                    xaxis=dict(title='Year', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', range=[m_start - 0.5, m_end + 0.5]),
                    yaxis=dict(tickformat='.1%', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', zeroline=True, zerolinecolor='#888888', zerolinewidth=1.5),
                    hovermode='x unified', height=280, margin=dict(t=50, b=20, l=10, r=10)
                )
                st.plotly_chart(fig_m_boom2, use_container_width=True)

            with mc6:
                roll_m_vol1 = d1_m['gdpGrowth'].rolling(window=m_vol, min_periods=2).std()
                roll_m_vol2 = d2_m['gdpGrowth'].rolling(window=m_vol, min_periods=2).std()

                fig_m_vol = go.Figure()
                fig_m_vol.add_trace(go.Scatter(
                    x=d1_m['year'], y=roll_m_vol1, mode='lines+markers', name=f'{n1_m} Volatility', 
                    line=dict(color=col1_m, width=2.5), 
                    hovertemplate=f"<b>{n1_m} Volatility</b>: %{{y:.2%}}<extra></extra>"
                ))
                fig_m_vol.add_trace(go.Scatter(
                    x=d2_m['year'], y=roll_m_vol2, mode='lines+markers', name=f'{n2_m} Volatility', 
                    line=dict(color=col2_m, width=2.5), 
                    hovertemplate=f"<b>{n2_m} Volatility</b>: %{{y:.2%}}<extra></extra>"
                ))
                fig_m_vol.update_layout(
                    title=dict(text=f"Comparative Macroeconomic Volatility ({m_vol}-Yr) ({m_start}-{m_end})", font=dict(size=17), x=0.5, xanchor="center"),
                    xaxis=dict(title='Year', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                    yaxis=dict(title='Standard Deviation', tickformat='.1%', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                    hovermode='x unified', height=580, margin=dict(t=60, b=20, l=10, r=10)
                )
                st.plotly_chart(fig_m_vol, use_container_width=True)

            st.divider()
            st.subheader("Cross-Campaign Performance & Convergence Tables")

            def get_meta_stats(g):
                g = g.dropna()
                gm = stats.gmean(1 + g) - 1 if not g.empty and (1 + g > 0).all() else np.nan
                return [f"{gm:.2%}" if pd.notnull(gm) else "N/A", f"{g.median():.2%}", f"{g.max():.2%}", f"{g.min():.2%}", f"{g.std():.2%}"]

            m_stats_df = pd.DataFrame({
                'Metric': ['CAGR', 'Median', 'Max', 'Min', 'Std Dev'],
                f'R1 ({meta_tag1}) GDP Gr': get_meta_stats(d1_m['gdpGrowth']),
                f'R2 ({meta_tag2}) GDP Gr': get_meta_stats(d2_m['gdpGrowth']),
                f'R1 ({meta_tag1}) GDP/c Gr': get_meta_stats(d1_m['gdpcGrowth']),
                f'R2 ({meta_tag2}) GDP/c Gr': get_meta_stats(d2_m['gdpcGrowth']),
                f'R1 ({meta_tag1}) SoL Gr': get_meta_stats(d1_m['solGrowth']),
                f'R2 ({meta_tag2}) SoL Gr': get_meta_stats(d2_m['solGrowth']),
                f'R1 ({meta_tag1}) Pop Gr': get_meta_stats(d1_m['popGrowth']),
                f'R2 ({meta_tag2}) Pop Gr': get_meta_stats(d2_m['popGrowth'])
            })

            def get_m_decomp(df_run):
                i_gdp, f_gdp = df_run['gdp'].iloc[0], df_run['gdp'].iloc[-1]
                i_gdpc, f_gdpc = df_run['gdpc'].iloc[0], df_run['gdpc'].iloc[-1]
                i_pop, f_pop = df_run['population'].iloc[0], df_run['population'].iloc[-1]
                ln_gdp = np.log(f_gdp / i_gdp)
                s_int = (np.log(f_gdpc / i_gdpc) / ln_gdp) if ln_gdp != 0 else 0
                s_ext = (np.log(f_pop / i_pop) / ln_gdp) if ln_gdp != 0 else 0
                return [f"£{i_gdp:,.2f}M -> £{f_gdp:,.2f}M", f"{s_int:.1%}", f"{s_ext:.1%}"]

            m_decomp_df = pd.DataFrame({
                'Component': ['Total GDP Span', 'Intensive Share (Productivity)', 'Extensive Share (Demographics)'],
                f'{n1_m}': get_m_decomp(d1_m),
                f'{n2_m}': get_m_decomp(d2_m)
            })

            t4_c1, t4_c2 = st.columns(2)
            with t4_c1:
                st.write(f"**Performance Stats: {meta_tag1} (Run 1) vs {meta_tag2} (Run 2)**")
                st.dataframe(m_stats_df, use_container_width=True, hide_index=True)
                st.write("**Growth Accounting Comparison**")
                st.dataframe(m_decomp_df, use_container_width=True, hide_index=True)

            with t4_c2:
                if not combined_m.empty:
                    init_r_m, final_r_m = combined_m['catchup_ratio'].iloc[0], combined_m['catchup_ratio'].iloc[-1]
                    net_spread_m = (final_r_m - init_r_m) * 100
                    
                    conv_m_summary_df = pd.DataFrame({
                        'Convergence Metric': [
                            'Initial Productivity Ratio', 
                            'Final Productivity Ratio', 
                            'Net Convergence Spread', 
                            f'Run 1 ({meta_tag1}) GDP/c CAGR', 
                            f'Run 2 ({meta_tag2}) GDP/c CAGR'
                        ],
                        'Result': [
                            f"{init_r_m:.2%}", 
                            f"{final_r_m:.2%}", 
                            f"{net_spread_m:+.2f} percentage points",
                            f"{(stats.gmean(1 + combined_m['target_g'].dropna()) - 1):.2%}",
                            f"{(stats.gmean(1 + combined_m['hegemon_g'].dropna()) - 1):.2%}"
                        ]
                    })
                    st.write(f"**Convergence Dynamics Summary**")
                    st.dataframe(conv_m_summary_df, use_container_width=True, hide_index=True)
