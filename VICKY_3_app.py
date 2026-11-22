import inspect
import os
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.special import gammaln
import streamlit as st
import streamlit.components.v1 as components

# ==============================================================================
# 1. PAGE CONFIG, COMPATIBILITY HELPERS & SESSION STATE
# ==============================================================================
st.set_page_config(
    page_title="Victoria 3 Macroeconomics & Bayesian Lab",
    page_icon="🌍",
    layout="wide",
)

# Version-safe stretch arguments across Streamlit releases
_USE_NEW_WIDTH = "width" in inspect.signature(st.button).parameters
STRETCH_KW = {"width": "stretch"} if _USE_NEW_WIDTH else {"use_container_width": True}
PLOT_STRETCH = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.plotly_chart).parameters
    else {"use_container_width": True}
)

FUSO_BR = timezone(timedelta(hours=-3))

if "bookmarks" not in st.session_state:
    st.session_state["bookmarks"] = []
if "bm_form_ver" not in st.session_state:
    st.session_state["bm_form_ver"] = 0
if "tab_jump_target" not in st.session_state:
    st.session_state["tab_jump_target"] = None
if "bookmark_notice" not in st.session_state:
    st.session_state["bookmark_notice"] = ""


def ensure_select(key, options, default_val):
    if not options:
        return None
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = default_val if default_val in options else options[0]
    return st.session_state[key]


def ensure_multiselect(key, options, default_vals):
    if key not in st.session_state:
        st.session_state[key] = [v for v in default_vals if v in options]
    else:
        valid = [v for v in st.session_state[key] if v in options]
        st.session_state[key] = valid if valid else [v for v in default_vals if v in options]
    return st.session_state[key]


def ensure_slider_range(key, min_yr, max_yr):
    if min_yr >= max_yr:
        st.session_state[key] = (min_yr, max_yr)
        return (min_yr, max_yr)
    if (
        key not in st.session_state
        or not isinstance(st.session_state[key], (tuple, list))
        or len(st.session_state[key]) != 2
    ):
        st.session_state[key] = (min_yr, max_yr)
    else:
        lo, hi = st.session_state[key]
        lo = max(min_yr, min(int(lo), max_yr))
        hi = max(lo, min(int(hi), max_yr))
        st.session_state[key] = (lo, hi)
    return st.session_state[key]


def renumber_bookmarks():
    for idx, bm in enumerate(st.session_state["bookmarks"], start=1):
        bm["id"] = idx


def save_bookmark(tab_name, tab_idx, note, file_label, summary_metrics, snapshot):
    bm_id = len(st.session_state["bookmarks"]) + 1
    now_str = datetime.now(FUSO_BR).strftime("%H:%M:%S")
    bm = {
        "uid": uuid.uuid4().hex,
        "id": bm_id,
        "time": now_str,
        "tab_name": tab_name,
        "tab_idx": tab_idx,
        "note": note.strip() if note else "",
        "file": file_label,
        "summary": summary_metrics,
        "snapshot": dict(snapshot),
    }
    st.session_state["bookmarks"].append(bm)
    renumber_bookmarks()
    label = f" [{bm['note']}]" if bm["note"] else ""
    st.toast(f"🔖 Bookmark #{bm_id} saved ({tab_name}{label})!")


def cb_load_bookmark(uid):
    for bm in st.session_state["bookmarks"]:
        if bm["uid"] == uid:
            for k, v in bm["snapshot"].items():
                st.session_state[k] = v
            st.session_state["tab_jump_target"] = int(bm["tab_idx"])
            note_txt = f" — *{bm['note']}*" if bm["note"] else ""
            st.session_state["bookmark_notice"] = (
                f"🔖 Loaded **Bookmark #{bm['id']}** ({bm['tab_name']}{note_txt}): `{bm['summary']}`"
            )
            break


def cb_discrete_post_to_prior(post_vals):
    for idx, val in enumerate(post_vals):
        st.session_state[f"t5_disc_p_{idx}"] = round(float(val), 4)


# Auto-jump JS when restoring a bookmark
if st.session_state.get("tab_jump_target") is not None:
    _jump_idx = int(st.session_state["tab_jump_target"])
    st.session_state["tab_jump_target"] = None
    components.html(
        f"""
        <script>
        (function() {{
            const doc = window.parent.document;
            const tabs = doc.querySelectorAll('button[data-baseweb="tab"]');
            if (tabs && tabs.length > {_jump_idx}) {{
                tabs[{_jump_idx}].click();
                window.parent.scrollTo({{ top: 0, behavior: 'smooth' }});
            }}
        }})();
        </script>
        """,
        height=0,
    )


# ==============================================================================
# 2. CONSTANTS, TAGS & PLOTLY STYLING
# ==============================================================================
VIC3_TAGS = {
    "GBR": "Great Britain", "FRA": "France", "USA": "United States", "GER": "Germany", "PRU": "Prussia",
    "NGF": "North German Federation", "SGF": "South German Federation", "HRE": "Central Europe",
    "RUS": "Russia", "ITA": "Italy", "SPA": "Spain", "IBE": "Iberia", "POR": "Portugal",
    "AUS": "Austria", "KUK": "Austria-Hungary", "TUR": "Ottoman Empire", "BYZ": "Byzantium",
    "SWE": "Sweden", "NOR": "Norway", "DEN": "Denmark", "FIN": "Finland", "SCA": "Scandinavia",
    "BEL": "Belgium", "NET": "Netherlands", "SWI": "Switzerland", "POL": "Poland", "PLC": "Poland-Lithuania",
    "JAP": "Japan", "CHI": "China", "QIN": "Qing", "EIC": "East India Company", "BIC": "British Raj",
    "KOR": "Korea", "DAI": "Dai Nam", "SIA": "Siam", "SIK": "Sikh Empire", "PER": "Persia", "AFG": "Afghanistan",
    "BRM": "Burma", "TIB": "Tibet", "MGL": "Mongolia",
    "BRZ": "Brazil", "PNI": "Riograndense Republic", "MEX": "Mexico", "CAN": "Canada", "HBC": "Hudson Bay Co.",
    "TEX": "Texas", "CSA": "Confederate States", "CUB": "Cuba", "HAI": "Haiti",
    "ARG": "Argentina", "RPL": "Rio de la Plata", "CHL": "Chile", "PEU": "Peru", "PBC": "Peru-Bolivia",
    "COL": "Colombia", "GCO": "Gran Colombia", "NGR": "New Granada", "VNZ": "Venezuela", "ECU": "Ecuador",
    "BOL": "Bolivia", "PAR": "Paraguay", "URU": "Uruguay", "FND": "Federation of the Americas",
    "SAR": "Sardinia-Piedmont", "SIC": "Two Sicilies", "PAP": "Papal States", "TUS": "Tuscany",
    "BAV": "Bavaria", "SAX": "Saxony", "WUR": "Württemberg", "HAN": "Hanover",
    "GRE": "Greece", "SER": "Serbia", "YUG": "Yugoslavia", "ROM": "Romania", "BUL": "Bulgaria",
    "HUN": "Hungary", "UKR": "Ukraine",
    "EGY": "Egypt", "MOR": "Morocco", "ETH": "Ethiopia", "SOK": "Sokoto", "ZUL": "Zulu",
    "SAF": "South Africa", "ORA": "Oranje", "TRN": "Transvaal", "MAD": "Madagascar",
    "DEI": "Dutch East Indies", "JAV": "Java", "AST": "Australia", "PHI": "Philippines",
}

PALETTE = [
    "#2E86AB", "#E84855", "#2A9D8F", "#F4A261", "#9B5DE5",
    "#E76F51", "#00BBF9", "#00F5D4", "#D62828", "#003049",
]


def tag_label(tag):
    return f"{tag} - {VIC3_TAGS.get(tag, tag)}"


def country_pretty(tag):
    return VIC3_TAGS.get(tag, tag)


def infer_played_tag(file_label):
    clean = file_label.replace("☁️ ", "")
    return clean.split("-")[0].upper() if "-" in clean else "UNKNOWN"


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
                dict(label="Log", method="relayout", args=[{"yaxis.type": "log"}]),
            ],
        )
    ]


# ==============================================================================
# 3. DATA ENGINE & ECONOMETRIC / BAYESIAN MATH CORE
# ==============================================================================
@st.cache_data
def load_data(file_source):
    if hasattr(file_source, "seek"):
        file_source.seek(0)
    df_raw = pd.read_csv(file_source)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw["gdpc"] = df_raw["gdpc"] / 100000.0
    safe_gdpc = df_raw["gdpc"].replace(0, np.nan)
    df_raw["population"] = df_raw["gdp"] / safe_gdpc
    df_raw["year"] = df_raw["date"].dt.year

    df = df_raw.sort_values("date").groupby(["country", "year"]).last().reset_index()
    df["gdpGrowth"] = df.groupby("country")["gdp"].pct_change(fill_method=None)
    df["gdpcGrowth"] = df.groupby("country")["gdpc"].pct_change(fill_method=None)
    df["popGrowth"] = df.groupby("country")["population"].pct_change(fill_method=None)
    df["solGrowth"] = df.groupby("country")["sol"].pct_change(fill_method=None)
    df["logGdpGrowth"] = np.log1p(df["gdpGrowth"].where(df["gdpGrowth"] > -0.999))
    df["logGdpcGrowth"] = np.log1p(df["gdpcGrowth"].where(df["gdpcGrowth"] > -0.999))
    return df


def safe_cagr(start_val, end_val, periods):
    if pd.notnull(start_val) and pd.notnull(end_val) and start_val > 0 and end_val > 0 and periods > 1:
        return (end_val / start_val) ** (1.0 / (periods - 1)) - 1.0
    return np.nan


def rolling_cagr(series, window):
    return series.rolling(window=window, min_periods=1).apply(
        lambda x: np.prod(1.0 + x) ** (1.0 / len(x)) - 1.0 if (1.0 + x > 0).all() else np.nan,
        raw=True,
    )


def hp_filter(y, lamb=6.25):
    """Hodrick-Prescott filter (Ravn-Uhlig annual default lambda=6.25). Returns (trend, cycle)."""
    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)
    if n < 4 or np.isnan(y_arr).any():
        return y_arr.copy(), np.zeros_like(y_arr)
    d = np.zeros((n - 2, n))
    for i in range(n - 2):
        d[i, i] = 1.0
        d[i, i + 1] = -2.0
        d[i, i + 2] = 1.0
    mat = np.eye(n) + lamb * (d.T @ d)
    trend = np.linalg.solve(mat, y_arr)
    cycle = y_arr - trend
    return trend, cycle


def detect_crises(df_country, min_drawdown=0.0):
    """
    Identifies peak-to-trough-to-recovery economic contractions.
    min_drawdown is a positive proportion (e.g., 0.02 means contractions of at least -2%).
    """
    df_c = df_country.dropna(subset=["gdp"]).sort_values("year").reset_index(drop=True)
    if df_c.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_c["peak_gdp"] = df_c["gdp"].cummax()
    safe_peak = df_c["peak_gdp"].replace(0, np.nan)
    df_c["drawdown"] = ((df_c["gdp"] - df_c["peak_gdp"]) / safe_peak).fillna(0.0)

    crises = []
    in_crisis = False
    s_yr, min_dd, t_yr = None, 0.0, None

    for i, row in df_c.iterrows():
        yr, dd = int(row["year"]), float(row["drawdown"])
        if dd < 0:
            if not in_crisis:
                in_crisis = True
                s_yr = int(df_c.loc[i - 1, "year"]) if i > 0 else yr
                min_dd = dd
                t_yr = yr
            elif dd < min_dd:
                min_dd = dd
                t_yr = yr
        elif dd >= 0 and in_crisis:
            in_crisis = False
            if abs(min_dd) >= min_drawdown:
                crises.append({
                    "Peak Year": s_yr,
                    "Trough Year": t_yr,
                    "Recovery Year": yr,
                    "Drawdown": min_dd,
                    "Duration (Yrs)": yr - s_yr,
                    "Recovered": True,
                })

    if in_crisis and abs(min_dd) >= min_drawdown:
        max_y = int(df_c["year"].max())
        crises.append({
            "Peak Year": s_yr,
            "Trough Year": t_yr,
            "Recovery Year": "Unrecovered",
            "Drawdown": min_dd,
            "Duration (Yrs)": max(1, max_y - s_yr),
            "Recovered": False,
        })

    return df_c, pd.DataFrame(crises)


def empirical_bayes_cagr(df_all, col="logGdpGrowth"):
    """Normal-Normal Empirical Bayes shrinkage on country annual log-growth rates."""
    summary = []
    for tag, grp in df_all.groupby("country"):
        vals = grp[col].dropna()
        if len(vals) >= 1:
            summary.append({
                "Tag": tag,
                "n": len(vals),
                "mean": float(vals.mean()),
                "var": float(vals.var(ddof=1)) if len(vals) > 1 else np.nan,
            })
    s_df = pd.DataFrame(summary)
    if s_df.empty:
        return {}, 0.0, 0.001

    mature = s_df[s_df["n"] >= 10]
    ref = mature if len(mature) >= 5 else s_df

    mu0 = float(np.average(ref["mean"], weights=ref["n"]))
    tau2 = max(float(ref["mean"].var(ddof=1)) if len(ref) > 1 else 0.0004, 1e-5)
    pooled_sigma2 = float(ref["var"].dropna().median()) if not ref["var"].dropna().empty else 0.0025
    pooled_sigma2 = max(pooled_sigma2, 1e-5)

    shrunk_map = {}
    for _, r in s_df.iterrows():
        n_i = r["n"]
        sig2_i = r["var"] if (pd.notnull(r["var"]) and n_i >= 5 and r["var"] > 1e-6) else pooled_sigma2
        prec_prior = 1.0 / tau2
        prec_data = n_i / sig2_i
        post_mu = (prec_prior * mu0 + prec_data * r["mean"]) / (prec_prior + prec_data)
        shrunk_map[r["Tag"]] = (np.expm1(post_mu)) * 100.0
    return shrunk_map, mu0, tau2


def fit_empirical_beta_prior(df_all, exclude_tag=None, threshold=0.0):
    """Method-of-moments Beta(a, b) prior fitted to cross-country contraction rates."""
    rates = []
    for tag, grp in df_all.groupby("country"):
        if tag == exclude_tag:
            continue
        g = grp["gdpGrowth"].dropna()
        if len(g) >= 15:
            rates.append(float((g < threshold).mean()))
    if len(rates) < 4:
        return 1.5, 4.5
    m = float(np.clip(np.mean(rates), 0.02, 0.98))
    v = float(np.var(rates, ddof=1))
    max_v = m * (1.0 - m) * 0.85
    if v <= 1e-6 or v >= max_v:
        return round(m * 10.0, 2), round((1.0 - m) * 10.0, 2)
    common = (m * (1.0 - m) / v) - 1.0
    a = max(0.5, min(50.0, m * common))
    b = max(0.5, min(50.0, (1.0 - m) * common))
    return round(a, 2), round(b, 2)


def nig_posterior(x, mu0, kappa0, alpha0, beta0):
    """Exact Normal-Inverse-Gamma conjugate update for unknown mean & variance."""
    x_arr = np.asarray(x, dtype=float)
    x_arr = x_arr[np.isfinite(x_arr)]
    n = len(x_arr)
    if n == 0:
        return mu0, kappa0, alpha0, beta0, 0, np.nan, np.nan
    x_bar = float(np.mean(x_arr))
    ssd = float(np.sum((x_arr - x_bar) ** 2))
    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * x_bar) / kappa_n
    alpha_n = alpha0 + 0.5 * n
    beta_n = beta0 + 0.5 * ssd + (kappa0 * n * (x_bar - mu0) ** 2) / (2.0 * kappa_n)
    return mu_n, kappa_n, alpha_n, beta_n, n, x_bar, (np.sqrt(ssd / (n - 1)) if n > 1 else np.nan)


def gamma_poisson_predictive_pmf(k_vals, alpha_post, beta_rate_post, exposure=1.0):
    """Exact Negative Binomial posterior predictive PMF for Gamma(alpha, rate=beta) + Poisson."""
    p = beta_rate_post / (beta_rate_post + exposure)
    k_arr = np.asarray(k_vals, dtype=float)
    log_p = (
        gammaln(k_arr + alpha_post)
        - gammaln(alpha_post)
        - gammaln(k_arr + 1.0)
        + alpha_post * np.log(p)
        + k_arr * np.log(1.0 - p)
    )
    return np.exp(log_p)


def analytical_mode(kind, p1, p2):
    """MAP estimate for Beta(a, b) or Gamma(alpha, rate=beta)."""
    if kind == "beta":
        if p1 > 1.0 and p2 > 1.0:
            return (p1 - 1.0) / (p1 + p2 - 2.0)
        if p1 <= 1.0 and p2 > 1.0:
            return 0.0
        if p1 > 1.0 and p2 <= 1.0:
            return 1.0
        return np.nan
    if kind == "gamma_rate":
        return (p1 - 1.0) / p2 if p1 >= 1.0 else 0.0
    return np.nan


def get_perf_stats(series):
    g = series.dropna()
    if g.empty:
        return ["N/A"] * 5
    gm = stats.gmean(1.0 + g) - 1.0 if (1.0 + g > 0).all() else np.nan
    return [
        f"{gm:.2%}" if pd.notnull(gm) else "N/A",
        f"{g.median():.2%}",
        f"{g.max():.2%}",
        f"{g.min():.2%}",
        f"{g.std():.2%}",
    ]


def get_growth_decomp_summary(df_run):
    i_gdp, f_gdp = df_run["gdp"].iloc[0], df_run["gdp"].iloc[-1]
    i_gdpc, f_gdpc = df_run["gdpc"].iloc[0], df_run["gdpc"].iloc[-1]
    i_pop, f_pop = df_run["population"].iloc[0], df_run["population"].iloc[-1]
    ln_gdp = np.log(f_gdp / i_gdp) if (i_gdp > 0 and f_gdp > 0) else 0.0
    s_int = (np.log(f_gdpc / i_gdpc) / ln_gdp) if (ln_gdp != 0 and i_gdpc > 0 and f_gdpc > 0) else 0.0
    s_ext = (np.log(f_pop / i_pop) / ln_gdp) if (ln_gdp != 0 and i_pop > 0 and f_pop > 0) else 0.0
    return [f"£{i_gdp:,.2f}M → £{f_gdp:,.2f}M", f"{s_int:.1%}", f"{s_ext:.1%}"], s_int, s_ext


# ==============================================================================
# 4. REUSABLE PLOTLY VISUALIZATION SUITE
# ==============================================================================
def plot_bayesian_density_plotly(
    x_grid, pri_pdf, pos_pdf, ci_low, ci_high, ci_level,
    title, x_label, pri_name, pos_name, is_pct_axis=False, post_mean=None
):
    fig = go.Figure()
    x_fmt = ".1%" if is_pct_axis else ".3f"
    hover_x = f"%{{x:{x_fmt}}}"

    fig.add_trace(go.Scatter(
        x=x_grid, y=pri_pdf, mode="lines", name=f"Prior: {pri_name}",
        line=dict(color="#6c757d", width=2, dash="dash"),
        hovertemplate=f"<b>Prior</b> ({hover_x}): %{{y:.4f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_grid, y=pos_pdf, mode="lines", name=f"Posterior: {pos_name}",
        line=dict(color="#E84855", width=2.8),
        hovertemplate=f"<b>Posterior</b> ({hover_x}): %{{y:.4f}}<extra></extra>",
    ))

    mask = (x_grid >= ci_low) & (x_grid <= ci_high)
    if mask.any():
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_grid[mask], x_grid[mask][::-1]]),
            y=np.concatenate([pos_pdf[mask], np.zeros(mask.sum())]),
            fill="toself",
            fillcolor="rgba(232, 72, 85, 0.22)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"{int(ci_level*100)}% Credible Interval",
            hoverinfo="skip",
        ))

    if post_mean is not None and np.isfinite(post_mean):
        fig.add_vline(
            x=post_mean, line_dash="dot", line_color="#1D3557",
            annotation_text=f"Mean: {post_mean:{x_fmt}}",
            annotation_position="top right",
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=17), x=0.5, xanchor="center"),
        xaxis=dict(title=x_label, tickformat=".0%" if is_pct_axis else None, showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(title="Density", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        hovermode="x unified",
        height=430,
        updatemenus=get_scale_menu(),
        margin=dict(t=85, b=35, l=15, r=15),
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, **PLOT_STRETCH)


def plot_decomp_pair(df_run, run_name, ma_window, x_range=None):
    r_start, r_end = (int(df_run["year"].min()), int(df_run["year"].max())) if not df_run.empty else ("N/A", "N/A")
    xaxis_cfg = dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)")
    if x_range is not None:
        xaxis_cfg["range"] = x_range

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=df_run["year"], y=df_run["gdpcGrowth"], name="Intensive (Productivity)", marker_color="#E76F51", hovertemplate="<b>Intensive</b>: %{y:.2%}<extra></extra>"))
    fig1.add_trace(go.Bar(x=df_run["year"], y=df_run["popGrowth"], name="Extensive (Demographics)", marker_color="#2A9D8F", hovertemplate="<b>Extensive</b>: %{y:.2%}<extra></extra>"))
    fig1.add_trace(go.Scatter(x=df_run["year"], y=df_run["gdpGrowth"], mode="lines+markers", name="Total GDP Growth", line=dict(color="#1D3557", width=2.5), hovertemplate="<b>Total Growth</b>: %{y:.2%}<extra></extra>"))
    fig1.update_layout(
        barmode="relative",
        title=dict(text=f"{run_name} — Annual Decomposition ({r_start}-{r_end})", font=dict(size=16), x=0.5, xanchor="center"),
        yaxis=dict(title="Annual Contribution", tickformat=".1%", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", zeroline=True, zerolinecolor="#888888", zerolinewidth=1.5),
        xaxis=xaxis_cfg,
        hovermode="x unified",
        height=390,
        margin=dict(t=60, b=20, l=10, r=10),
    )
    st.plotly_chart(fig1, **PLOT_STRETCH)

    roll_gdpc = rolling_cagr(df_run["gdpcGrowth"], ma_window)
    roll_pop = rolling_cagr(df_run["popGrowth"], ma_window)
    roll_gdp = rolling_cagr(df_run["gdpGrowth"], ma_window)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df_run["year"], y=roll_gdpc, name="Trend Intensive", marker_color="#E76F51", hovertemplate="<b>Trend Intensive</b>: %{y:.2%}<extra></extra>"))
    fig2.add_trace(go.Bar(x=df_run["year"], y=roll_pop, name="Trend Extensive", marker_color="#2A9D8F", hovertemplate="<b>Trend Extensive</b>: %{y:.2%}<extra></extra>"))
    fig2.add_trace(go.Scatter(x=df_run["year"], y=roll_gdp, mode="lines+markers", name="Trend GDP", line=dict(color="#1D3557", width=2.5), hovertemplate="<b>Trend GDP</b>: %{y:.2%}<extra></extra>"))
    fig2.update_layout(
        barmode="relative",
        title=dict(text=f"{run_name} — Structural Decomposition ({ma_window}-Yr CAGR) ({r_start}-{r_end})", font=dict(size=16), x=0.5, xanchor="center"),
        yaxis=dict(title="Trend Contribution", tickformat=".1%", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", zeroline=True, zerolinecolor="#888888", zerolinewidth=1.5),
        xaxis=xaxis_cfg,
        hovermode="x unified",
        height=390,
        margin=dict(t=60, b=20, l=10, r=10),
    )
    st.plotly_chart(fig2, **PLOT_STRETCH)


def render_comparison_suite(d1, d2, n1, n2, short1, short2, start_yr, end_yr, ma_win, vol_win, section_prefix=""):
    """Shared visual & statistical comparison engine for Tabs 3 (Bilateral) and 4 (Cross-Campaign)."""
    c1_hex, c2_hex = "#2E86AB", "#E84855"

    st.subheader(f"{section_prefix}Head-to-Head Trajectories")

    def plot_pair(col, title, y_title, is_pct=False):
        fig = go.Figure()
        if is_pct:
            roll1 = rolling_cagr(d1[col], ma_win)
            roll2 = rolling_cagr(d2[col], ma_win)
            t_lbl = f"{ma_win}-Yr CAGR"
        else:
            roll1 = d1[col].rolling(window=ma_win, min_periods=1).mean()
            roll2 = d2[col].rolling(window=ma_win, min_periods=1).mean()
            t_lbl = f"{ma_win}-Yr Trend"
        val_fmt = ".2%" if is_pct else ".2f"

        fig.add_trace(go.Scatter(x=d1["year"], y=roll1, mode="lines", name=f"{n1} {t_lbl}", line=dict(color=c1_hex, width=1.5, dash="dash"), hovertemplate=f"<b>{n1} {t_lbl}</b>: %{{y:{val_fmt}}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=d2["year"], y=roll2, mode="lines", name=f"{n2} {t_lbl}", line=dict(color=c2_hex, width=1.5, dash="dash"), hovertemplate=f"<b>{n2} {t_lbl}</b>: %{{y:{val_fmt}}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=d1["year"], y=d1[col], mode="lines", name=f"{n1} Data", line=dict(color=c1_hex, width=2.5), hovertemplate=f"<b>{n1} Data</b>: %{{y:{val_fmt}}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=d2["year"], y=d2[col], mode="lines", name=f"{n2} Data", line=dict(color=c2_hex, width=2.5), hovertemplate=f"<b>{n2} Data</b>: %{{y:{val_fmt}}}<extra></extra>"))

        fig.update_layout(
            title=dict(text=f"{title} ({start_yr}-{end_yr})", font=dict(size=17), x=0.5, xanchor="center"),
            xaxis=dict(title="Year", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
            yaxis=dict(title=y_title, tickformat=".1%" if is_pct else None, showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", zeroline=True, zerolinecolor="#888888", zerolinewidth=1.5),
            hovermode="x unified",
            height=380,
            updatemenus=[] if is_pct else get_scale_menu(),
            margin=dict(t=85, b=30, l=10, r=10),
        )
        st.plotly_chart(fig, **PLOT_STRETCH)

    col_a, col_b = st.columns(2)
    with col_a:
        plot_pair("gdp", f"Total GDP: {n1} vs {n2}", "£ Millions")
        plot_pair("gdpc", f"GDP per Capita: {n1} vs {n2}", "£")
        plot_pair("population", f"Population: {n1} vs {n2}", "Millions")
        plot_pair("sol", f"Standard of Living: {n1} vs {n2}", "Index")
    with col_b:
        plot_pair("gdpGrowth", f"GDP Growth Rate: {n1} vs {n2}", "Rate (%)", True)
        plot_pair("gdpcGrowth", f"GDP/c Growth Rate: {n1} vs {n2}", "Rate (%)", True)
        plot_pair("popGrowth", f"Population Growth Rate: {n1} vs {n2}", "Rate (%)", True)
        plot_pair("solGrowth", f"SoL Growth Rate: {n1} vs {n2}", "Rate (%)", True)

    st.divider()
    st.subheader(f"{section_prefix}Growth Accounting Comparison")
    x_rng = [start_yr - 0.5, end_yr + 0.5]
    dc1, dc2 = st.columns(2)
    with dc1:
        plot_decomp_pair(d1, n1, ma_win, x_range=x_rng)
    with dc2:
        plot_decomp_pair(d2, n2, ma_win, x_range=x_rng)

    st.divider()
    st.subheader(f"{section_prefix}Productivity Convergence & Bayesian Outgrowth Edge")
    combined = pd.DataFrame({
        "target_gdpc": d1.set_index("year")["gdpc"],
        "bench_gdpc": d2.set_index("year")["gdpc"],
        "target_g": d1.set_index("year")["gdpcGrowth"],
        "bench_g": d2.set_index("year")["gdpcGrowth"],
    }).dropna()

    conv_summary = {}
    if not combined.empty:
        c_start, c_end = int(combined.index.min()), int(combined.index.max())
        combined["catchup_ratio"] = combined["target_gdpc"] / combined["bench_gdpc"]
        combined["velocity"] = combined["target_g"] - combined["bench_g"]
        years = combined.index

        wins = int((combined["velocity"] > 0).sum())
        n_obs = len(combined)
        post_beta = stats.beta(1 + wins, 1 + (n_obs - wins))
        p_outgrow = post_beta.mean()
        ci_lo, ci_hi = post_beta.ppf(0.025), post_beta.ppf(0.975)

        init_r, final_r = float(combined["catchup_ratio"].iloc[0]), float(combined["catchup_ratio"].iloc[-1])
        net_spread = (final_r - init_r) * 100.0
        g1_cagr = stats.gmean(1.0 + combined["target_g"]) - 1.0 if (1.0 + combined["target_g"] > 0).all() else np.nan
        g2_cagr = stats.gmean(1.0 + combined["bench_g"]) - 1.0 if (1.0 + combined["bench_g"] > 0).all() else np.nan

        if final_r >= 1.0:
            parity_str = "Parity Reached"
        elif pd.notnull(g1_cagr) and pd.notnull(g2_cagr) and g1_cagr > g2_cagr:
            yrs_to_par = np.log(1.0 / final_r) / np.log((1.0 + g1_cagr) / (1.0 + g2_cagr))
            parity_str = f"~{yrs_to_par:.1f} yrs (by {int(c_end + np.ceil(yrs_to_par))})"
        else:
            parity_str = "Diverging / Stalled"

        conv_summary = {
            "init_r": init_r, "final_r": final_r, "net_spread": net_spread,
            "g1_cagr": g1_cagr, "g2_cagr": g2_cagr,
            "p_outgrow": p_outgrow, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "wins": wins, "n_obs": n_obs, "parity_str": parity_str,
        }

        km1, km2, km3, km4 = st.columns(4)
        km1.metric("Final Productivity Ratio", f"{final_r:.1%}", f"{net_spread:+.1f} pp")
        km2.metric(f"Bayesian P({short1} Outgrows {short2})", f"{p_outgrow:.1%}", f"95% CI [{ci_lo:.0%}, {ci_hi:.0%}]")
        km3.metric("Outperformance Years", f"{wins} / {n_obs}", f"{wins/n_obs:.0%} raw")
        km4.metric("Implied Parity Horizon", parity_str)

        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(x=years, y=combined["catchup_ratio"], mode="lines+markers", name=f"{n1} / {n2} Ratio", line=dict(color=c1_hex, width=2.5), hovertemplate="<b>Productivity Ratio</b>: %{y:.2%}<extra></extra>"))
        fig_conv.add_hline(y=1.0, line_dash="dash", line_color="#D62828", annotation_text="Productivity Parity (100%)", annotation_position="top right")
        fig_conv.update_layout(
            title=dict(text=f"Productivity Convergence: {n1} relative to {n2} ({c_start}-{c_end})", font=dict(size=17), x=0.5, xanchor="center"),
            xaxis=dict(title="Year", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
            yaxis=dict(title="GDP/c Ratio", tickformat=".1%", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
            hovermode="x unified", height=400, margin=dict(t=60, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_conv, **PLOT_STRETCH)

        vel_colors = np.where(combined["velocity"] >= 0, "#2A9D8F", "#E84855")
        fig_vel = go.Figure(go.Bar(x=years, y=combined["velocity"], marker_color=vel_colors, name="Catch-up Spread", hovertemplate="<b>Growth Rate Spread</b>: %{y:+.2%}<extra></extra>"))
        fig_vel.update_layout(
            title=dict(text=f"Convergence Velocity ({short1} GDP/c Growth minus {short2} GDP/c Growth) ({c_start}-{c_end})", font=dict(size=17), x=0.5, xanchor="center"),
            xaxis=dict(title="Year", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
            yaxis=dict(title="Annual Spread (%)", tickformat="+.1%", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", zeroline=True, zerolinecolor="#888888", zerolinewidth=1.5),
            hovermode="x unified", height=360, margin=dict(t=60, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_vel, **PLOT_STRETCH)

    st.divider()
    st.subheader(f"{section_prefix}Comparative Macroeconomic Volatility")
    vc1, vc2 = st.columns(2)
    with vc1:
        for df_sub, lbl in [(d1, n1), (d2, n2)]:
            y0, y1 = (int(df_sub["year"].min()), int(df_sub["year"].max())) if not df_sub.empty else ("N/A", "N/A")
            bar_c = np.where(df_sub["gdpGrowth"] >= 0, "#2A9D8F", "#E76F51")
            fig_b = go.Figure(go.Bar(x=df_sub["year"], y=df_sub["gdpGrowth"], marker_color=bar_c, name=lbl, hovertemplate=f"<b>{lbl}</b>: %{{y:.2%}}<extra></extra>"))
            fig_b.update_layout(
                title=dict(text=f"{lbl} — Expansion vs Contraction ({y0}-{y1})", font=dict(size=15), x=0.5, xanchor="center"),
                xaxis=dict(title="Year", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", range=x_rng),
                yaxis=dict(tickformat=".1%", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", zeroline=True, zerolinecolor="#888888", zerolinewidth=1.5),
                hovermode="x unified", height=280, margin=dict(t=50, b=20, l=10, r=10),
            )
            st.plotly_chart(fig_b, **PLOT_STRETCH)
    with vc2:
        rv1 = d1["gdpGrowth"].rolling(window=vol_win, min_periods=2).std()
        rv2 = d2["gdpGrowth"].rolling(window=vol_win, min_periods=2).std()
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=d1["year"], y=rv1, mode="lines+markers", name=f"{n1} Volatility", line=dict(color=c1_hex, width=2.5), hovertemplate=f"<b>{n1} Volatility</b>: %{{y:.2%}}<extra></extra>"))
        fig_v.add_trace(go.Scatter(x=d2["year"], y=rv2, mode="lines+markers", name=f"{n2} Volatility", line=dict(color=c2_hex, width=2.5), hovertemplate=f"<b>{n2} Volatility</b>: %{{y:.2%}}<extra></extra>"))
        fig_v.update_layout(
            title=dict(text=f"Comparative Macroeconomic Volatility ({vol_win}-Yr) ({start_yr}-{end_yr})", font=dict(size=17), x=0.5, xanchor="center"),
            xaxis=dict(title="Year", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
            yaxis=dict(title="Standard Deviation", tickformat=".1%", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
            hovermode="x unified", height=580, margin=dict(t=60, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_v, **PLOT_STRETCH)

    st.divider()
    st.subheader(f"{section_prefix}Performance & Convergence Tables")
    stats_tbl = pd.DataFrame({
        "Metric": ["CAGR", "Median", "Max", "Min", "Std Dev"],
        f"{short1} GDP Gr": get_perf_stats(d1["gdpGrowth"]),
        f"{short2} GDP Gr": get_perf_stats(d2["gdpGrowth"]),
        f"{short1} GDP/c Gr": get_perf_stats(d1["gdpcGrowth"]),
        f"{short2} GDP/c Gr": get_perf_stats(d2["gdpcGrowth"]),
        f"{short1} SoL Gr": get_perf_stats(d1["solGrowth"]),
        f"{short2} SoL Gr": get_perf_stats(d2["solGrowth"]),
        f"{short1} Pop Gr": get_perf_stats(d1["popGrowth"]),
        f"{short2} Pop Gr": get_perf_stats(d2["popGrowth"]),
    })

    decomp_1, _, _ = get_growth_decomp_summary(d1)
    decomp_2, _, _ = get_growth_decomp_summary(d2)
    decomp_tbl = pd.DataFrame({
        "Component": ["Total GDP Span", "Intensive Share (Productivity)", "Extensive Share (Demographics)"],
        n1: decomp_1,
        n2: decomp_2,
    })

    tc1, tc2 = st.columns(2)
    with tc1:
        st.write(f"**Comparative Performance Statistics: {short1} vs {short2}**")
        st.dataframe(stats_tbl, hide_index=True, **STRETCH_KW)
        st.write("**Growth Accounting Decomposition Comparison**")
        st.dataframe(decomp_tbl, hide_index=True, **STRETCH_KW)
    with tc2:
        if conv_summary:
            conv_tbl = pd.DataFrame({
                "Convergence Metric": [
                    "Initial Productivity Ratio",
                    "Final Productivity Ratio",
                    "Net Convergence Spread",
                    f"{n1} GDP/c CAGR",
                    f"{n2} GDP/c CAGR",
                    f"Bayesian P({short1} Outgrows {short2}) [95% CI]",
                    "Implied Parity Horizon",
                ],
                "Result": [
                    f"{conv_summary['init_r']:.2%}",
                    f"{conv_summary['final_r']:.2%}",
                    f"{conv_summary['net_spread']:+.2f} percentage points",
                    f"{conv_summary['g1_cagr']:.2%}" if pd.notnull(conv_summary["g1_cagr"]) else "N/A",
                    f"{conv_summary['g2_cagr']:.2%}" if pd.notnull(conv_summary["g2_cagr"]) else "N/A",
                    f"{conv_summary['p_outgrow']:.1%} [{conv_summary['ci_lo']:.1%}, {conv_summary['ci_hi']:.1%}]",
                    conv_summary["parity_str"],
                ],
            })
            st.write(f"**Convergence Dynamics Summary: {short1} vs {short2}**")
            st.dataframe(conv_tbl, hide_index=True, **STRETCH_KW)

    return conv_summary


# ==============================================================================
# 5. SIDEBAR & FILE DISCOVERY
# ==============================================================================
st.title("🌍 Victoria 3 Macroeconomic & Bayesian Analytics Suite")

if st.session_state.get("bookmark_notice"):
    st.info(st.session_state["bookmark_notice"])
    st.session_state["bookmark_notice"] = ""

with st.sidebar:
    st.markdown("### 📂 Analyze Your Campaign")
    st.write("Upload `.csv` exports to analyze them instantly. Files stay in your browser session and are never stored on a server.")
    uploaded_files = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)

    st.divider()
    st.markdown("### ⚙️ Analytics Settings")
    default_crisis_thresh = st.slider(
        "Default Crisis Threshold (% GDP Drop):",
        min_value=0.0, max_value=15.0, value=2.0, step=0.5,
        help="Minimum peak-to-trough GDP drawdown required to count as a macroeconomic crisis.",
    ) / 100.0
    hp_lambda_global = st.number_input(
        "HP Filter Smoothing (λ):",
        min_value=1.0, max_value=1000.0, value=6.25, step=1.0,
        help="Ravn-Uhlig standard for annual macro data is 6.25 (classic textbook is 100).",
    )

    st.divider()
    st.markdown("### ☕ Support the Project")
    st.write("If this dashboard helped analyze your campaign, consider supporting its development!")
    st.markdown("[**☕ Buy me a Coffee / Ko-fi**](https://ko-fi.com/YOUR_USERNAME)")
    st.divider()
    st.write("**Chave Pix Aleatória (Brazil):**")
    st.code("1c6eff73-ffa1-4297-8eda-622ca1a57d9d", language="text")

local_csvs = sorted([f for f in os.listdir(".") if f.endswith(".csv")])
file_options = {f: f for f in local_csvs}
if uploaded_files:
    for uf in uploaded_files:
        file_options[f"☁️ {uf.name}"] = uf

if not file_options:
    st.error("No CSV files found. Please upload a campaign file in the sidebar.")
    st.stop()

csv_files = list(file_options.keys())


# ==============================================================================
# 6. MAIN TABS
# ==============================================================================
n_bm = len(st.session_state["bookmarks"])
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌍 1. Global Landscape",
    "🔍 2. Single Economy Dossier",
    "⚔️ 3. Bilateral & Convergence",
    "📊 4. Cross-Campaign Meta-Analysis",
    "🎲 5. Bayesian Macro & Risk Lab",
    f"🔖 6. Bookmarks ({n_bm})",
])


# ==============================================================================
# TAB 1: GLOBAL LANDSCAPE
# ==============================================================================
with tab1:
    st.header("The Great Power Race")
    ensure_select("t1_file", csv_files, csv_files[0])
    selected_file_t1 = st.selectbox("Select Campaign Data:", csv_files, key="t1_file")
    played_sigla_t1 = infer_played_tag(selected_file_t1)
    df_t1 = load_data(file_options[selected_file_t1])

    latest_rank = df_t1.dropna(subset=["gdp"]).groupby("country").last().reset_index().sort_values(by="gdp", ascending=False)
    all_tags_t1 = latest_rank["country"].tolist()
    default_tags = latest_rank["country"].head(8).tolist()
    if played_sigla_t1 in all_tags_t1 and played_sigla_t1 not in default_tags:
        default_tags.append(played_sigla_t1)

    if st.session_state.get("t1_last_file") != selected_file_t1:
        st.session_state["t1_tags"] = default_tags
        st.session_state["t1_last_file"] = selected_file_t1
    ensure_multiselect("t1_tags", all_tags_t1, default_tags)

    c_t1_sel, c_t1_bm_in, c_t1_bm_btn = st.columns([3.0, 1.2, 0.8])
    with c_t1_sel:
        selected_tags = st.multiselect("Select Nations to Compare:", options=all_tags_t1, format_func=tag_label, key="t1_tags")
    with c_t1_bm_in:
        t1_note = st.text_input("Bookmark Note:", placeholder="e.g. Top 8 GPs", key="t1_bm_note")
    with c_t1_bm_btn:
        st.write("")
        st.write("")
        if st.button("🔖 Save View", key="t1_save_bm", **STRETCH_KW):
            save_bookmark(
                "1. Global Landscape", 0, t1_note, selected_file_t1,
                f"Nations: {', '.join(selected_tags[:6])}{'...' if len(selected_tags) > 6 else ''}",
                {"t1_file": selected_file_t1, "t1_last_file": selected_file_t1, "t1_tags": list(selected_tags)},
            )

    if selected_tags:
        filtered_df = df_t1[df_t1["country"].isin(selected_tags)].copy()
        g_start, g_end = int(filtered_df["year"].min()), int(filtered_df["year"].max())

        def plot_multitrace(col, title, y_title, is_indexed=False):
            fig = go.Figure()
            for i, tag in enumerate(selected_tags):
                c_data = filtered_df[filtered_df["country"] == tag].sort_values("year")
                if c_data.empty:
                    continue
                c_name = country_pretty(tag)
                base_val = c_data[col].iloc[0]
                if is_indexed and (pd.isna(base_val) or base_val == 0):
                    continue
                y_vals = (c_data[col] / base_val * 100.0) if is_indexed else c_data[col]
                color = "#2E86AB" if tag == played_sigla_t1 else PALETTE[i % len(PALETTE)]
                width = 3.5 if tag == played_sigla_t1 else 2.0
                fmt = ".2f" if not is_indexed else ".1f"
                trace_name = f"{'⭐ ' if tag == played_sigla_t1 else ''}{c_name} ({tag})"

                fig.add_trace(go.Scatter(
                    x=c_data["year"], y=y_vals, mode="lines", name=trace_name,
                    line=dict(color=color, width=width),
                    hovertemplate=f"<b>{trace_name}</b>: %{{y:{fmt}}}<extra></extra>",
                ))

            fig.update_layout(
                title=dict(text=f"{title} ({g_start}-{g_end})", font=dict(size=17), x=0.5, xanchor="center"),
                xaxis=dict(title="Year", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
                yaxis=dict(title=y_title, showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
                height=440,
                hovermode="x unified",
                updatemenus=get_scale_menu(),
                margin=dict(t=85, b=30, l=10, r=10),
            )
            st.plotly_chart(fig, **PLOT_STRETCH)

        col1, col2 = st.columns(2)
        with col1:
            plot_multitrace("gdp", "Total GDP", "Gross Domestic Product (£ Millions)")
            plot_multitrace("gdpc", "GDP per Capita", "GDP per Capita (£)")
            plot_multitrace("population", "Total Population", "Population (Millions)")
        with col2:
            plot_multitrace("gdp", "GDP Indexed", "Indexed GDP (100 = Base Year)", is_indexed=True)
            plot_multitrace("sol", "Standard of Living", "Standard of Living Index")
            plot_multitrace("population", "Population Indexed", "Indexed Population (100 = Base Year)", is_indexed=True)

        st.divider()
        st.subheader("Global Hegemony & Market Concentration")
        world_gdp = df_t1.groupby("year")["gdp"].sum().rename("world_gdp")
        share_df = df_t1.merge(world_gdp, on="year")
        share_df["gdp_share"] = share_df["gdp"] / share_df["world_gdp"].replace(0, np.nan)
        hhi_series = share_df.groupby("year")["gdp_share"].apply(lambda s: float(np.sum((s.dropna() * 100.0) ** 2)))

        ch1, ch2 = st.columns(2)
        with ch1:
            fig_sh = go.Figure()
            for i, tag in enumerate(selected_tags):
                c_sh = share_df[share_df["country"] == tag].sort_values("year")
                if c_sh.empty:
                    continue
                color = "#2E86AB" if tag == played_sigla_t1 else PALETTE[i % len(PALETTE)]
                tname = f"{'⭐ ' if tag == played_sigla_t1 else ''}{country_pretty(tag)} ({tag})"
                fig_sh.add_trace(go.Scatter(
                    x=c_sh["year"], y=c_sh["gdp_share"], mode="lines", name=tname,
                    line=dict(color=color, width=3.0 if tag == played_sigla_t1 else 1.8),
                    hovertemplate=f"<b>{tname}</b>: %{{y:.2%}}<extra></extra>",
                ))
            fig_sh.update_layout(
                title=dict(text=f"Share of Global GDP ({g_start}-{g_end})", font=dict(size=17), x=0.5, xanchor="center"),
                xaxis=dict(title="Year", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                yaxis=dict(title="% of World GDP", tickformat=".1%", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                hovermode="x unified", height=390, margin=dict(t=60, b=20, l=10, r=10),
            )
            st.plotly_chart(fig_sh, **PLOT_STRETCH)
        with ch2:
            fig_hhi = go.Figure(go.Scatter(
                x=hhi_series.index, y=hhi_series.values, mode="lines+markers",
                name="Global GDP HHI", line=dict(color="#9B5DE5", width=2.5), marker=dict(size=4),
                hovertemplate="<b>Global HHI</b>: %{y:,.0f}<extra></extra>",
            ))
            fig_hhi.add_hline(y=1500, line_dash="dot", line_color="#F4A261", annotation_text="Moderate Concentration (1500)")
            fig_hhi.add_hline(y=2500, line_dash="dot", line_color="#D62828", annotation_text="Hegemonic Concentration (2500)")
            fig_hhi.update_layout(
                title=dict(text="Global Economic Concentration (HHI: 0–10,000)", font=dict(size=17), x=0.5, xanchor="center"),
                xaxis=dict(title="Year", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                yaxis=dict(title="HHI Score", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                hovermode="x unified", height=390, margin=dict(t=60, b=20, l=10, r=10),
            )
            st.plotly_chart(fig_hhi, **PLOT_STRETCH)

        st.divider()
        st.subheader("Global Dataset Master Summary (with Empirical Bayes Shrinkage)")
        shrunk_cagr_map, global_mu0, _ = empirical_bayes_cagr(df_t1, col="logGdpGrowth")
        st.caption(
            f"💡 **Bayes Shrunk CAGR** uses Normal–Normal Empirical Bayes shrinkage toward the global cross-sectional prior "
            f"(**{np.expm1(global_mu0)*100:.2f}%**). Long-lived empires stay nearly identical to raw CAGR, while short-lived lucky minors shrink toward the global mean."
        )

        master_records = []
        for tag in df_t1["country"].unique():
            c_df = df_t1[(df_t1["country"] == tag) & (df_t1["gdp"].notnull())].sort_values("year")
            if len(c_df) < 2:
                continue
            yrs = len(c_df)
            f_gdp, i_gdp = c_df["gdp"].iloc[-1], c_df["gdp"].iloc[0]
            f_gdpc, i_gdpc = c_df["gdpc"].iloc[-1], c_df["gdpc"].iloc[0]
            f_sol, i_sol = c_df["sol"].iloc[-1], c_df["sol"].iloc[0]
            f_pop, i_pop = c_df["population"].iloc[-1], c_df["population"].iloc[0]

            master_records.append({
                "Tag": tag,
                "Country": f"⭐ {country_pretty(tag)}" if tag == played_sigla_t1 else country_pretty(tag),
                "Span": f"{int(c_df['year'].min())}-{int(c_df['year'].max())}",
                "Yrs": yrs,
                "Final GDP": f_gdp,
                "GDP CAGR": safe_cagr(i_gdp, f_gdp, yrs) * 100.0,
                "Bayes Shrunk CAGR": shrunk_cagr_map.get(tag, np.nan),
                "Final GDP/c": f_gdpc,
                "GDP/c CAGR": safe_cagr(i_gdpc, f_gdpc, yrs) * 100.0,
                "Final SoL": f_sol,
                "SoL CAGR": safe_cagr(i_sol, f_sol, yrs) * 100.0,
                "Final Pop": f_pop,
                "Pop CAGR": safe_cagr(i_pop, f_pop, yrs) * 100.0,
            })

        if master_records:
            master_df = pd.DataFrame(master_records).sort_values("Final GDP", ascending=False)
            st.dataframe(
                master_df,
                hide_index=True,
                column_config={
                    "Final GDP": st.column_config.NumberColumn("Final GDP", format="£%.2fM"),
                    "GDP CAGR": st.column_config.NumberColumn("GDP CAGR", format="%.2f%%"),
                    "Bayes Shrunk CAGR": st.column_config.NumberColumn("Bayes Shrunk CAGR", format="%.2f%%"),
                    "Final GDP/c": st.column_config.NumberColumn("Final GDP/c", format="£%.2f"),
                    "GDP/c CAGR": st.column_config.NumberColumn("GDP/c CAGR", format="%.2f%%"),
                    "Final SoL": st.column_config.NumberColumn("Final SoL", format="%.2f"),
                    "SoL CAGR": st.column_config.NumberColumn("SoL CAGR", format="%.2f%%"),
                    "Final Pop": st.column_config.NumberColumn("Final Pop", format="%.2fM"),
                    "Pop CAGR": st.column_config.NumberColumn("Pop CAGR", format="%.2f%%"),
                },
                **STRETCH_KW,
            )


# ==============================================================================
# TAB 2: SINGLE ECONOMY DOSSIER
# ==============================================================================
with tab2:
    st.header("Single Economy Dossier")
    ensure_select("t2_file", csv_files, csv_files[0])
    selected_file_t2 = st.selectbox("Select Campaign Data:", csv_files, key="t2_file")
    played_sigla_t2 = infer_played_tag(selected_file_t2)
    df_t2 = load_data(file_options[selected_file_t2])

    available_tags_t2 = sorted(df_t2.dropna(subset=["gdp"])["country"].unique().tolist())
    if st.session_state.get("t2_last_file") != selected_file_t2:
        st.session_state["t2_target"] = played_sigla_t2 if played_sigla_t2 in available_tags_t2 else available_tags_t2[0]
        st.session_state.pop("t2_slider", None)
        st.session_state["t2_last_file"] = selected_file_t2

    ensure_select("t2_target", available_tags_t2, played_sigla_t2)

    col_t2_1, col_t2_2, col_t2_3, col_t2_4 = st.columns([1.2, 2.0, 0.9, 1.1])
    with col_t2_1:
        target_tag = st.selectbox("Target Economy:", options=available_tags_t2, format_func=tag_label, key="t2_target")

    country_df = df_t2[df_t2["country"] == target_tag].sort_values("year").reset_index(drop=True)
    min_yr, max_yr = int(country_df["year"].min()), int(country_df["year"].max())
    ensure_slider_range("t2_slider", min_yr, max_yr)

    with col_t2_2:
        if min_yr < max_yr:
            start_year, end_year = st.slider("Timeline Range:", min_value=min_yr, max_value=max_yr, key="t2_slider")
        else:
            start_year, end_year = min_yr, max_yr
            st.caption(f"Single year available: {min_yr}")
    with col_t2_3:
        ma_window = st.number_input("Trend Window:", min_value=1, max_value=50, value=10, key="t2_ma")
        vol_window = st.number_input("Volatility Window:", min_value=2, max_value=50, value=10, key="t2_vol")
    with col_t2_4:
        t2_note = st.text_input("Bookmark Note:", placeholder="e.g. Industrial Boom", key="t2_bm_note")
        if st.button("🔖 Save Dossier", key="t2_save_bm", **STRETCH_KW):
            save_bookmark(
                "2. Single Economy Dossier", 1, t2_note, selected_file_t2,
                f"{country_pretty(target_tag)} ({target_tag}) | {start_year}-{end_year}",
                {
                    "t2_file": selected_file_t2, "t2_last_file": selected_file_t2,
                    "t2_target": target_tag, "t2_slider": (start_year, end_year),
                    "t2_ma": ma_window, "t2_vol": vol_window,
                },
            )

    p_df = country_df[(country_df["year"] >= start_year) & (country_df["year"] <= end_year)].copy().reset_index(drop=True)
    country_name = country_pretty(target_tag)

    if p_df.empty:
        st.warning("No data available for the selected timeframe.")
    else:
        p_df_dd, crises_df = detect_crises(p_df, min_drawdown=default_crisis_thresh)
        _, s_int_kpi, _ = get_growth_decomp_summary(p_df)
        gdp_cagr_kpi = safe_cagr(p_df["gdp"].iloc[0], p_df["gdp"].iloc[-1], len(p_df))
        sol_delta = p_df["sol"].iloc[-1] - p_df["sol"].iloc[0]
        max_dd_val = float(p_df_dd["drawdown"].min()) if not p_df_dd.empty else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Final GDP", f"£{p_df['gdp'].iloc[-1]:,.2f}M", f"{gdp_cagr_kpi:.2%} CAGR" if pd.notnull(gdp_cagr_kpi) else None)
        k2.metric("Final GDP per Capita", f"£{p_df['gdpc'].iloc[-1]:,.2f}", f"{s_int_kpi:.0%} Intensive Growth")
        k3.metric("Final Standard of Living", f"{p_df['sol'].iloc[-1]:.2f}", f"{sol_delta:+.2f} since {start_year}")
        k4.metric(f"Crises (≥{default_crisis_thresh:.1%} drop)", f"{len(crises_df)}", f"Max DD: {max_dd_val:.1%}", delta_color="inverse")

        def plot_metric(y_col, title, y_title, color, is_pct=False):
            fig = go.Figure()
            if is_pct:
                roll = rolling_cagr(p_df[y_col], ma_window)
                t_lbl = f"{ma_window}-Yr CAGR"
            else:
                roll = p_df[y_col].rolling(window=ma_window, min_periods=1).mean()
                t_lbl = f"{ma_window}-Yr Trend"
            val_fmt = ".2%" if is_pct else ".2f"

            fig.add_trace(go.Scatter(x=p_df["year"], y=roll, mode="lines", name=t_lbl, line=dict(color="rgba(150, 150, 150, 0.7)", width=2, dash="dash"), hovertemplate=f"<b>{t_lbl}</b>: %{{y:{val_fmt}}}<extra></extra>"))
            fig.add_trace(go.Scatter(x=p_df["year"], y=p_df[y_col], mode="lines+markers", name=title, line=dict(color=color, width=2.5), marker=dict(size=4), hovertemplate=f"<b>{title}</b>: %{{y:{val_fmt}}}<extra></extra>"))

            fig.update_layout(
                title=dict(text=f"{country_name} — {title} ({start_year}-{end_year})", font=dict(size=17), x=0.5, xanchor="center"),
                xaxis=dict(title="Year", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
                yaxis=dict(title=y_title, tickformat=".1%" if is_pct else None, zeroline=True, zerolinecolor="#888888", zerolinewidth=1.5, showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
                hovermode="x unified", height=360,
                updatemenus=[] if is_pct else get_scale_menu(),
                margin=dict(t=85, b=30, l=10, r=10),
            )
            st.plotly_chart(fig, **PLOT_STRETCH)

        st.subheader("Macroeconomic Trajectories")
        c1, c2 = st.columns(2)
        with c1:
            plot_metric("gdp", "Total GDP", "£ Millions", "#2E86AB")
            plot_metric("gdpc", "GDP per Capita", "£", "#E84855")
            plot_metric("population", "Total Population", "Millions", "#2A9D8F")
            plot_metric("sol", "Standard of Living", "Index", "#9B5DE5")
        with c2:
            plot_metric("gdpGrowth", "Annual GDP Growth", "Rate (%)", "#F4A261", True)
            plot_metric("gdpcGrowth", "Annual GDP/c Growth", "Rate (%)", "#E76F51", True)
            plot_metric("popGrowth", "Annual Population Growth", "Rate (%)", "#264653", True)
            plot_metric("solGrowth", "Annual SoL Growth", "Rate (%)", "#8338EC", True)

        st.divider()
        st.subheader(f"Hodrick–Prescott Business Cycle Decomposition (λ = {hp_lambda_global:g})")
        valid_hp = p_df[p_df["gdp"] > 0].copy()
        if len(valid_hp) >= 6:
            log_gdp = np.log(valid_hp["gdp"].values)
            hp_trend_log, hp_cycle = hp_filter(log_gdp, lamb=hp_lambda_global)
            valid_hp["hp_trend_gdp"] = np.exp(hp_trend_log)
            valid_hp["output_gap"] = hp_cycle

            chp1, chp2 = st.columns(2)
            with chp1:
                fig_hpt = go.Figure()
                fig_hpt.add_trace(go.Scatter(x=valid_hp["year"], y=valid_hp["gdp"], mode="lines", name="Actual GDP", line=dict(color="#2E86AB", width=2.5), hovertemplate="<b>Actual</b>: £%{y:,.2f}M<extra></extra>"))
                fig_hpt.add_trace(go.Scatter(x=valid_hp["year"], y=valid_hp["hp_trend_gdp"], mode="lines", name="HP Potential Trend", line=dict(color="#E84855", width=2.2, dash="dash"), hovertemplate="<b>HP Trend</b>: £%{y:,.2f}M<extra></extra>"))
                fig_hpt.update_layout(
                    title=dict(text=f"{country_name} — Actual vs Potential GDP", font=dict(size=16), x=0.5, xanchor="center"),
                    xaxis=dict(title="Year", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                    yaxis=dict(title="GDP (£ Millions)", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                    hovermode="x unified", height=360, updatemenus=get_scale_menu(), margin=dict(t=85, b=20, l=10, r=10),
                )
                st.plotly_chart(fig_hpt, **PLOT_STRETCH)
            with chp2:
                gap_colors = np.where(valid_hp["output_gap"] >= 0, "#2A9D8F", "#E84855")
                fig_gap = go.Figure(go.Bar(x=valid_hp["year"], y=valid_hp["output_gap"], marker_color=gap_colors, name="Output Gap", hovertemplate="<b>Output Gap</b>: %{y:+.2%}<extra></extra>"))
                fig_gap.update_layout(
                    title=dict(text=f"{country_name} — Cyclical Output Gap (% Deviation from Trend)", font=dict(size=16), x=0.5, xanchor="center"),
                    xaxis=dict(title="Year", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                    yaxis=dict(title="Gap (%)", tickformat="+.1%", zeroline=True, zerolinecolor="#888", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                    hovermode="x unified", height=360, margin=dict(t=60, b=20, l=10, r=10),
                )
                st.plotly_chart(fig_gap, **PLOT_STRETCH)
        else:
            st.caption("Select at least 6 years of positive GDP data to compute the HP-filter business cycle.")

        st.divider()
        st.subheader("Structural Growth Accounting")
        c3, c4 = st.columns(2)
        with c3:
            fig_decomp = go.Figure()
            fig_decomp.add_trace(go.Bar(x=p_df["year"], y=p_df["gdpcGrowth"], name="Intensive (Productivity)", marker_color="#E76F51", hovertemplate="<b>Intensive</b>: %{y:.2%}<extra></extra>"))
            fig_decomp.add_trace(go.Bar(x=p_df["year"], y=p_df["popGrowth"], name="Extensive (Demographics)", marker_color="#2A9D8F", hovertemplate="<b>Extensive</b>: %{y:.2%}<extra></extra>"))
            fig_decomp.add_trace(go.Scatter(x=p_df["year"], y=p_df["gdpGrowth"], mode="lines+markers", name="Total GDP Growth", line=dict(color="#1D3557", width=2.5), hovertemplate="<b>Total Growth</b>: %{y:.2%}<extra></extra>"))
            fig_decomp.update_layout(
                barmode="relative",
                title=dict(text=f"{country_name} — Annual Growth Decomposition ({start_year}-{end_year})", font=dict(size=16), x=0.5, xanchor="center"),
                yaxis=dict(tickformat=".1%", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", zeroline=True, zerolinecolor="#888888", zerolinewidth=1.5),
                xaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
                hovermode="x unified", height=390, margin=dict(t=60, b=20, l=10, r=10),
            )
            st.plotly_chart(fig_decomp, **PLOT_STRETCH)
        with c4:
            roll_gdpc = rolling_cagr(p_df["gdpcGrowth"], ma_window)
            roll_pop = rolling_cagr(p_df["popGrowth"], ma_window)
            roll_gdp = rolling_cagr(p_df["gdpGrowth"], ma_window)
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Bar(x=p_df["year"], y=roll_gdpc, name="Trend Intensive", marker_color="#E76F51", hovertemplate="<b>Trend Intensive</b>: %{y:.2%}<extra></extra>"))
            fig_trend.add_trace(go.Bar(x=p_df["year"], y=roll_pop, name="Trend Extensive", marker_color="#2A9D8F", hovertemplate="<b>Trend Extensive</b>: %{y:.2%}<extra></extra>"))
            fig_trend.add_trace(go.Scatter(x=p_df["year"], y=roll_gdp, mode="lines+markers", name="Trend GDP", line=dict(color="#1D3557", width=2.5), hovertemplate="<b>Trend GDP</b>: %{y:.2%}<extra></extra>"))
            fig_trend.update_layout(
                barmode="relative",
                title=dict(text=f"{country_name} — Structural Decomposition ({ma_window}-Yr CAGR) ({start_year}-{end_year})", font=dict(size=16), x=0.5, xanchor="center"),
                yaxis=dict(tickformat=".1%", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", zeroline=True, zerolinecolor="#888888", zerolinewidth=1.5),
                xaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
                hovermode="x unified", height=390, margin=dict(t=60, b=20, l=10, r=10),
            )
            st.plotly_chart(fig_trend, **PLOT_STRETCH)

        st.divider()
        st.subheader("Volatility & Crises")
        c5, c6 = st.columns(2)
        with c5:
            colors = np.where(p_df["gdpGrowth"] >= 0, "#2A9D8F", "#E76F51")
            fig_boom = go.Figure(go.Bar(x=p_df["year"], y=p_df["gdpGrowth"], marker_color=colors, name="GDP Growth", hovertemplate="<b>Growth</b>: %{y:.2%}<extra></extra>"))
            fig_boom.update_layout(
                title=dict(text=f"{country_name} — Economic Expansion vs Contraction ({start_year}-{end_year})", font=dict(size=16), x=0.5, xanchor="center"),
                yaxis=dict(tickformat=".1%", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", zeroline=True, zerolinecolor="#888888", zerolinewidth=1.5),
                xaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
                hovermode="x unified", height=350, margin=dict(t=60, b=20, l=10, r=10),
            )
            st.plotly_chart(fig_boom, **PLOT_STRETCH)
        with c6:
            roll_vol = p_df["gdpGrowth"].rolling(window=vol_window, min_periods=2).std()
            fig_vol = go.Figure(go.Scatter(x=p_df["year"], y=roll_vol, mode="lines+markers", name="Volatility", line=dict(color="#9B5DE5", width=2.5), hovertemplate="<b>Volatility</b>: %{y:.2%}<extra></extra>"))
            fig_vol.update_layout(
                title=dict(text=f"{country_name} — Macroeconomic Volatility ({vol_window}-Yr) ({start_year}-{end_year})", font=dict(size=16), x=0.5, xanchor="center"),
                yaxis=dict(tickformat=".1%", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", zeroline=True, zerolinecolor="#888888", zerolinewidth=1.5),
                xaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
                hovermode="x unified", height=350, margin=dict(t=60, b=20, l=10, r=10),
            )
            st.plotly_chart(fig_vol, **PLOT_STRETCH)

        st.divider()
        st.subheader("Dossier Master Tables")
        stats_df = pd.DataFrame({
            "Metric": ["CAGR", "Median", "Max", "Min", "Std. Dev"],
            "GDP Growth": get_perf_stats(p_df["gdpGrowth"]),
            "GDPpC Growth": get_perf_stats(p_df["gdpcGrowth"]),
            "SoL Growth": get_perf_stats(p_df["solGrowth"]),
            "Pop Growth": get_perf_stats(p_df["popGrowth"]),
        })

        i_gdp, f_gdp = p_df["gdp"].iloc[0], p_df["gdp"].iloc[-1]
        i_gdpc, f_gdpc = p_df["gdpc"].iloc[0], p_df["gdpc"].iloc[-1]
        i_pop, f_pop = p_df["population"].iloc[0], p_df["population"].iloc[-1]
        _, s_int, s_ext = get_growth_decomp_summary(p_df)
        decomp_df = pd.DataFrame({
            "Component": ["Total Expansion", "Intensive (Productivity)", "Extensive (Demographics)"],
            "Initial": [f"£{i_gdp:,.2f}M", f"£{i_gdpc:,.2f}", f"{i_pop:,.2f}M"],
            "Final": [f"£{f_gdp:,.2f}M", f"£{f_gdpc:,.2f}", f"{f_pop:,.2f}M"],
            "Share": ["100.0%", f"{s_int:.1%}", f"{s_ext:.1%}"],
        })

        if not crises_df.empty:
            crisis_disp = crises_df.sort_values(by="Drawdown").head(6).copy()
            crisis_disp["Drawdown"] = crisis_disp["Drawdown"].apply(lambda x: f"{x:.2%}")
            crisis_disp["Duration"] = crisis_disp.apply(
                lambda r: f"{r['Duration (Yrs)']} yrs" if r["Recovered"] else f">{r['Duration (Yrs)']} yrs", axis=1
            )
            crisis_disp = crisis_disp[["Peak Year", "Trough Year", "Recovery Year", "Drawdown", "Duration"]]
        else:
            crisis_disp = pd.DataFrame([{"Message": f"No contractions ≥ {default_crisis_thresh:.1%} detected."}])

        c7, c8 = st.columns(2)
        with c7:
            st.write("**General Performance Stats**")
            st.dataframe(stats_df, hide_index=True, **STRETCH_KW)
            st.write("**Growth Accounting Breakdown**")
            st.dataframe(decomp_df, hide_index=True, **STRETCH_KW)
        with c8:
            st.write(f"**Top Economic Crises Leaderboard (Threshold ≥ {default_crisis_thresh:.1%})**")
            st.dataframe(crisis_disp, hide_index=True, **STRETCH_KW)


# ==============================================================================
# TAB 3: BILATERAL & CONVERGENCE
# ==============================================================================
with tab3:
    st.header("Bilateral & Convergence Analysis")
    ensure_select("t3_file", csv_files, csv_files[0])
    selected_file_t3 = st.selectbox("Select Campaign Data:", csv_files, key="t3_file")
    played_sigla_t3 = infer_played_tag(selected_file_t3)
    df_t3 = load_data(file_options[selected_file_t3])
    available_tags_t3 = sorted(df_t3.dropna(subset=["gdp"])["country"].unique().tolist())

    if st.session_state.get("t3_last_file") != selected_file_t3:
        st.session_state["t3_tag1"] = played_sigla_t3 if played_sigla_t3 in available_tags_t3 else available_tags_t3[0]
        st.session_state["t3_tag2"] = "GBR" if "GBR" in available_tags_t3 else (available_tags_t3[1] if len(available_tags_t3) > 1 else available_tags_t3[0])
        st.session_state.pop("t3_slider", None)
        st.session_state["t3_last_file"] = selected_file_t3

    ensure_select("t3_tag1", available_tags_t3, played_sigla_t3)
    ensure_select("t3_tag2", available_tags_t3, "GBR")

    col_t3_1, col_t3_2, col_t3_3, col_t3_4, col_t3_5 = st.columns([1.1, 1.1, 1.8, 0.85, 1.0])
    with col_t3_1:
        tag1 = st.selectbox("Target Nation:", options=available_tags_t3, format_func=tag_label, key="t3_tag1")
    with col_t3_2:
        tag2 = st.selectbox("Benchmark / Rival:", options=available_tags_t3, format_func=tag_label, key="t3_tag2")

    df_t = df_t3[df_t3["country"] == tag1].copy().reset_index(drop=True)
    df_b = df_t3[df_t3["country"] == tag2].copy().reset_index(drop=True)

    if df_t.empty or df_b.empty:
        st.warning("Insufficient data for the selected pair.")
    else:
        min_b_yr = min(int(df_t["year"].min()), int(df_b["year"].min()))
        max_b_yr = max(int(df_t["year"].max()), int(df_b["year"].max()))
        ensure_slider_range("t3_slider", min_b_yr, max_b_yr)

        with col_t3_3:
            if min_b_yr < max_b_yr:
                b_start, b_end = st.slider("Common Historical Range:", min_value=min_b_yr, max_value=max_b_yr, key="t3_slider")
            else:
                b_start, b_end = min_b_yr, max_b_yr
        with col_t3_4:
            b_ma = st.number_input("Trend Win:", min_value=1, max_value=50, value=10, key="t3_ma")
            b_vol = st.number_input("Vol Win:", min_value=2, max_value=50, value=10, key="t3_vol")
        with col_t3_5:
            t3_note = st.text_input("Bookmark Note:", placeholder="e.g. Catch-up Era", key="t3_bm_note")
            if st.button("🔖 Save Pair", key="t3_save_bm", **STRETCH_KW):
                save_bookmark(
                    "3. Bilateral & Convergence", 2, t3_note, selected_file_t3,
                    f"{tag1} vs {tag2} | {b_start}-{b_end}",
                    {
                        "t3_file": selected_file_t3, "t3_last_file": selected_file_t3,
                        "t3_tag1": tag1, "t3_tag2": tag2, "t3_slider": (b_start, b_end),
                        "t3_ma": b_ma, "t3_vol": b_vol,
                    },
                )

        d1 = df_t[(df_t["year"] >= b_start) & (df_t["year"] <= b_end)].sort_values("year").reset_index(drop=True)
        d2 = df_b[(df_b["year"] >= b_start) & (df_b["year"] <= b_end)].sort_values("year").reset_index(drop=True)
        n1, n2 = f"{country_pretty(tag1)} ({tag1})", f"{country_pretty(tag2)} ({tag2})"

        if not d1.empty and not d2.empty:
            render_comparison_suite(d1, d2, n1, n2, tag1, tag2, b_start, b_end, b_ma, b_vol, section_prefix="Bilateral ")
        else:
            st.warning("No overlapping rows in the chosen year range.")


# ==============================================================================
# TAB 4: CROSS-CAMPAIGN META-ANALYSIS
# ==============================================================================
with tab4:
    st.header("Cross-Campaign Meta-Analysis")
    if len(csv_files) < 2:
        st.info("Cross-campaign analysis requires at least 2 CSV files. Upload another campaign in the sidebar.")
    else:
        ensure_select("t4_f1", csv_files, csv_files[0])
        ensure_select("t4_f2", csv_files, csv_files[1])
        col_t4_f1, col_t4_f2 = st.columns(2)
        with col_t4_f1:
            file_meta_1 = st.selectbox("First Campaign File (Run 1):", csv_files, key="t4_f1")
            played_t4_1 = infer_played_tag(file_meta_1)
        with col_t4_f2:
            file_meta_2 = st.selectbox("Second Campaign File (Run 2):", csv_files, key="t4_f2")
            played_t4_2 = infer_played_tag(file_meta_2)

        df_meta_1 = load_data(file_options[file_meta_1])
        df_meta_2 = load_data(file_options[file_meta_2])
        tags_meta_1 = sorted(df_meta_1.dropna(subset=["gdp"])["country"].unique().tolist())
        tags_meta_2 = sorted(df_meta_2.dropna(subset=["gdp"])["country"].unique().tolist())

        if st.session_state.get("t4_last_files") != (file_meta_1, file_meta_2):
            st.session_state["t4_tag1"] = played_t4_1 if played_t4_1 in tags_meta_1 else tags_meta_1[0]
            st.session_state["t4_tag2"] = played_t4_2 if played_t4_2 in tags_meta_2 else tags_meta_2[0]
            st.session_state.pop("t4_slider", None)
            st.session_state["t4_last_files"] = (file_meta_1, file_meta_2)

        ensure_select("t4_tag1", tags_meta_1, played_t4_1)
        ensure_select("t4_tag2", tags_meta_2, played_t4_2)

        col_t4_1, col_t4_2, col_t4_3, col_t4_4, col_t4_5 = st.columns([1.1, 1.1, 1.8, 0.85, 1.0])
        with col_t4_1:
            meta_tag1 = st.selectbox("Nation in Run 1:", tags_meta_1, format_func=tag_label, key="t4_tag1")
        with col_t4_2:
            meta_tag2 = st.selectbox("Nation in Run 2:", tags_meta_2, format_func=tag_label, key="t4_tag2")

        d1_m_temp = df_meta_1[df_meta_1["country"] == meta_tag1].copy().reset_index(drop=True)
        d2_m_temp = df_meta_2[df_meta_2["country"] == meta_tag2].copy().reset_index(drop=True)

        if d1_m_temp.empty or d2_m_temp.empty:
            st.warning("Insufficient data for the selected countries.")
        else:
            min_m_yr = min(int(d1_m_temp["year"].min()), int(d2_m_temp["year"].min()))
            max_m_yr = max(int(d1_m_temp["year"].max()), int(d2_m_temp["year"].max()))
            ensure_slider_range("t4_slider", min_m_yr, max_m_yr)

            with col_t4_3:
                if min_m_yr < max_m_yr:
                    m_start, m_end = st.slider("Cross-Campaign Timeline Range:", min_value=min_m_yr, max_value=max_m_yr, key="t4_slider")
                else:
                    m_start, m_end = min_m_yr, max_m_yr
            with col_t4_4:
                m_ma = st.number_input("Trend Win:", min_value=1, max_value=50, value=10, key="t4_ma")
                m_vol = st.number_input("Vol Win:", min_value=2, max_value=50, value=10, key="t4_vol")
            with col_t4_5:
                t4_note = st.text_input("Bookmark Note:", placeholder="e.g. LF vs Interv.", key="t4_bm_note")
                if st.button("🔖 Save Meta", key="t4_save_bm", **STRETCH_KW):
                    save_bookmark(
                        "4. Cross-Campaign Meta-Analysis", 3, t4_note, f"{file_meta_1} vs {file_meta_2}",
                        f"R1:{meta_tag1} vs R2:{meta_tag2} | {m_start}-{m_end}",
                        {
                            "t4_f1": file_meta_1, "t4_f2": file_meta_2, "t4_last_files": (file_meta_1, file_meta_2),
                            "t4_tag1": meta_tag1, "t4_tag2": meta_tag2, "t4_slider": (m_start, m_end),
                            "t4_ma": m_ma, "t4_vol": m_vol,
                        },
                    )

            d1_m = d1_m_temp[(d1_m_temp["year"] >= m_start) & (d1_m_temp["year"] <= m_end)].sort_values("year").reset_index(drop=True)
            d2_m = d2_m_temp[(d2_m_temp["year"] >= m_start) & (d2_m_temp["year"] <= m_end)].sort_values("year").reset_index(drop=True)

            if not d1_m.empty and not d2_m.empty:
                g1_r = d1_m["gdpGrowth"].dropna()
                g2_r = d2_m["gdpGrowth"].dropna()
                if len(g1_r) >= 2 and len(g2_r) >= 2:
                    k1, n1_c = int((g1_r < 0).sum()), len(g1_r)
                    k2, n2_c = int((g2_r < 0).sum()), len(g2_r)
                    b_r1 = stats.beta(1 + k1, 1 + n1_c - k1)
                    b_seq = stats.beta(1 + k1 + k2, 1 + (n1_c - k1) + (n2_c - k2))

                    mu1, kp1, al1, bt1, _, _, _ = nig_posterior(d1_m["logGdpGrowth"].dropna(), 0.03, 0.5, 2.0, 0.002)
                    mu2, kp2, al2, bt2, _, _, _ = nig_posterior(d2_m["logGdpGrowth"].dropna(), 0.03, 0.5, 2.0, 0.002)
                    rng = np.random.default_rng(42)
                    s1 = stats.t.rvs(df=2 * al1, loc=mu1, scale=np.sqrt(bt1 / (al1 * kp1)), size=10000, random_state=rng)
                    s2 = stats.t.rvs(df=2 * al2, loc=mu2, scale=np.sqrt(bt2 / (al2 * kp2)), size=10000, random_state=rng)
                    p_r1_beats_r2 = float(np.mean(s1 > s2))

                    with st.container(border=True):
                        st.markdown("##### 🎲 Bayesian Cross-Campaign Evidence Summary")
                        sq1, sq2, sq3 = st.columns(3)
                        sq1.metric("P(Run 1 Structural Growth > Run 2)", f"{p_r1_beats_r2:.1%}", "NIG Posterior Comparison")
                        sq2.metric("Run 1 Recession Risk (Posterior)", f"{b_r1.mean():.1%}", f"{k1}/{n1_c} yrs < 0%")
                        sq3.metric("Pooled Sequential Risk (R1 → R2)", f"{b_seq.mean():.1%}", f"95% CI [{b_seq.ppf(0.025):.1%}, {b_seq.ppf(0.975):.1%}]")

                n1_m = f"Run 1: {country_pretty(meta_tag1)} ({meta_tag1})"
                n2_m = f"Run 2: {country_pretty(meta_tag2)} ({meta_tag2})"
                render_comparison_suite(
                    d1_m, d2_m, n1_m, n2_m, f"R1({meta_tag1})", f"R2({meta_tag2})",
                    m_start, m_end, m_ma, m_vol, section_prefix="Cross-Campaign ",
                )


# ==============================================================================
# TAB 5: BAYESIAN MACRO & RISK LAB (BAYESLAB × VICTORIA 3 FUSION)
# ==============================================================================
with tab5:
    st.header("🎲 Bayesian Macro & Risk Lab")
    st.caption(
        "Conjugate Bayesian inference wired directly to your Victoria 3 campaign data—turning raw growth, crisis, and rival series into posterior distributions, credible intervals, and predictive forecasts."
    )

    ensure_select("t5_file", csv_files, csv_files[0])
    c5_f, c5_tag, c5_sl, c5_bm = st.columns([1.3, 1.2, 1.8, 1.0])
    with c5_f:
        file_t5 = st.selectbox("Campaign File:", csv_files, key="t5_file")
    played_t5 = infer_played_tag(file_t5)
    df_t5 = load_data(file_options[file_t5])
    tags_t5 = sorted(df_t5.dropna(subset=["gdp"])["country"].unique().tolist())

    if st.session_state.get("t5_last_file") != file_t5:
        st.session_state["t5_target"] = played_t5 if played_t5 in tags_t5 else tags_t5[0]
        st.session_state.pop("t5_slider", None)
        st.session_state["t5_last_file"] = file_t5

    ensure_select("t5_target", tags_t5, played_t5)
    with c5_tag:
        tag_t5 = st.selectbox("Target Nation:", tags_t5, format_func=tag_label, key="t5_target")

    c_df_t5 = df_t5[df_t5["country"] == tag_t5].sort_values("year").reset_index(drop=True)
    min_y5, max_y5 = int(c_df_t5["year"].min()), int(c_df_t5["year"].max())
    ensure_slider_range("t5_slider", min_y5, max_y5)

    with c5_sl:
        if min_y5 < max_y5:
            y5_start, y5_end = st.slider("Analysis Era:", min_value=min_y5, max_value=max_y5, key="t5_slider")
        else:
            y5_start, y5_end = min_y5, max_y5

    df_b5 = c_df_t5[(c_df_t5["year"] >= y5_start) & (c_df_t5["year"] <= y5_end)].copy().reset_index(drop=True)
    name_t5 = f"{country_pretty(tag_t5)} ({tag_t5})"

    with c5_bm:
        t5_note = st.text_input("Bookmark Note:", placeholder="e.g. Recession Risk", key="t5_bm_note")
        if st.button("🔖 Save Lab View", key="t5_save_bm", **STRETCH_KW):
            save_bookmark(
                "5. Bayesian Macro & Risk Lab", 4, t5_note, file_t5,
                f"{name_t5} | {y5_start}-{y5_end}",
                {
                    "t5_file": file_t5, "t5_last_file": file_t5,
                    "t5_target": tag_t5, "t5_slider": (y5_start, y5_end),
                },
            )

    bayes_sub = st.tabs([
        "📉 Beta–Binomial (Risk & Edge)",
        "⚡ Gamma–Poisson (Crisis Rate)",
        "⏱️ Gamma–Exponential (Recovery)",
        "📈 NIG Growth & Fan Forecast",
        "🌳 Discrete Bayes Diagnosis",
    ])

    # 5A: BETA-BINOMIAL
    with bayes_sub[0]:
        st.subheader(f"Beta–Binomial Posterior — {name_t5} ({y5_start}-{y5_end})")
        bb1, bb2, bb3, bb4 = st.columns([1.4, 1.3, 1.1, 0.9])
        bb_mode = bb1.selectbox(
            "Binary Event to Model:",
            ["Annual GDP Contraction (gdpGrowth < Threshold)", "Outgrowing a Rival (GDP/c Growth > Rival)"],
            key="t5_bb_mode",
        )
        prior_src = bb2.selectbox(
            "Prior Source:",
            ["Empirical Global Prior (This Save)", "Custom Beta(a, b)", "Sequential Prior (Another Campaign)"],
            key="t5_bb_prior_src",
        )
        ci_lvl_bb = bb3.selectbox("Credible Interval:", [0.90, 0.95, 0.99], index=1, format_func=lambda x: f"{int(x*100)}%", key="t5_bb_ci")

        if bb_mode.startswith("Annual GDP"):
            bb_thresh = bb4.number_input("Contraction < (%):", value=0.0, step=0.5, key="t5_bb_thresh") / 100.0
            valid_g = df_b5["gdpGrowth"].dropna()
            n_trials = len(valid_g)
            k_events = int((valid_g < bb_thresh).sum())
            event_lbl = f"Annual GDP Growth < {bb_thresh:.1%}"
        else:
            rival_opts = [t for t in tags_t5 if t != tag_t5] or tags_t5
            ensure_select("t5_bb_rival", rival_opts, "GBR" if "GBR" in rival_opts else rival_opts[0])
            rival_tag = bb4.selectbox("Rival:", rival_opts, format_func=tag_label, key="t5_bb_rival")
            df_riv = df_t5[df_t5["country"] == rival_tag].set_index("year")["gdpcGrowth"]
            paired = pd.DataFrame({"t": df_b5.set_index("year")["gdpcGrowth"], "r": df_riv}).dropna()
            n_trials = len(paired)
            k_events = int((paired["t"] > paired["r"]).sum())
            event_lbl = f"{tag_t5} Outgrows {rival_tag} in GDP/c"

        if prior_src == "Empirical Global Prior (This Save)" and bb_mode.startswith("Annual GDP"):
            a0, b0 = fit_empirical_beta_prior(df_t5, exclude_tag=tag_t5, threshold=bb_thresh)
            prior_desc = f"Empirical Global Beta({a0:g}, {b0:g})"
        elif prior_src == "Sequential Prior (Another Campaign)" and len(csv_files) >= 2:
            sc1, sc2 = st.columns(2)
            other_files = [f for f in csv_files if f != file_t5] or csv_files
            ensure_select("t5_bb_prior_file", other_files, other_files[0])
            pf = sc1.selectbox("Prior Campaign (Run A):", other_files, key="t5_bb_prior_file")
            df_pf = load_data(file_options[pf])
            pf_tags = sorted(df_pf.dropna(subset=["gdp"])["country"].unique().tolist())
            ensure_select("t5_bb_prior_tag", pf_tags, tag_t5 if tag_t5 in pf_tags else pf_tags[0])
            pt = sc2.selectbox("Nation in Prior Campaign:", pf_tags, format_func=tag_label, key="t5_bb_prior_tag")
            g_pf = df_pf[df_pf["country"] == pt]["gdpGrowth"].dropna()
            if bb_mode.startswith("Annual GDP"):
                k_pf = int((g_pf < bb_thresh).sum())
                n_pf = len(g_pf)
            else:
                df_riv_pf = df_pf[df_pf["country"] == rival_tag].set_index("year")["gdpcGrowth"]
                paired_pf = pd.DataFrame({"t": df_pf[df_pf["country"] == pt].set_index("year")["gdpcGrowth"], "r": df_riv_pf}).dropna()
                k_pf, n_pf = int((paired_pf["t"] > paired_pf["r"]).sum()), len(paired_pf)
            a0, b0 = 1.0 + k_pf, 1.0 + max(0, n_pf - k_pf)
            prior_desc = f"Run A ({pt}: {k_pf}/{n_pf}) → Beta({a0:g}, {b0:g})"
        else:
            if prior_src == "Sequential Prior (Another Campaign)" and len(csv_files) < 2:
                st.caption("ℹ️ Upload a 2nd CSV to use a Sequential Prior — using Custom Beta(a, b) below.")
            elif prior_src == "Empirical Global Prior (This Save)" and not bb_mode.startswith("Annual GDP"):
                st.caption("ℹ️ Empirical Global Prior applies to contraction rates — using Custom Beta(a, b) for rival outgrowth.")
            cp1, cp2 = st.columns(2)
            a0 = cp1.number_input("Prior α (a):", min_value=0.1, value=1.5, step=0.5, key="t5_bb_a")
            b0 = cp2.number_input("Prior β (b):", min_value=0.1, value=4.5, step=0.5, key="t5_bb_b")
            prior_desc = f"Custom Beta({a0:g}, {b0:g})"

        if n_trials == 0:
            st.warning("Not enough annual observations in this era to update Beta–Binomial.")
        else:
            a_post = a0 + k_events
            b_post = b0 + (n_trials - k_events)
            d_pri = stats.beta(a0, b0)
            d_pos = stats.beta(a_post, b_post)
            alpha_tail = (1.0 - ci_lvl_bb) / 2.0
            ci_lo, ci_hi = float(d_pos.ppf(alpha_tail)), float(d_pos.ppf(1.0 - alpha_tail))
            pred_next = a_post / (a_post + b_post)
            map_pos = analytical_mode("beta", a_post, b_post)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Observed Data", f"{k_events} / {n_trials} yrs", f"{k_events/n_trials:.1%} raw rate")
            m2.metric("Posterior Mean E[θ]", f"{d_pos.mean():.2%}", f"Prior: {d_pri.mean():.2%}")
            m3.metric(f"{int(ci_lvl_bb*100)}% Credible Interval", f"[{ci_lo:.1%}, {ci_hi:.1%}]", f"MAP: {map_pos:.1%}" if pd.notnull(map_pos) else None)
            m4.metric("Predictive P(Next Year)", f"{pred_next:.2%}", "Posterior Predictive")

            col_tbl, col_plt = st.columns([1.0, 1.45])
            with col_tbl:
                bb_tbl = pd.DataFrame({
                    "Statistic": ["Distribution", "Mean E[θ]", "Median", "Mode (MAP)", "Std. Dev", "90% CI", "95% CI", "99% CI"],
                    "Prior": [
                        f"Beta({a0:g}, {b0:g})", f"{d_pri.mean():.2%}", f"{d_pri.ppf(0.5):.2%}",
                        f"{analytical_mode('beta', a0, b0):.2%}" if pd.notnull(analytical_mode("beta", a0, b0)) else "Boundary",
                        f"{d_pri.std():.2%}",
                        f"[{d_pri.ppf(0.05):.1%}, {d_pri.ppf(0.95):.1%}]",
                        f"[{d_pri.ppf(0.025):.1%}, {d_pri.ppf(0.975):.1%}]",
                        f"[{d_pri.ppf(0.005):.1%}, {d_pri.ppf(0.995):.1%}]",
                    ],
                    "Posterior": [
                        f"Beta({a_post:g}, {b_post:g})", f"{d_pos.mean():.2%}", f"{d_pos.ppf(0.5):.2%}",
                        f"{map_pos:.2%}" if pd.notnull(map_pos) else "Boundary",
                        f"{d_pos.std():.2%}",
                        f"[{d_pos.ppf(0.05):.1%}, {d_pos.ppf(0.95):.1%}]",
                        f"[{d_pos.ppf(0.025):.1%}, {d_pos.ppf(0.975):.1%}]",
                        f"[{d_pos.ppf(0.005):.1%}, {d_pos.ppf(0.995):.1%}]",
                    ],
                })
                st.dataframe(bb_tbl, hide_index=True, **STRETCH_KW)
                st.caption(f"**Event:** {event_lbl} | **Prior:** {prior_desc}")
            with col_plt:
                xg = np.linspace(0.001, 0.999, 500)
                plot_bayesian_density_plotly(
                    xg, d_pri.pdf(xg), d_pos.pdf(xg), ci_lo, ci_hi, ci_lvl_bb,
                    f"Beta–Binomial Update: {event_lbl}", "Latent Annual Probability θ",
                    f"Beta({a0:g}, {b0:g})", f"Beta({a_post:g}, {b_post:g})",
                    is_pct_axis=True, post_mean=float(d_pos.mean()),
                )

    # 5B: GAMMA-POISSON
    with bayes_sub[1]:
        st.subheader(f"Gamma–Poisson Crisis Frequency — {name_t5} ({y5_start}-{y5_end})")
        gp1, gp2, gp3, gp4, gp5 = st.columns(5)
        gp_thresh = gp1.number_input("Min Drawdown (%):", min_value=0.5, max_value=30.0, value=float(default_crisis_thresh * 100.0), step=0.5, key="t5_gp_thresh") / 100.0
        gp_a0 = gp2.number_input("Prior α (shape):", min_value=0.1, value=2.0, step=0.5, key="t5_gp_a")
        gp_b0 = gp3.number_input("Prior β (rate/decades):", min_value=0.1, value=1.5, step=0.5, key="t5_gp_b")
        gp_fut = gp4.number_input("Forecast Window (Decades):", min_value=0.5, max_value=5.0, value=1.0, step=0.5, key="t5_gp_fut")
        ci_lvl_gp = gp5.selectbox("CI Level:", [0.90, 0.95, 0.99], index=1, format_func=lambda x: f"{int(x*100)}%", key="t5_gp_ci")

        _, crises_gp = detect_crises(df_b5, min_drawdown=gp_thresh)
        decades_obs = max(0.1, (y5_end - y5_start) / 10.0)
        n_crises = len(crises_gp)

        a_gp_post = gp_a0 + n_crises
        b_gp_post = gp_b0 + decades_obs
        d_gp_pri = stats.gamma(a=gp_a0, scale=1.0 / gp_b0)
        d_gp_pos = stats.gamma(a=a_gp_post, scale=1.0 / b_gp_post)

        tail_gp = (1.0 - ci_lvl_gp) / 2.0
        gp_lo, gp_hi = float(d_gp_pos.ppf(tail_gp)), float(d_gp_pos.ppf(1.0 - tail_gp))

        k_grid = np.arange(0, 8)
        pred_pmf = gamma_poisson_predictive_pmf(k_grid, a_gp_post, b_gp_post, exposure=gp_fut)
        p_zero = float(pred_pmf[0])
        p_two_plus = float(max(0.0, 1.0 - pred_pmf[0] - pred_pmf[1]))

        gm1, gm2, gm3, gm4 = st.columns(4)
        gm1.metric("Observed Crises", f"{n_crises} in {decades_obs:.1f} dec.", f"{n_crises/decades_obs:.2f} / decade raw")
        gm2.metric("Posterior Rate E[λ]", f"{d_gp_pos.mean():.2f} / dec.", f"Prior: {d_gp_pri.mean():.2f}")
        gm3.metric(f"P(0 Crises in Next {gp_fut:g} Dec.)", f"{p_zero:.1%}", "Exact NegBin Predictive")
        gm4.metric(f"P(≥2 Crises in Next {gp_fut:g} Dec.)", f"{p_two_plus:.1%}", f"{int(ci_lvl_gp*100)}% CI [{gp_lo:.2f}, {gp_hi:.2f}]")

        cgp_l, cgp_r = st.columns(2)
        with cgp_l:
            x_max_gp = max(float(d_gp_pri.ppf(0.995)), float(d_gp_pos.ppf(0.995)), 1.0)
            xg_gp = np.linspace(0.001, x_max_gp, 500)
            plot_bayesian_density_plotly(
                xg_gp, d_gp_pri.pdf(xg_gp), d_gp_pos.pdf(xg_gp), gp_lo, gp_hi, ci_lvl_gp,
                f"Posterior Crisis Rate λ (Drawdown ≥ {gp_thresh:.1%})", "Crises per Decade (λ)",
                f"Gamma({gp_a0:g}, rate={gp_b0:g})", f"Gamma({a_gp_post:g}, rate={b_gp_post:.2f})",
                is_pct_axis=False, post_mean=float(d_gp_pos.mean()),
            )
        with cgp_r:
            fig_nb = go.Figure(go.Bar(
                x=[str(k) for k in k_grid], y=pred_pmf,
                marker_color="#2E86AB",
                text=[f"{p:.1%}" for p in pred_pmf], textposition="outside",
                hovertemplate="<b>%{x} crises</b>: %{y:.2%}<extra></extra>",
            ))
            fig_nb.update_layout(
                title=dict(text=f"Posterior Predictive Crises in Next {gp_fut:g} Decade(s)", font=dict(size=17), x=0.5, xanchor="center"),
                xaxis=dict(title="Predicted Number of Crises"),
                yaxis=dict(title="Probability", tickformat=".0%", range=[0, min(1.05, float(pred_pmf.max()) * 1.25)], showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                height=430, margin=dict(t=85, b=35, l=15, r=15),
            )
            st.plotly_chart(fig_nb, **PLOT_STRETCH)

    # 5C: GAMMA-EXPONENTIAL
    with bayes_sub[2]:
        st.subheader(f"Gamma–Exponential Resilience & Recovery — {name_t5} ({y5_start}-{y5_end})")
        ge1, ge2, ge3, ge4, ge5 = st.columns([1.3, 0.9, 0.9, 0.9, 1.0])
        ge_metric = ge1.selectbox("Duration Variable:", ["Crisis Recovery Duration (Yrs)", "Inter-Crisis Calm Spells (Yrs)"], key="t5_ge_metric")
        ge_thresh = ge2.number_input("Min Drawdown (%):", min_value=0.5, max_value=30.0, value=float(default_crisis_thresh * 100.0), step=0.5, key="t5_ge_thresh") / 100.0
        ge_a0 = ge3.number_input("Prior α:", min_value=0.1, value=2.0, step=0.5, key="t5_ge_a")
        ge_b0 = ge4.number_input("Prior β (rate/yrs):", min_value=0.1, value=6.0, step=0.5, key="t5_ge_b")
        ge_horizon = ge5.number_input("Target Window T (Yrs):", min_value=1, max_value=30, value=4, step=1, key="t5_ge_horizon")

        _, crises_ge = detect_crises(df_b5, min_drawdown=ge_thresh)
        if ge_metric.startswith("Crisis Recovery"):
            rec_only = crises_ge[crises_ge["Recovered"]] if not crises_ge.empty else pd.DataFrame()
            durations = rec_only["Duration (Yrs)"].astype(float).tolist() if not rec_only.empty else []
            dur_lbl = "Recovery Duration"
        else:
            if not crises_ge.empty and len(crises_ge) >= 2:
                peaks = sorted(crises_ge["Peak Year"].astype(int).tolist())
                durations = [float(max(1, peaks[i + 1] - peaks[i])) for i in range(len(peaks) - 1)]
            else:
                durations = []
            dur_lbl = "Inter-Crisis Wait Time"

        if not durations:
            st.info(f"No completed {dur_lbl.lower()} episodes found at the ≥{ge_thresh:.1%} threshold. Lower the drawdown threshold or widen the era.")
        else:
            n_dur = len(durations)
            sum_dur = float(sum(durations))
            a_ge_post = ge_a0 + n_dur
            b_ge_post = ge_b0 + sum_dur
            d_ge_pri = stats.gamma(a=ge_a0, scale=1.0 / ge_b0)
            d_ge_pos = stats.gamma(a=a_ge_post, scale=1.0 / b_ge_post)

            ge_lo, ge_hi = float(d_ge_pos.ppf(0.025)), float(d_ge_pos.ppf(0.975))
            p_within_T = 1.0 - (b_ge_post / (b_ge_post + ge_horizon)) ** a_ge_post
            exp_dur_pred = (b_ge_post / (a_ge_post - 1.0)) if a_ge_post > 1.0 else np.nan

            em1, em2, em3, em4 = st.columns(4)
            em1.metric("Observed Episodes", f"{n_dur}", f"Sample Mean: {np.mean(durations):.2f} yrs")
            em2.metric("Posterior Hazard Rate E[λ]", f"{d_ge_pos.mean():.3f} / yr", f"95% CI [{ge_lo:.3f}, {ge_hi:.3f}]")
            em3.metric("Predictive Expected Duration", f"{exp_dur_pred:.2f} yrs" if pd.notnull(exp_dur_pred) else "∞", "Lomax Predictive Mean")
            em4.metric(f"P(Duration ≤ {ge_horizon} yrs)", f"{p_within_T:.1%}", "Exact Posterior Predictive")

            cge1, cge2 = st.columns(2)
            with cge1:
                x_max_ge = max(float(d_ge_pri.ppf(0.995)), float(d_ge_pos.ppf(0.995)), 0.5)
                xg_ge = np.linspace(0.001, x_max_ge, 500)
                plot_bayesian_density_plotly(
                    xg_ge, d_ge_pri.pdf(xg_ge), d_ge_pos.pdf(xg_ge), ge_lo, ge_hi, 0.95,
                    f"Posterior Hazard Rate λ ({dur_lbl})", "Annual Hazard Rate λ",
                    f"Gamma({ge_a0:g}, rate={ge_b0:g})", f"Gamma({a_ge_post:g}, rate={b_ge_post:.1f})",
                    is_pct_axis=False, post_mean=float(d_ge_pos.mean()),
                )
            with cge2:
                t_grid = np.linspace(0, max(15, int(ge_horizon * 2.5)), 200)
                cdf_pred = 1.0 - (b_ge_post / (b_ge_post + t_grid)) ** a_ge_post
                fig_cdf = go.Figure(go.Scatter(
                    x=t_grid, y=cdf_pred, mode="lines", name="Predictive CDF",
                    line=dict(color="#2A9D8F", width=2.8), fill="tozeroy", fillcolor="rgba(42,157,143,0.18)",
                    hovertemplate="<b>≤ %{x:.1f} yrs</b>: %{y:.1%}<extra></extra>",
                ))
                fig_cdf.add_vline(x=ge_horizon, line_dash="dash", line_color="#E84855", annotation_text=f"T={ge_horizon} yrs ({p_within_T:.1%})")
                fig_cdf.update_layout(
                    title=dict(text=f"Posterior Predictive Cumulative Probability ({dur_lbl})", font=dict(size=16), x=0.5, xanchor="center"),
                    xaxis=dict(title="Years (T)", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                    yaxis=dict(title="P(Duration ≤ T)", tickformat=".0%", range=[0, 1.02], showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                    height=430, margin=dict(t=85, b=35, l=15, r=15),
                )
                st.plotly_chart(fig_cdf, **PLOT_STRETCH)

    # 5D: NORMAL-INVERSE-GAMMA & MONTE CARLO FAN CHART
    with bayes_sub[3]:
        st.subheader(f"Normal–Inverse-Gamma Growth Inference & Forecast Fan — {name_t5}")
        ng1, ng2, ng3, ng4 = st.columns([1.2, 1.4, 0.9, 0.9])
        nig_series_opt = ng1.selectbox("Growth Series:", ["Total GDP (log growth)", "GDP per Capita (log growth)"], key="t5_nig_series")
        nig_prior_src = ng2.selectbox("Prior Calibration:", ["Global Cross-Sectional Prior", "Weakly Informative Custom"], key="t5_nig_prior_src")
        nig_horizon = ng3.number_input("Forecast Horizon (Yrs):", min_value=5, max_value=40, value=15, step=5, key="t5_nig_horizon")
        nig_ci = ng4.selectbox("CI Level:", [0.90, 0.95], index=0, format_func=lambda x: f"{int(x*100)}%", key="t5_nig_ci")

        is_gdpc = nig_series_opt.startswith("GDP per Capita")
        col_log = "logGdpcGrowth" if is_gdpc else "logGdpGrowth"
        col_lvl = "gdpc" if is_gdpc else "gdp"
        unit_lbl = "£" if is_gdpc else "£ Millions"

        if nig_prior_src == "Global Cross-Sectional Prior":
            _, g_mu0, g_tau2 = empirical_bayes_cagr(df_t5, col=col_log)
            mu0, kappa0, alpha0, beta0 = g_mu0, 2.0, 3.0, max(g_tau2 * 3.0, 0.0025)
        else:
            cn1, cn2, cn3, cn4 = st.columns(4)
            mu0 = cn1.number_input("Prior Mean μ₀ (%):", value=2.5, step=0.5, key="t5_nig_mu0") / 100.0
            kappa0 = cn2.number_input("Prior Strength κ₀ (pseudo-yrs):", min_value=0.1, value=1.0, step=0.5, key="t5_nig_kappa0")
            alpha0 = cn3.number_input("Prior α₀ (shape):", min_value=1.5, value=2.5, step=0.5, key="t5_nig_alpha0")
            beta0 = cn4.number_input("Prior β₀ (scale):", min_value=0.0001, value=0.0025, step=0.0005, format="%.4f", key="t5_nig_beta0")

        x_obs = df_b5[col_log].dropna().values
        if len(x_obs) < 3:
            st.warning("Need at least 3 annual growth observations for Normal–Inverse-Gamma inference.")
        else:
            mu_n, kp_n, al_n, bt_n, n_g, x_bar, s_sample = nig_posterior(x_obs, mu0, kappa0, alpha0, beta0)
            df_t_post = 2.0 * al_n
            scale_mu_post = np.sqrt(bt_n / (al_n * kp_n))
            scale_mu_pri = np.sqrt(beta0 / (alpha0 * kappa0))

            d_mu_pri = stats.t(df=2.0 * alpha0, loc=mu0, scale=scale_mu_pri)
            d_mu_pos = stats.t(df=df_t_post, loc=mu_n, scale=scale_mu_post)
            tail_n = (1.0 - nig_ci) / 2.0
            mu_lo, mu_hi = float(d_mu_pos.ppf(tail_n)), float(d_mu_pos.ppf(1.0 - tail_n))
            p_pos_growth = float(d_mu_pos.sf(0.0))

            nm1, nm2, nm3, nm4 = st.columns(4)
            nm1.metric("Raw Sample Mean", f"{np.expm1(x_bar):.2%}", f"n = {n_g} yrs")
            nm2.metric("Bayesian Posterior Mean", f"{np.expm1(mu_n):.2%}", f"Prior: {np.expm1(mu0):.2%}")
            nm3.metric(f"{int(nig_ci*100)}% CI for True Growth", f"[{np.expm1(mu_lo):.2%}, {np.expm1(mu_hi):.2%}]")
            nm4.metric("P(Structural Growth > 0%)", f"{p_pos_growth:.1%}", f"Post. Vol σ ≈ {np.sqrt(bt_n / max(al_n - 1, 0.5)):.2%}")

            rng = np.random.default_rng(1836)
            n_sims = 2000
            sig2_draws = stats.invgamma.rvs(a=al_n, scale=bt_n, size=n_sims, random_state=rng)
            sig_draws = np.sqrt(sig2_draws)
            mu_draws = rng.normal(loc=mu_n, scale=sig_draws / np.sqrt(kp_n))
            shocks = rng.normal(loc=mu_draws[:, None], scale=sig_draws[:, None], size=(n_sims, int(nig_horizon)))
            shocks = np.clip(shocks, -0.5, 0.5)

            last_yr = int(df_b5["year"].iloc[-1])
            last_val = float(df_b5[col_lvl].iloc[-1])
            fut_years = np.arange(last_yr, last_yr + int(nig_horizon) + 1)
            cum_growth = np.hstack([np.ones((n_sims, 1)), np.exp(np.cumsum(shocks, axis=1))])
            paths = last_val * cum_growth

            q05, q25, q50, q75, q95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)

            cng1, cng2 = st.columns(2)
            with cng1:
                x_span = max(scale_mu_post * 5.0, scale_mu_pri * 2.5, 0.04)
                xg_mu = np.linspace(mu_n - x_span, mu_n + x_span, 500)
                plot_bayesian_density_plotly(
                    xg_mu, d_mu_pri.pdf(xg_mu), d_mu_pos.pdf(xg_mu), mu_lo, mu_hi, nig_ci,
                    "Posterior Distribution of True Annual Log-Growth Rate (μ)", "Annual Log-Growth Rate (μ)",
                    f"Prior t(μ₀={mu0:.2%})", f"Posterior t(μₙ={mu_n:.2%})",
                    is_pct_axis=True, post_mean=mu_n,
                )
            with cng2:
                fig_fan = go.Figure()
                hist_tail = df_b5.tail(30)
                fig_fan.add_trace(go.Scatter(
                    x=hist_tail["year"], y=hist_tail[col_lvl], mode="lines",
                    name="Historical", line=dict(color="#1D3557", width=2.8),
                    hovertemplate="<b>Historical</b>: %{y:,.2f}<extra></extra>",
                ))
                fig_fan.add_trace(go.Scatter(
                    x=np.concatenate([fut_years, fut_years[::-1]]),
                    y=np.concatenate([q95, q05[::-1]]),
                    fill="toself", fillcolor="rgba(46, 134, 171, 0.16)",
                    line=dict(color="rgba(255,255,255,0)"), name="90% Predictive Band", hoverinfo="skip",
                ))
                fig_fan.add_trace(go.Scatter(
                    x=np.concatenate([fut_years, fut_years[::-1]]),
                    y=np.concatenate([q75, q25[::-1]]),
                    fill="toself", fillcolor="rgba(46, 134, 171, 0.34)",
                    line=dict(color="rgba(255,255,255,0)"), name="50% Predictive Band", hoverinfo="skip",
                ))
                fig_fan.add_trace(go.Scatter(
                    x=fut_years, y=q50, mode="lines", name="Median Forecast",
                    line=dict(color="#E84855", width=2.5, dash="dash"),
                    hovertemplate="<b>Median</b>: %{y:,.2f}<extra></extra>",
                ))
                fig_fan.update_layout(
                    title=dict(text=f"{int(nig_horizon)}-Year Posterior Predictive Fan Chart ({nig_series_opt.split(' ')[0]})", font=dict(size=16), x=0.5, xanchor="center"),
                    xaxis=dict(title="Year", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                    yaxis=dict(title=unit_lbl, showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                    hovermode="x unified", height=430, updatemenus=get_scale_menu(),
                    margin=dict(t=85, b=35, l=15, r=15),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.26, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig_fan, **PLOT_STRETCH)

    # 5E: DISCRETE BAYES REGIME DIAGNOSIS
    with bayes_sub[4]:
        st.subheader(f"Discrete Bayes Macroeconomic Regime Diagnosis — {name_t5}")
        recent_3 = df_b5.tail(3)
        if not recent_3.empty:
            r_gdp = recent_3["gdpGrowth"].mean()
            r_gdpc = recent_3["gdpcGrowth"].mean()
            r_pop = recent_3["popGrowth"].mean()
            r_sol = recent_3["solGrowth"].mean()
            st.info(
                f"📡 **Observed Save Signals ({int(recent_3['year'].min())}-{int(recent_3['year'].max())} avg):** "
                f"GDP Growth `{r_gdp:+.2%}` | GDP/c Growth `{r_gdpc:+.2%}` | Pop Growth `{r_pop:+.2%}` | SoL Growth `{r_sol:+.2%}`"
            )

        PRESETS = {
            "GDP Contraction Culprit": {
                "event": "GDP Contraction > 3% with Stable Population",
                "states": [
                    ("War / Mobilization & Convoy Shock", 0.40, 0.75),
                    ("MAPI / Input Bottleneck (Iron/Coal)", 0.35, 0.45),
                    ("Peasant Depletion / Wage Squeeze", 0.25, 0.20),
                ],
            },
            "SoL Stagnation During GDP Boom": {
                "event": "GDP/c +3% while SoL Flat/Falling",
                "states": [
                    ("Regressive Taxation / High Consumption Taxes", 0.45, 0.70),
                    ("Grain & Staple Goods Price Spike", 0.35, 0.65),
                    ("Rapid Capitalist Profit Concentration", 0.20, 0.35),
                ],
            },
            "Custom Hypotheses": {
                "event": "Observed Macro Signal (B)",
                "states": [
                    ("Hypothesis H1", 0.50, 0.70),
                    ("Hypothesis H2", 0.50, 0.25),
                ],
            },
        }

        cd1, cd2 = st.columns([1.4, 2.2])
        preset_name = cd1.selectbox("Scenario Preset:", list(PRESETS.keys()), key="t5_disc_preset")
        if st.session_state.get("t5_disc_last_preset") != preset_name:
            pr = PRESETS[preset_name]
            st.session_state["t5_disc_ev"] = pr["event"]
            st.session_state["t5_disc_n"] = len(pr["states"])
            for idx, (nm, p_v, l_v) in enumerate(pr["states"]):
                st.session_state[f"t5_disc_name_{idx}"] = nm
                st.session_state[f"t5_disc_p_{idx}"] = float(p_v)
                st.session_state[f"t5_disc_v_{idx}"] = float(l_v)
            st.session_state["t5_disc_last_preset"] = preset_name

        n_hyp = cd2.slider("Number of Competing Hypotheses:", min_value=2, max_value=6, key="t5_disc_n")

        for i in range(n_hyp):
            st.session_state.setdefault(f"t5_disc_name_{i}", f"Hypothesis H{i+1}")
            st.session_state.setdefault(f"t5_disc_p_{i}", round(1.0 / n_hyp, 4))
            st.session_state.setdefault(f"t5_disc_v_{i}", 0.5)

        with st.form("form_t5_discrete_bayes"):
            ev_name = st.text_input("Observed Signal / Evidence (E):", key="t5_disc_ev")
            hyps = []
            for i in range(n_hyp):
                hc1, hc2, hc3 = st.columns([2.2, 1.0, 1.0])
                h_nm = hc1.text_input(f"Hypothesis #{i+1}:", key=f"t5_disc_name_{i}")
                h_pri = hc2.number_input(f"Prior P(H{i+1}):", min_value=0.0, max_value=1.0, step=0.05, format="%.4f", key=f"t5_disc_p_{i}")
                h_lik = hc3.number_input(f"Likelihood P(E | H{i+1}):", min_value=0.0, max_value=1.0, step=0.05, format="%.4f", key=f"t5_disc_v_{i}")
                hyps.append((h_nm.strip() or f"H{i+1}", float(h_pri), float(h_lik)))
            st.form_submit_button("🔄 Update Posterior Probabilities", type="primary")

        sum_pri = sum(h[1] for h in hyps)
        if not np.isclose(sum_pri, 1.0, atol=0.01) and sum_pri > 0:
            st.warning(f"Prior probabilities sum to {sum_pri:.4f} (automatically normalized to 1.0 for Bayes calculation).")
            hyps = [(nm, p / sum_pri, lk) for nm, p, lk in hyps]

        p_evidence = sum(p * lk for _, p, lk in hyps)
        if p_evidence <= 0:
            st.error("Total evidence probability P(E) is 0. Increase at least one likelihood.")
        else:
            rows_d = []
            post_vals = []
            for nm, pri, lik in hyps:
                joint = pri * lik
                post = joint / p_evidence
                post_vals.append(post)
                rows_d.append({
                    "Hypothesis": nm,
                    "Prior P(H)": f"{pri:.2%}",
                    "Likelihood P(E|H)": f"{lik:.2%}",
                    "Joint P(H∩E)": f"{joint:.4f}",
                    "Posterior P(H|E)": f"{post:.2%}",
                    "Bayes Shift": f"{(post - pri):+.1%}",
                })

            dc_l, dc_r = st.columns([1.25, 1.15])
            with dc_l:
                st.metric(f"Total Evidence Probability P({ev_name})", f"{p_evidence:.2%}")
                st.dataframe(pd.DataFrame(rows_d), hide_index=True, **STRETCH_KW)
                st.button(
                    "🔁 Use Posterior as New Prior (Chain Next Signal)",
                    on_click=cb_discrete_post_to_prior,
                    args=(post_vals,),
                    key="t5_disc_seq_btn",
                    **STRETCH_KW,
                )
            with dc_r:
                h_labels = [h[0] for h in hyps]
                fig_db = go.Figure()
                fig_db.add_trace(go.Bar(name="Prior", x=h_labels, y=[h[1] for h in hyps], marker_color="#95a5a6", hovertemplate="<b>Prior</b>: %{y:.1%}<extra></extra>"))
                fig_db.add_trace(go.Bar(name="Posterior", x=h_labels, y=post_vals, marker_color="#2E86AB", hovertemplate="<b>Posterior</b>: %{y:.1%}<extra></extra>"))
                fig_db.update_layout(
                    barmode="group",
                    title=dict(text=f"Belief Update Given: {ev_name}", font=dict(size=15), x=0.5, xanchor="center"),
                    yaxis=dict(title="Probability", tickformat=".0%", range=[0, 1.05], showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                    height=370, margin=dict(t=60, b=30, l=10, r=10),
                )
                st.plotly_chart(fig_db, **PLOT_STRETCH)


# ==============================================================================
# TAB 6: CAMPAIGN BOOKMARKS & FINDINGS LOG
# ==============================================================================
with tab6:
    st.header("🔖 Campaign Bookmarks & Findings Log")
    st.caption(
        "Save any interesting chart configuration or Bayesian finding from Tabs 1–5. Rename notes, delete entries in batch, export your session log to CSV, or jump straight back to any saved view."
    )

    bms = st.session_state["bookmarks"]
    if not bms:
        st.info("No bookmarks saved yet. Use the **🔖 Save View** button on any tab to log key campaign milestones.")
    else:
        bf1, bf2, bf3, bf4 = st.columns([1.6, 1.2, 1.0, 1.0])
        tab_opts = ["All"] + sorted(list({b["tab_name"] for b in bms}))
        bm_filter = bf1.selectbox("Filter by Tab:", tab_opts, key="t6_filter")
        bm_order = bf2.selectbox("Order:", ["Newest first", "Oldest first"], key="t6_order")

        bms_view = [b for b in bms if bm_filter == "All" or b["tab_name"] == bm_filter]
        if bm_order == "Newest first":
            bms_view = list(reversed(bms_view))

        export_df = pd.DataFrame([
            {
                "#": b["id"], "Time": b["time"], "Note": b["note"],
                "Tab": b["tab_name"], "Campaign File": b["file"], "Summary": b["summary"],
            }
            for b in bms_view
        ])
        bf3.write("")
        bf3.write("")
        bf3.download_button(
            "📥 Export CSV",
            data=export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="vic3_campaign_findings.csv",
            mime="text/csv",
            **STRETCH_KW,
        )
        bf4.write("")
        bf4.write("")
        if bf4.button("🗑️ Clear All", type="primary", key="t6_del_all", **STRETCH_KW):
            st.session_state["bookmarks"] = []
            st.session_state["bm_form_ver"] += 1
            st.rerun()

        ver = st.session_state["bm_form_ver"]
        with st.form("form_bookmarks_manage"):
            st.markdown("##### 📋 Saved Findings (Edit Notes or Mark to Delete)")
            ch1, ch2, ch3, ch4, ch5, ch6 = st.columns([0.4, 0.7, 1.5, 1.6, 2.8, 0.5])
            ch1.markdown("**#**")
            ch2.markdown("**Time**")
            ch3.markdown("**🏷️ Note**")
            ch4.markdown("**Tab**")
            ch5.markdown("**Summary**")
            ch6.markdown("**🗑️**")
            st.divider()

            for b in bms_view:
                uid = b["uid"]
                r1, r2, r3, r4, r5, r6 = st.columns([0.4, 0.7, 1.5, 1.6, 2.8, 0.5])
                r1.write(f"**{b['id']}**")
                r2.write(b["time"])
                r3.text_input("Note", value=b["note"], key=f"t6_ren_{uid}_{ver}", label_visibility="collapsed")
                r4.write(b["tab_name"])
                r5.write(f"`{b['file']}` — {b['summary']}")
                r6.checkbox("Delete", value=False, key=f"t6_del_{uid}_{ver}", label_visibility="collapsed")

            sub_bm = st.form_submit_button("✅ Confirm Changes", type="primary")

        if sub_bm:
            to_del = set()
            for orig in st.session_state["bookmarks"]:
                uid = orig["uid"]
                if st.session_state.get(f"t6_del_{uid}_{ver}", False):
                    to_del.add(uid)
                    continue
                new_note = st.session_state.get(f"t6_ren_{uid}_{ver}", orig["note"])
                orig["note"] = str(new_note).strip()
            if to_del:
                st.session_state["bookmarks"] = [b for b in st.session_state["bookmarks"] if b["uid"] not in to_del]
            renumber_bookmarks()
            st.session_state["bm_form_ver"] += 1
            st.rerun()

        st.markdown("##### 🔎 Restore Any Saved View")
        for b in bms_view:
            note_part = f" [{b['note']}]" if b["note"] else ""
            with st.expander(f"#{b['id']} ({b['time']}) — {b['tab_name']}{note_part} | {b['summary']}", expanded=False):
                bc1, bc2 = st.columns([3.2, 1.2])
                with bc1:
                    st.write(f"**Campaign:** `{b['file']}`")
                    st.write(f"**Details:** `{b['summary']}`")
                with bc2:
                    st.button(
                        "🔎 Load View in Tab",
                        key=f"t6_load_{b['uid']}",
                        on_click=cb_load_bookmark,
                        args=(b["uid"],),
                        **STRETCH_KW,
                    )
