"""Victoria 3 Macroeconomic Dashboard.

Run with:  streamlit run app.py

Campaign exports are CSVs with the columns ``date, country, gdp, gdpc, sol``. The part of the
file name before the first "-" is taken as the nation you played (``GBR-run1.csv`` -> GBR).
"""
from __future__ import annotations

import inspect
import io
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ======================================================================================
# Configuration
# ======================================================================================
DATA_DIR = Path(".")  # folder scanned for campaign CSVs
REQUIRED_COLUMNS = {"date", "country", "gdp", "gdpc", "sol"}
GDPC_DIVISOR = 100_000  # the raw export stores GDP per capita scaled by 1e5
CLOUD_PREFIX = "☁️ "  # marks uploaded files in the pickers

KOFI_URL = "https://ko-fi.com/YOUR_USERNAME"
PIX_KEY = "insert-your-random-pix-key-here"

# column -> name of its year-over-year growth column
GROWTH_COLUMNS = {
    "gdp": "gdpGrowth",
    "gdpc": "gdpcGrowth",
    "population": "popGrowth",
    "sol": "solGrowth",
}

TAG_NAMES = {
    "GBR": "Great Britain", "FRA": "France", "USA": "United States", "GER": "Germany",
    "PRU": "Prussia", "RUS": "Russia", "ITA": "Italy", "SPA": "Spain", "POR": "Portugal",
    "AUS": "Austria", "TUR": "Ottoman Empire", "SWE": "Sweden", "NOR": "Norway",
    "DEN": "Denmark", "FIN": "Finland", "BEL": "Belgium", "NET": "Netherlands",
    "SWI": "Switzerland", "JAP": "Japan", "CHI": "China", "QIN": "Qing",
    "EIC": "East India Company", "KOR": "Korea", "DAI": "Dai Nam", "SIA": "Siam",
    "SIK": "Sikh Empire", "PER": "Persia", "AFG": "Afghanistan",
    "BRZ": "Brazil", "PNI": "Riograndense Republic", "MEX": "Mexico", "CAN": "Canada",
    "TEX": "Texas", "CUB": "Cuba", "HAI": "Haiti", "ARG": "Argentina", "CHL": "Chile",
    "PEU": "Peru", "COL": "Colombia", "GCO": "Gran Colombia", "NGR": "New Granada",
    "VNZ": "Venezuela", "ECU": "Ecuador", "BOL": "Bolivia", "PAR": "Paraguay",
    "URU": "Uruguay",
    "SAR": "Sardinia-Piedmont", "SIC": "Two Sicilies", "PAP": "Papal States",
    "TUS": "Tuscany", "BAV": "Bavaria", "SAX": "Saxony", "WUR": "Württemberg",
    "HAN": "Hanover",
    "GRE": "Greece", "SER": "Serbia", "ROM": "Romania", "BUL": "Bulgaria",
    "HUN": "Hungary", "POL": "Poland", "EGY": "Egypt", "MOR": "Morocco",
    "DEI": "Dutch East Indies", "AST": "Australia", "PHI": "Philippines",
    "SAF": "South Africa", "ETH": "Ethiopia", "SOK": "Sokoto", "ZUL": "Zulu",
    "MAD": "Madagascar",
}

# --- colours ---------------------------------------------------------------------------
GRID = "rgba(128, 128, 128, 0.2)"
ZERO_LINE = "#888888"
TREND_GREY = "rgba(150, 150, 150, 0.7)"
TARGET, RIVAL = "#2E86AB", "#E84855"  # the two sides of a comparison
GAIN, LOSS = "#2A9D8F", "#E76F51"  # expansion / contraction bars
INTENSIVE, EXTENSIVE, TOTAL = "#E76F51", "#2A9D8F", "#1D3557"  # growth accounting
VOLATILITY = "#9B5DE5"
# The played nation is drawn in TARGET, so TARGET is deliberately not in this palette.
PALETTE = ["#E84855", "#2A9D8F", "#F4A261", "#9B5DE5", "#E76F51",
           "#00BBF9", "#00F5D4", "#D62828", "#003049"]


@dataclass(frozen=True)
class Metric:
    col: str
    title: str
    y_title: str
    color: str


LEVELS = (
    Metric("gdp", "Total GDP", "£ Millions", "#2E86AB"),
    Metric("gdpc", "GDP per Capita", "£", "#E84855"),
    Metric("population", "Total Population", "Millions", "#2A9D8F"),
    Metric("sol", "Standard of Living", "Index", "#9B5DE5"),
)
RATES = (
    Metric("gdpGrowth", "Annual GDP Growth", "Rate (%)", "#F4A261"),
    Metric("gdpcGrowth", "Annual GDP/c Growth", "Rate (%)", "#E76F51"),
    Metric("popGrowth", "Annual Population Growth", "Rate (%)", "#264653"),
    Metric("solGrowth", "Annual SoL Growth", "Rate (%)", "#8338EC"),
)
# (column, title, y-axis title, rebase to 100 at the first year?) for the Global tab
GLOBAL_CHARTS = (
    (("gdp", "Total GDP", "Gross Domestic Product (£ Millions)", False),
     ("gdpc", "GDP per Capita", "GDP per Capita (£)", False),
     ("population", "Total Population", "Population (Millions)", False)),
    (("gdp", "GDP Indexed", "Indexed GDP (100 = Base Year)", True),
     ("sol", "Standard of Living", "Standard of Living Index", False),
     ("population", "Population Indexed", "Indexed Population (100 = Base Year)", True)),
)
# (label, column, number format) for the Global master summary
SUMMARY_METRICS = (
    ("GDP", "gdp", "£%.2fM"),
    ("GDP/c", "gdpc", "£%.2f"),
    ("SoL", "sol", "%.2f"),
    ("Pop", "population", "%.2fM"),
)
STAT_ROWS = ["CAGR", "Median", "Max", "Min", "Std. Dev"]
RIVAL_PREFERENCE = ("GBR", "USA", "FRA", "GER", "PRU", "RUS")  # default benchmarks, in order


# ======================================================================================
# Streamlit helpers
# ======================================================================================
def _fill_width(widget) -> dict:
    """Kwargs that make `widget` span its container on any Streamlit version.

    Recent releases deprecate ``use_container_width=True`` in favour of ``width="stretch"``.
    """
    try:
        width = inspect.signature(widget).parameters.get("width")
    except (TypeError, ValueError):
        width = None
    if width is not None and width.default == "stretch":
        return {"width": "stretch"}
    return {"use_container_width": True}


FILL_CHART = _fill_width(st.plotly_chart)
FILL_TABLE = _fill_width(st.dataframe)

# Run each tab as a fragment so touching a widget re-renders that tab only, not all four.
fragment = getattr(st, "fragment", None) or (lambda func: func)


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, **FILL_CHART)


def show_table(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, hide_index=True, **FILL_TABLE, **kwargs)


# ======================================================================================
# Loading campaigns
# ======================================================================================
@dataclass(frozen=True)
class Campaign:
    label: str  # text shown in the pickers
    name: str  # file name
    path: Path | None = None  # set for files found on disk ...
    content: bytes | None = None  # ... or the raw bytes of an upload

    @property
    def played_tag(self) -> str:
        return self.name.split("-")[0].upper() if "-" in self.name else "UNKNOWN"


def _csv_problem(source) -> str | None:
    """Return why `source` is not a usable campaign export, or None if it is."""
    try:
        columns = set(pd.read_csv(source, nrows=0).columns)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as exc:
        return f"could not be read as CSV ({type(exc).__name__})"
    missing = REQUIRED_COLUMNS - columns
    return f"missing columns: {', '.join(sorted(missing))}" if missing else None


def discover_campaigns(uploads) -> dict[str, Campaign]:
    """CSVs next to the app plus the user's uploads, keyed by their picker label."""
    campaigns: dict[str, Campaign] = {}
    for path in sorted(DATA_DIR.glob("*.csv")):
        # Unrelated CSVs in the folder are skipped quietly; only uploads get a warning.
        if path.is_file() and _csv_problem(path) is None:
            campaigns[path.name] = Campaign(path.name, path.name, path=path)
    for upload in uploads or []:
        content = upload.getvalue()
        problem = _csv_problem(io.BytesIO(content))
        if problem:
            st.sidebar.warning(f"`{upload.name}` was skipped: {problem}.")
            continue
        label = f"{CLOUD_PREFIX}{upload.name}"
        campaigns[label] = Campaign(label, upload.name, content=content)
    return campaigns


@st.cache_data(max_entries=16, ttl=3600, show_spinner="Loading campaign…")
def _prepare(source: str | bytes, version: float) -> pd.DataFrame:
    """Turn one export into a (country, year) panel with year-over-year growth columns.

    `version` is not used in the body; it makes the cache key change when a file is edited.
    """
    raw = pd.read_csv(io.BytesIO(source) if isinstance(source, bytes) else source)
    raw["date"] = pd.to_datetime(raw["date"])
    raw["gdpc"] = raw["gdpc"] / GDPC_DIVISOR
    raw["population"] = raw["gdp"] / raw["gdpc"]
    raw["year"] = raw["date"].dt.year

    yearly = raw.sort_values("date").groupby(["country", "year"]).last().reset_index()
    growth = yearly.groupby("country")[list(GROWTH_COLUMNS)].pct_change(fill_method=None)
    return yearly.join(growth.rename(columns=GROWTH_COLUMNS))


def load_campaign(campaign: Campaign) -> pd.DataFrame:
    if campaign.content is not None:
        return _prepare(campaign.content, 0.0)
    return _prepare(str(campaign.path), campaign.path.stat().st_mtime)


def pick_campaign(campaigns: dict[str, Campaign], label: str, key: str, index: int = 0):
    choice = st.selectbox(label, list(campaigns), index=index, key=key)
    campaign = campaigns[choice]
    return campaign, load_campaign(campaign)


# ======================================================================================
# Data helpers
# ======================================================================================
def tag_name(tag: str) -> str:
    return TAG_NAMES.get(tag, tag)


def tag_option(tag: str) -> str:  # "GBR - Great Britain"
    return f"{tag} - {tag_name(tag)}"


def tag_label(tag: str) -> str:  # "Great Britain (GBR)"
    return f"{tag_name(tag)} ({tag})"


def available_tags(df: pd.DataFrame) -> list[str]:
    return sorted(df.dropna(subset=["gdp"])["country"].unique().tolist())


def default_index(options: list, preferred, fallback: int = 0) -> int:
    return options.index(preferred) if preferred in options else fallback


def default_rival_index(tags: list[str], target: str) -> int:
    """Pre-select a sensible benchmark: a major power that isn't the target itself."""
    for tag in RIVAL_PREFERENCE:
        if tag in tags and tag != target:
            return tags.index(tag)
    return next(i for i, tag in enumerate(tags) if tag != target)


def nation(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    return df[df["country"] == tag].reset_index(drop=True)


def slice_years(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return df[df["year"].between(start, end)].reset_index(drop=True)


def year_span(df: pd.DataFrame) -> str:
    return f"{int(df['year'].min())}-{int(df['year'].max())}" if not df.empty else "N/A"


@dataclass
class Run:
    """One nation in one campaign, as shown in the two-sided comparison tabs."""
    label: str  # long name for titles and legends
    short: str  # compact name for table headers
    color: str
    df: pd.DataFrame


# ======================================================================================
# Maths
# ======================================================================================
def rolling_cagr(growth: pd.Series, window: int) -> pd.Series:
    """Rolling geometric-mean growth rate. Missing years (e.g. the first one) are skipped."""
    factor = growth + 1
    log_factor = np.log(factor.where(factor > 0))
    return np.expm1(log_factor.rolling(window, min_periods=1).mean())


def geometric_mean_growth(growth: pd.Series) -> float:
    growth = growth.dropna()
    if growth.empty or (growth <= -1).any():
        return np.nan
    return float(np.expm1(np.log1p(growth).mean()))


def smooth(series: pd.Series, window: int, *, pct: bool) -> tuple[pd.Series, str]:
    """Trend line for a level (rolling mean) or a growth rate (rolling CAGR), plus its label."""
    if pct:
        return rolling_cagr(series, window), f"{window}-Yr CAGR"
    return series.rolling(window, min_periods=1).mean(), f"{window}-Yr Trend"


def cagr_pct(first: float, last: float, years: int) -> float:
    if years > 0 and first > 0 and last > 0:
        return ((last / first) ** (1 / years) - 1) * 100
    return np.nan


@dataclass
class Accounting:
    """Split of log-GDP growth into productivity (GDP/c) and demographics (population)."""
    gdp: tuple[float, float]
    gdpc: tuple[float, float]
    population: tuple[float, float]
    intensive: float
    extensive: float


def growth_accounting(df: pd.DataFrame) -> Accounting | None:
    valid = df.dropna(subset=["gdp", "gdpc", "population"])
    if valid.empty:
        return None
    first, last = valid.iloc[0], valid.iloc[-1]
    total = np.log(last["gdp"] / first["gdp"])

    def share(col: str) -> float:
        if not np.isfinite(total) or total == 0:
            return np.nan
        return np.log(last[col] / first[col]) / total

    return Accounting(
        gdp=(first["gdp"], last["gdp"]),
        gdpc=(first["gdpc"], last["gdpc"]),
        population=(first["population"], last["population"]),
        intensive=share("gdpc"),
        extensive=share("population"),
    )


def find_crises(df: pd.DataFrame) -> pd.DataFrame:
    """The five deepest peak-to-recovery GDP contractions."""
    peak = df["gdp"].cummax()
    drawdown = (df["gdp"] - peak) / peak
    years = df["year"].astype(int).tolist()

    crises, current = [], None
    for i, (year, dd) in enumerate(zip(years, drawdown)):
        if dd < 0:
            if current is None:
                current = {"peak": years[i - 1] if i else year, "trough": year, "depth": dd}
            elif dd < current["depth"]:
                current.update(trough=year, depth=dd)
        elif dd == 0 and current is not None:
            crises.append({"Peak Year": current["peak"], "Trough Year": current["trough"],
                           "Recovery Year": str(year), "Drawdown": current["depth"],
                           "Duration": f"{year - current['peak']} yrs"})
            current = None
    if current is not None:
        crises.append({"Peak Year": current["peak"], "Trough Year": current["trough"],
                       "Recovery Year": "Unrecovered", "Drawdown": current["depth"],
                       "Duration": f">{years[-1] - current['peak']} yrs"})

    if not crises:
        return pd.DataFrame({"Message": ["No major economic contractions detected."]})
    table = pd.DataFrame(crises).sort_values("Drawdown").head(5)
    table["Drawdown"] = table["Drawdown"].map(fmt_pct)
    return table


# ======================================================================================
# Formatting and tables
# ======================================================================================
def fmt_pct(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2%}"


def fmt_share(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.1%}"


def growth_stats(growth: pd.Series) -> list[str]:
    g = growth.dropna()
    return [fmt_pct(geometric_mean_growth(g)), fmt_pct(g.median()), fmt_pct(g.max()),
            fmt_pct(g.min()), fmt_pct(g.std())]


def stats_table(columns: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({"Metric": STAT_ROWS,
                         **{name: growth_stats(series) for name, series in columns.items()}})


def accounting_table(acc: Accounting) -> pd.DataFrame:
    return pd.DataFrame({
        "Component": ["Total Expansion", "Intensive (Productivity)", "Extensive (Demographics)"],
        "Initial": [f"£{acc.gdp[0]:,.2f}M", f"£{acc.gdpc[0]:,.2f}", f"{acc.population[0]:,.2f}M"],
        "Final": [f"£{acc.gdp[1]:,.2f}M", f"£{acc.gdpc[1]:,.2f}", f"{acc.population[1]:,.2f}M"],
        "Share": ["100.0%", fmt_share(acc.intensive), fmt_share(acc.extensive)],
    })


def accounting_column(df: pd.DataFrame) -> list[str]:
    acc = growth_accounting(df)
    if acc is None:
        return ["N/A"] * 3
    return [f"£{acc.gdp[0]:,.2f}M -> £{acc.gdp[1]:,.2f}M",
            fmt_share(acc.intensive), fmt_share(acc.extensive)]


def master_summary(df: pd.DataFrame, played: str) -> pd.DataFrame:
    """One row per nation: span, final levels and compound growth over its whole history."""
    rows = []
    for tag, c in df.dropna(subset=["gdp"]).groupby("country"):
        if len(c) < 2:
            continue
        first, last = c.iloc[0], c.iloc[-1]
        years = int(last["year"] - first["year"])
        row = {
            "Tag": tag,
            "Country": f"⭐ {tag_name(tag)}" if tag == played else tag_name(tag),
            "Span": f"{int(first['year'])}-{int(last['year'])}",
            "Yrs": len(c),
        }
        for label, col, _ in SUMMARY_METRICS:
            row[f"Final {label}"] = last[col]
            row[f"{label} CAGR"] = cagr_pct(first[col], last[col], years)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Final GDP", ascending=False)


def summary_column_config() -> dict:
    config = {}
    for label, _, fmt in SUMMARY_METRICS:
        config[f"Final {label}"] = st.column_config.NumberColumn(f"Final {label}", format=fmt)
        config[f"{label} CAGR"] = st.column_config.NumberColumn(f"{label} CAGR", format="%.2f%%")
    return config


# ======================================================================================
# Figure building blocks
# ======================================================================================
def hover(label: str, fmt: str) -> str:
    return f"<b>{label}</b>: %{{y:{fmt}}}<extra></extra>"


def line_trace(x, y, name, color, *, width=2.5, dash=None, markers=False, fmt=".2f",
               hover_label=None) -> go.Scatter:
    return go.Scatter(
        x=x, y=y, name=name, mode="lines+markers" if markers else "lines",
        line=dict(color=color, width=width, dash=dash),
        marker=dict(size=4) if markers else None,
        hovertemplate=hover(hover_label or name, fmt),
    )


def bar_trace(x, y, name, color, *, fmt=".2%", hover_label=None) -> go.Bar:
    return go.Bar(x=x, y=y, name=name, marker_color=color,
                  hovertemplate=hover(hover_label or name, fmt))


def sign_colors(values: pd.Series, negative: str = LOSS) -> np.ndarray:
    return np.where(values >= 0, GAIN, negative)


def scale_menu() -> list[dict]:
    """Linear / Log toggle for the y-axis."""
    return [dict(
        type="buttons", direction="right", active=0,
        x=0.0, xanchor="left", y=1.18, yanchor="top",
        bgcolor="rgba(128, 128, 128, 0.15)", bordercolor="rgba(128, 128, 128, 0.4)",
        font=dict(size=12),
        buttons=[
            dict(label="Linear", method="relayout", args=[{"yaxis.type": "linear"}]),
            dict(label="Log", method="relayout", args=[{"yaxis.type": "log"}]),
        ],
    )]


def style(fig: go.Figure, title: str, *, height: int, y_title: str | None = None,
          pct: bool = False, signed: bool = False, toggle: bool = False,
          x_title: str | None = None, x_range: list | None = None,
          title_size: int = 18, top: int | None = None) -> go.Figure:
    """Apply the dashboard's shared look so each chart only states what is specific to it."""
    tick_format = None if not pct else ("+.1%" if signed else ".1%")
    fig.update_layout(
        title=dict(text=title, font=dict(size=title_size), x=0.5, xanchor="center"),
        xaxis=dict(title=x_title, range=x_range, showgrid=True, gridcolor=GRID),
        yaxis=dict(title=y_title, tickformat=tick_format, showgrid=True, gridcolor=GRID,
                   zeroline=True, zerolinecolor=ZERO_LINE, zerolinewidth=1.5),
        hovermode="x unified",
        height=height,
        updatemenus=scale_menu() if toggle else [],
        margin=dict(t=top or (85 if toggle else 60), b=30 if toggle else 20, l=10, r=10),
    )
    return fig


# ======================================================================================
# Figures
# ======================================================================================
def country_comparison_figure(frames, selected, colors, played, span,
                              col, title, y_title, indexed) -> go.Figure:
    """One line per nation; the played nation is highlighted. `indexed` rebases to 100."""
    fig = go.Figure()
    for tag in selected:
        data = frames.get(tag)
        if data is None or data.empty:
            continue
        y = data[col]
        if indexed:
            valid = y.dropna()
            y = y / valid.iloc[0] * 100 if not valid.empty else y
        name = f"{'⭐ ' if tag == played else ''}{tag_label(tag)}"
        fig.add_trace(line_trace(
            data["year"], y, name, colors[tag],
            width=3.5 if tag == played else 2.0, fmt=".1f" if indexed else ".2f"))
    return style(fig, f"{title} ({span})", height=450, y_title=y_title, toggle=True)


def single_trend_figure(df, metric: Metric, country: str, window: int, *, pct: bool):
    """Dossier chart: a metric over time with its rolling trend."""
    fmt = ".2%" if pct else ".2f"
    smoothed, label = smooth(df[metric.col], window, pct=pct)
    fig = go.Figure()
    fig.add_trace(line_trace(df["year"], smoothed, label, TREND_GREY, width=2, dash="dash",
                             fmt=fmt))
    fig.add_trace(line_trace(df["year"], df[metric.col], metric.title, metric.color,
                             markers=True, fmt=fmt))
    return style(fig, f"{country} — {metric.title} ({year_span(df)})", height=360,
                 y_title=metric.y_title, pct=pct, toggle=not pct, x_title="Year")


def comparison_figure(runs: list[Run], metric: Metric, window: int, *, pct: bool, span: str):
    """Two-sided chart: both runs' data (solid) and their rolling trends (dashed)."""
    fmt = ".2%" if pct else ".2f"
    fig = go.Figure()
    for run in runs:
        smoothed, label = smooth(run.df[metric.col], window, pct=pct)
        fig.add_trace(line_trace(run.df["year"], smoothed, f"{run.label} {label}", run.color,
                                 width=1.5, dash="dash", fmt=fmt))
    for run in runs:
        fig.add_trace(line_trace(run.df["year"], run.df[metric.col], f"{run.label} Data",
                                 run.color, fmt=fmt))
    title = f"{metric.title}: {runs[0].label} vs {runs[1].label} ({span})"
    return style(fig, title, height=380, y_title=metric.y_title, pct=pct, toggle=not pct,
                 x_title="Year")


def decomposition_figure(df, name: str, *, window: int | None = None,
                         x_range: list | None = None, title_size: int = 18):
    """Growth split into productivity (GDP/c) and demographics (population).

    With `window` the bars are rolling CAGRs instead of annual rates.
    """
    def values(col: str) -> pd.Series:
        return df[col] if window is None else rolling_cagr(df[col], window)

    if window is None:
        names = ("Intensive (Productivity)", "Extensive (Demographics)", "Total GDP Growth")
        hovers = ("Intensive", "Extensive", "Total Growth")
        title = f"{name} — Annual Growth Decomposition ({year_span(df)})"
        y_title = "Annual Contribution"
    else:
        names = hovers = ("Trend Intensive", "Trend Extensive", "Trend GDP")
        title = f"{name} — Structural Decomposition ({window}-Yr CAGR) ({year_span(df)})"
        y_title = "Trend Contribution"

    fig = go.Figure()
    fig.add_trace(bar_trace(df["year"], values("gdpcGrowth"), names[0], INTENSIVE,
                            hover_label=hovers[0]))
    fig.add_trace(bar_trace(df["year"], values("popGrowth"), names[1], EXTENSIVE,
                            hover_label=hovers[1]))
    fig.add_trace(line_trace(df["year"], values("gdpGrowth"), names[2], TOTAL, markers=True,
                             fmt=".2%", hover_label=hovers[2]))
    fig.update_layout(barmode="relative")
    return style(fig, title, height=400, y_title=y_title, pct=True, x_range=x_range,
                 title_size=title_size)


def expansion_figure(df, name: str, *, height: int, title_size: int = 18, top: int = 60,
                     x_title: str | None = None, x_range: list | None = None):
    growth = df["gdpGrowth"]
    fig = go.Figure(bar_trace(df["year"], growth, name, sign_colors(growth)))
    return style(fig, f"{name} — Expansion vs Contraction ({year_span(df)})", height=height,
                 pct=True, x_title=x_title, x_range=x_range, title_size=title_size, top=top)


def volatility_figure(runs: list[Run], window: int, *, title: str, height: int):
    fig = go.Figure()
    for run in runs:
        rolling_std = run.df["gdpGrowth"].rolling(window, min_periods=2).std()
        fig.add_trace(line_trace(run.df["year"], rolling_std, f"{run.label} Volatility",
                                 run.color, markers=True, fmt=".2%"))
    span = year_span(pd.concat([run.df for run in runs]))
    return style(fig, f"{title} ({window}-Yr) ({span})", height=height,
                 y_title="Standard Deviation", pct=True, x_title="Year")


def convergence_frame(a: Run, b: Run) -> pd.DataFrame:
    """GDP/c of both sides in the years where both have levels and growth rates."""
    a_by_year, b_by_year = a.df.set_index("year"), b.df.set_index("year")
    frame = pd.DataFrame({
        "a_gdpc": a_by_year["gdpc"], "b_gdpc": b_by_year["gdpc"],
        "a_growth": a_by_year["gdpcGrowth"], "b_growth": b_by_year["gdpcGrowth"],
    }).dropna()
    frame["ratio"] = frame["a_gdpc"] / frame["b_gdpc"]
    frame["spread"] = frame["a_growth"] - frame["b_growth"]
    return frame


def convergence_figures(a: Run, b: Run, frame: pd.DataFrame) -> list[go.Figure]:
    span = f"{int(frame.index.min())}-{int(frame.index.max())}"

    ratio = go.Figure()
    ratio.add_trace(line_trace(frame.index, frame["ratio"], f"{a.label} / {b.label} Ratio",
                               a.color, markers=True, fmt=".2%",
                               hover_label="Productivity Ratio"))
    ratio.add_hline(y=1.0, line_dash="dash", line_color="#D62828",
                    annotation_text="Productivity Parity (100%)", annotation_position="top right")
    style(ratio, f"Productivity Convergence: {a.label} relative to {b.label} ({span})",
          height=420, y_title=f"GDP/c as % of {b.label}", pct=True, x_title="Year")

    spread = go.Figure(bar_trace(frame.index, frame["spread"], "Catch-up Spread",
                                 sign_colors(frame["spread"], RIVAL), fmt="+.2%",
                                 hover_label="Growth Rate Spread"))
    style(spread, f"Convergence Velocity ({a.short} GDP/c Growth minus {b.short} GDP/c Growth) "
                  f"({span})", height=380, y_title="Annual Spread (%)", pct=True, signed=True,
          x_title="Year")
    return [ratio, spread]


def convergence_summary(a: Run, b: Run, frame: pd.DataFrame) -> pd.DataFrame:
    first, last = frame["ratio"].iloc[0], frame["ratio"].iloc[-1]
    return pd.DataFrame({
        "Convergence Metric": ["Initial Productivity Ratio", "Final Productivity Ratio",
                               "Net Convergence Spread", f"{a.short} GDP/c CAGR",
                               f"{b.short} GDP/c CAGR"],
        "Result": [fmt_pct(first), fmt_pct(last),
                   f"{(last - first) * 100:+.2f} percentage points",
                   fmt_pct(geometric_mean_growth(frame["a_growth"])),
                   fmt_pct(geometric_mean_growth(frame["b_growth"]))],
    })


# ======================================================================================
# Widgets shared by several tabs
# ======================================================================================
def year_range(label: str, *frames: pd.DataFrame, key: str) -> tuple[int, int]:
    lo = min(int(f["year"].min()) for f in frames)
    hi = max(int(f["year"].max()) for f in frames)
    if lo == hi:  # st.slider needs min < max
        st.caption(f"Only {lo} is available.")
        return lo, hi
    return st.slider(label, min_value=lo, max_value=hi, value=(lo, hi), key=key)


def window_inputs(prefix: str) -> tuple[int, int]:
    trend = st.number_input("Trend Window (Years):", min_value=1, max_value=50, value=10,
                            key=f"{prefix}_ma")
    vol = st.number_input("Volatility Window (Years):", min_value=2, max_value=50, value=10,
                          key=f"{prefix}_vol")
    return trend, vol


# ======================================================================================
# Tab 1: Global landscape
# ======================================================================================
@fragment
def render_global(campaigns: dict[str, Campaign]) -> None:
    st.header("The Great Power Race")
    campaign, df = pick_campaign(campaigns, "Select Campaign Data:", "t1_file")
    played = campaign.played_tag

    ranking = df.dropna(subset=["gdp"]).groupby("country")["gdp"].last()
    ranking = ranking.sort_values(ascending=False)
    defaults = ranking.index[:8].tolist()
    if played in ranking.index and played not in defaults:
        defaults.append(played)
    selected = st.multiselect("Select Nations to Compare:", options=ranking.index.tolist(),
                              default=defaults, format_func=tag_option)

    if selected:
        chosen = df[df["country"].isin(selected)]
        frames = {tag: g.sort_values("year") for tag, g in chosen.groupby("country")}
        others = [tag for tag in selected if tag != played]
        colors = {tag: PALETTE[i % len(PALETTE)] for i, tag in enumerate(others)}
        colors[played] = TARGET
        span = year_span(chosen)

        for column, specs in zip(st.columns(2), GLOBAL_CHARTS):
            with column:
                for col, title, y_title, indexed in specs:
                    show_chart(country_comparison_figure(
                        frames, selected, colors, played, span, col, title, y_title, indexed))

    st.divider()
    st.subheader("Global Dataset Master Summary")
    summary = master_summary(df, played)
    if summary.empty:
        st.info("No nation has at least two years of GDP data.")
    else:
        show_table(summary, column_config=summary_column_config())


# ======================================================================================
# Tab 2: Single economy dossier
# ======================================================================================
@fragment
def render_dossier(campaigns: dict[str, Campaign]) -> None:
    st.header("Single Economy Dossier")
    campaign, df = pick_campaign(campaigns, "Select Campaign Data:", "t2_file")
    tags = available_tags(df)
    if not tags:
        st.warning("This campaign has no GDP data.")
        return

    col_tag, col_range, col_windows = st.columns([1, 2, 1])
    with col_tag:
        tag = st.selectbox("Target Economy:", options=tags,
                           index=default_index(tags, campaign.played_tag),
                           format_func=tag_option, key="t2_target")
    country = nation(df, tag)
    with col_range:
        start, end = year_range("Timeline Range:", country, key="t2_slider")
    with col_windows:
        trend_window, vol_window = window_inputs("t2")

    data = slice_years(country, start, end)
    if data.empty:
        st.warning("No data available for the selected timeframe.")
        return
    name = tag_name(tag)

    st.subheader("Macroeconomic Trajectories")
    left, right = st.columns(2)
    with left:
        for metric in LEVELS:
            show_chart(single_trend_figure(data, metric, name, trend_window, pct=False))
    with right:
        for metric in RATES:
            show_chart(single_trend_figure(data, metric, name, trend_window, pct=True))

    st.divider()
    st.subheader("Structural Growth Accounting")
    left, right = st.columns(2)
    with left:
        show_chart(decomposition_figure(data, name))
    with right:
        show_chart(decomposition_figure(data, name, window=trend_window))

    st.divider()
    st.subheader("Volatility & Crises")
    left, right = st.columns(2)
    with left:
        show_chart(expansion_figure(data, name, height=350))
    with right:
        show_chart(volatility_figure([Run(name, tag, VOLATILITY, data)], vol_window,
                                     title=f"{name} — Macroeconomic Volatility", height=350))

    st.divider()
    st.subheader("Dossier Master Tables")
    stats = stats_table({
        "GDP Growth": data["gdpGrowth"], "GDPpC Growth": data["gdpcGrowth"],
        "SoL Growth": data["solGrowth"], "Pop Growth": data["popGrowth"],
    })
    acc = growth_accounting(data)
    left, right = st.columns(2)
    with left:
        st.write("**General Performance Stats**")
        show_table(stats)
        st.write("**Growth Accounting Breakdown**")
        if acc is None:
            st.info("Not enough data for growth accounting.")
        else:
            show_table(accounting_table(acc))
    with right:
        st.write("**Top Economic Crises Leaderboard**")
        show_table(find_crises(data))


# ======================================================================================
# Tabs 3 and 4: two-sided comparisons (two nations / two campaigns)
# ======================================================================================
STAT_COLUMNS = (("gdpGrowth", "GDP Gr"), ("gdpcGrowth", "GDP/c Gr"),
                ("solGrowth", "SoL Gr"), ("popGrowth", "Pop Gr"))


def render_comparison(a: Run, b: Run, *, scope: str, key: str, range_label: str,
                      range_col, window_col) -> None:
    """Body shared by the Bilateral and Cross-Campaign tabs (`a` = target, `b` = benchmark)."""
    if a.df.empty or b.df.empty:
        st.warning("Insufficient data for the selected pair.")
        return

    with range_col:
        start, end = year_range(range_label, a.df, b.df, key=f"{key}_slider")
    with window_col:
        trend_window, vol_window = window_inputs(key)

    a = replace(a, df=slice_years(a.df, start, end))
    b = replace(b, df=slice_years(b.df, start, end))
    if a.df.empty or b.df.empty:
        st.warning("One of the two has no data in the selected range.")
        return
    runs = [a, b]
    span = f"{start}-{end}"
    x_range = [start - 0.5, end + 0.5]

    # --- trajectories --------------------------------------------------------------
    st.subheader(f"{scope} Trajectories")
    left, right = st.columns(2)
    with left:
        for metric in LEVELS:
            show_chart(comparison_figure(runs, metric, trend_window, pct=False, span=span))
    with right:
        for metric in RATES:
            show_chart(comparison_figure(runs, metric, trend_window, pct=True, span=span))

    # --- growth accounting ---------------------------------------------------------
    st.divider()
    st.subheader(f"{scope} Growth Accounting")
    for column, run in zip(st.columns(2), runs):
        with column:
            show_chart(decomposition_figure(run.df, run.label, x_range=x_range, title_size=17))
            show_chart(decomposition_figure(run.df, run.label, window=trend_window,
                                            x_range=x_range, title_size=17))

    # --- convergence ---------------------------------------------------------------
    st.divider()
    st.subheader(f"{scope} Convergence & Catch-Up Dynamics")
    frame = convergence_frame(a, b)
    if not frame.empty:
        for fig in convergence_figures(a, b, frame):
            show_chart(fig)

    # --- volatility ----------------------------------------------------------------
    st.divider()
    st.subheader(f"{scope} Macroeconomic Volatility")
    left, right = st.columns(2)
    with left:
        for run in runs:
            show_chart(expansion_figure(run.df, run.label, height=280, title_size=15, top=50,
                                        x_title="Year", x_range=x_range))
    with right:
        show_chart(volatility_figure(runs, vol_window, title="Comparative Macroeconomic Volatility",
                                     height=580))

    # --- tables --------------------------------------------------------------------
    st.divider()
    st.subheader(f"{scope} Performance & Convergence Tables")
    stats = stats_table({f"{run.short} {abbr}": run.df[col]
                         for col, abbr in STAT_COLUMNS for run in runs})
    accounting = pd.DataFrame({
        "Component": ["Total GDP Span", "Intensive Share (Productivity)",
                      "Extensive Share (Demographics)"],
        **{run.label: accounting_column(run.df) for run in runs},
    })
    left, right = st.columns(2)
    with left:
        st.write(f"**Comparative Performance Statistics: {a.short} vs {b.short}**")
        show_table(stats)
        st.write("**Growth Accounting Decomposition Comparison**")
        show_table(accounting)
    with right:
        if not frame.empty:
            st.write(f"**Convergence Dynamics Summary: {a.short} vs {b.short}**")
            show_table(convergence_summary(a, b, frame))


# ======================================================================================
# Tab 3: Bilateral & convergence
# ======================================================================================
@fragment
def render_bilateral(campaigns: dict[str, Campaign]) -> None:
    st.header("Bilateral & Convergence Analysis")
    campaign, df = pick_campaign(campaigns, "Select Campaign Data:", "t3_file")
    tags = available_tags(df)
    if len(tags) < 2:
        st.warning("This campaign needs at least two nations with GDP data.")
        return

    target_default = default_index(tags, campaign.played_tag)
    controls = st.columns([1, 1, 2, 1])
    with controls[0]:
        tag_a = st.selectbox("Target Nation:", options=tags, index=target_default,
                             format_func=tag_option, key="t3_tag1")
    with controls[1]:
        tag_b = st.selectbox("Benchmark / Rival:", options=tags,
                             index=default_rival_index(tags, tags[target_default]),
                             format_func=tag_option, key="t3_tag2")
    if tag_a == tag_b:
        st.warning("Choose two different nations to compare.")
        return

    render_comparison(
        Run(tag_label(tag_a), tag_a, TARGET, nation(df, tag_a)),
        Run(tag_label(tag_b), tag_b, RIVAL, nation(df, tag_b)),
        scope="Bilateral", key="t3", range_label="Common Historical Range:",
        range_col=controls[2], window_col=controls[3],
    )


# ======================================================================================
# Tab 4: Cross-campaign meta-analysis
# ======================================================================================
@fragment
def render_meta(campaigns: dict[str, Campaign]) -> None:
    st.header("Cross-Campaign Meta-Analysis")
    if len(campaigns) < 2:
        st.info("Cross-campaign analysis requires at least 2 CSV files. "
                "Please upload another campaign in the sidebar.")
        return

    col_1, col_2 = st.columns(2)
    with col_1:
        campaign_1, df_1 = pick_campaign(campaigns, "First Campaign File (Run 1):", "t4_f1", 0)
    with col_2:
        campaign_2, df_2 = pick_campaign(campaigns, "Second Campaign File (Run 2):", "t4_f2", 1)

    tags_1, tags_2 = available_tags(df_1), available_tags(df_2)
    if not tags_1 or not tags_2:
        st.warning("Both campaigns need GDP data.")
        return

    controls = st.columns([1, 1, 2, 1])
    with controls[0]:
        tag_1 = st.selectbox("Nation in Run 1:", tags_1,
                             index=default_index(tags_1, campaign_1.played_tag),
                             format_func=tag_option, key="t4_tag1")
    with controls[1]:
        tag_2 = st.selectbox("Nation in Run 2:", tags_2,
                             index=default_index(tags_2, campaign_2.played_tag),
                             format_func=tag_option, key="t4_tag2")

    render_comparison(
        Run(f"Run 1: {tag_label(tag_1)}", f"Run 1 ({tag_1})", TARGET, nation(df_1, tag_1)),
        Run(f"Run 2: {tag_label(tag_2)}", f"Run 2 ({tag_2})", RIVAL, nation(df_2, tag_2)),
        scope="Cross-Campaign", key="t4", range_label="Cross-Campaign Timeline Range:",
        range_col=controls[2], window_col=controls[3],
    )


# ======================================================================================
# App
# ======================================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("### 📂 Analyze Your Campaign")
        st.write("Upload your own `.csv` exports to analyze them instantly. "
                 "Files are processed in memory and never written to disk.")
        uploads = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)

        st.divider()
        st.markdown("### ☕ Support the Project")
        st.write("If this dashboard helped analyze your campaign, "
                 "consider supporting its development!")
        st.markdown(f"[**☕ Buy me a Coffee / Ko-fi**]({KOFI_URL})")

        st.divider()
        st.write("**Pix (Brazil):**")
        st.code(PIX_KEY, language="text")
    return uploads


def main() -> None:
    st.set_page_config(page_title="Victoria 3 Macroeconomics", layout="wide")
    st.title("🌍 Victoria 3 Macroeconomic Dashboard")

    campaigns = discover_campaigns(render_sidebar())
    if not campaigns:
        st.error("No CSV files found. Please upload a campaign file in the sidebar.")
        st.stop()

    tabs = st.tabs([
        "🌍 1. Global Landscape",
        "🔍 2. Single Economy Dossier",
        "⚔️ 3. Bilateral & Convergence",
        "📊 4. Cross-Campaign Meta-Analysis",
    ])
    for tab, render in zip(tabs, (render_global, render_dossier, render_bilateral, render_meta)):
        with tab:
            render(campaigns)


main()
