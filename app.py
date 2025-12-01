from __future__ import annotations

from flask import Flask, render_template, request, redirect, url_for, session, flash, abort
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from pathlib import Path
import mimetypes
import os
import re
import uuid
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sqlalchemy import text
from plotly.io import to_html
from plotly.utils import PlotlyJSONEncoder
import json

pio.renderers.default = 'iframe'
pio.templates.default = "plotly_white"

# Load .env file if available
load_dotenv()

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")
if MAPBOX_TOKEN:
    px.set_mapbox_access_token(MAPBOX_TOKEN)
else:
    # Map charts will gracefully degrade if no token is supplied
    os.environ.setdefault("MAPBOX_TOKEN", "")

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    raise RuntimeError("SECRET_KEY must be set for session security. Add it to your environment.")

UPLOAD_BASE = BASE_DIR / 'uploads'
UPLOAD_BASE.mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_BASE
app.config.setdefault('MAX_CONTENT_LENGTH', 25 * 1024 * 1024)

# Basic in-memory request throttle (per IP per route)
_rate_limits: dict[str, list[datetime]] = defaultdict(list)
RATE_LIMIT_WINDOW = timedelta(minutes=1)
RATE_LIMIT_COUNT = 30

# Database configuration with PostgreSQL preference and SQLite fallback
def _determine_database_uri():
    """Return the preferred database URI, falling back to SQLite if needed."""
    database_url = os.getenv('DATABASE_URL')

    if database_url:
        # SQLAlchemy expects the ``postgresql`` scheme. Some providers still
        # supply ``postgres`` so we normalise it here to avoid runtime errors.
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)

        return database_url

    sqlite_path = BASE_DIR / 'app.db'
    return f"sqlite:///{sqlite_path}"


app.config['SQLALCHEMY_DATABASE_URI'] = _determine_database_uri()
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


def _client_ip() -> str:
    return request.headers.get('X-Forwarded-For', request.remote_addr or 'unknown')


def _enforce_rate_limit(key: str) -> None:
    now = datetime.utcnow()
    window_start = now - RATE_LIMIT_WINDOW
    entries = [ts for ts in _rate_limits[key] if ts > window_start]
    entries.append(now)
    _rate_limits[key] = entries
    if len(entries) > RATE_LIMIT_COUNT:
        abort(429, description="Too many requests; please slow down.")


def _user_upload_dir(user_id: int) -> Path:
    path = UPLOAD_BASE / f"user_{user_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_upload_file(file) -> None:
    allowed_mimes = {
        'text/csv',
        'application/csv',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    }
    allowed_extensions = {'.csv', '.xlsx', '.xls'}

    mime_guess, _ = mimetypes.guess_type(file.filename)
    content_type = (file.mimetype or '').split(';')[0].strip()
    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise ValueError('Invalid file type. Please upload a CSV or Excel file.')

    if content_type and content_type not in allowed_mimes and mime_guess not in allowed_mimes:
        raise ValueError('Invalid file type. Please upload a CSV or Excel file.')


_GSHEET_PATTERN = re.compile(r"docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)")


def _google_sheet_export_url(url: str) -> str:
    match = _GSHEET_PATTERN.search(url)
    if not match:
        raise ValueError('Please provide a valid Google Sheets sharing link.')

    sheet_id = match.group(1)
    gid_match = re.search(r"gid=([0-9]+)", url)
    gid_suffix = f"&gid={gid_match.group(1)}" if gid_match else ""
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv{gid_suffix}"


def _load_google_sheet(url: str) -> pd.DataFrame:
    export_url = _google_sheet_export_url(url)
    df = pd.read_csv(export_url)
    if df.empty:
        raise ValueError('The provided Google Sheet is empty or could not be read.')
    return df


def _enforce_dataframe_limits(df: pd.DataFrame, max_rows: int = 150_000, max_cols: int = 150) -> pd.DataFrame:
    if df.shape[1] > max_cols:
        raise ValueError(f"CSV has too many columns ({df.shape[1]}). Please upload a file with <= {max_cols} columns.")
    if df.shape[0] > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    return df


def _ensure_unique_columns(columns) -> list[str]:
    seen: dict[str, int] = {}
    unique_cols: list[str] = []
    for col in columns:
        base = col
        if base in seen:
            seen[base] += 1
            col = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        unique_cols.append(col)
    return unique_cols


def _coerce_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce obvious datetime columns into datetime64 with safeguards."""

    datetime_hints = ('date', 'time', 'timestamp')
    string_like = df.select_dtypes(include=['object', 'string'])

    for column in string_like.columns:
        lowered = column.lower()
        if not any(hint in lowered for hint in datetime_hints):
            continue

        sample = string_like[column].dropna().astype(str).head(200)
        if sample.empty:
            continue

        parsed_sample = pd.to_datetime(
            sample,
            errors='coerce',
            cache=True,
            utc=True,
        )
        # Only coerce if the column is plausibly datetime to avoid noisy warnings
        if parsed_sample.notna().mean() >= 0.6:
            df[column] = pd.to_datetime(
                df[column],
                errors='coerce',
                cache=True,
                utc=True,
            )

    return df


def _analysis_subset(df: pd.DataFrame, max_rows: int = 20_000) -> pd.DataFrame:
    """Return a representative sample to keep downstream analysis responsive."""

    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


def infer_column_roles(df: pd.DataFrame) -> dict:
    """
    Inspect df and classify columns into semantic roles:

    - "time": true datetime columns.
    - "year": numeric columns with all 4-digit integers in [1800, 2100].
    - "metric": numeric columns with many distinct values (e.g. > 10).
    - "numeric_category": numeric columns with low cardinality (e.g. 2–20 unique values).
    - "category": string/object columns with low cardinality (2–50 unique values).
    - "id": high-cardinality string columns (likely identifiers).
    """
    roles = {
        "time": [],
        "year": [],
        "metric": [],
        "numeric_category": [],
        "category": [],
        "id": [],
        "dtypes": {},
        "cardinalities": {},
    }

    for col in df.columns:
        series = df[col]
        roles["dtypes"][col] = str(series.dtype)
        card = int(series.nunique(dropna=True))
        roles["cardinalities"][col] = card

        if pd.api.types.is_datetime64_any_dtype(series):
            roles["time"].append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if numeric_series.empty:
                continue
            is_int_like = numeric_series.round().eq(numeric_series).all()
            if is_int_like and numeric_series.between(1800, 2100).all() and numeric_series.nunique() >= 2:
                roles["year"].append(col)
                continue
            unique_numeric = numeric_series.nunique()
            if unique_numeric >= 10:
                roles["metric"].append(col)
            elif 2 <= unique_numeric <= 20:
                roles["numeric_category"].append(col)
            continue

        if series.dtype == 'object' or series.dtype.name == 'string':
            if 2 <= card <= 50:
                roles["category"].append(col)
            elif card > 50:
                roles["id"].append(col)

    return roles


def _basic_column_types(df: pd.DataFrame, *, max_category_cardinality: int = 50) -> dict[str, list[str]]:
    """Classify dataframe columns into numeric, datetime, and categorical buckets."""

    df = df.dropna(axis=1, how='all')
    numeric_cols: list[str] = []
    datetime_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        if non_null.empty:
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            coerced = pd.to_numeric(series, errors='coerce')
            if coerced.notna().sum() == 0:
                continue
            if coerced.nunique(dropna=True) <= max_category_cardinality:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
            continue

        if series.dtype == 'object' or series.dtype.name == 'string' or pd.api.types.is_bool_dtype(series):
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "datetime": datetime_cols,
        "categorical": categorical_cols,
    }


def _numeric_series_map(df: pd.DataFrame) -> dict[str, pd.Series]:
    numeric_series_map: dict[str, pd.Series] = {}
    for col in df.columns:
        series = pd.to_numeric(df[col], errors='coerce')
        if series.notna().sum() > 0:
            numeric_series_map[col] = series
    return numeric_series_map


def _select_primary_numeric(numeric_series_map: dict[str, pd.Series]) -> str | None:
    if not numeric_series_map:
        return None
    variances = {col: series.var(skipna=True) for col, series in numeric_series_map.items()}
    return max(variances, key=lambda c: (variances[c] if not pd.isna(variances[c]) else -np.inf), default=None)


def build_tableau_overview(df: pd.DataFrame) -> tuple[dict, list[str], list[str]]:
    """Construct a smarter overview profile, headline insights, and charts."""

    roles = infer_column_roles(df)
    numeric_series_map: dict[str, pd.Series] = {}
    for col in df.columns:
        series = pd.to_numeric(df[col], errors='coerce')
        if series.notna().sum() > 0:
            numeric_series_map[col] = series
    metric_cols = [c for c in roles["metric"] if c in numeric_series_map]
    category_cols = roles["category"] + roles["numeric_category"]
    time_cols = roles["time"]
    year_cols = roles["year"]

    primary_time_col = None
    if time_cols:
        primary_time_col = time_cols[0]
    elif year_cols:
        primary_time_col = year_cols[0]

    time_start = time_end = None
    if primary_time_col:
        if primary_time_col in year_cols:
            time_series = pd.to_datetime(pd.to_numeric(df[primary_time_col], errors='coerce'), format='%Y', errors='coerce', utc=True)
        else:
            time_series = pd.to_datetime(df[primary_time_col], errors='coerce', utc=True)
        time_series = time_series.dropna()
        if not time_series.empty:
            time_start = time_series.min()
            time_end = time_series.max()

    num_numeric = len(df.select_dtypes(include=['number']).columns)
    num_datetime = len(time_cols) + len(year_cols)
    num_categorical = len(category_cols)

    inferred_type = "unknown"
    if primary_time_col and metric_cols:
        inferred_type = "time-series"
    elif len(metric_cols) >= 2:
        inferred_type = "numeric-matrix"
    elif metric_cols and category_cols:
        inferred_type = "categorical-metric"
    elif category_cols:
        inferred_type = "categorical"

    dataset_summary = {
        "row_count": len(df),
        "col_count": len(df.columns),
        "num_numeric": num_numeric,
        "num_datetime": num_datetime,
        "num_categorical": num_categorical,
        "primary_time_col": primary_time_col,
        "time_start": time_start,
        "time_end": time_end,
        "inferred_dataset_type": inferred_type,
        "roles": roles,
    }

    overview_insights: list[str] = []
    if category_cols:
        cat = category_cols[0]
        vc = df[cat].value_counts(dropna=True)
        if not vc.empty:
            top_cat = vc.index[0]
            overview_insights.append(f"Most common category in {cat} is '{top_cat}' ({int(vc.iloc[0])} rows).")

    if metric_cols:
        ranges: dict[str, float] = {}
        for col in metric_cols:
            series = numeric_series_map.get(col, pd.Series(dtype=float)).dropna()
            if series.empty:
                continue
            ranges[col] = float(series.max() - series.min())
        if ranges:
            top_metric = max(ranges, key=ranges.get)
            overview_insights.append(f"Most volatile metric is {top_metric} (range {ranges[top_metric]:.2f}).")

    if len(metric_cols) >= 2:
        corr_df = pd.DataFrame({col: numeric_series_map[col] for col in metric_cols})
        corr = corr_df.corr()
        if not corr.empty:
            corr_values = corr.abs()
            mask = corr_values.where(~np.eye(corr_values.shape[0], dtype=bool))
            strongest = mask.stack().sort_values(ascending=False)
            if not strongest.empty and not pd.isna(strongest.iloc[0]):
                m1, m2 = strongest.index[0]
                val = float(strongest.iloc[0])
                overview_insights.append(
                    f"Strongest correlation: {m1} vs {m2} (r = {val:.2f})."
                )

    overview_charts_html: list[str] = []

    def _style_and_render(fig) -> None:
        fig.update_layout(
            margin=dict(l=40, r=20, t=60, b=40),
            title_x=0.0,
            font=dict(size=14)
        )
        overview_charts_html.append(to_html(fig, full_html=False, include_plotlyjs=False))

    if primary_time_col and metric_cols:
        time_df = df.copy()
        if primary_time_col in year_cols:
            time_df[primary_time_col] = pd.to_datetime(pd.to_numeric(time_df[primary_time_col], errors='coerce'), format='%Y', errors='coerce', utc=True)
        else:
            time_df[primary_time_col] = pd.to_datetime(time_df[primary_time_col], errors='coerce', utc=True)
        time_df = time_df.dropna(subset=[primary_time_col]).sort_values(primary_time_col)
        if not time_df.empty:
            metric_var = {
                col: numeric_series_map[col].loc[time_df.index].var(skipna=True) if col in numeric_series_map else -np.inf
                for col in metric_cols
            }
            top_metrics = [
                col for col, _ in sorted(
                    metric_var.items(),
                    key=lambda item: (item[1] if pd.notna(item[1]) else -np.inf),
                    reverse=True,
                )
                if pd.notna(metric_var.get(col, np.nan)) and metric_var.get(col, -np.inf) > -np.inf
            ][:3]
            for col in top_metrics:
                if col in numeric_series_map:
                    time_df[col] = numeric_series_map[col]
            if top_metrics:
                fig_line = px.line(
                    time_df,
                    x=primary_time_col,
                    y=top_metrics,
                    title="Key metrics over time",
                )
                _style_and_render(fig_line)

                if category_cols and top_metrics:
                    cat = category_cols[0]
                    temp_df = time_df[[primary_time_col, cat] + top_metrics].dropna(subset=[cat])
                    if not temp_df.empty:
                        melt_df = temp_df.melt(id_vars=[primary_time_col, cat], value_vars=top_metrics, var_name="Metric", value_name="Value")
                        grouped = melt_df.groupby([cat, "Metric"])["Value"].mean().reset_index()
                        if not grouped.empty:
                            fig_cat = px.bar(
                                grouped,
                                x=cat,
                                y="Value",
                                color="Metric",
                                title="Average metrics by category",
                            )
                            _style_and_render(fig_cat)
    elif len(metric_cols) >= 2:
        corr_df = pd.DataFrame({col: numeric_series_map[col] for col in metric_cols})
        corr = corr_df.corr().fillna(0)
        fig_corr = px.imshow(
            corr,
            text_auto='.2f',
            color_continuous_scale='Blues',
            title='Correlation matrix of metrics',
            labels={'color': 'Correlation'}
        )
        _style_and_render(fig_corr)

        metric_var = {col: numeric_series_map[col].var(skipna=True) for col in metric_cols}
        primary_metric = max(metric_var, key=lambda c: (metric_var[c] if not pd.isna(metric_var[c]) else -np.inf), default=None)
        if primary_metric and not pd.isna(metric_var.get(primary_metric, np.nan)):
            series = numeric_series_map[primary_metric].dropna()
            if not series.empty:
                fig_hist = px.histogram(series, x=series, nbins=30, title=f"Distribution of {primary_metric}")
                _style_and_render(fig_hist)
    elif metric_cols and category_cols:
        metric = metric_cols[0]
        cat = category_cols[0]
        temp_df = df[[cat]].copy()
        temp_df[metric] = numeric_series_map[metric]
        grouped = temp_df.groupby(cat)[metric].mean(numeric_only=True).reset_index()
        if not grouped.empty:
            fig_bar = px.bar(grouped, x=cat, y=metric, title=f"Average {metric} by {cat}")
            _style_and_render(fig_bar)
        counts = df[cat].value_counts().reset_index()
        counts.columns = [cat, 'Count']
        if not counts.empty:
            fig_count = px.bar(counts, x=cat, y='Count', title=f"Count by {cat}")
            _style_and_render(fig_count)
    elif category_cols:
        cat = category_cols[0]
        counts = df[cat].value_counts().reset_index()
        counts.columns = [cat, 'Count']
        if not counts.empty:
            fig_cat = px.bar(counts, x=cat, y='Count', title=f"Count by {cat}")
            _style_and_render(fig_cat)

    if not overview_charts_html and metric_cols:
        fallback_metric = metric_cols[0]
        series = numeric_series_map[fallback_metric].dropna()
        if not series.empty:
            fig_fallback = px.histogram(series, x=series, nbins=30, title=f"Distribution of {fallback_metric}")
            _style_and_render(fig_fallback)

    return dataset_summary, overview_insights, overview_charts_html


TRADING_CORE_COLUMNS = {
    'row_type',
    'symbol',
    'asset_class',
    'qty',
    'avg_entry_price',
    'current_price',
    'market_value',
    'unrealized_pl',
    'unrealized_plpc',
}

TRADING_OPTIONAL_COLUMNS = {
    'mode_or_strategy',
    'strategy_name',
    'realized_pl',
    'realized_plpc',
}


def classify_dataset(df: pd.DataFrame) -> str:
    """Classify the dataset to switch between generic and trading views."""

    lowered = {col.lower() for col in df.columns}
    if TRADING_CORE_COLUMNS.issubset(lowered):
        return "trading_paper_trades"
    return "generic"


def split_trading_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a trading CSV into account, position, and trade frames."""

    if 'row_type' not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    normalized = df.copy()
    normalized['row_type'] = normalized['row_type'].astype(str).str.lower()
    account_df = normalized[normalized['row_type'] == 'account_summary'].copy()
    positions_df = normalized[normalized['row_type'] == 'position'].copy()
    trades_df = normalized[normalized['row_type'] == 'trade'].copy()
    return account_df, positions_df, trades_df


def _load_dataframe(data_path: Path, max_rows: int = 150_000) -> pd.DataFrame:
    """Load a CSV or Excel file into a cleaned dataframe with limits and streaming."""

    suffix = data_path.suffix.lower()

    if suffix in {'.xlsx', '.xls'}:
        df = pd.read_excel(data_path)
    else:
        dfs = []
        rows_read = 0
        for chunk in pd.read_csv(data_path, chunksize=25_000, low_memory=False):
            dfs.append(chunk)
            rows_read += len(chunk)
            if rows_read >= max_rows:
                break
        df = pd.concat(dfs, ignore_index=True)

    df.columns = df.columns.str.strip()
    df.columns = _ensure_unique_columns(df.columns)
    df = _coerce_datetime_columns(df)
    df = _enforce_dataframe_limits(df, max_rows=max_rows)
    return df


def _summarise_dataframe(df: pd.DataFrame) -> dict[str, str]:
    """Generate human-readable insights for each column."""

    summary: dict[str, str] = {}

    for column in df.columns:
        col_data = df[column]

        if pd.api.types.is_numeric_dtype(col_data):
            non_missing = col_data.dropna()
            if non_missing.empty:
                summary[column] = "No numeric data available"
            else:
                summary[column] = (
                    f"Mean: {non_missing.mean():.2f}, "
                    f"Median: {non_missing.median():.2f}, "
                    f"Min: {non_missing.min()}, "
                    f"Max: {non_missing.max()}"
                )
        elif col_data.dtype == 'object' or col_data.dtype.name == 'category':
            non_missing = col_data.dropna()
            if non_missing.empty:
                summary[column] = "No values available"
            else:
                value_counts = non_missing.value_counts()
                top_value = value_counts.idxmax()
                top_count = value_counts.max()
                summary[column] = f"Most common: {top_value} ({top_count} times)"
        else:
            summary[column] = "No insight available"

    missing_data = df.isnull().sum()
    for column, count in missing_data.items():
        if count > 0:
            existing = summary.get(column, "")
            missing_message = f"Missing: {count}"
            summary[column] = f"{existing} | {missing_message}".strip(" |")

    return summary


def _data_quality_signals(df: pd.DataFrame) -> dict[str, str]:
    signals: dict[str, str] = {}
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        signals['duplicates'] = f"{duplicate_rows} duplicate rows detected"
    numeric_cols = df.select_dtypes(include=['number'])
    outlier_messages = []
    for col in numeric_cols.columns:
        series = numeric_cols[col].dropna()
        if len(series) < 10:
            continue
        zscores = ((series - series.mean()) / (series.std() or 1)).abs()
        outliers = int((zscores > 3).sum())
        if outliers:
            outlier_messages.append(f"{col}: {outliers} potential outliers")
    if outlier_messages:
        signals['outliers'] = '; '.join(outlier_messages)
    return signals


def _generate_additional_insights(df: pd.DataFrame) -> dict[str, str]:
    insights: dict[str, str] = {}
    dtypes = df.dtypes.apply(lambda x: str(x)).value_counts()
    insights['dataset shape'] = f"{len(df):,} rows × {df.shape[1]} columns"
    insights['column types'] = ', '.join(f"{dtype}: {count}" for dtype, count in dtypes.items())

    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls:
        percent_missing = 100 * total_nulls / (len(df) * max(len(df.columns), 1))
        top_missing = (
            null_counts[null_counts > 0]
            .sort_values(ascending=False)
            .head(3)
            .apply(lambda c: f"{c} ({c / len(df):.1%})")
        )
        top_missing_str = ', '.join(f"{col}: {val}" for col, val in top_missing.items())
        insights['missing data'] = (
            f"{total_nulls:,} missing values (~{percent_missing:.1f}% of the dataset); "
            f"top gaps in {top_missing_str}"
        )

    numeric_cols = df.select_dtypes(include=['number'])
    if numeric_cols.shape[1] >= 2:
        corr = numeric_cols.corr().abs()
        mask = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
        stacked = mask.stack()
        strong = stacked[stacked >= 0.65].sort_values(ascending=False).head(3)
        if not strong.empty:
            pairs = ', '.join(f"{a} vs {b} (ρ={val:.2f})" for (a, b), val in strong.items())
            insights['strong relationships'] = f"Notable correlations: {pairs}"

    categorical_cols = df.select_dtypes(include=['object', 'category'])
    if not categorical_cols.empty:
        top_categories: list[str] = []
        for col in categorical_cols.columns[:3]:
            freq = categorical_cols[col].value_counts(dropna=True)
            if not freq.empty:
                top = freq.iloc[0]
                top_categories.append(f"{col}: '{freq.index[0]}' appears {top} times ({top / len(df):.1%})")
        if top_categories:
            insights['dominant categories'] = '; '.join(top_categories)
    return insights


def _dataset_brief(df: pd.DataFrame) -> str:
    """Return a concise, human-friendly dataset synopsis."""

    parts = [f"{len(df):,} rows across {df.shape[1]} columns"]
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

    if len(numeric_cols) > 0:
        parts.append(f"{len(numeric_cols)} numeric fields for trending and aggregation")
    if len(categorical_cols) > 0:
        parts.append(f"{len(categorical_cols)} categorical fields for grouping and filters")
    if datetime_cols:
        parts.append("time-based fields available for timelines")

    if not parts:
        return "Upload data to see a quick synopsis."
    return "; ".join(parts)


def _chart_payload(title: str, description: str, fig) -> dict[str, str]:
    return {
        'title': title,
        'description': description,
        'html': fig.to_html(full_html=False),
    }


def _build_trading_visuals(df: pd.DataFrame) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    """Construct trading-specific visuals and supporting data health charts."""

    trading_charts: list[dict[str, str]] = []
    chart_notes: list[str] = []
    account_df, positions_df, trades_df = split_trading_frames(df)

    # Equity curve
    if not account_df.empty:
        value_col = None
        if 'equity' in account_df.columns:
            value_col = 'equity'
        elif 'portfolio_value' in account_df.columns:
            value_col = 'portfolio_value'

        if value_col and 'timestamp' in account_df.columns:
            account_df = account_df.copy()
            account_df['timestamp'] = pd.to_datetime(account_df['timestamp'], errors='coerce')
            account_df = account_df.dropna(subset=['timestamp', value_col]).sort_values('timestamp')
            if not account_df.empty:
                fig_equity = px.line(
                    account_df,
                    x='timestamp',
                    y=value_col,
                    title='Equity over time',
                )
                fig_equity.update_layout(yaxis_title='Equity' if value_col == 'equity' else 'Portfolio value')
                trading_charts.append(
                    _chart_payload(
                        'Equity over time',
                        'Tracks account equity based on account_summary rows in the CSV.',
                        fig_equity,
                    )
                )
            else:
                chart_notes.append('Skipped equity curve: timestamp or equity values were missing after cleaning.')
        else:
            chart_notes.append("Skipped equity curve: required columns 'timestamp' and 'equity'/'portfolio_value' not found.")

    # Open positions overview
    if not positions_df.empty:
        positions_df = positions_df.copy()
        if 'market_value' in positions_df.columns:
            positions_df['market_value'] = pd.to_numeric(positions_df['market_value'], errors='coerce')
        else:
            positions_df['market_value'] = np.nan
        if 'unrealized_pl' in positions_df.columns:
            positions_df['unrealized_pl'] = pd.to_numeric(positions_df['unrealized_pl'], errors='coerce')
        top_positions = positions_df.sort_values('market_value', ascending=False).head(20)
        if 'symbol' in top_positions.columns and top_positions['market_value'].notna().any():
            fig_positions = px.bar(
                top_positions,
                x='symbol',
                y='market_value',
                title='Open positions by market value',
                hover_data=['asset_class', 'qty', 'avg_entry_price', 'current_price', 'unrealized_pl', 'unrealized_plpc'],
            )
            fig_positions.update_layout(yaxis_title='Market value', xaxis_title='Symbol')
            trading_charts.append(
                _chart_payload(
                    'Open positions by market value',
                    'Shows where capital is allocated right now, sorted by position size.',
                    fig_positions,
                )
            )
        else:
            chart_notes.append('Skipped positions overview: missing symbol or market_value data.')

        if top_positions['unrealized_pl'].notna().any():
            fig_unrealized = px.bar(
                top_positions.sort_values('unrealized_pl', ascending=False),
                x='symbol',
                y='unrealized_pl',
                title='Unrealized P/L by symbol',
                color='unrealized_pl',
                color_continuous_scale='RdYlGn',
            )
            fig_unrealized.update_layout(yaxis_title='Unrealized P/L', xaxis_title='Symbol', coloraxis_showscale=False)
            trading_charts.append(
                _chart_payload(
                    'Unrealized P/L by symbol',
                    'Highlights winners and losers among open positions using unrealized profit/loss.',
                    fig_unrealized,
                )
            )
    else:
        chart_notes.append('Skipped positions overview: no position rows detected.')

    # Strategy performance breakdown
    if not trades_df.empty:
        trades_df = trades_df.copy()
        if 'realized_pl' in trades_df.columns:
            trades_df['realized_pl'] = pd.to_numeric(trades_df['realized_pl'], errors='coerce')
        strategy_field = 'strategy_name' if 'strategy_name' in trades_df.columns else 'mode_or_strategy'

        if strategy_field in trades_df.columns:
            grouped = trades_df.groupby(strategy_field)
            aggregated = grouped['realized_pl'].agg(['sum', 'count', 'mean']) if 'realized_pl' in trades_df.columns else None
            if aggregated is not None and aggregated['sum'].notna().any():
                aggregated = aggregated.rename(columns={'sum': 'total_realized_pl', 'count': 'trade_count', 'mean': 'avg_realized_pl'})
                fig_strategy = px.bar(
                    aggregated.reset_index(),
                    x=strategy_field,
                    y='total_realized_pl',
                    title='Strategy performance (realized P/L)',
                    hover_data=['trade_count', 'avg_realized_pl'],
                )
                fig_strategy.update_layout(yaxis_title='Total realized P/L', xaxis_title='Strategy')
                trading_charts.append(
                    _chart_payload(
                        'Strategy performance (realized P/L)',
                        'Aggregates realized profit and trade count per strategy or mode to reveal what is working.',
                        fig_strategy,
                    )
                )
            else:
                volume = grouped.size().reset_index(name='trade_count')
                fig_volume = px.bar(
                    volume,
                    x=strategy_field,
                    y='trade_count',
                    title='Trade volume by strategy',
                )
                fig_volume.update_layout(yaxis_title='Trade count', xaxis_title='Strategy')
                trading_charts.append(
                    _chart_payload(
                        'Trade volume by strategy',
                        'Counts trades per strategy when realized profit/loss is unavailable.',
                        fig_volume,
                    )
                )
        else:
            chart_notes.append('Skipped strategy performance: no strategy identifiers found.')

        # Symbol performance
        trades_df['symbol'] = trades_df.get('symbol')
        if 'symbol' in trades_df.columns:
            symbol_grouped = trades_df.groupby('symbol')
            symbol_perf = symbol_grouped['realized_pl'].agg(['sum', 'count']) if 'realized_pl' in trades_df.columns else None
            if symbol_perf is not None and symbol_perf['sum'].notna().any():
                symbol_perf = symbol_perf.rename(columns={'sum': 'total_realized_pl', 'count': 'trade_count'})
                top_symbols = symbol_perf.reindex(symbol_perf['total_realized_pl'].abs().sort_values(ascending=False).head(20).index)
                fig_symbol = px.bar(
                    top_symbols.reset_index(),
                    x='symbol',
                    y='total_realized_pl',
                    title='Symbol performance (realized P/L)',
                    color='total_realized_pl',
                    color_continuous_scale='RdYlGn',
                )
                fig_symbol.update_layout(yaxis_title='Total realized P/L', xaxis_title='Symbol', coloraxis_showscale=False)
                trading_charts.append(
                    _chart_payload(
                        'Symbol performance (realized P/L)',
                        'Ranks symbols by realized profit/loss to spotlight the biggest contributors.',
                        fig_symbol,
                    )
                )
            else:
                trade_counts = symbol_grouped.size().reset_index(name='trade_count')
                fig_symbol_volume = px.bar(
                    trade_counts.sort_values('trade_count', ascending=False).head(20),
                    x='symbol',
                    y='trade_count',
                    title='Trade volume by symbol',
                )
                fig_symbol_volume.update_layout(yaxis_title='Trade count', xaxis_title='Symbol')
                trading_charts.append(
                    _chart_payload(
                        'Trade volume by symbol',
                        'Counts trades per symbol when realized profit/loss is unavailable.',
                        fig_symbol_volume,
                    )
                )
        else:
            chart_notes.append('Skipped symbol performance: symbol column not found in trades.')

        # Trade timeline
        if 'timestamp' in trades_df.columns and 'side' in trades_df.columns:
            price_candidates = [
                col for col in ['filled_avg_price', 'avg_entry_price', 'current_price'] if col in trades_df.columns
            ]
            price_col = price_candidates[0] if price_candidates else None
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], errors='coerce')
            if price_col:
                timeline_df = trades_df.dropna(subset=['timestamp', price_col, 'side']).sort_values('timestamp')
                if not timeline_df.empty:
                    fig_timeline = px.scatter(
                        timeline_df,
                        x='timestamp',
                        y=price_col,
                        color='side',
                        title='Trade timeline',
                        symbol='side',
                        hover_data=['symbol', 'qty', price_col],
                    )
                    fig_timeline.update_layout(xaxis_title='Timestamp', yaxis_title=price_col.replace('_', ' ').title())
                    trading_charts.append(
                        _chart_payload(
                            'Trade timeline',
                            'Plots trade timestamps and prices to reveal entry/exit timing by side.',
                            fig_timeline,
                        )
                    )
                else:
                    chart_notes.append('Skipped trade timeline: missing timestamp or price data after cleaning.')
            else:
                chart_notes.append('Skipped trade timeline: no price column (filled_avg_price/avg_entry_price/current_price) available.')
        else:
            chart_notes.append("Skipped trade timeline: required columns 'timestamp' and 'side' not found.")
    else:
        chart_notes.append('Skipped trading visuals: no trade rows detected.')

    data_health_charts, health_notes = _build_data_health_charts(df, 'trading_paper_trades')
    chart_notes.extend(health_notes)
    return trading_charts, data_health_charts, chart_notes


def _build_generic_visuals(df: pd.DataFrame) -> tuple[list[dict[str, str]], list[str]]:
    charts, notes = _build_data_health_charts(df, 'generic')
    return charts, notes


def _prepare_visual_context(df: pd.DataFrame) -> tuple[str, list[dict[str, str]], list[dict[str, str]], list[str]]:
    dataset_type = classify_dataset(df)
    if dataset_type == 'trading_paper_trades':
        trading_charts, data_health_charts, chart_notes = _build_trading_visuals(df)
    else:
        trading_charts = []
        data_health_charts, chart_notes = _build_generic_visuals(df)
    return dataset_type, trading_charts, data_health_charts, chart_notes


def _build_data_health_charts(df: pd.DataFrame, dataset_type: str) -> tuple[list[dict[str, str]], list[str]]:
    charts: list[dict[str, str]] = []
    notes: list[str] = []
    numeric_cols = df.select_dtypes(include=['number'])

    corr_condition = (
        numeric_cols.shape[1] >= 3 and len(df) > 10
        if dataset_type == 'trading_paper_trades'
        else numeric_cols.shape[1] >= 2
    )
    if corr_condition:
        corr = numeric_cols.corr().fillna(0)
        fig_corr = px.imshow(
            corr,
            text_auto='.2f',
            color_continuous_scale='Blues',
            title='Correlation matrix of numeric columns',
            labels={'color': 'Correlation'}
        )
        charts.append(
            _chart_payload(
                'Correlation matrix of numeric columns',
                'Shows how numeric fields move together; strong values hint at drivers of performance or redundant fields.',
                fig_corr,
            )
        )
    else:
        notes.append('Skipped correlation matrix: insufficient numeric columns or rows to compute reliable correlations.')

    null_matrix = df.isnull()
    if null_matrix.values.any():
        fig_null = px.imshow(
            null_matrix.astype(int),
            color_continuous_scale=[[0, 'rgb(0,123,255)'], [1, 'rgb(220,53,69)']],
            title='Null heatmap by column',
            labels={'color': 'Null (1=yes)'}
        )
        charts.append(
            _chart_payload(
                'Null heatmap by column',
                'Highlights where values are missing so you can judge data reliability before charting.',
                fig_null,
            )
        )
    else:
        notes.append('Skipped null heatmap: no missing values detected.')

    return charts, notes


def _render_preview_table(df: pd.DataFrame) -> tuple[str, list[str]]:
    preview_df = df.head(100)
    table_html = preview_df.to_html(classes='table table-striped table-bordered', index=False)
    columns = df.columns.tolist()
    return table_html, columns


def _save_profile(upload: Upload, df: pd.DataFrame) -> None:
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    null_summary = df.isnull().sum().to_dict()
    quality_notes = _data_quality_signals(df)
    profile = DatasetProfile(
        upload_id=upload.id,
        schema=json.dumps(schema),
        null_summary=json.dumps(null_summary),
        quality_notes=json.dumps(quality_notes)
    )
    db.session.add(profile)


def _schema_drift_message(user_id: int, original_filename: str, current_schema: dict[str, str], *, exclude_upload_id: int | None = None) -> str | None:
    last_upload = (
        Upload.query
        .filter(Upload.user_id == user_id)
        .filter(Upload.filename.endswith(f"__{original_filename}"))
        .filter(Upload.id != exclude_upload_id)
        .order_by(Upload.timestamp.desc())
        .first()
    )
    if not last_upload:
        return None
    profile = DatasetProfile.query.filter_by(upload_id=last_upload.id).order_by(DatasetProfile.created_at.desc()).first()
    if not profile:
        return None
    previous_schema = json.loads(profile.schema)
    added = set(current_schema) - set(previous_schema)
    removed = set(previous_schema) - set(current_schema)
    changes = []
    if added:
        changes.append(f"New columns: {', '.join(sorted(added))}")
    if removed:
        changes.append(f"Removed columns: {', '.join(sorted(removed))}")
    if not changes:
        for col, dtype in current_schema.items():
            if col in previous_schema and previous_schema[col] != dtype:
                changes.append(f"Column {col} type changed from {previous_schema[col]} to {dtype}")
    if changes:
        return '; '.join(changes)
    return None

def build_generated_chart(df: pd.DataFrame) -> tuple[str, str | None, str | None, str | None]:
    """Build a sensible default Plotly chart for the dashboard hero slot.

    Returns a tuple of (chart_html, figure_json, x_column, y_column).
    """

    if df is None or df.empty:
        return "<div class='text-muted'>No suitable data for chart.</div>", None, None, None

    df = df.dropna(axis=1, how='all')
    if df.empty:
        return "<div class='text-muted'>No suitable data for chart.</div>", None, None, None

    column_types = _basic_column_types(df)
    numeric_map = _numeric_series_map(df)
    numeric_cols = list(numeric_map.keys())
    datetime_cols = column_types["datetime"]
    categorical_cols = column_types["categorical"]

    def _safe_payload(fig, x_col: str | None = None, y_col: str | None = None):
        fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
        return (
            fig.to_html(full_html=False),
            json.dumps(fig, cls=PlotlyJSONEncoder),
            x_col,
            y_col,
        )

    if datetime_cols and numeric_cols:
        time_col = datetime_cols[0]
        y_col = _select_primary_numeric(numeric_map) or numeric_cols[0]
        time_df = df[[time_col]].copy()
        time_df[time_col] = pd.to_datetime(time_df[time_col], errors='coerce', utc=True)
        time_df[y_col] = numeric_map[y_col]
        time_df = time_df.dropna(subset=[time_col, y_col]).sort_values(time_col)
        if not time_df.empty:
            fig = px.line(time_df, x=time_col, y=y_col, title=f"Trend of {y_col} over time")
            return _safe_payload(fig, x_col=time_col, y_col=y_col)

    if len(numeric_cols) == 1 and categorical_cols:
        metric = numeric_cols[0]
        cat = categorical_cols[0]
        work_df = df[[cat]].copy()
        work_df[metric] = numeric_map[metric]
        grouped = work_df.groupby(cat)[metric].mean(numeric_only=True).reset_index()
        if grouped.empty:
            return "<div class='text-muted'>No suitable data for chart.</div>"
        grouped = grouped.sort_values(metric, ascending=False)
        if grouped[cat].nunique(dropna=True) > 50:
            grouped = grouped.head(20)
        fig = px.bar(grouped, x=cat, y=metric, title=f"Average {metric} by {cat}")
        return _safe_payload(fig, x_col=cat, y_col=metric)

    if len(numeric_cols) == 1:
        metric = numeric_cols[0]
        series = numeric_map[metric].dropna()
        if not series.empty:
            fig = px.histogram(series, x=series, nbins=30, title=f"Distribution of {metric}")
            return _safe_payload(fig, x_col=metric)

    if len(numeric_cols) >= 2:
        corr_df = pd.DataFrame({col: numeric_map[col] for col in numeric_cols})
        corr = corr_df.corr().fillna(0)
        if not corr.empty:
            fig = px.imshow(
                corr,
                text_auto='.2f',
                color_continuous_scale='Blues',
                title='Correlation heatmap',
                labels={'color': 'Correlation'},
            )
            return _safe_payload(fig)

    if categorical_cols:
        cat = categorical_cols[0]
        counts = df[cat].value_counts(dropna=True).head(20).reset_index()
        if not counts.empty:
            counts.columns = [cat, 'Count']
            fig = px.bar(counts, x=cat, y='Count', title=f"Counts of {cat}")
            return _safe_payload(fig, x_col=cat)

    return "<div class='text-muted'>No suitable data for chart.</div>", None, None, None



def generate_chart_suggestions(df, max_suggestions=10):
    suggestions = []
    df = df.dropna(axis=1, how='all')
    if df.empty:
        return [{
            "title": "No suitable data for chart.",
            "chart_type": None,
            "x": None,
            "y": None,
            "score": 0,
        }]

    column_types = _basic_column_types(df)
    numeric_series_map = _numeric_series_map(df)
    numeric_cols = list(numeric_series_map.keys())
    datetime_cols = column_types["datetime"]
    categorical_cols = column_types["categorical"]

    if datetime_cols and numeric_cols:
        time_col = datetime_cols[0]
        metric = _select_primary_numeric(numeric_series_map) or numeric_cols[0]
        time_df = df[[time_col]].copy()
        time_df[time_col] = pd.to_datetime(time_df[time_col], errors='coerce', utc=True)
        time_df[metric] = numeric_series_map[metric]
        time_df = time_df.dropna(subset=[time_col, metric])
        if not time_df.empty:
            variability = float(time_df[metric].var(skipna=True) or 0)
            suggestions.append({
                "title": f"Trend of {metric} over {time_col}",
                "chart_type": "line",
                "x": time_col,
                "y": metric,
                "score": variability + 1.0,
            })

    if len(numeric_cols) >= 2:
        corr_df = pd.DataFrame({col: numeric_series_map[col] for col in numeric_cols})
        corr = corr_df.corr()
        if not corr.empty:
            strongest_pair = corr.abs().where(~np.eye(len(corr), dtype=bool)).stack().sort_values(ascending=False)
            if not strongest_pair.empty:
                c1, c2 = strongest_pair.index[0]
                corr_val = float(strongest_pair.iloc[0])
                suggestions.append({
                    "title": "Correlation heatmap for numeric columns",
                    "chart_type": "heatmap",
                    "x": c1,
                    "y": c2,
                    "score": corr_val + 0.8,
                })
                suggestions.append({
                    "title": f"Scatter: {c1} vs {c2} (r={corr_val:.2f})",
                    "chart_type": "scatter",
                    "x": c1,
                    "y": c2,
                    "score": corr_val + 0.6,
                })

    if len(numeric_cols) == 1 and categorical_cols:
        metric = numeric_cols[0]
        cat = categorical_cols[0]
        work_df = df[[cat]].copy()
        work_df[metric] = numeric_series_map[metric]
        grouped = work_df.groupby(cat)[metric].mean(numeric_only=True)
        if not grouped.empty:
            variability = float(grouped.std() / (abs(grouped.mean()) + 1e-9)) if grouped.mean() != 0 else float(grouped.std())
            suggestions.append({
                "title": f"Average {metric} by {cat}",
                "chart_type": "bar",
                "x": cat,
                "y": metric,
                "score": variability + 0.5,
            })

    if len(numeric_cols) == 1 and not categorical_cols:
        metric = numeric_cols[0]
        suggestions.append({
            "title": f"Distribution of {metric}",
            "chart_type": "histogram",
            "x": metric,
            "y": None,
            "score": float(abs(numeric_series_map[metric].var(skipna=True) or 0) + 0.4),
        })

    if not numeric_cols and categorical_cols:
        cat = categorical_cols[0]
        if df[cat].nunique(dropna=True) > 1:
            suggestions.append({
                "title": f"Counts of top {cat} values",
                "chart_type": "bar",  # handled as count plot
                "x": cat,
                "y": None,
                "score": 0.3,
            })

    if not suggestions:
        fallback_message = (
            "No strong chart suggestions found. Try exploring columns manually "
            "or adjust thresholds."
        )
        return [{
            "title": fallback_message,
            "chart_type": None,
            "x": None,
            "y": None,
            "score": 0,
        }]

    suggestions = sorted(suggestions, key=lambda s: s["score"], reverse=True)
    return suggestions[:max_suggestions]

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(300), nullable=False, unique=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    @property
    def original_filename(self) -> str:
        if '__' in self.filename:
            return self.filename.split('__', 1)[1]
        return self.filename


class SavedChart(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    upload_id = db.Column(db.Integer, db.ForeignKey('upload.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    chart_type = db.Column(db.String(50), nullable=True)
    x_column = db.Column(db.String(255), nullable=True)
    y_column = db.Column(db.String(255), nullable=True)
    filter_column = db.Column(db.String(255), nullable=True)
    filter_value = db.Column(db.String(255), nullable=True)
    sample_fraction = db.Column(db.Float, nullable=True)
    figure_json = db.Column(db.Text, nullable=True)
    chart_html = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    shared_token = db.Column(db.String(64), unique=True, nullable=True)


class DatasetProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    upload_id = db.Column(db.Integer, db.ForeignKey('upload.id'), nullable=False)
    schema = db.Column(db.Text, nullable=False)
    null_summary = db.Column(db.Text, nullable=True)
    quality_notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


def _render_saved_chart_html(saved_chart: SavedChart) -> str:
    if saved_chart.figure_json:
        try:
            fig = pio.from_json(saved_chart.figure_json)
            return fig.to_html(full_html=False, include_plotlyjs=False)
        except Exception:
            pass
    return saved_chart.chart_html


def _prepare_saved_chart_views(saved_charts: list[SavedChart]) -> list[dict]:
    views: list[dict] = []
    for chart in saved_charts:
        views.append({
            "id": chart.id,
            "title": chart.title,
            "created_at": chart.created_at,
            "shared_token": chart.shared_token,
            "chart_type": chart.chart_type,
            "x_column": chart.x_column,
            "y_column": chart.y_column,
            "filter_column": chart.filter_column,
            "filter_value": chart.filter_value,
            "sample_fraction": chart.sample_fraction,
            "html": _render_saved_chart_html(chart),
        })
    return views


def ensure_saved_chart_schema() -> None:
    statements = [
        text('ALTER TABLE saved_chart ADD COLUMN IF NOT EXISTS chart_type VARCHAR(50);'),
        text('ALTER TABLE saved_chart ADD COLUMN IF NOT EXISTS x_column VARCHAR(255);'),
        text('ALTER TABLE saved_chart ADD COLUMN IF NOT EXISTS y_column VARCHAR(255);'),
        text('ALTER TABLE saved_chart ADD COLUMN IF NOT EXISTS filter_column VARCHAR(255);'),
        text('ALTER TABLE saved_chart ADD COLUMN IF NOT EXISTS filter_value VARCHAR(255);'),
        text('ALTER TABLE saved_chart ADD COLUMN IF NOT EXISTS sample_fraction DOUBLE PRECISION;'),
        text('ALTER TABLE saved_chart ADD COLUMN IF NOT EXISTS figure_json TEXT;'),
        text(
            'ALTER TABLE saved_chart '
            "ADD COLUMN IF NOT EXISTS chart_html TEXT NOT NULL DEFAULT '';"
        ),
        text('ALTER TABLE saved_chart ADD COLUMN IF NOT EXISTS shared_token VARCHAR(64) UNIQUE;'),
    ]

    with db.engine.begin() as connection:
        for statement in statements:
            connection.execute(statement)


with app.app_context():
    db.create_all()
    ensure_saved_chart_schema()

# Routes
@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.template_filter('file_exists')
def file_exists_filter(filename):
    user_id = session.get('user_id')
    if not user_id:
        return False
    return (_user_upload_dir(user_id) / filename).is_file()

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    _enforce_rate_limit(f"signup:{_client_ip()}")
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('User already exists')
            return render_template('signup.html')

        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully. Please log in.')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    _enforce_rate_limit(f"login:{_client_ip()}")
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['user'] = user.username
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))

        flash('Invalid username or password')
        return render_template('login.html')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    filename = request.args.get('filename')
    uploads = (
        Upload.query
        .filter_by(user_id=session['user_id'])
        .order_by(Upload.timestamp.desc())
        .all()
    )
    table_html = None
    columns: list[str] = []
    summary = None
    dataset_synopsis = None
    suggestions = None
    dataset_summary: dict = {}
    overview_insights: list[str] = []
    overview_charts_html: list[str] = []
    chart_html: str | None = None
    generated_figure_json: str | None = None
    generated_x_column: str | None = None
    generated_y_column: str | None = None
    saved_chart_views: list[dict] = []

    selected_upload = None
    if not filename and uploads:
        selected_upload = uploads[0]
        filename = selected_upload.filename
    elif filename:
        selected_upload = Upload.query.filter_by(filename=filename, user_id=session['user_id']).first()

    if selected_upload:
        saved_chart_views = _prepare_saved_chart_views(
            SavedChart.query
            .filter_by(user_id=session['user_id'], upload_id=selected_upload.id)
            .order_by(SavedChart.created_at.desc())
            .all()
        )

    dataset_type = 'generic'
    trading_charts: list[dict[str, str]] = []
    data_health_charts: list[dict[str, str]] = []
    chart_notes: list[str] = []
    quality_signals = {}
    extra_insights = {}

    if filename:
        filepath = _user_upload_dir(session['user_id']) / filename
        if filepath.exists():
            try:
                df = _load_dataframe(filepath)
                dataset_summary, overview_insights, overview_charts_html = build_tableau_overview(df)
                analysis_df = _analysis_subset(df)
                table_html, columns = _render_preview_table(df)
                summary = _summarise_dataframe(analysis_df)
                suggestions = generate_chart_suggestions(analysis_df)
                dataset_type, trading_charts, data_health_charts, chart_notes = _prepare_visual_context(analysis_df)
                quality_signals = _data_quality_signals(analysis_df)
                extra_insights = _generate_additional_insights(analysis_df)
                dataset_synopsis = _dataset_brief(analysis_df)
                (
                    chart_html,
                    generated_figure_json,
                    generated_x_column,
                    generated_y_column,
                ) = build_generated_chart(analysis_df)
            except Exception as e:
                flash(f"Error loading file '{filename}': {e}")
        else:
            flash('Selected file not found on disk')

    suggestions = suggestions or []

    return render_template(
        'dashboard.html',
        user=session['user'],
        table=table_html,
        columns=columns,
        uploads=uploads,
        selected_file=filename,
        summary=summary,
        dataset_synopsis=dataset_synopsis,
        suggestions=suggestions,
        dataset_type=dataset_type,
        trading_charts=trading_charts,
        data_health_charts=data_health_charts,
        chart_notes=chart_notes,
        quality_signals=quality_signals,
        extra_insights=extra_insights,
        saved_charts=saved_chart_views,
        dataset_summary=dataset_summary,
        overview_insights=overview_insights,
        overview_charts_html=overview_charts_html,
        chart=chart_html,
        generated_chart_html=chart_html,
        generated_figure_json=generated_figure_json,
        generated_x_column=generated_x_column,
        generated_y_column=generated_y_column,
        current_upload=selected_upload,
    )

@app.route('/upload', methods=['POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    _enforce_rate_limit(f"upload:{_client_ip()}")

    file = request.files.get('data_file')
    google_sheet_url = request.form.get('gsheet_url', '').strip()

    if (not file or file.filename == '') and not google_sheet_url:
        flash('Please upload a file or provide a Google Sheets link.')
        return redirect(url_for('dashboard'))

    user_dir = _user_upload_dir(session['user_id'])

    df: pd.DataFrame
    safe_name: str
    stored_name: str
    filepath: Path
    dataset_summary: dict = {}
    overview_insights: list[str] = []
    overview_charts_html: list[str] = []
    chart_html: str | None = None

    if google_sheet_url:
        try:
            df = _load_google_sheet(google_sheet_url)
        except Exception as exc:
            flash(f'Google Sheets import failed: {exc}')
            return redirect(url_for('dashboard'))

        safe_name = secure_filename(f"google-sheet-{datetime.utcnow().strftime('%Y%m%d')}.csv") or 'google-sheet.csv'
        stored_name = f"{uuid.uuid4().hex}__{safe_name}"
        filepath = user_dir / stored_name
        df.to_csv(filepath, index=False)
        df = _load_dataframe(filepath)
    else:
        try:
            _validate_upload_file(file)
        except Exception as exc:
            flash(str(exc))
            return redirect(url_for('dashboard'))

        safe_name = secure_filename(file.filename)
        if not safe_name:
            flash('Invalid filename')
            return redirect(url_for('dashboard'))

        stored_name = f"{uuid.uuid4().hex}__{safe_name}"
        filepath = user_dir / stored_name
        file.save(filepath)

        try:
            df = _load_dataframe(filepath)
        except Exception:
            filepath.unlink(missing_ok=True)
            raise

    new_upload = Upload(filename=stored_name, user_id=session['user_id'])
    db.session.add(new_upload)

    try:
        dataset_summary, overview_insights, overview_charts_html = build_tableau_overview(df)
        analysis_df = _analysis_subset(df)
        summary = _summarise_dataframe(analysis_df)
        suggestions = generate_chart_suggestions(analysis_df)
        dataset_type, trading_charts, data_health_charts, chart_notes = _prepare_visual_context(analysis_df)
        table_html, columns = _render_preview_table(df)
        (
            chart_html,
            generated_figure_json,
            generated_x_column,
            generated_y_column,
        ) = build_generated_chart(analysis_df)
        db.session.commit()
        _save_profile(new_upload, analysis_df)
        db.session.commit()
        drift = _schema_drift_message(
            session['user_id'],
            safe_name,
            {col: str(dtype) for col, dtype in analysis_df.dtypes.items()},
            exclude_upload_id=new_upload.id
        )
        if drift:
            flash(f"Schema drift detected: {drift}")
    except Exception as e:
        db.session.rollback()
        filepath.unlink(missing_ok=True)
        flash(f'Error processing file: {e}')
        return redirect(url_for('dashboard'))

    uploads = (
        Upload.query
        .filter_by(user_id=session['user_id'])
        .order_by(Upload.timestamp.desc())
        .all()
    )

    suggestions = suggestions or []

    dataset_synopsis = _dataset_brief(analysis_df)

    return render_template(
        'dashboard.html',
        user=session['user'],
        table=table_html,
        columns=columns,
        uploads=uploads,
        summary=summary,
        dataset_synopsis=dataset_synopsis,
        selected_file=stored_name,
        suggestions=suggestions,
        dataset_type=dataset_type,
        trading_charts=trading_charts,
        data_health_charts=data_health_charts,
        chart_notes=chart_notes,
        quality_signals=_data_quality_signals(analysis_df),
        extra_insights=_generate_additional_insights(analysis_df),
        saved_charts=_prepare_saved_chart_views(
            SavedChart.query
            .filter_by(user_id=session['user_id'], upload_id=new_upload.id)
            .order_by(SavedChart.created_at.desc())
            .all()
        ),
        dataset_summary=dataset_summary,
        overview_insights=overview_insights,
        overview_charts_html=overview_charts_html,
        chart=chart_html,
        generated_chart_html=chart_html,
        generated_figure_json=generated_figure_json,
        generated_x_column=generated_x_column,
        generated_y_column=generated_y_column,
        current_upload=new_upload,
    )

@app.route('/visualize', methods=['POST'])
def visualize():
    if 'user' not in session:
        return redirect(url_for('login'))

    _enforce_rate_limit(f"visualize:{_client_ip()}")

    filename = request.form.get('filename')
    x_column = request.form.get('x_column')
    y_column = request.form.get('y_column', '')
    chart_type = request.form.get('chart_type')
    filter_column = request.form.get('filter_column')
    filter_value = request.form.get('filter_value')
    sample_fraction = request.form.get('sample_fraction')
    save_title = request.form.get('save_title')
    shareable = request.form.get('shareable') == 'on'
    dataset_summary: dict = {}
    overview_insights: list[str] = []
    overview_charts_html: list[str] = []
    generated_chart_html: str | None = None
    generated_figure_json: str | None = None
    generated_x_column: str | None = None
    generated_y_column: str | None = None

    if not filename or not x_column or not chart_type:
        flash("Missing form data")
        return redirect(url_for('dashboard'))
    needs_y = chart_type not in {'pie', 'histogram', 'heatmap'} and not (chart_type == 'bar' and not y_column)
    if needs_y and not y_column:
        flash('Y axis is required for this chart type')
        return redirect(url_for('dashboard'))

    upload = Upload.query.filter_by(filename=filename, user_id=session['user_id']).first()
    filepath = _user_upload_dir(session['user_id']) / filename
    if not filepath.exists():
        flash("Selected file not found")
        return redirect(url_for('dashboard'))

    try:
        df = _load_dataframe(filepath)

        sample_fraction_value: float | None = None

        if filter_column and filter_column in df.columns and filter_value:
            mask = df[filter_column].astype(str).str.contains(re.escape(filter_value), case=False, na=False)
            df = df[mask]
        if sample_fraction:
            try:
                frac = float(sample_fraction)
                if 0 < frac < 1:
                    df = df.sample(frac=frac, random_state=42)
                    sample_fraction_value = frac
            except ValueError:
                flash('Invalid sample fraction; ignoring.')

        # Chart rendering logic
        if chart_type == 'pie':
            value_counts = df[x_column].value_counts().reset_index()
            value_counts.columns = [x_column, 'Count']
            fig = px.pie(value_counts, names=x_column, values='Count')
        elif chart_type == 'bar':
            if y_column:
                fig = px.bar(df, x=x_column, y=y_column)
            else:
                counts = df[x_column].value_counts(dropna=True).head(20).reset_index()
                if counts.empty:
                    raise ValueError('No suitable data for bar chart.')
                counts.columns = [x_column, 'Count']
                fig = px.bar(counts, x=x_column, y='Count')
        elif chart_type == 'line':
            fig = px.line(df.sort_values(by=x_column), x=x_column, y=y_column)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_column, y=y_column)
        elif chart_type == 'histogram':
            series = pd.to_numeric(df[x_column], errors='coerce').dropna()
            if series.empty:
                raise ValueError('No numeric data available for histogram.')
            fig = px.histogram(series, x=series, nbins=30, title=f"Distribution of {x_column}")
        elif chart_type == 'heatmap':
            numeric_df = df.select_dtypes(include=['number']).apply(pd.to_numeric, errors='coerce')
            corr = numeric_df.corr().fillna(0)
            if corr.empty:
                raise ValueError('No numeric columns available for heatmap.')
            fig = px.imshow(
                corr,
                text_auto='.2f',
                color_continuous_scale='Blues',
                title='Correlation heatmap',
                labels={'color': 'Correlation'},
            )
        elif chart_type == 'scatter_map':
            if not MAPBOX_TOKEN:
                flash('Mapbox token missing; map charts may not render correctly.')
            if df[x_column].dtype.kind in 'iuf' and df[y_column].dtype.kind in 'iuf':
                df = df[(df[y_column].between(-90, 90)) & (df[x_column].between(-180, 180))]
            else:
                raise ValueError('Selected latitude/longitude columns must be numeric.')
            fig = px.scatter_mapbox(
                df,
                lat=y_column,
                lon=x_column,
                hover_name='city' if 'city' in df.columns else None,
                zoom=1,
                height=500,
                mapbox_style='carto-positron'
            )
        elif chart_type == 'density_map':
            if not MAPBOX_TOKEN:
                flash('Mapbox token missing; map charts may not render correctly.')
            if df[x_column].dtype.kind in 'iuf' and df[y_column].dtype.kind in 'iuf':
                df = df[(df[y_column].between(-90, 90)) & (df[x_column].between(-180, 180))]
            else:
                raise ValueError('Selected latitude/longitude columns must be numeric.')
            fig = px.density_mapbox(
                df,
                lat=y_column,
                lon=x_column,
                z='population' if 'population' in df.columns else None,
                radius=10,
                zoom=1,
                height=500,
                mapbox_style='carto-positron'
            )
        else:
            flash("Invalid chart type")
            return redirect(url_for('dashboard'))

        chart_html = fig.to_html(full_html=False)
        figure_json = json.dumps(fig, cls=PlotlyJSONEncoder)

        if save_title:
            token = uuid.uuid4().hex if shareable else None
            if not upload:
                raise ValueError('Upload not found for saving chart')
            saved = SavedChart(
                user_id=session['user_id'],
                upload_id=upload.id,
                title=save_title,
                chart_type=chart_type,
                x_column=x_column,
                y_column=y_column or None,
                filter_column=filter_column,
                filter_value=filter_value,
                sample_fraction=sample_fraction_value,
                figure_json=figure_json,
                chart_html=chart_html,
                shared_token=token
            )
            db.session.add(saved)
            db.session.commit()

        # Update suggestions, preview, summary
        dataset_summary, overview_insights, overview_charts_html = build_tableau_overview(df)
        analysis_df = _analysis_subset(df)
        (
            generated_chart_html,
            generated_figure_json,
            generated_x_column,
            generated_y_column,
        ) = build_generated_chart(analysis_df)
        suggestions = generate_chart_suggestions(analysis_df)
        dataset_type, trading_charts, data_health_charts, chart_notes = _prepare_visual_context(analysis_df)
        table_html, columns = _render_preview_table(df)
        summary = _summarise_dataframe(analysis_df)
        dataset_synopsis = _dataset_brief(analysis_df)

        return render_template(
            'dashboard.html',
            user=session['user'],
            table=table_html,
            columns=columns,
            uploads=(
                Upload.query
                .filter_by(user_id=session['user_id'])
                .order_by(Upload.timestamp.desc())
                .all()
            ),
            selected_file=filename,
            chart=chart_html,
            summary=summary,
            dataset_synopsis=dataset_synopsis,
            suggestions=suggestions,
            dataset_type=dataset_type,
            trading_charts=trading_charts,
            data_health_charts=data_health_charts,
            chart_notes=chart_notes,
            quality_signals=_data_quality_signals(analysis_df),
            extra_insights=_generate_additional_insights(analysis_df),
            dataset_summary=dataset_summary,
            overview_insights=overview_insights,
            overview_charts_html=overview_charts_html,
            saved_charts=_prepare_saved_chart_views(
                SavedChart.query
                .filter_by(user_id=session['user_id'], upload_id=upload.id if upload else None)
                .order_by(SavedChart.created_at.desc())
                .all()
            ) if upload else [],
            generated_chart_html=generated_chart_html,
            generated_figure_json=generated_figure_json,
            generated_x_column=generated_x_column,
            generated_y_column=generated_y_column,
            current_upload=upload,
        )

    except Exception as e:
        flash(f"Error generating chart: {e}")
        return redirect(url_for('dashboard'))


@app.route('/save_generated_chart', methods=['POST'])
def save_generated_chart():
    if 'user' not in session:
        return redirect(url_for('login'))

    upload_id = request.form.get('upload_id', type=int)
    title = request.form.get('title', '').strip()
    generate_shareable = request.form.get('generate_shareable') in {'1', 'on', 'true', 'yes'}

    if not upload_id:
        flash('No upload selected for saving chart.')
        return redirect(url_for('dashboard'))

    upload = Upload.query.filter_by(id=upload_id, user_id=session['user_id']).first()
    if not upload:
        flash('Upload not found for saving chart.')
        return redirect(url_for('dashboard'))

    if not title:
        base_name = upload.original_filename if upload else 'Generated chart'
        title = f"{base_name} (auto chart)"

    figure_json = request.form.get('figure_json')
    chart_html = request.form.get('chart_html')
    x_column = request.form.get('x_column') or None
    y_column = request.form.get('y_column') or None

    if not chart_html:
        flash('No generated chart available to save.')
        return redirect(url_for('dashboard', filename=upload.filename))

    token = uuid.uuid4().hex if generate_shareable else None

    saved = SavedChart(
        user_id=session['user_id'],
        upload_id=upload.id,
        title=title,
        chart_type='auto',
        x_column=x_column,
        y_column=y_column,
        filter_column=None,
        filter_value=None,
        sample_fraction=None,
        figure_json=figure_json,
        chart_html=chart_html,
        shared_token=token,
    )
    db.session.add(saved)
    db.session.commit()

    flash('Generated chart saved.', 'success')
    return redirect(url_for('dashboard', filename=upload.filename))

@app.route('/delete_file', methods=['POST'])
def delete_file():
    if 'user' not in session:
        return redirect(url_for('login'))

    filename = request.form['filename']
    filepath = _user_upload_dir(session['user_id']) / filename

    # Delete file from disk
    if filepath.exists():
        filepath.unlink()

    # Delete file record from database
    upload = Upload.query.filter_by(user_id=session['user_id'], filename=filename).first()
    display_name = upload.original_filename if upload else filename
    if upload:
        DatasetProfile.query.filter_by(upload_id=upload.id).delete()
        SavedChart.query.filter_by(upload_id=upload.id).delete()
        db.session.delete(upload)
        db.session.commit()

    # Find the next most recent file, if any
    uploads = (
        Upload.query
        .filter_by(user_id=session['user_id'])
        .order_by(Upload.timestamp.desc())
        .all()
    )
    if uploads:
        next_filename = uploads[0].filename
        flash(f"Deleted {display_name}")
        return redirect(url_for('dashboard', filename=next_filename))
    else:
        flash(f"Deleted {display_name}")
        return redirect(url_for('dashboard'))


@app.route('/delete_all_files', methods=['POST'])
def delete_all_files():
    if 'user' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    uploads = Upload.query.filter_by(user_id=user_id).all()
    user_dir = _user_upload_dir(user_id)

    for upload in uploads:
        filepath = user_dir / upload.filename
        if filepath.exists():
            filepath.unlink()
        DatasetProfile.query.filter_by(upload_id=upload.id).delete()
        SavedChart.query.filter_by(upload_id=upload.id).delete()
        db.session.delete(upload)

    db.session.commit()
    flash('All uploaded files have been deleted.')
    return redirect(url_for('dashboard'))


@app.route('/save_chart/<int:chart_id>/delete', methods=['POST'])
def delete_saved_chart(chart_id: int):
    if 'user' not in session:
        return redirect(url_for('login'))

    chart = SavedChart.query.filter_by(id=chart_id, user_id=session['user_id']).first_or_404()
    upload = Upload.query.get(chart.upload_id)
    db.session.delete(chart)
    db.session.commit()
    flash('Saved chart deleted')
    if upload:
        return redirect(url_for('dashboard', filename=upload.filename))
    return redirect(url_for('dashboard'))


@app.route('/charts/<token>')
def shared_chart(token: str):
    chart = SavedChart.query.filter_by(shared_token=token).first_or_404()
    chart_html = _render_saved_chart_html(chart)
    return render_template('shared_chart.html', chart_html=chart_html, title=chart.title)



# Entry point
if __name__ == '__main__':
    app.run(debug=True)
