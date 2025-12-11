from __future__ import annotations

from flask import Flask, render_template, request, redirect, url_for, session, flash, abort
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from pathlib import Path
import mimetypes
import os
import uuid
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
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
    }
    allowed_extensions = {'.csv'}

    mime_guess, _ = mimetypes.guess_type(file.filename)
    content_type = (file.mimetype or '').split(';')[0].strip()
    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise ValueError('Invalid file type. Please upload a CSV file.')

    if content_type and content_type not in allowed_mimes and mime_guess not in allowed_mimes:
        raise ValueError('Invalid file type. Please upload a CSV file.')


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


def _load_dataframe(data_path: Path, max_rows: int = 150_000) -> pd.DataFrame:
    """Load a CSV file into a cleaned dataframe with limits and streaming."""

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


def _render_preview_table(df: pd.DataFrame) -> tuple[str, list[str]]:
    preview_df = df.head(100)
    table_html = preview_df.to_html(classes='table table-striped table-bordered', index=False)
    columns = df.columns.tolist()
    return table_html, columns



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


with app.app_context():
    db.create_all()

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
    chart_html: str | None = None
    generated_chart_html: str | None = None
    generated_figure_json: str | None = None
    generated_x_column: str | None = None
    generated_y_column: str | None = None
    display_chart_html: str | None = None
    display_figure_json: str | None = None
    display_x_column: str | None = None
    display_y_column: str | None = None
    selected_upload = None
    if not filename and uploads:
        selected_upload = uploads[0]
        filename = selected_upload.filename
    elif filename:
        selected_upload = Upload.query.filter_by(filename=filename, user_id=session['user_id']).first()

    quality_signals = {}
    extra_insights = {}

    if filename:
        filepath = _user_upload_dir(session['user_id']) / filename
        if filepath.exists():
            try:
                df = _load_dataframe(filepath)
                analysis_df = _analysis_subset(df)
                table_html, columns = _render_preview_table(df)
                summary = _summarise_dataframe(analysis_df)
                suggestions = generate_chart_suggestions(analysis_df)
                quality_signals = _data_quality_signals(analysis_df)
                extra_insights = _generate_additional_insights(analysis_df)
                dataset_synopsis = _dataset_brief(analysis_df)
            except Exception as e:
                flash(f"Error loading file '{filename}': {e}")
        else:
            flash('Selected file not found on disk')

    suggestions = suggestions or []
    display_chart_html = chart_html  # leave blank until user explicitly visualizes
    display_figure_json = generated_figure_json
    display_x_column = generated_x_column
    display_y_column = generated_y_column

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
        quality_signals=quality_signals,
        extra_insights=extra_insights,
        chart=chart_html,
        generated_chart_html=generated_chart_html,
        generated_figure_json=generated_figure_json,
        generated_x_column=generated_x_column,
        generated_y_column=generated_y_column,
        display_chart_html=display_chart_html,
        display_figure_json=display_figure_json,
        display_x_column=display_x_column,
        display_y_column=display_y_column,
        current_upload=selected_upload,
    )

@app.route('/upload', methods=['POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    _enforce_rate_limit(f"upload:{_client_ip()}")

    file = request.files.get('data_file')

    if not file or file.filename == '':
        flash('Please upload a CSV file.')
        return redirect(url_for('dashboard'))

    user_dir = _user_upload_dir(session['user_id'])

    df: pd.DataFrame
    safe_name: str
    stored_name: str
    filepath: Path
    chart_html: str | None = None
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
        analysis_df = _analysis_subset(df)
        summary = _summarise_dataframe(analysis_df)
        suggestions = generate_chart_suggestions(analysis_df)
        table_html, columns = _render_preview_table(df)
        # Do not auto-generate a chart on upload; keep the hero slot blank until the user visualizes.
        display_chart_html = None
        display_figure_json = None
        display_x_column = None
        display_y_column = None
        db.session.commit()
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
        quality_signals=_data_quality_signals(analysis_df),
        extra_insights=_generate_additional_insights(analysis_df),
        chart=display_chart_html,
        generated_chart_html=display_chart_html,
        generated_figure_json=display_figure_json,
        generated_x_column=display_x_column,
        generated_y_column=display_y_column,
        display_chart_html=display_chart_html,
        display_figure_json=display_figure_json,
        display_x_column=display_x_column,
        display_y_column=display_y_column,
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
    generated_chart_html: str | None = None
    generated_figure_json: str | None = None
    generated_x_column: str | None = None
    generated_y_column: str | None = None

    if not filename or not x_column or not chart_type:
        flash("Missing form data")
        return redirect(url_for('dashboard'))
    needs_y = chart_type not in {'pie', 'histogram'} and not (chart_type == 'bar' and not y_column)
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

        # Update suggestions, preview, summary
        analysis_df = _analysis_subset(df)
        (
            generated_chart_html,
            generated_figure_json,
            generated_x_column,
            generated_y_column,
        ) = build_generated_chart(analysis_df)
        suggestions = generate_chart_suggestions(analysis_df)
        table_html, columns = _render_preview_table(df)
        summary = _summarise_dataframe(analysis_df)
        dataset_synopsis = _dataset_brief(analysis_df)

        display_chart_html = chart_html or generated_chart_html
        display_figure_json = figure_json or generated_figure_json
        display_x_column = x_column
        display_y_column = y_column

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
            quality_signals=_data_quality_signals(analysis_df),
            extra_insights=_generate_additional_insights(analysis_df),
            generated_chart_html=generated_chart_html,
            generated_figure_json=generated_figure_json,
            generated_x_column=generated_x_column,
            generated_y_column=generated_y_column,
            display_chart_html=display_chart_html,
            display_figure_json=display_figure_json,
            display_x_column=display_x_column,
            display_y_column=display_y_column,
            current_upload=upload,
        )

    except Exception as e:
        flash(f"Error generating chart: {e}")
        return redirect(url_for('dashboard'))


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
        db.session.delete(upload)

    db.session.commit()
    flash('All uploaded files have been deleted.')
    return redirect(url_for('dashboard'))



# Entry point
if __name__ == '__main__':
    app.run(debug=True)
