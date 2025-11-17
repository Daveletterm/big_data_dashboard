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
import json

pio.renderers.default = 'iframe'

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
            infer_datetime_format=True,
            cache=True,
        )
        # Only coerce if the column is plausibly datetime to avoid noisy warnings
        if parsed_sample.notna().mean() >= 0.6:
            df[column] = pd.to_datetime(
                df[column],
                errors='coerce',
                infer_datetime_format=True,
                cache=True,
            )

    return df


def _analysis_subset(df: pd.DataFrame, max_rows: int = 20_000) -> pd.DataFrame:
    """Return a representative sample to keep downstream analysis responsive."""

    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


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


def _build_profile_charts(df: pd.DataFrame) -> dict[str, str]:
    charts: dict[str, str] = {}
    numeric_cols = df.select_dtypes(include=['number'])
    if numeric_cols.shape[1] >= 2:
        corr = numeric_cols.corr().fillna(0)
        fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale='Blues', title='Correlation matrix')
        charts['correlation'] = fig_corr.to_html(full_html=False)
    null_matrix = df.isnull()
    if null_matrix.values.any():
        fig_null = px.imshow(
            null_matrix.astype(int),
            color_continuous_scale=[[0, 'rgb(0,123,255)'], [1, 'rgb(220,53,69)']],
            title='Null heatmap',
            labels={'color': 'Null (1=yes)'}
        )
        charts['nulls'] = fig_null.to_html(full_html=False)
    return charts


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

def generate_chart_suggestions(df, max_suggestions=10):
    suggestions = []
    df = df.dropna(axis=1, how='all')

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = [
        column
        for column in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[column])
    ]

    # 1. Time trends: look for columns that change steadily over time
    for tcol in datetime_cols:
        for num in numeric_cols:
            non_missing = df[[tcol, num]].dropna()
            if len(non_missing) < 5:
                continue

            non_missing = non_missing.sort_values(by=tcol)
            values = non_missing[num].values
            if len(values) > 1:
                corr_time = pd.Series(range(len(values))).corr(pd.Series(values))
                score = float(abs(corr_time))
                if score > 0.3:  # modest trend threshold
                    suggestions.append({
                        "title": f"Trend of {num} over {tcol} (trend strength {score:.2f})",
                        "chart_type": "line",
                        "x": tcol,
                        "y": num,
                        "score": score + 1.0  # bump time charts slightly
                    })

    # 2. Category vs numeric: only if categories differ meaningfully
    for cat in categorical_cols:
        unique_count = df[cat].nunique()
        if unique_count < 2 or unique_count > 20:
            continue

        for num in numeric_cols:
            group_means = df.groupby(cat)[num].mean()
            if group_means.empty or group_means.std() == 0:
                continue
            # relative variability of group means
            var_ratio = group_means.std() / (group_means.mean() + 1e-9)
            score = float(min(var_ratio, 1.0))
            if score > 0.15:  # meaningful group separation
                suggestions.append({
                    "title": f"Average {num} by {cat} (variation {score:.2f})",
                    "chart_type": "bar",
                    "x": cat,
                    "y": num,
                    "score": score + 0.5
                })
        if unique_count <= 8:
            # pie charts for small, diverse categories
            counts = df[cat].value_counts(normalize=True)
            balance = float(1 - abs(counts.max() - counts.mean()))  # penalize dominance
            if balance > 0.4:
                suggestions.append({
                    "title": f"Distribution of {cat}",
                    "chart_type": "pie",
                    "x": cat,
                    "y": None,
                    "score": balance
                })

    # 3. Numeric vs numeric: measure actual correlation strength
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i+1:]:
            subset = df[[c1, c2]].dropna()
            if len(subset) < 5:
                continue
            corr = subset[c1].corr(subset[c2])
            if pd.isna(corr):
                continue
            if abs(corr) >= 0.5:
                suggestions.append({
                    "title": f"Correlation between {c1} and {c2} (r={corr:.2f})",
                    "chart_type": "scatter",
                    "x": c1,
                    "y": c2,
                    "score": float(abs(corr) + 1.2)  # rank strong correlations highest
                })

    # Rank and trim
    if not suggestions:
        fallback_suggestions = []
        if datetime_cols and numeric_cols:
            fallback_suggestions.append({
                "title": f"Trend of {numeric_cols[0]} over {datetime_cols[0]}",
                "chart_type": "line",
                "x": datetime_cols[0],
                "y": numeric_cols[0],
                "score": 0.3,
            })
        if numeric_cols:
            fallback_suggestions.append({
                "title": f"Distribution of {numeric_cols[0]}",
                "chart_type": "histogram",
                "x": numeric_cols[0],
                "y": None,
                "score": 0.25,
            })
        if categorical_cols and numeric_cols:
            fallback_suggestions.append({
                "title": f"Average {numeric_cols[0]} by {categorical_cols[0]}",
                "chart_type": "bar",
                "x": categorical_cols[0],
                "y": numeric_cols[0],
                "score": 0.2,
            })

        if fallback_suggestions:
            suggestions = fallback_suggestions
        else:
            fallback_message = (
                "No strong chart suggestions found. Try exploring columns manually "
                "or adjust thresholds."
            )
            return [{
                "title": fallback_message,
                "chart_type": None,
                "x": None,
                "y": None,
                "score": 0
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

    if not filename and uploads:
        filename = uploads[0].filename

    profile_charts = {}
    quality_signals = {}
    extra_insights = {}
    saved_charts = (
        SavedChart.query
        .filter_by(user_id=session['user_id'])
        .order_by(SavedChart.created_at.desc())
        .all()
    )

    if filename:
        filepath = _user_upload_dir(session['user_id']) / filename
        if filepath.exists():
            try:
                df = _load_dataframe(filepath)
                analysis_df = _analysis_subset(df)
                table_html, columns = _render_preview_table(df)
                summary = _summarise_dataframe(analysis_df)
                suggestions = generate_chart_suggestions(analysis_df)
                profile_charts = _build_profile_charts(analysis_df)
                quality_signals = _data_quality_signals(analysis_df)
                extra_insights = _generate_additional_insights(analysis_df)
                dataset_synopsis = _dataset_brief(analysis_df)
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
        profile_charts=profile_charts,
        quality_signals=quality_signals,
        extra_insights=extra_insights,
        saved_charts=saved_charts
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
        analysis_df = _analysis_subset(df)
        summary = _summarise_dataframe(analysis_df)
        suggestions = generate_chart_suggestions(analysis_df)
        table_html, columns = _render_preview_table(df)
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
        profile_charts=_build_profile_charts(analysis_df),
        quality_signals=_data_quality_signals(analysis_df),
        extra_insights=_generate_additional_insights(analysis_df),
        saved_charts=(
            SavedChart.query
            .filter_by(user_id=session['user_id'])
            .order_by(SavedChart.created_at.desc())
            .all()
        )
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

    if not filename or not x_column or not chart_type:
        flash("Missing form data")
        return redirect(url_for('dashboard'))
    if chart_type != 'pie' and not y_column:
        flash('Y axis is required for this chart type')
        return redirect(url_for('dashboard'))

    filepath = _user_upload_dir(session['user_id']) / filename
    if not filepath.exists():
        flash("Selected file not found")
        return redirect(url_for('dashboard'))

    try:
        df = _load_dataframe(filepath)

        if filter_column and filter_column in df.columns and filter_value:
            mask = df[filter_column].astype(str).str.contains(re.escape(filter_value), case=False, na=False)
            df = df[mask]
        if sample_fraction:
            try:
                frac = float(sample_fraction)
                if 0 < frac < 1:
                    df = df.sample(frac=frac, random_state=42)
            except ValueError:
                flash('Invalid sample fraction; ignoring.')

        # Chart rendering logic
        if chart_type == 'pie':
            value_counts = df[x_column].value_counts().reset_index()
            value_counts.columns = [x_column, 'Count']
            fig = px.pie(value_counts, names=x_column, values='Count')
        elif chart_type == 'bar':
            fig = px.bar(df, x=x_column, y=y_column)
        elif chart_type == 'line':
            fig = px.line(df, x=x_column, y=y_column)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_column, y=y_column)
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

        if save_title:
            token = uuid.uuid4().hex if shareable else None
            upload = Upload.query.filter_by(filename=filename, user_id=session['user_id']).first()
            if not upload:
                raise ValueError('Upload not found for saving chart')
            saved = SavedChart(
                user_id=session['user_id'],
                upload_id=upload.id,
                title=save_title,
                chart_html=chart_html,
                shared_token=token
            )
            db.session.add(saved)
            db.session.commit()

        # Update suggestions, preview, summary
        analysis_df = _analysis_subset(df)
        suggestions = generate_chart_suggestions(analysis_df)
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
            profile_charts=_build_profile_charts(analysis_df),
            quality_signals=_data_quality_signals(analysis_df),
            extra_insights=_generate_additional_insights(analysis_df),
            saved_charts=(
                SavedChart.query
                .filter_by(user_id=session['user_id'])
                .order_by(SavedChart.created_at.desc())
                .all()
            )
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


@app.route('/save_chart/<int:chart_id>/delete', methods=['POST'])
def delete_saved_chart(chart_id: int):
    if 'user' not in session:
        return redirect(url_for('login'))

    chart = SavedChart.query.filter_by(id=chart_id, user_id=session['user_id']).first_or_404()
    db.session.delete(chart)
    db.session.commit()
    flash('Saved chart deleted')
    return redirect(url_for('dashboard'))


@app.route('/charts/<token>')
def shared_chart(token: str):
    chart = SavedChart.query.filter_by(shared_token=token).first_or_404()
    return render_template('shared_chart.html', chart_html=chart.chart_html, title=chart.title)



# Entry point
if __name__ == '__main__':
    app.run(debug=True)
