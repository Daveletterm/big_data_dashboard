from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from pathlib import Path
import os
import uuid
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'iframe'
px.set_mapbox_access_token('pk.eyJ1IjoiZGF2ZWxldHRlcm0iLCJhIjoiY21kZmZrcWR3MGQ2MDJpcTNwejFkb2d5byJ9.hNgFSC79KzFzyBrgMBLKbA')

# Load .env file if available
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))
app.config['UPLOAD_FOLDER'] = BASE_DIR / 'uploads'
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config.setdefault('MAX_CONTENT_LENGTH', 25 * 1024 * 1024)

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


def _coerce_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce obvious datetime columns into datetime64."""

    datetime_hints = ('date', 'time', 'timestamp')
    string_like = df.select_dtypes(include=['object', 'string'])

    for column in string_like.columns:
        lowered = column.lower()
        if any(hint in lowered for hint in datetime_hints):
            parsed = pd.to_datetime(df[column], errors='coerce')
            if not parsed.isna().all():
                df[column] = parsed

    return df


def _load_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load a CSV file into a cleaned dataframe."""

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = _coerce_datetime_columns(df)
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


def _render_preview_table(df: pd.DataFrame) -> tuple[str, list[str]]:
    preview_df = df.head(100)
    table_html = preview_df.to_html(classes='table table-striped table-bordered', index=False)
    columns = df.columns.tolist()
    return table_html, columns

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
        return [{
            "title": "No strong chart suggestions found. Try exploring columns manually.",
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


with app.app_context():
    db.create_all()

# Routes
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.template_filter('file_exists')
def file_exists_filter(filename):
    return (app.config['UPLOAD_FOLDER'] / filename).is_file()

@app.route('/signup', methods=['GET', 'POST'])
def signup():
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
    suggestions = None

    if not filename and uploads:
        filename = uploads[0].filename

    if filename:
        filepath = app.config['UPLOAD_FOLDER'] / filename
        if filepath.exists():
            try:
                df = _load_dataframe(filepath)
                table_html, columns = _render_preview_table(df)
                summary = _summarise_dataframe(df)
                suggestions = generate_chart_suggestions(df)
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
        suggestions=suggestions
    )

@app.route('/upload', methods=['POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    if 'csv_file' not in request.files:
        flash('No file part')
        return redirect(url_for('dashboard'))

    file = request.files['csv_file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('dashboard'))

    if not file.filename.lower().endswith('.csv'):
        flash('Only CSV files are allowed')
        return redirect(url_for('dashboard'))

    safe_name = secure_filename(file.filename)
    if not safe_name:
        flash('Invalid filename')
        return redirect(url_for('dashboard'))

    stored_name = f"{uuid.uuid4().hex}__{safe_name}"
    filepath = app.config['UPLOAD_FOLDER'] / stored_name
    file.save(filepath)

    new_upload = Upload(filename=stored_name, user_id=session['user_id'])
    db.session.add(new_upload)

    try:
        df = _load_dataframe(filepath)
        summary = _summarise_dataframe(df)
        suggestions = generate_chart_suggestions(df)
        table_html, columns = _render_preview_table(df)
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

    return render_template(
        'dashboard.html',
        user=session['user'],
        table=table_html,
        columns=columns,
        uploads=uploads,
        summary=summary,
        selected_file=stored_name,
        suggestions=suggestions
    )

@app.route('/visualize', methods=['POST'])
def visualize():
    if 'user' not in session:
        return redirect(url_for('login'))

    filename = request.form.get('filename')
    x_column = request.form.get('x_column')
    y_column = request.form.get('y_column', '')
    chart_type = request.form.get('chart_type')

    if not filename or not x_column or not chart_type:
        flash("Missing form data")
        return redirect(url_for('dashboard'))

    filepath = app.config['UPLOAD_FOLDER'] / filename
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
            fig = px.bar(df, x=x_column, y=y_column)
        elif chart_type == 'line':
            fig = px.line(df, x=x_column, y=y_column)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_column, y=y_column)
        elif chart_type == 'scatter_map':
            if df[x_column].dtype.kind in 'iuf' and df[y_column].dtype.kind in 'iuf':
                df = df[(df[y_column].between(-90, 90)) & (df[x_column].between(-180, 180))]
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
            if df[x_column].dtype.kind in 'iuf' and df[y_column].dtype.kind in 'iuf':
                df = df[(df[y_column].between(-90, 90)) & (df[x_column].between(-180, 180))]
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

        # Update suggestions, preview, summary
        suggestions = generate_chart_suggestions(df)
        table_html, columns = _render_preview_table(df)
        summary = _summarise_dataframe(df)

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
            suggestions=suggestions
        )

    except Exception as e:
        flash(f"Error generating chart: {e}")
        return redirect(url_for('dashboard'))

@app.route('/delete_file', methods=['POST'])
def delete_file():
    if 'user' not in session:
        return redirect(url_for('login'))

    filename = request.form['filename']
    filepath = app.config['UPLOAD_FOLDER'] / filename

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



# Entry point
if __name__ == '__main__':
    app.run(debug=True)
