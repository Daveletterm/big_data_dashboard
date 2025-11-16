# Big Data Dashboard

A Flask app for uploading CSVs, exploring quick insights, and building Plotly charts. The app expects a few environment variables for security and storage; use a `.env` file so you only set them once.

## 1) Local setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the example environment file and fill in the values:
   ```bash
   cp .env.example .env
   # edit .env to set SECRET_KEY, DATABASE_URL, and MAPBOX_TOKEN (optional)
   ```
   - **SECRET_KEY** is required; use a long random string to avoid the runtime error you saw.
   - **DATABASE_URL** can point to SQLite by leaving it blank, but for hosting use PostgreSQL (e.g., Supabase).
   - **MAPBOX_TOKEN** is only needed for map charts; leave it empty otherwise.
4. Start the app (Flask automatically loads `.env` because `python-dotenv` is installed):
   ```bash
   flask run  # or: FLASK_APP=app.py flask run
   ```
   Uploaded files are saved under `uploads/` by default; each user gets a subfolder.

## 2) Supabase setup (PostgreSQL)
1. Sign up at [Supabase](https://supabase.com/) and create a new project (the free tier works for small datasets).
2. From the project **Settings → Database**, copy the connection string. It usually looks like:
   ```
   postgresql://postgres:<password>@db.<hash>.supabase.co:5432/postgres
   ```
3. Paste that value into your `.env` as `DATABASE_URL`. The app will normalize `postgres://` to `postgresql://` automatically.
4. (Optional) Create a dedicated database and user in Supabase and update the connection string accordingly to avoid using the default superuser.
5. Run database migrations on first boot. Since the app uses SQLAlchemy models without Alembic, you can trigger table creation by opening the site once; the models will call `db.create_all()` at startup.

## 3) Hosting (cheap/free path)
- **Platform**: Render, Fly.io, or Railway free tiers work; choose one that supports long-running web services and environment variables.
- **Runtime**: Use Gunicorn instead of the Flask dev server. Example start command:
  ```bash
  gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT
  ```
- **Environment variables**: Add `SECRET_KEY`, `DATABASE_URL`, and optional `MAPBOX_TOKEN` in the platform dashboard. Do not commit real secrets.
- **Persistent storage**: Attach a disk/volume for the `uploads/` directory so user files survive deploys. If the platform lacks disks, point `UPLOAD_BASE` (in `.env`) to a mounted path or switch to an object store (S3-compatible) and mount it via a fuse driver.
- **Build steps**: Install dependencies (`pip install -r requirements.txt`) and ensure Python 3.10+.
- **Health checks**: Expose the default Flask port (5000) or rely on `$PORT` provided by the host.

### Minimal Render example
1. Create a new **Web Service** from this repo.
2. Set **Environment** to Python and **Build Command** to `pip install -r requirements.txt`.
3. Set **Start Command** to `gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT`.
4. Add environment variables (`SECRET_KEY`, `DATABASE_URL`, optional `MAPBOX_TOKEN`).
5. Add a **Persistent Disk** and mount it at `/workspace/big_data_dashboard/uploads` (or update `UPLOAD_BASE` accordingly).

### Minimal Fly.io example
1. Install `flyctl` and run `fly launch --no-deploy` in the repo to generate a `fly.toml`.
2. Set secrets with `fly secrets set SECRET_KEY=... DATABASE_URL=... MAPBOX_TOKEN=...`.
3. Add a volume: `fly volumes create uploads --size 1 --region <region>` and mount it in `fly.toml` under `[mounts]` at `/workspace/big_data_dashboard/uploads`.
4. Set the `CMD` in `fly.toml` to `gunicorn app:app --workers 2 --bind 0.0.0.0:8080` and map internal port `8080` to external.
5. Deploy with `fly deploy`.

## 4) Avoid re-exporting variables every time
- The app loads `.env` automatically via `python-dotenv`, so once you create `.env` you can simply run `flask run` without re-exporting variables.
- If you prefer shell exports, add them to your shell profile (e.g., `~/.bashrc` or `~/.zshrc`) so they load on every terminal start:
  ```bash
  export SECRET_KEY="your-secret"
  export DATABASE_URL="postgresql://..."
  export MAPBOX_TOKEN=""
  ```
