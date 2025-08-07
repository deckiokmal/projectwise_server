#!/usr/bin/env bash
#===============================================================================
# entrypoint.sh
#-----------------------------------------------------------------------------
# Skrip inisialisasi container sebelum menjalankan proses utama (e.g. Gunicorn)
#===============================================================================

# 1) STRICT MODE: 
#   -e : exit on any command failure  
#   -u : treat unset variables as errors  
#   -o pipefail : catch failures in piped commands  
set -euo pipefail

# 2) FUNCTIONS FOR SIGNAL HANDLING
cleanup() {
  echo "[entrypoint] ⚙️  Trapped SIGTERM/SIGINT, melakukan shutdown..."
  # Jika perlu: panggil perintah cleanup, flush logs, dsb.
  exit 0
}
trap cleanup SIGINT SIGTERM

# 3) LOAD ENV VARS & DEFAULTS
# (Jika menggunakan file .env, uncomment baris berikut)
# export $(grep -v '^#' /app/.env | xargs)

: "${DB_HOST:?Environment variable DB_HOST tidak diset}"
: "${DB_PORT:?Environment variable DB_PORT tidak diset}"
: "${API_KEY:=default_api_key}"

# 4) PERMISSION FIXES
# Pastikan direktori log/cache dimiliki oleh user yang tepat
echo "[entrypoint] 🔐 Memperbaiki permission pada /app/logs dan /app/tmp"
chown -R appuser:appgroup /app/logs /app/tmp || true

# 5) WAIT FOR DEPENDENCIES
# Contoh: tunggu database dan message broker (Redis) siap
echo "[entrypoint] ⏳ Menunggu database di $DB_HOST:$DB_PORT..."
until nc -z "$DB_HOST" "$DB_PORT"; do
  sleep 1
done
echo "[entrypoint] ✅ Database siap!"

# Jika Anda menggunakan Redis:
if [ -n "${REDIS_HOST-}" ]; then
  echo "[entrypoint] ⏳ Menunggu Redis di $REDIS_HOST:$REDIS_PORT..."
  until nc -z "$REDIS_HOST" "$REDIS_PORT"; do
    sleep 1
  done
  echo "[entrypoint] ✅ Redis siap!"
fi

# 6) DATABASE MIGRATIONS / SEEDING
# Contoh Alembic:
if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
  echo "[entrypoint] 🗄️  Menjalankan database migration (alembic)..."
  alembic upgrade head
fi

# (Opsional) Seed data awal:
# if [ "${RUN_SEED:-false}" = "true" ]; then
#   echo "[entrypoint] 🌱 Menjalankan data seeding..."
#   python scripts/seed_db.py
# fi

# 7) CACHE WARM-UP atau PRE-COMPILE
# Contoh: compile Jinja template, preload ML model, dsb.
# echo "[entrypoint] 🛠️  Warming up cache dan template..."
# python - <<EOF
# from app import create_app
# app = create_app()
# with app.app_context():
#     app.jinja_env.cache.clear()
# EOF

# 8) EXECUTE MAIN PROCESS
# Jika ada argumen yang diteruskan saat `docker run`, jalankan perintah itu:
if [ $# -gt 0 ]; then
  echo "[entrypoint] ▶️  Menjalankan perintah kustom: $*"
  exec "$@"
else
  # Default command: jalankan Gunicorn sebagai app PID 1 agar sinyal diteruskan
  echo "[entrypoint] ▶️  Menjalankan aplikasi dengan Gunicorn"
  exec su-exec appuser:appgroup \
    gunicorn --workers "${GUNICORN_WORKERS:-4}" \
             --bind "0.0.0.0:${PORT:-8000}" \
             --access-logfile "-" \
             --error-logfile "-" \
             app:create_app()
fi
