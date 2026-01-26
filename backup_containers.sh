#!/bin/bash
# ============================================
# MLOps Container Backup Script
# Creates clean backup of all running containers
# ============================================

BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "🔵 Starting MLOps Container Backup..."
echo "📁 Backup directory: $BACKUP_DIR"

# ============================================
# 1. POSTGRES BACKUP
# ============================================
echo ""
echo "📦 [1/6] Backing up PostgreSQL (MLflow backend)..."
docker exec postgres_db pg_dumpall -U airflow > "$BACKUP_DIR/postgres_full.sql"
echo "✅ PostgreSQL backup saved: postgres_full.sql ($(du -h "$BACKUP_DIR/postgres_full.sql" | cut -f1))"

# ============================================
# 2. MARIADB COLUMNSTORE BACKUP
# ============================================
echo ""
echo "📦 [2/6] Backing up MariaDB ColumnStore (lending_club)..."
docker exec mcs1 mariadb-dump -u admin -pAdmin@1234Strong! --all-databases > "$BACKUP_DIR/mariadb_full.sql"
echo "✅ MariaDB backup saved: mariadb_full.sql ($(du -h "$BACKUP_DIR/mariadb_full.sql" | cut -f1))"

# ============================================
# 3. REDIS BACKUP
# ============================================
echo ""
echo "📦 [3/6] Backing up Redis (pipeline state)..."
# Trigger save and copy RDB file
docker exec redis1 redis-cli -a mysecurepassword SAVE
docker cp redis1:/data/dump.rdb "$BACKUP_DIR/redis_dump.rdb"
echo "✅ Redis backup saved: redis_dump.rdb ($(du -h "$BACKUP_DIR/redis_dump.rdb" | cut -f1))"

# ============================================
# 4. MLFLOW ARTIFACTS BACKUP
# ============================================
echo ""
echo "📦 [4/6] Backing up MLflow artifacts..."
# Copy artifacts from host volume
tar -czf "$BACKUP_DIR/mlflow_artifacts.tar.gz" -C ./mlflow_artifacts . 2>/dev/null
echo "✅ MLflow artifacts saved: mlflow_artifacts.tar.gz ($(du -h "$BACKUP_DIR/mlflow_artifacts.tar.gz" | cut -f1))"

# ============================================
# 5. AIRFLOW DAGS & LOGS BACKUP
# ============================================
echo ""
echo "📦 [5/6] Backing up Airflow DAGs and logs..."
tar -czf "$BACKUP_DIR/airflow_dags.tar.gz" -C ./airflow/dags . 2>/dev/null
tar -czf "$BACKUP_DIR/airflow_logs.tar.gz" -C ./airflow/logs . 2>/dev/null
echo "✅ Airflow DAGs saved: airflow_dags.tar.gz ($(du -h "$BACKUP_DIR/airflow_dags.tar.gz" | cut -f1))"
echo "✅ Airflow logs saved: airflow_logs.tar.gz ($(du -h "$BACKUP_DIR/airflow_logs.tar.gz" | cut -f1))"

# ============================================
# 6. DATA FILES BACKUP
# ============================================
echo ""
echo "📦 [6/6] Backing up data files..."
tar -czf "$BACKUP_DIR/data_files.tar.gz" -C ./data . 2>/dev/null
echo "✅ Data files saved: data_files.tar.gz ($(du -h "$BACKUP_DIR/data_files.tar.gz" | cut -f1))"

# ============================================
# METADATA & DOCKER CONFIG
# ============================================
echo ""
echo "📋 Saving container metadata..."

# Save container info
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" > "$BACKUP_DIR/containers_info.txt"

# Save docker-compose
cp docker-compose.yml "$BACKUP_DIR/docker-compose.yml"

# Save environment info
cat > "$BACKUP_DIR/backup_manifest.txt" << EOF
MLOps Backup Manifest
=====================
Date: $(date)
Hostname: $(hostname)
Backup Directory: $BACKUP_DIR

Containers Backed Up:
- postgres_db (PostgreSQL 13)
- mcs1 (MariaDB ColumnStore)
- redis1 (Redis)
- mlflow_server
- airflow_webserver
- airflow_scheduler

Contents:
- postgres_full.sql: Full PostgreSQL dump (MLflow tracking database)
- mariadb_full.sql: Full MariaDB dump (lending_club database)
- redis_dump.rdb: Redis snapshot (pipeline state)
- mlflow_artifacts.tar.gz: MLflow experiment artifacts & models
- airflow_dags.tar.gz: Airflow DAG definitions
- airflow_logs.tar.gz: Airflow execution logs
- data_files.tar.gz: Raw and processed data files
- docker-compose.yml: Container orchestration config
- containers_info.txt: Running containers metadata

Restoration Steps:
1. Start containers: docker-compose up -d
2. Restore PostgreSQL: docker exec -i postgres_db psql -U airflow < postgres_full.sql
3. Restore MariaDB: docker exec -i mcs1 mariadb -u admin -pAdmin@1234Strong! < mariadb_full.sql
4. Restore Redis: docker cp redis_dump.rdb redis1:/data/dump.rdb && docker restart redis1
5. Extract artifacts: tar -xzf mlflow_artifacts.tar.gz -C ./mlflow_artifacts
6. Extract DAGs: tar -xzf airflow_dags.tar.gz -C ./airflow/dags
7. Extract data: tar -xzf data_files.tar.gz -C ./data
EOF

echo "✅ Manifest saved: backup_manifest.txt"

# ============================================
# SUMMARY
# ============================================
echo ""
echo "======================================"
echo "🎉 Backup Complete!"
echo "======================================"
echo "📁 Location: $BACKUP_DIR"
echo "💾 Total size: $(du -sh "$BACKUP_DIR" | cut -f1)"
echo ""
echo "📋 Contents:"
ls -lh "$BACKUP_DIR"
echo ""
echo "✨ To restore: See backup_manifest.txt for instructions"
echo "======================================"
