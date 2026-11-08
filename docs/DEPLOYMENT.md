# SkinAI Production Deployment & Staging Runbook

## 1. Architecture & Environment Overview

SkinAI operates under a decoupled, privacy-first microservice architecture:
- **`skinai_api` (FastAPI + ONNX Runtime)**: Ephemeral, CPU-optimized inference service with strictly zero raw photo retention.
- **`skinai_bot` (aiogram 3)**: Telegram bot handling interactive user dialogs, language selection, and Glasgow 7-point checklist anamnesis.
- **`skinai_postgres` (PostgreSQL 16)**: Internal isolated database storing salted anonymous telemetry audit logs (`PredictionAuditLog`).
- **`skinai_nginx` (Nginx 1.25 Reverse Proxy)**: Front-facing proxy enforcing TLS/SSL termination, rate limiting (10 req/s, burst 20), strict 15MB upload caps, and OWASP security headers.

### VPS Recommended Specifications
| Component | Minimum Spec | Recommended Staging/Prod |
|---|---|---|
| **OS** | Ubuntu 22.04 LTS or 24.04 LTS | Ubuntu 24.04 LTS |
| **vCPU** | 2 vCPU | 4 vCPU |
| **RAM** | 4 GB | 8 GB |
| **Storage** | 25 GB SSD | 50 GB NVMe |
| **Network** | 100 Mbps, public IPv4 | 1 Gbps, static public IPv4 |

---

## 2. Server Provisioning & Firewall Configuration

### Step 2.1: Update OS and Install Base Utilities
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y curl git ufw fail2ban jq
```

### Step 2.2: Install Docker Engine & Compose Plugin
```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Enable and start Docker service
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

### Step 2.3: Configure Host Firewall (UFW)
```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh          # Port 22
sudo ufw allow http         # Port 80
sudo ufw allow https        # Port 443
# Strictly DO NOT expose port 8000 (API) or port 5432 (Postgres) externally!
sudo ufw enable
```

---

## 3. Directory Layout & Repository Setup

On the VPS, deploy the application under `/opt/skinai`:
```bash
sudo mkdir -p /opt/skinai
sudo chown -R $USER:$USER /opt/skinai
cd /opt/skinai

# Clone repository
git clone https://github.com/your-org/skinai.git .
```

Directory structure on the host:
```text
/opt/skinai/
├── artifacts/
│   └── models/
│       └── smartphone_clinical/
│           ├── model.onnx
│           └── manifest.yaml
├── backend/
├── bot/
├── deploy/
│   ├── Dockerfile.api
│   ├── Dockerfile.bot
│   ├── docker-compose.prod.yml
│   └── nginx.conf
├── .env
└── pyproject.toml
```

---

## 4. Production Secret Generation & `.env` Setup

Generate cryptographically secure salts and passwords:
```bash
# Generate secrets
TELEMETRY_SALT=$(openssl rand -hex 32)
API_SECRET_KEY=$(openssl rand -hex 32)
POSTGRES_PASS=$(openssl rand -hex 24)
```

Create `/opt/skinai/.env`:
```ini
# Production Environment
ENVIRONMENT=production
DEBUG=false

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
API_URL=http://api:8000

# Backend & Security
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=${API_SECRET_KEY}
TELEMETRY_SALT=${TELEMETRY_SALT}
MAX_UPLOAD_SIZE_BYTES=15728640

# PostgreSQL Telemetry Database
POSTGRES_USER=skinai_prod_user
POSTGRES_PASSWORD=${POSTGRES_PASS}
POSTGRES_DB=skinai_telemetry
DATABASE_URL=postgresql://skinai_prod_user:${POSTGRES_PASS}@db:5432/skinai_telemetry

# Model Artifacts
ARTIFACTS_DIR=artifacts
SKINAI_ACTIVE_SMARTPHONE_MODEL=smartphone_clinical
```

Set strict permissions:
```bash
chmod 600 /opt/skinai/.env
```

---

## 5. Model Transfer & Cryptographic Verification

1. Copy the versioned model bundle from the workstation to the VPS:
   ```bash
   scp -r artifacts/models/smartphone_clinical user@vps-ip:/opt/skinai/artifacts/models/
   ```
2. On the VPS, verify SHA-256 integrity:
   ```bash
   sha256sum /opt/skinai/artifacts/models/smartphone_clinical/model.onnx
   # Confirm output matches the sha256 checksum recorded in manifest.yaml
   ```

---

## 6. Service Deployment & Verification

### Step 6.1: Build and Launch Containers
```bash
cd /opt/skinai
docker compose -f deploy/docker-compose.prod.yml up -d --build
```

### Step 6.2: Verify Health
```bash
# Check container status
docker compose -f deploy/docker-compose.prod.yml ps

# Inspect API health endpoint
curl -s http://localhost/health | jq .

# Inspect active profiles
curl -s http://localhost/v1/profiles | jq .

# Check container logs
docker compose -f deploy/docker-compose.prod.yml logs -f api
docker compose -f deploy/docker-compose.prod.yml logs -f bot
```

Expected health response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "active_profiles": ["smartphone_clinical"],
  "model_status": {
    "smartphone_clinical": "loaded"
  }
}
```

---

## 7. Systemd Service Configuration (Auto-Start)

Create systemd unit file at `/etc/systemd/system/skinai.service`:
```ini
[Unit]
Description=SkinAI Docker Compose Application Service
Requires=docker.service
After=docker.service network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/skinai
ExecStart=/usr/bin/docker compose -f deploy/docker-compose.prod.yml up -d --remove-orphans
ExecStop=/usr/bin/docker compose -f deploy/docker-compose.prod.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Enable and activate the unit:
```bash
sudo systemctl daemon-reload
sudo systemctl enable skinai.service
```

---

## 8. Database Telemetry Backup Runbook

To safeguard anonymous audit logs and metadata without stopping services:

### Step 8.1: Create Backup Script `/opt/skinai/scripts/backup_db.sh`
```bash
#!/bin/bash
set -euo pipefail

BACKUP_DIR="/opt/skinai/backups/postgres"
DATE=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$BACKUP_DIR"

echo "Starting SkinAI telemetry backup..."
docker exec skinai_postgres pg_dump -U skinai_prod_user -d skinai_telemetry | gzip > "${BACKUP_DIR}/telemetry_${DATE}.sql.gz"

# Retain only last 14 daily backups
find "$BACKUP_DIR" -type f -name "telemetry_*.sql.gz" -mtime +14 -delete
echo "Backup completed: ${BACKUP_DIR}/telemetry_${DATE}.sql.gz"
```

```bash
chmod +x /opt/skinai/scripts/backup_db.sh
```

### Step 8.2: Configure Daily Cron
Add entry via `crontab -e`:
```cron
0 3 * * * /opt/skinai/scripts/backup_db.sh >> /var/log/skinai_backup.log 2>&1
```

---

## 9. Zero-Downtime Rolling Update & Rollback

### Updating Application or Bot Code
```bash
cd /opt/skinai
git pull origin master
docker compose -f deploy/docker-compose.prod.yml up -d --build --no-deps api
docker compose -f deploy/docker-compose.prod.yml up -d --build --no-deps bot
```

### Hot Model Rollback
If a newly deployed model exhibits anomalies:
1. Edit `/opt/skinai/.env` to point `SKINAI_ACTIVE_SMARTPHONE_MODEL` back to previous version.
2. Restart only the API service:
   ```bash
   docker compose -f deploy/docker-compose.prod.yml restart api
   ```
3. Confirm active profile version in `curl -s http://localhost/v1/profiles | jq .`.

---

## 10. Incident Triage & Emergency Abstention

If the model behaves erratically or upstream issues are detected:
- The bot features an instant abstention fallback: when API returns HTTP 503 or quality checks fail, the bot safely displays:
  > *"Не удалось получить достоверную оценку снимка. Пожалуйста, обратитесь на очный приём к сертифицированному дерматологу."*
- To shut down services immediately in an incident:
  ```bash
  docker compose -f deploy/docker-compose.prod.yml stop
  ```
