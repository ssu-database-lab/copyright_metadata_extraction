# Oracle Cloud Deployment Guide

Step-by-step record of deploying the Copyright Metadata Extraction API to Oracle Cloud Infrastructure (OCI). Includes every issue encountered and how it was resolved.

---

## Server Details

| Item | Value |
|------|-------|
| Provider | Oracle Cloud Infrastructure (OCI) |
| Region | uk-london-1 |
| Shape | VM.Standard3.Flex |
| CPU | 2 OCPU (4 threads) |
| RAM | 32 GB |
| Disk | 45 GB |
| OS | Ubuntu 24.04 LTS |
| Username | ubuntu |
| Public IP | 150.230.114.9 |
| API Port | 5000 |

---

## Step 1: Create the Instance

Done through the OCI Console (https://cloud.oracle.com). Key settings:
- Shape: VM.Standard3.Flex (2 OCPU, 32GB RAM)
- Image: Canonical Ubuntu 24.04
- VCN: auto-created
- SSH key: generated and downloaded (`ssh-key-2026-03-25.key`)

### Issue: No Public IP

After creating the instance, the public IP showed `-` (not assigned).

**Cause:** By default, OCI instances in a public subnet don't automatically get a public IP. Two things are needed:
1. An **Internet Gateway** + route table (so the subnet can reach the internet)
2. A **public IP** assigned to the VNIC

**Solution:**
1. In the instance page → Quick Actions → clicked **"Connect public subnet to internet"**
   - This auto-creates: internet gateway, NSG (network security group), route table
2. Scrolled to Primary VNIC → clicked the VNIC → IPv4 Addresses → Edit → selected **"Ephemeral public IP"**
3. Got public IP: `150.230.114.9`

---

## Step 2: Configure Network Security

### Opening Port 22 (SSH) and Port 5000 (API)

In OCI Console → Networking → VCN → Network Security Groups → `ig-quick-action-NSG`:

Added two **Ingress Rules**:

| Field | Port 22 (SSH) | Port 5000 (API) |
|-------|--------------|-----------------|
| Stateless | No | No |
| Direction | Ingress | Ingress |
| Source Type | CIDR | CIDR |
| Source | 0.0.0.0/0 | 0.0.0.0/0 |
| IP Protocol | TCP | TCP |
| Destination Port Range | 22 | 5000 |

**Important:** This is the OCI-level firewall (cloud network). There's also an OS-level firewall (see Step 5).

---

## Step 3: SSH Connection

### From WSL2

```bash
# Copy key and set permissions
cp /path/to/ssh-key-2026-03-25.key ~/.ssh/oracle_key.pem
chmod 600 ~/.ssh/oracle_key.pem

# Connect
ssh -i ~/.ssh/oracle_key.pem ubuntu@150.230.114.9
```

### SSH Config (for convenience)

Added to `~/.ssh/config`:
```
Host oracle
    HostName 150.230.114.9
    User ubuntu
    IdentityFile ~/.ssh/oracle_key.pem
```

Now just: `ssh oracle`

### From Windows

Added to `C:\Users\mbmk9\.ssh\config`:
```
Host oracle_cr
    HostName 150.230.114.9
    User ubuntu
    IdentityFile C:\Users\mbmk9\.ssh\oracle_key.pem
```

**Issue:** "Host key verification failed"

**Cause:** First-time connection asks "Are you sure you want to continue connecting (yes/no)?". If you don't type `yes`, it fails.

**Solution:** Type `yes` when prompted.

**Issue:** "Permission denied (publickey)"

**Cause:** The private key file wasn't at the path specified in the config.

**Solution:** Copy the key to `C:\Users\mbmk9\.ssh\oracle_key.pem`:
```powershell
copy "C:\path\to\ssh-key-2026-03-25.key" "C:\Users\mbmk9\.ssh\oracle_key.pem"
```

### Key Concept: SSH Key Pair

```
Your laptop:   ssh-key-2026-03-25.key      (private key → you keep this)
Oracle server: ssh-key-2026-03-25.key.pub   (public key → already installed on server)
```

You only need the private key on your laptop. The public key was uploaded to the server during instance creation. You never need to copy the `.pub` file anywhere.

---

## Step 4: Deploy the Application

### 4.1 Install System Dependencies

```bash
ssh oracle "sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip python3-venv"
```

The server had Python 3.12 and git pre-installed, but not pip or venv.

### 4.2 Transfer Package

```bash
scp copyright_extraction_cli.tar.gz oracle:~/
```

Transferred the 1.2GB package to the server.

### 4.3 Extract and Setup

```bash
ssh oracle << 'DEPLOY'
cd ~
tar xzf copyright_extraction_cli.tar.gz
cd copyright_extraction_cli

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q

# Install dependencies
pip install -r requirements.txt -q

# Configure Python path
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$(pwd)/api" > "$SITE/copyright-metadata.pth"

# Verify
python extract.py --list-models
DEPLOY
```

### 4.4 Transfer Web Server Files

The CLI package didn't include the web server (`app.py`, templates). These were transferred separately:

```bash
# From local machine
tar czf /tmp/web_server_files.tar.gz api/web/app.py api/web/templates/ api/module/pdf_to_image.py api/module/ocr_system.py api/api.py api/__init__.py api/module/__init__.py
scp /tmp/web_server_files.tar.gz oracle:~/copyright_extraction_cli/
ssh oracle "cd ~/copyright_extraction_cli && tar xzf web_server_files.tar.gz && rm web_server_files.tar.gz"
```

**Lesson learned:** The CLI package (`copyright_extraction_cli.tar.gz`) was designed for CLI-only use. For the web server, additional files are needed: `app.py`, `templates/`, `pdf_to_image.py`, `ocr_system.py`, and the full `api/__init__.py` and `module/__init__.py`.

### Issue: Missing `pdf_to_image` Module

```
ModuleNotFoundError: No module named 'module.pdf_to_image'
```

**Cause:** The CLI package excluded `pdf_to_image.py` since the CLI tool doesn't use it directly. But `app.py` imports it through `api/__init__.py`.

**Solution:** Transferred the missing file from the local machine.

### Issue: `mistralai` Import Error

```
ImportError: cannot import name 'Mistral' from 'mistralai'
```

**Cause:** `pip install mistralai` installed version 2.1.3, which has a different API from version 1.x. Our code uses `from mistralai import Mistral` which is the v1 API.

**Solution:**
```bash
pip install 'mistralai>=1.0.0,<2.0.0'
```

**Lesson learned:** Always pin major versions in `requirements.txt` to avoid breaking changes. Updated requirements should specify `mistralai>=1.0.0,<2.0.0`.

### 4.5 Start the Server

```bash
cd ~/copyright_extraction_cli
source venv/bin/activate
mkdir -p api/web/uploads api/web/results api/web/temp

# Start in background
nohup python -m uvicorn web.app:app --host 0.0.0.0 --port 5000 --app-dir api > server.log 2>&1 &
```

- `nohup` — keeps running after SSH disconnects
- `--host 0.0.0.0` — listen on all interfaces (not just localhost)
- `--port 5000` — our API port
- `--app-dir api` — tells uvicorn the app is in the `api/` directory
- `> server.log 2>&1 &` — log to file, run in background

---

## Step 5: OS-Level Firewall (iptables)

### Issue: API Not Accessible from Outside

After starting the server, `curl http://150.230.114.9:5000/docs` timed out. But from inside the server, `curl http://localhost:5000/docs` returned 200.

**Cause:** Oracle Cloud Ubuntu images come with **iptables** pre-configured. The default rules only allow SSH (port 22) and reject everything else:

```
Chain INPUT (policy ACCEPT)
1  ACCEPT  state RELATED,ESTABLISHED
2  ACCEPT  icmp
3  ACCEPT  loopback
4  ACCEPT  tcp dpt:22 (SSH only)
5  REJECT  everything else    ← port 5000 blocked here!
```

This is **separate from** the OCI Network Security Group. OCI has two firewalls:
1. **Cloud-level:** NSG / Security Lists (configured in OCI Console) → we already opened port 5000 here
2. **OS-level:** iptables on the Ubuntu instance → this was still blocking

**Solution:**
```bash
# Add port 5000 rule BEFORE the REJECT rule (insert at position 5)
sudo iptables -I INPUT 5 -p tcp --dport 5000 -j ACCEPT

# Save rules to persist across reboots
sudo mkdir -p /etc/iptables
sudo sh -c 'iptables-save > /etc/iptables/rules.v4'
```

After this change:
```
Chain INPUT (policy ACCEPT)
1  ACCEPT  state RELATED,ESTABLISHED
2  ACCEPT  icmp
3  ACCEPT  loopback
4  ACCEPT  tcp dpt:22
5  ACCEPT  tcp dpt:5000    ← new rule
6  REJECT  everything else
```

**Key lesson:** On Oracle Cloud Ubuntu, you must open ports in BOTH:
1. OCI Console → NSG (cloud firewall)
2. Server → iptables (OS firewall)

---

## Step 6: Verify

```bash
# From inside server
curl -s http://localhost:5000/docs | head -1

# From outside (your laptop)
curl -s http://150.230.114.9:5000/docs | head -1

# Swagger UI in browser
# http://150.230.114.9:5000/docs
```

---

## Server Management Commands

### Check if server is running
```bash
ssh oracle "ps aux | grep uvicorn | grep -v grep"
```

### View server logs
```bash
ssh oracle "tail -50 ~/copyright_extraction_cli/server.log"
```

### Restart server
```bash
ssh oracle << 'RESTART'
cd ~/copyright_extraction_cli
source venv/bin/activate
pkill -f "uvicorn web.app" 2>/dev/null || true
sleep 1
nohup python -m uvicorn web.app:app --host 0.0.0.0 --port 5000 --app-dir api > server.log 2>&1 &
echo "Restarted. PID: $!"
RESTART
```

### Stop server
```bash
ssh oracle "pkill -f 'uvicorn web.app'"
```

### Check server resource usage
```bash
ssh oracle "free -h && echo '' && df -h / && echo '' && uptime"
```

---

## Auto-Start on Reboot (Optional)

To make the server start automatically when the instance reboots:

```bash
ssh oracle << 'AUTOSTART'
sudo tee /etc/systemd/system/copyright-api.service << 'EOF'
[Unit]
Description=Copyright Metadata Extraction API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/copyright_extraction_cli
Environment="PATH=/home/ubuntu/copyright_extraction_cli/venv/bin:/usr/bin"
ExecStart=/home/ubuntu/copyright_extraction_cli/venv/bin/python -m uvicorn web.app:app --host 0.0.0.0 --port 5000 --app-dir api
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable copyright-api
sudo systemctl start copyright-api
echo "Service created and started"
sudo systemctl status copyright-api
AUTOSTART
```

Then manage with:
```bash
sudo systemctl start copyright-api
sudo systemctl stop copyright-api
sudo systemctl restart copyright-api
sudo systemctl status copyright-api
journalctl -u copyright-api -f   # follow logs
```

---

## Summary of Issues and Solutions

| # | Issue | Cause | Solution |
|---|-------|-------|----------|
| 1 | No public IP after instance creation | OCI doesn't auto-assign public IPs | Quick Action → "Connect public subnet to internet" + assign ephemeral IP |
| 2 | SSH "Host key verification failed" | First connection needs manual approval | Type `yes` when prompted |
| 3 | SSH "Permission denied" on Windows | Key file not at the path in SSH config | Copy key to `C:\Users\mbmk9\.ssh\oracle_key.pem` |
| 4 | `ModuleNotFoundError: pdf_to_image` | CLI package didn't include web server modules | Transfer missing files: `pdf_to_image.py`, `ocr_system.py`, `api/__init__.py` |
| 5 | `ImportError: cannot import name 'Mistral'` | `mistralai` v2.x has different API from v1.x | `pip install 'mistralai>=1.0.0,<2.0.0'` |
| 6 | API works locally but not from outside | Oracle Ubuntu iptables blocks all ports except 22 | `sudo iptables -I INPUT 5 -p tcp --dport 5000 -j ACCEPT` |
| 7 | iptables rules lost on reboot | Rules are in-memory by default | `sudo iptables-save > /etc/iptables/rules.v4` |
| 8 | Web UI returns "Internal Server Error" (500) | FastAPI/Starlette version mismatch — latest pip installed 0.135/1.0 but code written for 0.109 | `pip install 'fastapi==0.109.0' 'starlette<1.0.0'` |

### Issue #8 Detail: FastAPI Version Mismatch

The `requirements.txt` had `fastapi>=0.109.0` which allowed pip to install the latest (0.135.2). Between 0.109 and 0.135, Starlette upgraded from 0.x to 1.0 and changed the `TemplateResponse` API:

```python
# FastAPI 0.109 (our code) — positional arguments
templates.TemplateResponse("index.html", {"request": request, "models": AVAILABLE_MODELS})

# FastAPI 0.135 / Starlette 1.0 — different internal handling
# The same call triggers: TypeError: unhashable type: 'dict'
```

**Root cause:** Using `>=` version specifier allows breaking upgrades. On our dev machine we had 0.109 installed already. On the fresh server, pip installed the latest.

**Fix:** Pin exact version: `fastapi==0.109.0`

**Lesson:** Always pin major dependencies to exact versions in production deployments:
```
# Bad (allows breaking upgrades)
fastapi>=0.109.0

# Good (reproducible)
fastapi==0.109.0
```

---

## Architecture

```
Internet
    │
    ▼
OCI NSG (cloud firewall) ── allows port 22, 5000
    │
    ▼
Ubuntu iptables (OS firewall) ── allows port 22, 5000
    │
    ▼
uvicorn (FastAPI server) ── listens on 0.0.0.0:5000
    │
    ▼
PipelineOrchestrator
├── OCR (Alibaba Cloud API)
├── LLM Extraction (Alibaba Cloud API)
├── NER (local KLUE-RoBERTa-Large)
└── Consolidation (Alibaba Cloud API)
```

## Access Points

| URL | Purpose |
|-----|---------|
| http://150.230.114.9:5000/docs | Swagger API documentation |
| http://150.230.114.9:5000/api/llm-extract | Main extraction endpoint |
| http://150.230.114.9:5000/health | Health check |
