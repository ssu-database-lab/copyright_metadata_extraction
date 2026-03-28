# Runpod Deployment Guide

This guide will help you pull the latest changes and test the web application on your Runpod server.

## Prerequisites

- SSH access to your Runpod server
- Project already cloned on the server (if not, clone it first)

## Step 1: Connect to Runpod Server

```bash
# SSH into your Runpod instance
ssh root@<your-runpod-ip>
# Or if you have a specific username/path:
# ssh <username>@<your-runpod-ip>
```

## Step 2: Navigate to Project Directory

```bash
# Navigate to where your project is located
cd /path/to/copyright_metadata_extraction
# Common locations:
# cd ~/copyright_metadata_extraction
# cd /workspace/copyright_metadata_extraction
```

## Step 3: Pull Latest Changes

```bash
# Check current status
git status

# Pull latest changes from GitHub
git pull origin main

# If you have local changes that conflict, you may need to:
# git stash  # Save local changes
# git pull origin main
# git stash pop  # Reapply local changes if needed
```

## Step 4: Verify Environment Variables

The application requires environment variables for OCR providers. Check if `.env` files exist:

```bash
# Check for environment files
ls -la api/.env_alibaba
ls -la api/web/.env_alibaba
ls -la OCR/google_vision/.env_alibaba

# If missing, you'll need to create them with your API keys
# Example for Alibaba Cloud:
# echo "ALIBABA_ACCESS_KEY_ID=your_key" > api/.env_alibaba
# echo "ALIBABA_ACCESS_KEY_SECRET=your_secret" >> api/.env_alibaba
# echo "ALIBABA_ENDPOINT=your_endpoint" >> api/.env_alibaba
```

## Step 5: Install/Update Dependencies

```bash
# Make sure you're in the project root or api directory
cd api

# Install/update Python dependencies
pip install -r requirements.txt

# If you need to install dependencies from other modules:
pip install -r module/llm_extraction/requirements.txt
```

## Step 6: Check Required Directories

The application creates these directories automatically, but verify they exist:

```bash
# Check/create directories
mkdir -p api/web/uploads
mkdir -p api/web/results
mkdir -p api/web/temp
```

## Step 7: Run the Web Application

### Option A: Direct Python Execution

```bash
# Navigate to the web directory
cd api/web

# Run the application
python app.py
```

### Option B: Using uvicorn directly

```bash
# From api/web directory
cd api/web

# Run with uvicorn
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

### Option C: Background Process (for testing)

```bash
# Run in background
cd api/web
nohup python app.py > app.log 2>&1 &

# Or with screen/tmux for better process management
screen -S webapp
cd api/web
python app.py
# Press Ctrl+A then D to detach
```

## Step 8: Access the Web Application

### ⚠️ IMPORTANT: RunPod Port Configuration

**RunPod only exposes specific ports.** By default, RunPod exposes:
- Port `8888` for Jupyter Lab (HTTP)
- Port `22` for SSH (via exposed TCP port)

**Port 5000 is NOT automatically exposed.** You have several options:

### Option 1: SSH Port Forwarding (Recommended for Testing)

Create an SSH tunnel from your local machine to access port 5000:

```bash
# On your LOCAL machine (not RunPod), create SSH tunnel:
ssh -L 5000:localhost:5000 root@50.145.48.94 -p 13346 -i ~/.ssh/id_ed25519

# Replace with your actual:
# - IP address: 50.145.48.94
# - SSH port: 13346 (from RunPod Connect tab)
# - SSH key path: ~/.ssh/id_ed25519

# Then access via your LOCAL browser:
# http://localhost:5000
```

**Keep the SSH terminal open** - closing it will disconnect the tunnel.

### Option 2: Use RunPod's HTTP Proxy (Port 8888)

Change your application to run on port 8888 instead:

```bash
# On RunPod, edit the app.py file:
cd api/web
sed -i 's/port=5000/port=8888/' app.py

# Or modify the uvicorn.run line at the bottom:
# Change: uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
# To:     uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")

# Restart the application
python app.py
```

Then access via RunPod's HTTP proxy URL (shown in the Connect tab).

### Option 3: Configure NGINX Reverse Proxy

Since RunPod has NGINX pre-installed, configure it to proxy port 8888 to your app on 5000:

```bash
# On RunPod, create NGINX config:
sudo nano /etc/nginx/sites-available/default

# Add location block:
# location / {
#     proxy_pass http://localhost:5000;
#     proxy_set_header Host $host;
#     proxy_set_header X-Real-IP $remote_addr;
# }

# Reload NGINX:
sudo nginx -t
sudo systemctl reload nginx
```

Then access via RunPod's HTTP proxy URL (port 8888).

### Option 4: Expose Port 5000 via RunPod UI (If Available)

1. Go to RunPod pod settings
2. Look for "Ports" or "Network" configuration
3. Add port mapping: `5000 → 5000` (HTTP)
4. Save and wait for the pod to restart

### Access URLs (after configuration):

- **Web Interface**: `http://localhost:5000` (with SSH tunnel) or `http://<runpod-http-proxy-url>` (via RunPod proxy)
- **API Documentation**: `http://localhost:5000/docs` (with SSH tunnel)
- **Health Check**: `http://localhost:5000/health` (with SSH tunnel)

## Step 9: Check Exposed Ports

### Check Ports on RunPod Server

```bash
# Check all listening ports on the server
netstat -tulpn | grep LISTEN

# Or using ss (more modern)
ss -tulpn | grep LISTEN

# Check specific port
lsof -i :5000
lsof -i :8888

# Check all listening ports (simplified)
netstat -tlnp
```

### Check RunPod Port Configuration

**Via RunPod Web UI:**
1. Go to your pod's "Connect" tab
2. Look at the "HTTP Services" section - shows proxied HTTP ports
3. Look at the "Direct TCP Ports" section - shows exposed TCP ports

**Via RunPod API (if available):**
```bash
# Check RunPod environment variables (may contain port info)
env | grep -i port
env | grep -i runpod

# Check RunPod config files (if accessible)
cat /runpod/config.json 2>/dev/null || echo "Config not accessible"
```

### Check What's Running on Each Port

```bash
# Check what process is using port 5000
sudo lsof -i :5000

# Check what process is using port 8888
sudo lsof -i :8888

# Check all processes listening on ports
sudo netstat -tulpn | grep LISTEN

# More detailed view with process names
sudo ss -tulpn | grep LISTEN
```

### Quick Port Check Script

Create a script `check_ports.sh`:

```bash
#!/bin/bash
echo "=== Listening Ports ==="
sudo netstat -tulpn | grep LISTEN | awk '{print $4}' | awk -F: '{print $NF}' | sort -n | uniq

echo ""
echo "=== Port 5000 Status ==="
if sudo lsof -i :5000 &>/dev/null; then
    echo "✓ Port 5000 is in use"
    sudo lsof -i :5000
else
    echo "✗ Port 5000 is not in use"
fi

echo ""
echo "=== Port 8888 Status ==="
if sudo lsof -i :8888 &>/dev/null; then
    echo "✓ Port 8888 is in use"
    sudo lsof -i :8888
else
    echo "✗ Port 8888 is not in use"
fi

echo ""
echo "=== Test Port Accessibility ==="
echo "Testing localhost:5000..."
curl -s http://localhost:5000/health > /dev/null && echo "✓ Port 5000 accessible" || echo "✗ Port 5000 not accessible"

echo "Testing localhost:8888..."
curl -s http://localhost:8888 > /dev/null && echo "✓ Port 8888 accessible" || echo "✗ Port 8888 not accessible"
```

Make it executable:
```bash
chmod +x check_ports.sh
./check_ports.sh
```

## Step 10: Test the Application

1. **Health Check**:
   ```bash
   curl http://localhost:5000/health
   ```

2. **Open in Browser**:
   - Navigate to `http://<runpod-ip>:5000`
   - You should see the web interface

3. **Test Features**:
   - Upload a PDF/image
   - Select OCR provider
   - Select NER model
   - Process and verify results

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 5000
lsof -i :5000
# or
netstat -tulpn | grep 5000

# Kill the process if needed
kill -9 <PID>
```

### Missing Dependencies

```bash
# Check Python version (should be 3.8+)
python3 --version

# Install missing packages
pip install --upgrade pip
pip install -r api/requirements.txt
```

### Permission Issues

```bash
# Make sure directories are writable
chmod -R 755 api/web/uploads
chmod -R 755 api/web/results
chmod -R 755 api/web/temp
```

### Environment Variables Not Loading

```bash
# Check if .env files exist and are readable
ls -la api/.env*
cat api/.env_alibaba  # Verify content (be careful with secrets)

# Test loading
python3 -c "from dotenv import load_dotenv; load_dotenv('api/.env_alibaba'); import os; print(os.getenv('ALIBABA_ACCESS_KEY_ID'))"
```

### Check Logs

```bash
# If running in background
tail -f app.log

# If running with screen
screen -r webapp

# Check application output
journalctl -u your-service-name  # If running as service
```

## Quick Test Script

Create a test script `test_server.sh`:

```bash
#!/bin/bash
echo "Testing Runpod Web Application..."

# Check if server is running
if curl -s http://localhost:5000/health > /dev/null; then
    echo "✓ Server is running"
    curl http://localhost:5000/health | python3 -m json.tool
else
    echo "✗ Server is not responding"
    echo "Starting server..."
    cd api/web
    python app.py &
fi
```

Make it executable:
```bash
chmod +x test_server.sh
./test_server.sh
```

## Notes

- The application runs on port 5000 by default
- Make sure the port is open in Runpod's firewall settings
- For production, consider using a process manager like `systemd`, `supervisor`, or `pm2`
- For HTTPS, you'll need to set up a reverse proxy (nginx, Caddy, etc.)

