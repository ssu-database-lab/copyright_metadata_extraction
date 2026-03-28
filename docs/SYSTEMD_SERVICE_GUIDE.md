# Systemd Service Guide — Auto-Start API on Boot

## What is systemd?

systemd is Linux's service manager. It controls what programs start automatically when the server boots, and monitors them to restart if they crash. Every long-running process on a Linux server (web servers, databases, etc.) typically runs as a systemd service.

Without systemd, you'd need to manually SSH into the server and run the command every time the server reboots. With systemd, it just starts automatically.

## The Problem

When we first deployed, we started the server with:
```bash
nohup python -m uvicorn web.app:app --host 0.0.0.0 --port 5000 --app-dir api > server.log 2>&1 &
```

This works, but:
- **Server reboots** → the process is gone, API is down
- **Process crashes** → stays dead until you manually restart
- **No standard way to check status** → have to use `ps aux | grep uvicorn`
- **Logs are in a random file** → `server.log` in the home directory

## The Solution: systemd Service

### The Service File

We created `/etc/systemd/system/copyright-api.service`:

```ini
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
```

### Line-by-Line Explanation

#### `[Unit]` Section — What is this service?

```ini
Description=Copyright Metadata Extraction API
```
Human-readable name. Shows up in `systemctl status` and logs.

```ini
After=network.target
```
Don't start this service until the network is ready. Our API needs internet (for Alibaba Cloud API calls), so we wait for network.

#### `[Service]` Section — How to run it?

```ini
Type=simple
```
The process runs in the foreground (not a daemon). uvicorn runs this way by default.

```ini
User=ubuntu
```
Run as the `ubuntu` user, not root. Security best practice — if the API is compromised, the attacker only has `ubuntu` permissions, not root.

```ini
WorkingDirectory=/home/ubuntu/copyright_extraction_cli
```
`cd` to this directory before running the command. Important because the app expects to find `.env`, `api/`, etc. relative to this path.

```ini
Environment="PATH=/home/ubuntu/copyright_extraction_cli/venv/bin:/usr/bin"
```
Sets the PATH to include our virtual environment's `bin/` directory. This is how systemd knows to use the venv's Python (with all our pip packages) instead of the system Python (which doesn't have our dependencies).

Without this: `ModuleNotFoundError: No module named 'fastapi'`
With this: uses the venv Python that has fastapi, torch, transformers, etc.

```ini
ExecStart=/home/ubuntu/copyright_extraction_cli/venv/bin/python -m uvicorn web.app:app --host 0.0.0.0 --port 5000 --app-dir api
```
The actual command to start the server. Note: **full absolute path** to the venv Python, not just `python`. systemd doesn't activate virtual environments — we point directly to the venv's Python binary.

```ini
Restart=always
```
If the process exits (crash, OOM kill, unhandled exception), systemd automatically restarts it. Options:
- `always` — restart no matter why it stopped
- `on-failure` — restart only if it crashed (non-zero exit code)
- `no` — never restart

```ini
RestartSec=5
```
Wait 5 seconds between restart attempts. Prevents rapid restart loops if there's a persistent error (like a missing config file).

#### `[Install]` Section — When to start it?

```ini
WantedBy=multi-user.target
```
Start this service when the system reaches "multi-user" mode (normal boot, command-line ready). This is the standard target for server applications. Other options:
- `graphical.target` — after desktop GUI is ready (not relevant for servers)
- `network-online.target` — after network is fully online (stricter than `network.target`)

### Setup Commands

```bash
# 1. Create the service file
sudo tee /etc/systemd/system/copyright-api.service << 'EOF'
[Unit]
Description=Copyright Metadata Extraction API
After=network.target
...
EOF

# 2. Tell systemd to reload its configuration (it reads the new file)
sudo systemctl daemon-reload

# 3. Enable — makes it start automatically on boot
sudo systemctl enable copyright-api
# This creates a symlink:
# /etc/systemd/system/multi-user.target.wants/copyright-api.service
# → /etc/systemd/system/copyright-api.service

# 4. Start — start it right now (without rebooting)
sudo systemctl start copyright-api
```

## Management Commands

### Check status
```bash
sudo systemctl status copyright-api
```
Output:
```
● copyright-api.service - Copyright Metadata Extraction API
     Loaded: loaded (/etc/systemd/system/copyright-api.service; enabled; preset: enabled)
     Active: active (running) since Sat 2026-03-28 17:33:54 UTC; 3s ago
   Main PID: 14605 (python)
     Memory: 370.5M
```

Key fields:
- `enabled` — will auto-start on boot
- `active (running)` — currently running
- `Main PID: 14605` — the process ID
- `Memory: 370.5M` — current memory usage

### Start / Stop / Restart
```bash
sudo systemctl start copyright-api      # start
sudo systemctl stop copyright-api       # stop
sudo systemctl restart copyright-api    # stop + start
```

### View Logs
```bash
# Follow logs in real-time (like tail -f)
journalctl -u copyright-api -f

# Last 50 lines
journalctl -u copyright-api -n 50

# Logs since last boot
journalctl -u copyright-api -b

# Logs from the last hour
journalctl -u copyright-api --since "1 hour ago"
```

systemd captures all stdout/stderr from the process and stores it in the journal. No need for `> server.log` redirection — `journalctl` handles it.

### Disable auto-start
```bash
sudo systemctl disable copyright-api    # won't start on boot
sudo systemctl enable copyright-api     # re-enable auto-start
```

### Check if enabled
```bash
systemctl is-enabled copyright-api
# Output: enabled  or  disabled
```

## How It Works on Reboot

```
Server powers on
    │
    ▼
Linux kernel boots
    │
    ▼
systemd starts (PID 1)
    │
    ▼
Reaches network.target (network ready)
    │
    ▼
Reaches multi-user.target
    │
    ├── starts sshd (SSH server)
    ├── starts copyright-api.service    ← our API starts here
    └── starts other enabled services
    │
    ▼
API is running on port 5000
(no manual intervention needed)
```

## How It Works on Crash

```
API process crashes (exception, OOM, etc.)
    │
    ▼
systemd detects process exit
    │
    ▼
Waits 5 seconds (RestartSec=5)
    │
    ▼
Starts the process again (Restart=always)
    │
    ▼
API is back up
(logged in journalctl: "Started Copyright Metadata Extraction API")
```

## Comparison: nohup vs systemd

| Feature | nohup | systemd |
|---------|-------|---------|
| Survives SSH disconnect | Yes | Yes |
| Survives server reboot | **No** | **Yes** |
| Auto-restart on crash | No | Yes |
| Standard status command | No (`ps aux \| grep`) | Yes (`systemctl status`) |
| Centralized logs | No (custom log file) | Yes (`journalctl`) |
| Resource monitoring | No | Yes (Memory, CPU in status) |
| Start/stop commands | `kill PID` | `systemctl start/stop` |
| Dependency management | No | Yes (`After=network.target`) |

## Common Issues

### Service won't start
```bash
# Check what went wrong
journalctl -u copyright-api -n 30
```

### "Failed to start" — Python not found
```
ExecStart=/home/ubuntu/.../venv/bin/python: No such file or directory
```
**Fix:** Check the venv path exists: `ls /home/ubuntu/copyright_extraction_cli/venv/bin/python`

### ModuleNotFoundError
The `Environment` line might not include the venv path. Check:
```ini
Environment="PATH=/home/ubuntu/copyright_extraction_cli/venv/bin:/usr/bin"
```

### Port already in use
```
ERROR: [Errno 98] Address already in use
```
**Fix:** Kill the old process first:
```bash
sudo systemctl stop copyright-api
pkill -f "uvicorn web.app"  # kill any stray processes
sudo systemctl start copyright-api
```

### After editing the service file
```bash
sudo systemctl daemon-reload    # re-read the file
sudo systemctl restart copyright-api  # apply changes
```
