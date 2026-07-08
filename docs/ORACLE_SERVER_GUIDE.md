# Oracle Cloud Server — Access & Infrastructure Guide

> Handoff doc for hosting a website on the shared Oracle Cloud VM. Snapshot taken **2026-07-08**.
> The box already runs several production services behind nginx — **read the "Do not disturb" and
> "Disk" sections before deploying anything.**

---

## 1. Connecting

| | |
|---|---|
| Public IP | **150.230.114.9** |
| SSH user | `ubuntu` |
| SSH alias | `ssh oracle` (configured in `~/.ssh/config`) |
| Private key | `~/.ssh/oracle_key.pem` (mode 600; **on the operator's machine only — never commit or paste it**) |

```bash
ssh oracle                 # if ~/.ssh/config has the Host oracle block
# or explicitly:
ssh -i ~/.ssh/oracle_key.pem ubuntu@150.230.114.9
```

`~/.ssh/config` block (recreate on a new machine; copy the key file over securely, `chmod 600`):
```
Host oracle
    HostName 150.230.114.9
    User ubuntu
    IdentityFile ~/.ssh/oracle_key.pem
```
`sudo` works without a password for the `ubuntu` user.

---

## 2. Host summary

| Resource | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS (kernel 6.17 oracle) |
| CPU / RAM | 4 vCPU / 31 GiB (ARM? no — x86_64) |
| Disk | **45 GB root, 94% FULL — only ~2.8 GB free** ⚠️ |
| Web server | **nginx** (reverse proxy, 80/443, Certbot TLS) |
| Containers | **Docker** (2 containers running) |

⚠️ **Disk is the #1 constraint.** Free space before deploying — see §6.

---

## 3. What's running (port map)

Publicly reachable (through the OS firewall, see §5):

| Port | Service | Notes |
|---|---|---|
| 22 | sshd | |
| 80 / 443 | **nginx** | reverse proxy → the apps below; Certbot TLS |
| 3000 | Next.js (`next-server`) | public bind `0.0.0.0` |
| **5000** | **`copyright-api`** (our FastAPI) | **our metadata pipeline — do not disturb** |
| 8000 | `fastapi_backend` (Docker) | container `backend-backend` |
| 8001 | uvicorn (`tfg` conversational AI agent) | bound localhost, proxied by nginx |
| 5432 | `postgres_db` (Docker, postgres:15) | ⚠️ exposed publicly |

Localhost-only (proxied via nginx or internal): 3002 (admin-tfg Next.js), 3001 (langfuse target), plus assorted node worker ports.

**nginx sites (`/etc/nginx/sites-enabled/`):**
| Site | Domain | proxy_pass |
|---|---|---|
| `tfg` | `tfg-seasea.duckdns.org` | `127.0.0.1:8001` |
| `admin-tfg` | `admin.tfg-seasea.duckdns.org` | `127.0.0.1:3002` |
| `langfuse` | `langfuse.tfg-seasea.duckdns.org` | `127.0.0.1:3001` |
| `default` | `_` | `/var/www/html` |

TLS: **Certbot + Let's Encrypt**, certs under `/etc/letsencrypt/live/<domain>/`. Domains use **DuckDNS** (`*.tfg-seasea.duckdns.org`).

**Other apps on disk:** `/home/ubuntu/tfg-app` (the TFG Next.js + AI agent stack), `/home/ubuntu/tfg-app-claude`, `/home/ubuntu/bondeal`.

---

## 4. ⚠️ Do NOT disturb (shared production box)

- **`copyright-api`** on port 5000 — our project. Managed by systemd (`/etc/systemd/system/copyright-api.service`), venv at `/home/ubuntu/copyright_extraction_cli/venv`. See the project's `docs/DEPLOYMENT_GUIDE.md`.
- **The TFG stack** (tfg-app: ports 8001/3000/3002, langfuse, the Docker `fastapi_backend` + `postgres_db`). These belong to another project on the same VM.
- Don't `docker system prune -a --volumes` blindly — the postgres volume (**14.9 GB**) holds another project's DB.
- Don't take ports **22, 80, 443, 3000, 3002, 5000, 8000, 8001, 5432**.

---

## 5. OS firewall (iptables)

Open INPUT ports: **22, 80, 443, 3000, 5000, 8000, 8001** (+ ESTABLISHED/RELATED). Rules are persisted (iptables-persistent). Oracle Cloud **also** has a cloud-level Security List / NSG in the OCI console — a port must be open in *both* iptables and the OCI Security List to be reachable from the internet.

**Best practice for a new site: don't open a new public app port at all.** Bind your app to `127.0.0.1:<port>` and let **nginx (already open on 80/443)** reverse-proxy to it. No firewall change needed.

If you truly must expose a raw port:
```bash
sudo iptables -I INPUT -p tcp --dport <PORT> -j ACCEPT
sudo netfilter-persistent save        # persist
# AND open <PORT> in the OCI console → VCN → Security List → Ingress Rules
```

---

## 6. Free disk space FIRST (disk is 94% full)

Safe reclaims (do at least the first two before deploying):
```bash
rm /home/ubuntu/copyright_extraction_cli.tar.gz         # old CLI bundle → +1.2 GB
sudo docker image prune -f                              # dangling images → ~+0.84 GB
sudo apt-get clean && sudo journalctl --vacuum-size=200M
```
Do **NOT** prune the postgres Docker volume (14.9 GB — another project's live data).

---

## 7. How to host a NEW website (recommended pattern)

The box already does this for 3 sites — follow the same recipe.

**A. Run your app on a free localhost port.** Free right now: **4000, 8080, 8081, 9000, 3003, 3004** (verify with `sudo ss -tlnp | grep :<port>` first). Bind to `127.0.0.1`, not `0.0.0.0`. Run it under **systemd** (like `copyright-api`) or **Docker** so it survives reboots.

Example systemd unit `/etc/systemd/system/<myapp>.service`:
```ini
[Unit]
Description=<My Site>
After=network.target
[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/<myapp>
ExecStart=/home/ubuntu/<myapp>/... --port 8080   # bind 127.0.0.1:8080
Restart=always
[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload && sudo systemctl enable --now <myapp>
```

**B. Point a domain at the server.** Either a subdomain of the existing DuckDNS (`<name>.tfg-seasea.duckdns.org`) or your own domain's A-record → `150.230.114.9`.

**C. Add an nginx site** `/etc/nginx/sites-available/<myapp>` (mirror the `tfg` block):
```nginx
server {
    server_name <name>.example.org;
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    listen 80;
}
```
```bash
sudo ln -s /etc/nginx/sites-available/<myapp> /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

**D. Get TLS** (Certbot is installed; it edits the nginx block in place):
```bash
sudo certbot --nginx -d <name>.example.org
```

**E. Verify:** `curl -I https://<name>.example.org`.

---

## 8. Managing our copyright-api (reference)

```bash
sudo systemctl status  copyright-api
sudo systemctl restart copyright-api
sudo journalctl -u copyright-api -n 100 --no-pager
```
- App root: `/home/ubuntu/copyright_extraction_cli/` (git repo + `.env` + `api/`)
- Runs: `venv/bin/python -m uvicorn web.app:app --host 0.0.0.0 --port 5000 --app-dir api`
- Deploy from a dev machine: `rsync api/ oracle:~/copyright_extraction_cli/api/` then restart (see project `docs/DEPLOYMENT_GUIDE.md`).
- Live URL: http://150.230.114.9:5000/  ·  /v2  ·  /docs

---

## 9. Quick reference

```bash
ssh oracle                                        # connect
sudo ss -tlnp                                     # what's listening
sudo docker ps                                    # containers
ls /etc/nginx/sites-enabled/                      # active sites
sudo nginx -t && sudo systemctl reload nginx      # apply nginx changes
df -h /                                            # disk (watch the 94%)
systemctl list-units --type=service --state=running
```
