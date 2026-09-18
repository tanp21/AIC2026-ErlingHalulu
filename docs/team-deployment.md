# Direct-IP Team Deployment

This runbook originally exposed AIC Search directly through a Viettel router to a five-person
team, authenticating each teammate with HAProxy Basic Auth. **The deployment has since moved to
Tailscale instead of router port forwarding** (see "Tailscale deployment model" below); the
Basic Auth runbook is kept below for reference in case port forwarding is ever reintroduced. It
intentionally uses plain HTTP because the operator accepted the risk. Basic Auth credentials,
queries, images, and videos are not encrypted in transit.

## Tailscale deployment model

```text
Owner + four teammates, all on the same tailnet
          │
          ▼
tailscale serve (https://<host>.<tailnet>.ts.net, on the host's own tailscaled)
          │  proxies to 127.0.0.1:8000, adding Tailscale-User-Login
          ▼
HAProxy (trusts that header from loopback only) ──► FastAPI ──► Elasticsearch
```

Access goes through `tailscale serve` (already configured on this host as
`https://<device>.<tailnet>.ts.net → http://127.0.0.1:8000`), not a raw `<tailscale-ip>:8000`
URL — that matters because `tailscale serve` terminates the tailnet HTTPS connection itself and
re-issues the request to HAProxy over loopback, so HAProxy never sees the original peer's IP.
Instead it reads the `Tailscale-User-Login` header tailscaled adds to every proxied request,
naming exactly which tailnet member is connecting, and copies it into `X-AIC-User` — but only
when the request truly arrived over loopback/the Docker bridge (`acl local_proxy` in
`haproxy.cfg`); anything else hitting the frontend directly is rejected with 403. `compose.yaml`
additionally publishes the ingress port as `127.0.0.1:8000` (not `0.0.0.0`), so nothing external
can reach it at all except through tailscaled's own local proxy.

There is no more per-teammate login — whoever is signed into Tailscale on a given device is who
the app sees. Each of the five entries in `AIC_ALLOWED_USERS` must be a teammate's **real
Tailscale account login** (the email they sign into Tailscale with), or they get 403 once
invited onto the tailnet; anything not in that list is rejected the same way regardless of who it
is. Deploying is then just: keep `tailscale serve` pointed at `127.0.0.1:8000` (`tailscale serve
status` shows the current mapping), invite the four teammates onto this tailnet from the
[Tailscale admin console](https://login.tailscale.com/admin/users) with those exact emails, and
run `docker compose up -d --build --wait`. No router port forward and no `haproxy-users.cfg` are
needed.

## Direct-IP (Basic Auth) deployment model — legacy

```text
Owner + four teammates
          │
          ▼
Viettel public IPv4 TCP :80
          │
          ▼
ZXHN F6601P NAT ──► Windows :8000 ──► HAProxy Basic Auth ──► FastAPI ──► Elasticsearch
```

HAProxy authenticates each teammate, overwrites the internal identity header, and FastAPI checks
that identity against the five-user allowlist. Only the owner receives administrative controls.

## 1. Create the five-user password file

Use the same five lowercase email addresses configured in `AIC_ALLOWED_USERS`. In WSL, install
`mkpasswd` from the `whois` package if it is unavailable. Run it without a password argument so
the password is not recorded in shell history:

```bash
cp haproxy-users.cfg.example haproxy-users.cfg
mkpasswd -m sha-256
```

Add exactly five entries below `userlist aic_users`, using a separately generated hash for each:

```haproxy
userlist aic_users
  user owner@example.com password $5$generated-hash
  user teammate1@example.com password $5$generated-hash
```

Never commit `haproxy-users.cfg`; it is gitignored. Validate and start the stack:

```bash
docker compose run --rm --no-deps ingress haproxy -c \
  -f /usr/local/etc/haproxy/haproxy.cfg \
  -f /usr/local/etc/haproxy/users.cfg
docker compose up -d --build --wait
curl -fsS http://127.0.0.1:8000/healthz
curl -u 'owner@example.com' http://127.0.0.1:8000/session
```

The first command must report a valid configuration. `/healthz` returns `200`; the authenticated
session returns the owner email and `"is_admin": true`.

## 2. Reserve the server address and allow Windows Firewall

The Ethernet-connected Windows host must keep `192.168.1.172`. In the F6601P interface, create a
DHCP reservation/binding for `192.168.1.172` using the Ethernet adapter's physical address.

Open PowerShell as Administrator:

```powershell
New-NetFirewallRule -DisplayName "AIC HTTP ingress" -Direction Inbound `
  -Action Allow -Protocol TCP -LocalPort 8000
```

From another LAN device, verify `http://192.168.1.172:8000/healthz` returns `200` before changing
the router.

## 3. Confirm public IPv4 and forward one port

In `Internet > WAN`, select the connected Internet entry. Its IPv4 address must equal the address
reported by a public “what is my IP” service. If it differs or is in a private/CGNAT range, ask
Viettel to provide public IPv4 before continuing.

In `Security > Port Forwarding > Create New Item`, add:

- Name: `AIC_HTTP`
- Protocol: `TCP`
- WAN Connection: the connected Internet entry
- WAN Host IP Address: blank
- LAN Host: `192.168.1.172`
- WAN Port: `80` through `80`
- LAN Host Port: `8000`

Do not enable DMZ or UPnP and do not forward UDP, 22, 443, 8000, or 9200.

## 4. Verify from mobile data

Disable Wi-Fi on a phone and open `http://<public-ip>`. Authenticate once with each account and
confirm `/session` reports the matching email; only the owner may receive `"is_admin": true`.
Supplying a forged `X-AIC-User` header must not override the Basic Auth identity. Confirm public
ports 22, 443, 8000, and 9200 remain closed.

## Competition collaboration workspace

The top bar is shared by all five authenticated teammates. The owner uploads each organizer query packet as a ZIP using **Upload vòng** and gives it a unique round name. A packet may contain `.txt` query files directly or inside one top-level folder. Uploading a new packet makes it active; the owner can reactivate an older round from the round selector.

For each query, a teammate can claim it, add the selected search keyframe to the answer editor, save drafts, and mark a valid answer as reviewed. Changes from other browsers appear within about two seconds. If two people edit the same version, the later save is rejected and its local draft remains visible for manual reconciliation.

Before submitting, use **Xuất ZIP**. The preflight reports missing, invalid, and unreviewed queries. A warned export is still available after confirmation: valid unreviewed files are included, while missing or invalid query files are omitted. The downloaded archive already has the required `submission/*.csv` structure.

Collaboration state persists in the Docker volume `collab-data` at `/workspace/aic-collab.sqlite3`. Back it up before host maintenance:

```bash
docker compose stop app
docker run --rm -v aic_hcmc_collab-data:/source -v "$PWD:/backup" alpine \
  cp /source/aic-collab.sqlite3 /backup/aic-collab.sqlite3.backup
docker compose start app
```

The Compose project prefix may differ from `aic_hcmc`; confirm the volume name with `docker volume ls`. Restore only while the app container is stopped so SQLite cannot be writing concurrently.

## Security boundaries

The application container runs as a non-root user with a read-only root filesystem, no Linux capabilities, and no access to the Docker socket, home directory, or SSH keys. Code, datasets, indexes, and model caches are read-only. Only the thumbnail cache, collaboration database volume, and `/tmp` are writable.

The internal Docker network permits application-to-Elasticsearch traffic but no Internet or LAN egress. HAProxy is the only container published on `0.0.0.0:8000`, and it has no access to data, indexes, models, application secrets, or the Docker socket.

## Remove or replace a teammate

Replace the old email in both `.env.secure` and `haproxy-users.cfg`, then recreate the app and ingress:

```bash
docker compose up -d --force-recreate app ingress
```

Confirm the removed credentials receive `401` and the replacement identity sees its own login at `/session`.
