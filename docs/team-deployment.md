# Secure Team Deployment

This runbook exposes AIC Search to a five-person team without router port forwarding or a public endpoint.

## Deployment model

```text
Owner + four teammates
          │
          ▼
Tailscale HTTPS :443
          │
          ▼
Windows loopback :8000 ──► HAProxy ──► FastAPI ──► Elasticsearch
```

Tailscale authenticates each teammate, FastAPI checks the exact login against a five-user allowlist, and only the owner receives administrative search controls. Do not enable Tailscale Funnel.

## 1. Start the local service

Complete [Getting Started](getting-started.md), then confirm:

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -i http://127.0.0.1:8000/session
```

Expected results are `200` for `/healthz` and `401` for `/session`.

## 2. Configure Tailscale Serve

Install Tailscale on the Windows 11 host. Open PowerShell **as Administrator**:

```powershell
tailscale set --ssh=false --webclient=false
tailscale serve --bg --https=443 http://127.0.0.1:8000
tailscale serve status
```

Use the reported `https://<machine>.<tailnet>.ts.net` URL. Everyone, including the owner, should access the application through this URL so the request contains a verified Tailscale identity.

## 3. Share only the server device

In the Tailscale Admin Console:

1. Share only the Windows server device with each of the four teammate emails.
2. Do not add teammates as full members of the owner's tailnet.
3. Keep Funnel, Tailscale SSH, and the device web client disabled.
4. Revoke unused or unexpected device shares immediately.

Each teammate must accept the device invitation and sign in to Tailscale with the exact email listed in `.env.secure`.

## 4. Apply a least-privilege policy

Replace `100.x.y.z` and the email addresses with real values. Keep the owner rule only if the owner needs broader access to this device.

```jsonc
{
  "hosts": {
    "aic-server": "100.x.y.z"
  },
  "grants": [
    {
      "src": ["owner@example.com"],
      "dst": ["aic-server"],
      "ip": ["*"]
    },
    {
      "src": ["autogroup:shared"],
      "dst": ["aic-server"],
      "ip": ["tcp:443"]
    }
  ],
  "tests": [
    {
      "src": "teammate1@example.com",
      "accept": ["aic-server:443"],
      "deny": ["aic-server:22", "aic-server:5252", "aic-server:8000", "aic-server:9200"]
    },
    {
      "src": "teammate2@example.com",
      "accept": ["aic-server:443"],
      "deny": ["aic-server:22", "aic-server:5252", "aic-server:8000", "aic-server:9200"]
    }
  ]
}
```

Add equivalent tests for teammates three and four before saving. Remove any broader allow-all grant that would override this restriction.

## 5. Verify from a teammate device

```bash
curl -I https://<machine>.<tailnet>.ts.net/
curl https://<machine>.<tailnet>.ts.net/session
```

The session response must contain the teammate's exact email and `"is_admin": false`. Verify that direct connections to ports 22, 5252, 8000, and 9200 fail. A login outside the application allowlist must receive `403`; a request that bypasses Tailscale Serve must receive `401`.

## Security boundaries

The application container runs as a non-root user with a read-only root filesystem, no Linux capabilities, and no access to the Docker socket, home directory, or SSH keys. Code, datasets, indexes, and model caches are read-only. Only the thumbnail cache and `/tmp` are writable.

The internal Docker network permits application-to-Elasticsearch traffic but no Internet or LAN egress. HAProxy is the only container bound to `127.0.0.1:8000`, and it has no access to data, indexes, models, secrets, or the Docker socket.

## Remove or replace a teammate

Revoke the device share first. The application requires exactly five identities, so replace the old email in `.env.secure` before recreating the app:

```bash
docker compose up -d --force-recreate app
```

Confirm the removed identity receives `403` and the replacement identity sees its own login at `/session`.
