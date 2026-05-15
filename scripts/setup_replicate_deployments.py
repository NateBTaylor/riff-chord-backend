"""
Create private Replicate Deployments for the 3 models Riff uses.

Why: public Replicate models share container pools and queue users by tier,
which is why we see 1-8 minute cold starts. Private deployments give us a
dedicated container pool that:
  - Stays warm for `scaledown_window` seconds after each call
  - Has no idle cost (min_instances=0)
  - Skips cross-tenant queueing

Run once:
    REPLICATE_API_TOKEN=r8_xxx python scripts/setup_replicate_deployments.py

Idempotent — re-running updates existing deployments instead of failing.
After running, add the printed env vars to Railway:
    RIFF_DEPLOY_SPLEETER=<username>/riff-spleeter
    RIFF_DEPLOY_CHORD=<username>/riff-chord
    RIFF_DEPLOY_WHISPER=<username>/riff-whisper
"""

import os
import sys
import json

try:
    import requests
except ImportError:
    print(
        "ERROR: `requests` is not installed. Run:\n"
        "    pip3 install requests\n"
        "(it's already a backend dependency, so `pip3 install -r requirements.txt` "
        "from riff-chord-backend/ also works).",
        file=sys.stderr,
    )
    sys.exit(1)


API_BASE = "https://api.replicate.com/v1"

# Same model versions the codebase currently calls — keep these in sync with
# spleeter_service.py, replicate_chord_detector.py, lyrics_transcription_service.py.
DEPLOYMENTS = [
    {
        "name": "riff-spleeter",
        "model": "soykertje/spleeter",
        "version": "cd128044253523c86abfd743dea680c88559ad975ccd72378c8433f067ab5d0a",
        "hardware": "gpu-t4",  # Spleeter is tiny — T4 is plenty
    },
    {
        "name": "riff-chord",
        "model": "triadmusic/chord-detection-cnn-lstm",
        "version": "be95be0303fd42000c413aec595922499f8b946d65416f31fb0034c2daf81f19",
        "hardware": "gpu-t4",
    },
    {
        "name": "riff-whisper",
        "model": "vaibhavs10/incredibly-fast-whisper",
        "version": "3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
        # T4's 16GB VRAM fits Whisper large-v3 fine. Chord runs in parallel
        # on T4 too, so a faster GPU here wouldn't reduce wall-time.
        "hardware": "gpu-t4",
    },
]

# Keep instances warm for 10 minutes after each call.
# Average analyze pipeline is < 30s, so back-to-back songs stay on a warm pool.
DEFAULT_CONFIG = {
    "min_instances": 0,
    "max_instances": 2,
}


def _request(method: str, path: str, token: str, body: dict | None = None) -> tuple[int, dict]:
    url = f"{API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = requests.request(method, url, headers=headers, json=body, timeout=30)
    try:
        parsed = resp.json() if resp.text else {}
    except json.JSONDecodeError:
        parsed = {"error": resp.text}
    return resp.status_code, parsed


def get_account(token: str) -> dict:
    status, body = _request("GET", "/account", token)
    if status != 200:
        raise RuntimeError(f"Failed to fetch account: HTTP {status} {body}")
    return body


def deployment_exists(token: str, owner: str, name: str) -> bool:
    status, _ = _request("GET", f"/deployments/{owner}/{name}", token)
    return status == 200


def create_deployment(token: str, owner: str, spec: dict) -> dict:
    body = {
        "name": spec["name"],
        "model": spec["model"],
        "version": spec["version"],
        "hardware": spec["hardware"],
        **DEFAULT_CONFIG,
    }
    status, payload = _request("POST", "/deployments", token, body)
    if status not in (200, 201):
        raise RuntimeError(f"Create {spec['name']} failed: HTTP {status} {payload}")
    return payload


def update_deployment(token: str, owner: str, spec: dict) -> dict:
    # PATCH lets us change hardware / min_instances / max_instances / version
    body = {
        "version": spec["version"],
        "hardware": spec["hardware"],
        **DEFAULT_CONFIG,
    }
    status, payload = _request(
        "PATCH",
        f"/deployments/{owner}/{spec['name']}",
        token,
        body,
    )
    # Replicate returns 409 when the requested values match the current state.
    # That's a no-op from our perspective, not a failure.
    if status == 409:
        return {}
    if status not in (200, 201):
        raise RuntimeError(f"Update {spec['name']} failed: HTTP {status} {payload}")
    return payload


def main() -> int:
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        print("ERROR: REPLICATE_API_TOKEN env var not set", file=sys.stderr)
        return 1

    account = get_account(token)
    owner = account.get("username")
    if not owner:
        print(f"ERROR: could not resolve account username: {account}", file=sys.stderr)
        return 1

    print(f"Replicate account: {owner} ({account.get('type')})")
    print(f"Creating/updating {len(DEPLOYMENTS)} deployments...\n")

    env_lines = []
    for spec in DEPLOYMENTS:
        slug = f"{owner}/{spec['name']}"
        exists = deployment_exists(token, owner, spec["name"])
        action = "Updating" if exists else "Creating"
        print(f"  {action} {slug} ({spec['hardware']}) → {spec['model']}")
        if exists:
            update_deployment(token, owner, spec)
        else:
            create_deployment(token, owner, spec)
        env_var = {
            "riff-spleeter": "RIFF_DEPLOY_SPLEETER",
            "riff-chord": "RIFF_DEPLOY_CHORD",
            "riff-whisper": "RIFF_DEPLOY_WHISPER",
        }[spec["name"]]
        env_lines.append(f"{env_var}={slug}")

    print("\n✓ Done.\n")
    print("Add these to Railway env vars:")
    print("-" * 50)
    for line in env_lines:
        print(line)
    print("-" * 50)
    print(
        "\nNote: Replicate's API does not expose `scaledown_window` for\n"
        "deployments — it defaults to a few minutes of warm idle. If you\n"
        "need a longer warm window, adjust it in the Replicate dashboard\n"
        "(Deployments → riff-* → Settings).\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
