from __future__ import annotations

import os
from pathlib import Path

import paramiko


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=os.environ["REMOTE_HOST"],
        port=int(os.environ.get("REMOTE_PORT", "22")),
        username=os.environ["REMOTE_USER"],
        password=os.environ["REMOTE_PASS"],
        timeout=30,
        banner_timeout=45,
        auth_timeout=30,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def main() -> int:
    remote_path = os.environ["REMOTE_FILE"]
    local_path = Path(os.environ["LOCAL_FILE"]).resolve()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client = connect()
    try:
        sftp = client.open_sftp()
        sftp.get(remote_path, str(local_path))
        sftp.close()
    finally:
        client.close()
    print(f"downloaded {remote_path} -> {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
