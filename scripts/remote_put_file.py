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


def mkdir_p(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts: list[str] = []
    current = remote_dir.rstrip("/")
    while current and current != "/":
        parts.append(current)
        current = str(Path(current).parent).replace("\\", "/")
    for directory in reversed(parts):
        try:
            sftp.stat(directory)
        except FileNotFoundError:
            sftp.mkdir(directory)


def main() -> int:
    local_path = Path(os.environ["LOCAL_FILE"]).resolve()
    remote_path = os.environ["REMOTE_FILE"]
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    client = connect()
    try:
        sftp = client.open_sftp()
        mkdir_p(sftp, str(Path(remote_path).parent).replace("\\", "/"))
        sftp.put(str(local_path), remote_path)
        sftp.close()
    finally:
        client.close()
    print(f"uploaded {local_path} -> {remote_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
