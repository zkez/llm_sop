from __future__ import annotations

import os
from pathlib import Path

import paramiko


EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "data",
    "models",
    "outputs",
    "deps",
    ".venv",
    "venv",
    "env",
}


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=os.environ["REMOTE_HOST"],
        port=int(os.environ.get("REMOTE_PORT", "22")),
        username=os.environ["REMOTE_USER"],
        password=os.environ["REMOTE_PASS"],
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def mkdir_p(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = []
    current = remote_dir
    while current not in ("", "/"):
        parts.append(current)
        current = str(Path(current).parent).replace("\\", "/")
    for directory in reversed(parts):
        try:
            sftp.stat(directory)
        except FileNotFoundError:
            sftp.mkdir(directory)


def upload_tree(sftp: paramiko.SFTPClient, local_root: Path, remote_root: str) -> None:
    for path in local_root.rglob("*"):
        relative = path.relative_to(local_root)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        remote_path = f"{remote_root}/{relative.as_posix()}"
        if path.is_dir():
            mkdir_p(sftp, remote_path)
            continue
        mkdir_p(sftp, str(Path(remote_path).parent).replace("\\", "/"))
        print(f"upload {relative.as_posix()}")
        sftp.put(str(path), remote_path)


def main() -> int:
    local_root = Path(os.environ["LOCAL_ROOT"]).resolve()
    remote_root = os.environ["REMOTE_PROJECT"].rstrip("/")
    client = connect()
    try:
        sftp = client.open_sftp()
        mkdir_p(sftp, remote_root)
        upload_tree(sftp, local_root, remote_root)
        sftp.close()
    finally:
        client.close()
    print(f"uploaded to {remote_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
