from __future__ import annotations

import os
import sys

import paramiko


def main() -> int:
    host = os.environ["REMOTE_HOST"]
    port = int(os.environ.get("REMOTE_PORT", "22"))
    username = os.environ["REMOTE_USER"]
    password = os.environ["REMOTE_PASS"]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=username,
        password=password,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
        look_for_keys=False,
        allow_agent=False,
    )

    commands = [
        "whoami",
        "uname -a",
        "pwd",
        "python3 --version || python --version || true",
        "git --version || true",
        "ffmpeg -version | head -n 1 || true",
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true",
        "df -h /mnt/project || true",
        "ls -ld /mnt/project /mnt/project/zy || true",
    ]

    for command in commands:
        print(f"\n$ {command}")
        stdin, stdout, stderr = client.exec_command(command, timeout=60)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace").strip()
        err = stderr.read().decode("utf-8", errors="replace").strip()
        if out:
            print(out)
        if err:
            print(err, file=sys.stderr)
        print(f"[exit={exit_code}]")

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
