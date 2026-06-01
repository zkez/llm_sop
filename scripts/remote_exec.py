from __future__ import annotations

import os
import sys
import time

import paramiko


def connect() -> paramiko.SSHClient:
    last_error: Exception | None = None
    for attempt in range(1, 6):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
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
        except Exception as error:
            last_error = error
            client.close()
            print(f"SSH 连接失败，重试 {attempt}/5: {error}", file=sys.stderr)
            time.sleep(2 * attempt)
    raise RuntimeError(f"SSH 连接失败: {last_error}")


def run(client: paramiko.SSHClient, command: str, timeout: int = 600) -> int:
    print(f"\n$ {command}")
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout, get_pty=True)
    channel = stdout.channel
    stdout_encoding = sys.stdout.encoding or "utf-8"
    stderr_encoding = sys.stderr.encoding or "utf-8"

    def write_stream(stream: object, data: bytes, encoding: str) -> None:
        text = data.decode("utf-8", errors="replace")
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        stream.write(safe_text)
        stream.flush()

    while not channel.exit_status_ready():
        if channel.recv_ready():
            write_stream(sys.stdout, channel.recv(4096), stdout_encoding)
        if channel.recv_stderr_ready():
            write_stream(sys.stderr, channel.recv_stderr(4096), stderr_encoding)
    while channel.recv_ready():
        write_stream(sys.stdout, channel.recv(4096), stdout_encoding)
    while channel.recv_stderr_ready():
        write_stream(sys.stderr, channel.recv_stderr(4096), stderr_encoding)
    exit_code = channel.recv_exit_status()
    print(f"\n[exit={exit_code}]")
    return exit_code


def main() -> int:
    command = os.environ["REMOTE_COMMAND"]
    timeout = int(os.environ.get("REMOTE_TIMEOUT", "600"))
    client = connect()
    try:
        return run(client, command, timeout)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
