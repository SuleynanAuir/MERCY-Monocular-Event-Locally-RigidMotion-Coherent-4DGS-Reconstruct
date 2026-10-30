"""Pause only explicitly configured viewer processes during offline training.

Registries are administrator-owned JSON, never supplied by an HTTP client.
Hybrid serving and comparison/export services are not selected.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def command(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    except (OSError, UnicodeError):
        return []


def viewer_command(cmd):
    return (len(cmd) > 2 and Path(cmd[1]).name == 'viewer.py'
            and '--scenes_path' in cmd and '--port' in cmd)


def pause(registries, records):
    for registry in registries:
        path = Path(registry)
        for row in json.loads(path.read_text()):
            cmd = row['command']
            if not viewer_command(cmd):
                continue
            # A cloud restart may invalidate stored PIDs. Match the entire
            # configured argv, not a substring or an unverified old PID.
            matches = [int(p.name) for p in Path('/proc').iterdir()
                       if p.name.isdigit() and command(int(p.name)) == cmd]
            if len(matches) > 1:
                raise RuntimeError('Multiple viewers match a training resource entry')
            if not matches:
                continue
            pid = matches[0]
            env = dict(part.decode().split('=', 1) for part in
                       Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if b'=' in part)
            records.append(dict(registry=str(path), port=row['port'], command=cmd,
                                pid=pid, cwd=os.readlink(f'/proc/{pid}/cwd'), env=env))
    for record in records:
        if command(record['pid']) == record['command']:
            os.kill(record['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not any(command(r['pid']) == r['command'] for r in records):
            return
        time.sleep(.1)
    for record in records:
        if command(record['pid']) == record['command']:
            os.kill(record['pid'], signal.SIGKILL)
    deadline = time.monotonic() + 5
    while any(command(r['pid']) == r['command'] for r in records):
        if time.monotonic() > deadline:
            raise RuntimeError('Viewer processes did not release the GPU')
        time.sleep(.1)


def resume(records):
    errors = []
    for record in records:
        try:
            if command(record['pid']) == record['command']:
                continue
            path = Path(record['registry'])
            with (path.parent / f"training-resume-{record['port']}.log").open('ab') as log:
                process = subprocess.Popen(record['command'], cwd=record['cwd'], env=record['env'],
                                           stdout=log, stderr=log, start_new_session=True)
            rows = json.loads(path.read_text())
            for row in rows:
                if row['port'] == record['port']:
                    row['pid'] = process.pid
            temporary = path.with_suffix('.resume.tmp')
            temporary.write_text(json.dumps(rows, indent=2))
            temporary.replace(path)
        except Exception as error:
            errors.append(f"port {record['port']}: {error}")
    if errors:
        raise RuntimeError('Failed to restore viewer services: ' + '; '.join(errors))
