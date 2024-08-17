import argparse
import os
from typing import Optional


def run(path: str, command: Optional[str] = None) -> None:
    fdir, fname = os.path.split(path)

    assert fdir[-7:] == "scripts", fdir
    assert fname[-3:] == ".py", fname
    assert os.path.exists(path), path
    assert fname in command if command is not None else True, (fname, command)

    os.system(f"cp {path} .")
    if command is None:
        os.system(f"python {fname}")
    else:
        os.system(f"{command}")
    os.system(f"rm {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--command", type=str, default=None)
    args = parser.parse_args()
    run(args.path, args.command)
