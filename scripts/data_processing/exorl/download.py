# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
from pathlib import Path

import tyro


def main(output_folder: str):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for domain in ["pointmass"]: #["walker", "cheetah", "quadruped", "pointmass"]:
        orig_domain = "point_mass_maze" if domain == "pointmass" else domain
        (output_folder / domain).mkdir(parents=True, exist_ok=True)
        print("Downloading: " + domain)
        result = subprocess.run(
            ["wget", "-O", f"{str(output_folder / domain / 'rnd.zip')}", f"https://dl.fbaipublicfiles.com/exorl/{orig_domain}/rnd.zip"],
            capture_output=True,
        )
        assert "100%" in result.stdout.decode() or "100%" in result.stderr.decode(), (
            f"Error downloading {domain} rnd dataset: {result.stdout.decode()}"
        )
        print("Download done. Unzipping...")
        result = subprocess.run(
            ["unzip", "-qq", f"{str(output_folder / domain / 'rnd.zip')}", "-d", f"{str(output_folder / domain / 'rnd')}"],
            capture_output=True,
        )
        print(result)


if __name__ == "__main__":
    tyro.cli(main)
