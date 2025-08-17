from pathlib import Path
import time
import shutil
from dataclasses import dataclass
from typing import Optional
import subprocess


def check_hardware_acceleration():
    command = f"ffmpeg -hwaccels && ffmpeg -encoders | grep nvenc"

    # get the output of the command
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Check if FFmpeg command was successful
        if result.returncode == 0:
            print(result.stdout.decode())
            return True
        else:
            print(f"FFmpeg failed with return code: {result.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg check: {e}")
        print(f"Command output: {e.stdout.decode()}")
        print(f"Error message: {e.stderr.decode()}")
        return False


def reencode_and_replace(
    tmp_dir: Path,
    video_fp: Path,
    hwaccel=False,
    fps: Optional[int] = None,
) -> Path:

    tmp_video_name = video_fp.with_suffix(".tmp.mp4").name
    tmp_video_fp = tmp_dir / tmp_video_name

    start_encoding = time.time()
    encoder_str = "-c:v libx264" if not hwaccel else "-c:v h264_nvenc "

    command = (
        f"ffmpeg -y -i {video_fp} {encoder_str} -crf 23 -preset fast {tmp_video_fp}"
    )

    success = False
    try:
        # Use subprocess.run for better error handling (instead of subprocess.call)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Check if FFmpeg command was successful
        if result.returncode == 0:
            success = True
        else:
            print(f"FFmpeg failed with return code: {result.returncode}")

    except subprocess.CalledProcessError as e:
        print(f"Error during video encoding: {e}")
        print(f"Command output: {e.stdout.decode()}")
        print(f"Error message: {e.stderr.decode()}")

    except Exception as e:
        # Catch any other unforeseen errors
        print(f"An unexpected error occurred: {e}")

    end_encoding = time.time()

    print(f"Encoding time: {end_encoding - start_encoding:.4f}sec.")

    if success:
        shutil.move(str(tmp_video_fp), str(video_fp))
    else:
        print(f"Error reencoding {video_fp}")
        if tmp_video_fp.exists():
            tmp_video_fp.unlink()

    return video_fp
