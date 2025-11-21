import yt_dlp
import os
import random
import subprocess
import json
from typing import Dict


def download_video(url: str, output_path: str, output_name: str):
    """
    Downloads the best quality video and audio from a given URL,
    merges them into an MP4 file, and saves it to the specified path
    with the given filename.

    Args:
        url (str): The URL of the video to download.
        output_path (str): The directory where the video should be saved.
                           e.g., '/home/user/Videos'
        output_name (str): The desired filename for the downloaded video (without extension).
                           e.g., 'my_awesome_video'
    """
    # Ensure the output directory exists
    if not os.path.exists(output_path):
        print(f"Creating output directory: {output_path}")
        os.makedirs(output_path)

    # Construct the full output template for yt-dlp
    output_template = os.path.join(output_path, f"{output_name}.mp4")

    # Define yt-dlp options
    ydl_opts = {
        "format": "bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "verbose": True,
    }

    print(f"\nAttempting to download: {url}")
    print(f"Saving to: {output_template}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"\nSuccessfully downloaded and saved to: {output_template}")
    except Exception as e:
        print(f"\nAn error occurred during download: {e}")
        print("Please ensure yt-dlp and ffmpeg are correctly installed and accessible in your environment.")
        print("If using Conda, try: conda install -c conda-forge yt-dlp ffmpeg")


def get_video_unique_identifier():
    return (lambda s: f"{s[:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:]}")(f"{random.getrandbits(128):032x}")


def check_video_streams(file_path: str) -> Dict[str, bool]:
    """
    Checks if a video file has both video and audio streams using ffprobe.

    Args:
        file_path (str): The full path to the video file.

    Returns:
        Dict[str, bool]: A dictionary indicating the presence of video and audio streams.
                         Example: {'has_video': True, 'has_audio': True}
    """
    has_video = False
    has_audio = False

    try:
        # Run ffprobe to get stream information in JSON format
        # -v quiet: Suppress verbose output from ffprobe
        # -show_streams: Show information about each stream
        # -of json: Output in JSON format
        command = ["ffprobe", "-v", "quiet", "-show_streams", "-of", "json", file_path]

        # Execute the command and capture its output
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Parse the JSON output
        info = json.loads(result.stdout)

        # Iterate through streams to find video and audio
        if "streams" in info:
            for stream in info["streams"]:
                if stream.get("codec_type") == "video":
                    has_video = True
                elif stream.get("codec_type") == "audio":
                    has_audio = True

                # If both are found, no need to check further
                if has_video and has_audio:
                    break

    except FileNotFoundError:
        print(f"Error: ffprobe not found. Please ensure FFmpeg is installed and in your system's PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error checking {file_path}: ffprobe returned an error.")
        print(f"Stderr: {e.stderr}")
    except json.JSONDecodeError:
        print(f"Error: Could not parse ffprobe JSON output for {file_path}.")
    except Exception as e:
        print(f"An unexpected error occurred for {file_path}: {e}")

    return {"has_video": has_video, "has_audio": has_audio}


def analyze_videos_in_folder(folder_path: str) -> Dict[str, Dict]:
    """
    Analyzes all video files in a given folder and returns statistics.

    Args:
        folder_path (str): The path to the folder containing video files.

    Returns:
        Dict[str, Dict]: A dictionary containing statistics.
                         Example:
                         {
                             'total_files_checked': 5,
                             'with_video_and_audio': 3,
                             'only_video': 1,
                             'only_audio': 0,
                             'other_issues': 1,
                             'problematic_files': ['path/to/problem_file.mp4']
                         }
    """
    stats = {
        "total_files_checked": 0,
        "with_video_and_audio": 0,
        "only_video": 0,
        "only_audio": 0,
        "other_issues": 0,
        "problematic_files": [],
    }

    # Common video extensions to filter files
    video_extensions = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv")

    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist or is not a directory.")
        return stats

    print(f"Analyzing videos in: {os.path.abspath(folder_path)}")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Skip directories and non-video files
        if os.path.isdir(file_path) or not filename.lower().endswith(video_extensions):
            continue

        stats["total_files_checked"] += 1
        print(f"  Checking: {filename}...")

        stream_info = check_video_streams(file_path)

        if stream_info["has_video"] and stream_info["has_audio"]:
            stats["with_video_and_audio"] += 1
        elif stream_info["has_video"] and not stream_info["has_audio"]:
            stats["only_video"] += 1
            stats["problematic_files"].append(file_path)
        elif not stream_info["has_video"] and stream_info["has_audio"]:
            stats["only_audio"] += 1
            stats["problematic_files"].append(file_path)
        else:  # Neither video nor audio stream found, or an error occurred
            stats["other_issues"] += 1
            stats["problematic_files"].append(file_path)

    return stats


if __name__ == "__main__":
    directory = os.path.join(os.path.expanduser("~"), "auvire/real-world-data")
    path = f"{directory}/videos"
    urls_file = f"{directory}/urls"
    report = f"{directory}/report.json"

    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Found {len(urls)} URLs in '{urls_file}'.")
    unique_names = [get_video_unique_identifier() for _ in urls]
    for unique_name, url in zip(unique_names, urls):
        if os.path.exists(os.path.join(path, f"{unique_name}.mp4")):
            print(f"Video with unique name '{unique_name}' already exists. Skipping download for URL: {url}")
            continue
        else:
            print(f"Downloading video from URL: {url} with unique name: {unique_name}")
            download_video(url, path, unique_name)
    with open(report, "w") as f:
        json.dump({url: unique_name for unique_name, url in zip(unique_names, urls)}, f, indent=4)
    print("\nAll download tasks completed.")
    statistics = analyze_videos_in_folder(path)
    print("\n--- Video Analysis Statistics ---")
    for key, value in statistics.items():
        if key == "problematic_files":
            print(f"{key.replace('_', ' ').capitalize()}:")
            if value:
                for f in value:
                    print(f"  - {f}")
            else:
                print("  None")
        else:
            print(f"{key.replace('_', ' ').capitalize()}: {value}")

    print("\nNote: 'Problematic files' include those with only video, only audio, or neither.")
    print("Ensure FFmpeg (which includes ffprobe) is installed and in your system's PATH.")
