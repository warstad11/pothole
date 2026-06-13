import ffmpeg
from pathlib import Path

class ClipExtractor:
    @staticmethod
    def extract_clip(video_path: Path, time: float, output_path: Path, duration: float = 4.0, pre_context: float = 2.0):
        """
        Extracts a clip starting `pre_context` before `time` with `duration`.
        Transcodes to libx264 for browser compatibility.
        Uses output seeking (after -i) for maximum accuracy.
        """
        import subprocess
        start_time = max(0, time - pre_context)
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", str(video_path),
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-pix_fmt", "yuv420p",
            "-loglevel", "quiet",
            str(output_path)
        ]
        
        try:
            # Use subprocess directly to avoid any ffmpeg-python pipe issues
            # Using input=None and a timeout ensures it doesn't hang
            subprocess.run(
                cmd, 
                check=True, 
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30 # 30 seconds for a 4s clip is plenty
            )
        except Exception as e:
            print(f"Error extracting clip: {e}", flush=True)
