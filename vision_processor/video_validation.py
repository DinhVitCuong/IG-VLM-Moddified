import os
from moviepy.video.io.VideoFileClip import VideoFileClip

class VideoLoader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_capture = None

    def file_exists(self):
        return os.path.isfile(self.video_path)

    def check_open_video(self):
        try:
            VideoFileClip(self.video_path)
            return True
        except Exception as e:
            print(f"Error opening video file: {self.video_path}. Reason: {str(e)}")
            return False

    def open_video(self, ts=None):
        try:
            self.video_capture = VideoFileClip(self.video_path)
            fps = self.video_capture.fps
            duration = self.video_capture.duration
            total_frames = int(duration * fps)  # Estimate total frames from duration and FPS

            if ts is not None:
                start_time, end_time = ts.split("-")
                start_frame = int(float(start_time) * fps)
                end_frame = int(float(end_time) * fps)
                self.video_capture = self.video_capture.subclip(start_time, end_time)

                return (
                    self.video_capture,
                    fps,
                    end_frame - start_frame + 1,
                    (end_frame - start_frame + 1) / fps,
                )

            return self.video_capture, fps, total_frames, duration

        except Exception as e:
            print(f"Error opening video file: {self.video_path}. Reason: {str(e)}")
            return False

    def release_video(self):
        if self.video_capture is not None:
            self.video_capture.close()  # Properly close the VideoFileClip
            self.video_capture = None  # Clear the reference


class VideoValidator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_loader = VideoLoader(video_path)
        self.check_and_open_video()

    def check_and_open_video(self):
        if not self.video_loader.file_exists():
            raise FileNotFoundError(f"File not found: {self.video_path}")

        if not self.video_loader.check_open_video():
            raise Exception(f"Unable to open video file: {self.video_path}")