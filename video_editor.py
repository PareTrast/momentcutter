import tempfile
import os
import io
from moviepy import AudioFileClip, ImageClip, VideoFileClip


def create_short_clip(
    input_file_path, start_time, end_time, is_video, background_image_path=None
):
    """
    Creates a clip from a video or audio file and returns it as an in-memory byte buffer.
    If a background image is provided, it creates a video with that image and the audio.
    If no image is provided, it performs standard sub-clipping.

    Args:
        input_file_path (str): Path to the source video/audio file.
        start_time (float): Start time of the clip in seconds.
        end_time (float): End time of the clip in seconds.
        is_video (bool): True if the input file is a video, False if audio.
        background_image_path (str, optional): Path to the background image file.

    Returns:
        bytes: The content of the final MP4 or MP3 file.
    """
    output_temp_file_path = None

    try:
        if background_image_path:
            # --- VIDEO/IMAGE ASSEMBLY (No cropping/resizing) ---

            # Extract the audio segment from the main input file
            if is_video:
                # Get audio from a video subclip
                # Clip the entire video first to get the correct audio segment
                clip_source = VideoFileClip(input_file_path)
                audio_subclip = clip_source.subclip(start_time, end_time).audio
            else:
                # Get audio from an audio subclip
                clip_source = AudioFileClip(input_file_path)
                audio_subclip = clip_source.subclipped(start_time, end_time)

            # 1. Load image and set duration to audio length
            img_clip = ImageClip(background_image_path)

            # The base clip for combining (set to duration of audio)
            # This resultant clip (final_clip) is now an ImageClip with an audio track.
            final_clip = img_clip.with_duration(audio_subclip.duration).with_audio(
                audio_subclip
            )

            # 2. Write to temporary file as MP4
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as tmp_out_file:
                output_temp_file_path = tmp_out_file.name

            # Write the file, defaulting to the image's original resolution/aspect ratio.
            final_clip.write_videofile(
                output_temp_file_path,
                codec="libx264",
                audio_codec="aac",
                fps=24,  # Explicitly set FPS for the ImageClip
                # file_format="mp4",
                logger=None,
                # Explicitly manage temp audio file for robustness
                temp_audiofile=os.path.join(tempfile.gettempdir(), "temp-audio.m4a"),
            )

        else:
            # --- STANDARD SUB-CLIPPING (No image background) ---
            if is_video:
                # Clip video
                clip = VideoFileClip(input_file_path)
                suffix = ".mp4"
                codec = "libx264"
                audio_codec = "aac"
            else:
                # Clip audio
                clip = AudioFileClip(input_file_path)
                suffix = ".mp3"
                codec = "mp3"
                audio_codec = None  # Audio only output

            sub_clip = clip.subclip(start_time, end_time)

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp_out_file:
                output_temp_file_path = tmp_out_file.name

            if is_video:
                sub_clip.write_videofile(
                    output_temp_file_path,
                    codec=codec,
                    audio_codec=audio_codec,
                    logger=None,
                )
            else:
                sub_clip.write_audiofile(
                    output_temp_file_path, codec=codec, logger=None
                )

        # Read the content of the temporary output file into memory
        with open(output_temp_file_path, "rb") as f:
            clip_data = f.read()

        return clip_data

    finally:
        # Clean up the temporary output file
        if output_temp_file_path and os.path.exists(output_temp_file_path):
            os.remove(output_temp_file_path)

        # Cleanup for the explicit temp audio file (if created)
        temp_audio_path = os.path.join(tempfile.gettempdir(), "temp-audio.m4a")
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
