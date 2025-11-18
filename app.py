import streamlit as st
import tempfile
import os
import whisper
import cv2
import io
import numpy as np
from textblob import TextBlob
from moviepy import VideoFileClip, AudioFileClip
from streamlit_extras.bottom_container import bottom
# set title of the web app
st.title("MomentCutter")

# add a description
st.write(
    "Upload a video or audio file to find interesting moments and create short clips."
)

with bottom():
    col1, col2 = st.columns(2)
    with col1:
        st.write(
            "If you find MomentCutter useful, consider supporting its development:"
        )

    with col2:
        st.link_button("Buy Me a Coffee", "https://www.buymeacoffee.com/kdickerson")


# function to transcribe audio and return segments with timestamps
def transcribe_audio(file_path):
    """Transcribes the audio from a given file path using OpeanAI's Whisper model"""
    st.info("Loading the transcription model...")
    model = whisper.load_model("base")
    st.info("Model loaded. Starting transcription...")
    result = model.transcribe(file_path)
    return result["segments"]


# Function to analyze sentiment of a text
def analyze_sentiment(text):
    """
    Analyzes the sentiment of a given text using TextBlob.
    """
    return TextBlob(text).sentiment


# Function to clip video or audio
def create_clip_in_memory(input_path, start_time, end_time, is_video):
    """
    Creates a clip from a video or audio file and returns it as an in-memory byte buffer.
    This function now writes to a temporary file first, then reads its content into memory,
    and finally deletes the temporary file.
    """
    output_temp_file_path = None
    try:
        if is_video:
            suffix = ".mp4"
            codec = "libx264"
            audio_codec = "aac"
            with VideoFileClip(input_path) as clip:
                end_time = min(end_time + 1.0, clip.duration)
                sub_clip = clip.subclipped(start_time, end_time)
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp_out_file:
                    output_temp_file_path = tmp_out_file.name
                sub_clip.write_videofile(
                    output_temp_file_path,
                    codec=codec,
                    audio_codec=audio_codec,
                    logger=None,
                )
        else:
            suffix = ".mp3"
            codec = "mp3"
            with AudioFileClip(input_path) as clip:
                end_time = min(end_time + 1.0, clip.duration)
                sub_clip = clip.subclipped(start_time, end_time)
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp_out_file:
                    output_temp_file_path = tmp_out_file.name
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


# Function to analyze video for scene changes
def analyze_video_for_scene_changes(video_path, threshold=30.0):
    """
    Analyzes a video file to detect scene changes.
    Returns a list of timestamps (in seconds) where scene changes are detected.
    """
    st.info("Analyzing video for scene changes...")
    scene_changes = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file for scene analysis.")
        return scene_changes

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25  # A sensible default
        st.warning(f"Could not determine video FPS. Assuming {fps} FPS.")

    prev_frame_gray = None
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame_gray is not None:
            # Calculate the absolute difference between the current and previous frame
            diff = cv2.absdiff(prev_frame_gray, gray_frame)
            # Calculate a score for the difference
            diff_score = np.mean(diff)

            if diff_score > threshold:
                timestamp = frame_number / fps
                scene_changes.append(timestamp)

        prev_frame_gray = gray_frame
        frame_number += 1

    cap.release()
    st.info(f"Found {len(scene_changes)} potential scene changes.")
    return scene_changes


# file uploader
uploaded_file = st.file_uploader(
    "Choose a file", type=["mp4", "mov", "avi", "mp3", "wav", "m4a"]
)

# Add a text input for keywords
keywords_input = st.text_input(
    "Optional: Enter keywords to look for (comma-separated)", "AI, technology, future"
)

if uploaded_file is not None:
    # this block executes when a file is uploaded
    st.write(f"You've uploaded a file named: {uploaded_file.name}")

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmpfile:
        tmpfile.write(uploaded_file.getvalue())
        temp_filename = tmpfile.name

    st.write("File uploaded. Here's a preview: ")

    # display the video or audio file
    is_video = uploaded_file.type.startswith("video/")
    if is_video:
        st.video(temp_filename)
    else:
        st.audio(temp_filename)

    if st.button("Find Interesting Moments"):
        # Clear any previous clips from session state
        if "final_clips" in st.session_state:
            del st.session_state["final_clips"]

        with st.spinner("Analyzing content... This may take a moment."):
            segments = transcribe_audio(temp_filename)

            # --- NEW: Analyze video for scene changes ---
            scene_changes = []
            # Only run this if the uploaded file is a video
            if is_video:
                scene_changes = analyze_video_for_scene_changes(temp_filename)

            # --- NEW STRATEGY: Find the most "interesting" windows of time ---

            # 1. Pre-calculate an "interest score" for each segment
            keywords = [
                keyword.strip().lower()
                for keyword in keywords_input.split(",")
                if keyword.strip()
            ]
            keyword_boost = 2.0  # A factor to boost score for keyword presence
            scene_change_boost = 1.5  # A factor to boost score for scene changes

            for segment in segments:
                sentiment = analyze_sentiment(segment["text"])
                # Score is the sum of how emotional and how subjective the text is
                base_score = abs(sentiment.polarity) + sentiment.subjectivity

                # Boost the score if any keywords are present in the segment text
                if any(keyword in segment["text"].lower() for keyword in keywords):
                    base_score *= keyword_boost

                # --- NEW: Boost score based on scene changes ---
                num_scene_changes_in_segment = 0
                if is_video:
                    for change_time in scene_changes:
                        if segment["start"] <= change_time <= segment["end"]:
                            num_scene_changes_in_segment += 1

                # Apply boost for each scene change found in the segment's timeframe
                if num_scene_changes_in_segment > 0:
                    base_score *= scene_change_boost * num_scene_changes_in_segment

                segment["interest_score"] = base_score
            # 2. Use a sliding window to find the most interesting contiguous block
            window_size = 90  # seconds - the ideal length of a clip
            clip_candidates = []

            if not segments:
                st.warning("Could not find any text segments in the file.")
            else:
                # Iterate through all possible start times for a clip
                for i in range(len(segments)):
                    window_start_time = segments[i]["start"]
                    current_window_score = 0
                    segments_in_window = []

                    # Find all segments that fall within the window_size
                    for j in range(i, len(segments)):
                        segment = segments[j]
                        if segment["end"] - window_start_time <= window_size:
                            current_window_score += segment["interest_score"]
                            segments_in_window.append(segment)
                        else:
                            break  # Window is full

                    # We have a potential clip candidate
                    if segments_in_window:
                        clip_start = segments_in_window[0]["start"]
                        clip_end = segments_in_window[-1]["end"]
                        clip_text = " ".join([s["text"] for s in segments_in_window])

                        # Add it as a candidate, we will sort them later
                        clip_candidates.append(
                            {
                                "start": clip_start,
                                "end": clip_end,
                                "score": current_window_score,
                                "text": clip_text,
                            }
                        )

            # 3. Sort candidates by score and filter out overlapping clips
            if clip_candidates:
                sorted_candidates = sorted(
                    clip_candidates, key=lambda x: x["score"], reverse=True
                )

                final_clips_data = []
                for candidate in sorted_candidates:
                    is_overlapping = False
                    for final_clip in final_clips_data:
                        if max(candidate["start"], final_clip["start"]) < min(
                            candidate["end"], final_clip["end"]
                        ):
                            is_overlapping = True
                            break
                    if not is_overlapping:
                        final_clips_data.append(candidate)

                # Store the results in the session state
                st.session_state["final_clips"] = final_clips_data[:5]  # Store top 5

    # Always check session state to display clips, so they persist across reruns
    if "final_clips" in st.session_state:
        st.success(f"Found {len(st.session_state['final_clips'])} high-interest clips!")
        st.subheader("Download Your Clips:")

        for i, clip_data in enumerate(st.session_state["final_clips"], start=1):
            st.write(
                f"**Clip {i} ({clip_data['start']:.2f}s - {clip_data['end']:.2f}s):**"
            )
            st.write(f"> {clip_data['text']}")

            # Create a button to generate this specific clip
            if st.button(f"Generate Clip {i}", key=f"clip_{i}"):
                with st.spinner(f"Generating Clip {i}..."):
                    # Generate the clip in memory instead of writing to a file
                    clip_data_bytes = create_clip_in_memory(
                        temp_filename,
                        clip_data["start"],
                        clip_data["end"],
                        is_video,
                    )
                    # Store the byte data in session state
                    st.session_state[f"download_clip_{i}"] = clip_data_bytes

            # Check if there is a generated clip ready for download
            if f"download_clip_{i}" in st.session_state:
                clip_filename = (
                    f"clip_{i}_{int(clip_data['start'])}-{int(clip_data['end'])}.mp4"
                    if is_video
                    else f"clip_{i}_{int(clip_data['start'])}-{int(clip_data['end'])}.mp3"
                )
                st.download_button(
                    label=f"Download {clip_filename}",
                    data=st.session_state[f"download_clip_{i}"],
                    file_name=clip_filename,
                    mime="video/mp4" if is_video else "audio/mpeg",
                )
            st.write("---")

