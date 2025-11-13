# AI Short-Form Content Clipper

This is a web application built with Streamlit that uses AI to automatically identify interesting moments in long-form video or audio content and create short, shareable clips.

The application analyzes content based on transcription, language sentiment, and visual cues (for videos) to find the most engaging segments.



## Features

- **File Upload**: Supports common video (`.mp4`, `.mov`, `.avi`) and audio (`.mp3`, `.wav`, `.m4a`) formats.
- **Automatic Transcription**: Uses OpenAI's Whisper model to generate a full transcript with accurate timestamps.
- **Interest Scoring**: Each segment of the transcript is scored based on a combination of factors:
  - **Sentiment Analysis**: How emotional (polarity) and subjective the language is, analyzed using TextBlob.
  - **Keyword Boosting**: Users can provide a list of keywords to boost the score of relevant segments.
  - **Scene Change Detection**: For videos, it detects camera cuts using OpenCV, giving a higher score to visually dynamic moments.
- **Clip Candidate Generation**: A sliding window algorithm identifies the highest-scoring contiguous segments that are ideal for short-form content (e.g., 90 seconds).
- **On-Demand Clipping**: Users can review the top 5 suggested clips, including the transcribed text, and generate and download them with a single click.
- **Efficient Processing**: Clips are generated in memory and served directly for download, keeping the project directory clean.

---

## How It Works

1.  **Upload**: The user uploads a video or audio file through the Streamlit interface.
2.  **Preprocessing**: The file is saved to a temporary location on the server.
3.  **Analysis**: When the user clicks "Find Interesting Moments":
    - The audio is transcribed by **Whisper**.
    - If it's a video, **OpenCV** scans the frames to log all scene-change timestamps.
    - An "interest score" is calculated for each transcribed segment, combining sentiment, subjectivity, keyword presence, and the number of scene changes within its timeframe.
4.  **Clipping Logic**:
    - The application looks for the best "windows" of text (e.g., 90 seconds long) that have the highest cumulative interest score.
    - The top candidates are sorted, and any overlapping clips are filtered out to ensure variety.
5.  **Output**: The top 5 non-overlapping clips are presented to the user. When a user chooses to generate a clip, **MoviePy** creates it, and Streamlit provides a download button.

---

## Setup and Installation

To run this application locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd ai-video-clipper
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The first run of the Whisper model will download its weights (~461 MB for the 'base' model).*

4.  **(Optional) Configure Upload Limit:**
    The default file upload limit for Streamlit is 200MB. To handle larger files, this project includes a `.streamlit/config.toml` file. You can adjust the `maxUploadSize` value as needed, but be mindful of your system's RAM.
    ```toml
    # .streamlit/config.toml
    [server]
    maxUploadSize = 500 # Sets the limit to 500 MB
    ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your default web browser.

---

## Technologies Used

- **Framework**: Streamlit
- **Transcription**: OpenAI Whisper
- **Video/Audio Processing**: MoviePy
- **Visual Analysis**: OpenCV-Python
- **Text Analysis**: TextBlob
- **Numerical Operations**: NumPy

---
