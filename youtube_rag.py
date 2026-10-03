import re
from youtube_transcript_api import YouTubeTranscriptApi


def get_video_id(url_or_id: str) -> str:
    """Extract the video ID from a YouTube URL (or return it if already an ID)."""
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url_or_id)
    return match.group(1) if match else url_or_id
def clean_text(text: str) -> str:
    # 1. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 2. Remove common transcript artifacts
    text = re.sub(
        r"\[(?:music|applause|laughter|silence)\]",
        "",
        text,
        flags=re.IGNORECASE
    )

    # 3. Remove HTML tags, if present
    text = re.sub(r"<[^>]+>", "", text)

    # 4. Remove extra whitespace and newlines
    text = re.sub(r"\s+", " ", text)

    # 5. Remove spaces before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # 6. Remove spaces at the beginning and end
    text = text.strip()

    return text

def load_transcript(url_or_id: str, languages=("en",)) -> str:
    video_id = get_video_id(url_or_id)
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=list(languages))
    transcript = " ".join(snippet.text for snippet in transcript)
    transcript = clean_text(transcript)

    return transcript
if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=TX1A3Nn4zbQ"
    text = load_transcript(url)
    print(text)