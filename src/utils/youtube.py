import re
import uuid
import json

from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR

downloader = YoutubeCommentDownloader()

def get_video_id(url) -> str | None:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def load_comments(video_id) -> list[dict] | None:

    try:
        comments_gen = downloader.get_comments(video_id, sort_by=SORT_BY_POPULAR)
    except Exception as e:
        return []

    comments_data = []

    for _, comment in enumerate(comments_gen):
        
        if comment['reply']:
            continue
        
        raw_text = comment["text"]
        
        comments_data.append({
            "author": comment["author"],
            "raw_text": raw_text,
            "uuid": str(uuid.uuid4()),
        })

    return comments_data