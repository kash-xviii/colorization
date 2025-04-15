import re

def clean_caption(caption):
    caption = caption.lower()
    caption = re.sub(r"[^a-z0-9\s]", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption