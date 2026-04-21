"""Channel learning — mines top-performing videos to extract hook patterns.

Outputs go into `channel_corpus.payload` as:
    {"top_titles":  [str, ...],
     "top_descriptions": [str, ...],
     "hook_patterns": [str, ...],
     "emotional_triggers": [str, ...],
     "power_words": [str, ...],
     "political_framing": str,
     "sample_size": int,
     "youtube_channel_id": "UCxxx",
     "refreshed_at": "ISO8601"}

Consumed at SEO-generation time by seo/prompts.py::build_user_prompt.
"""
