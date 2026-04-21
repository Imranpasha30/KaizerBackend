"""Multi-language rebroadcast — translate SEO to target language + queue a
sibling UploadJob against a channel that speaks that language. v1 reuses
the same source MP4 (description explains the language). v2 should re-run
the pipeline with translated transcript + TTS."""
