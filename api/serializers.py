from rest_framework import serializers
from .models import Job, Clip


class ClipSerializer(serializers.ModelSerializer):
    thumb_url  = serializers.SerializerMethodField()
    clip_url   = serializers.SerializerMethodField()
    image_url  = serializers.SerializerMethodField()
    raw_url    = serializers.SerializerMethodField()

    class Meta:
        model  = Clip
        fields = [
            "id", "index", "text", "frame_type",
            "clip_path", "raw_path", "thumb_path", "image_path",
            "thumb_url", "clip_url", "image_url", "raw_url",
            "card_params", "section_pct", "follow_params", "split_params", "preset",
        ]

    def _url(self, path):
        if not path:
            return ""
        return f"/api/file/?path={path}"

    def get_thumb_url(self, obj):  return self._url(obj.thumb_path)
    def get_clip_url(self,  obj):  return self._url(obj.clip_path)
    def get_image_url(self, obj):  return self._url(obj.image_path)
    def get_raw_url(self,   obj):  return self._url(obj.raw_path)


class JobSerializer(serializers.ModelSerializer):
    clips = ClipSerializer(many=True, read_only=True)

    class Meta:
        model  = Job
        fields = [
            "id", "video_name", "platform", "frame_layout",
            "status", "progress_pct", "log", "error",
            "output_dir", "created_at", "updated_at", "clips",
        ]


class JobListSerializer(serializers.ModelSerializer):
    clip_count = serializers.SerializerMethodField()

    class Meta:
        model  = Job
        fields = [
            "id", "video_name", "platform", "frame_layout",
            "status", "progress_pct", "created_at", "clip_count",
        ]

    def get_clip_count(self, obj):
        return obj.clips.count()
