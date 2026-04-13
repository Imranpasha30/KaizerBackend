import uuid
from django.db import models


class Job(models.Model):
    STATUS = [
        ("pending",  "Pending"),
        ("running",  "Running"),
        ("done",     "Done"),
        ("failed",   "Failed"),
    ]

    id           = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video_name   = models.CharField(max_length=255)
    video_path   = models.CharField(max_length=1024)
    platform     = models.CharField(max_length=50)
    frame_layout = models.CharField(max_length=50)
    status       = models.CharField(max_length=20, choices=STATUS, default="pending")
    progress_pct = models.IntegerField(default=0)
    log          = models.TextField(default="")
    error        = models.TextField(default="")
    output_dir   = models.CharField(max_length=1024, default="")
    meta_path    = models.CharField(max_length=1024, default="")
    created_at   = models.DateTimeField(auto_now_add=True)
    updated_at   = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.video_name} [{self.status}]"


class Clip(models.Model):
    job          = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="clips")
    index        = models.IntegerField()
    clip_path    = models.CharField(max_length=1024)
    raw_path     = models.CharField(max_length=1024, default="")
    thumb_path   = models.CharField(max_length=1024, default="")
    image_path   = models.CharField(max_length=1024, default="")
    text         = models.TextField(default="")
    frame_type   = models.CharField(max_length=50, default="torn_card")
    card_params  = models.JSONField(default=dict)
    section_pct  = models.JSONField(default=dict)
    follow_params = models.JSONField(default=dict)
    split_params = models.JSONField(default=dict)
    preset       = models.JSONField(default=dict)

    class Meta:
        ordering = ["index"]

    def __str__(self):
        return f"Clip {self.index + 1} [{self.job.video_name}]"
