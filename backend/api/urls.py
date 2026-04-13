from django.urls import path
from . import views

urlpatterns = [
    # Config
    path("platforms/",      views.platforms,     name="platforms"),
    path("frame-layouts/",  views.frame_layouts,  name="frame-layouts"),

    # Jobs
    path("jobs/",           views.job_list,   name="job-list"),
    path("jobs/create/",    views.job_create, name="job-create"),
    path("jobs/<uuid:job_id>/",        views.job_detail, name="job-detail"),
    path("jobs/<uuid:job_id>/status/", views.job_status, name="job-status"),
    path("jobs/<uuid:job_id>/export/", views.job_export, name="job-export"),
    path("jobs/<uuid:job_id>/delete/", views.job_delete, name="job-delete"),

    # Clips
    path("clips/<int:clip_id>/",              views.clip_detail,       name="clip-detail"),
    path("clips/<int:clip_id>/rerender/",     views.clip_rerender,     name="clip-rerender"),
    path("clips/<int:clip_id>/upload-image/", views.clip_upload_image, name="clip-upload-image"),

    # File serving
    path("file/",              views.serve_file,        name="serve-file"),
    path("fonts/<str:filename>", views.serve_font,      name="serve-font"),
]
