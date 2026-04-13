from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import FileResponse, Http404
import os

def serve_spa(request, *args, **kwargs):
    """Serve React index.html for all non-api routes (production build)."""
    # React build is copied into staticfiles/spa/ by the build step
    candidates = [
        os.path.join(settings.BASE_DIR, "staticfiles", "spa", "index.html"),
        os.path.join(settings.BASE_DIR, "staticfiles", "index.html"),
    ]
    for index in candidates:
        if os.path.exists(index):
            return FileResponse(open(index, "rb"), content_type="text/html")
    from django.http import HttpResponse
    return HttpResponse(
        "<h1>KAIZER NEWS API is running.</h1>"
        "<p>Frontend not built. Run <code>npm run build</code> in the frontend folder "
        "then copy <code>dist/</code> to <code>backend/staticfiles/spa/</code>.</p>",
        status=200
    )

urlpatterns = [
    path("api/", include("api.urls")),
    # SPA catch-all (must be last)
    path("", serve_spa),
    path("<path:path>", serve_spa),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
