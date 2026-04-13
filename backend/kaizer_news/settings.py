from pathlib import Path
import os
import dj_database_url

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
# BASE_DIR = kaizer/backend/
# Everything the pipeline needs lives INSIDE kaizer/backend/ — fully portable.

SECRET_KEY   = os.environ.get("SECRET_KEY", "kaizer-news-django-dev-secret-change-in-prod")
DEBUG        = os.environ.get("DEBUG", "true").lower() == "true"
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "django.contrib.staticfiles",
    "rest_framework",
    "corsheaders",
    "api",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.middleware.common.CommonMiddleware",
]

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": [],
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.MultiPartParser",
        "rest_framework.parsers.FormParser",
    ],
}

ROOT_URLCONF = "kaizer_news.urls"
WSGI_APPLICATION = "kaizer_news.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "staticfiles" / "spa"],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    }
]

_DATABASE_URL = os.environ.get("DATABASE_URL")
if _DATABASE_URL:
    DATABASES = {"default": dj_database_url.parse(_DATABASE_URL, conn_max_age=600)}
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

MEDIA_URL  = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

STATIC_URL  = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

CORS_ALLOW_ALL_ORIGINS  = True
CORS_ALLOW_CREDENTIALS  = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ── Pipeline paths — all inside kaizer/backend/ ─────────────
PIPELINE_SCRIPT      = str(BASE_DIR / "pipeline_core" / "pipeline.py")
PIPELINE_OUTPUT_ROOT = os.environ.get("KAIZER_OUTPUT_ROOT",
                       str(BASE_DIR / "output"))
PIPELINE_RESOURCES   = str(BASE_DIR / "resources")
PIPELINE_ASSETS      = str(BASE_DIR / "assests")
