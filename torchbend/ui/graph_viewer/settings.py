SECRET_KEY = "torchbend-graph-viewer-insecure-dev-key-do-not-use-in-production"
DEBUG = True
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "torchbend.ui.graph_viewer",
]

ROOT_URLCONF = "torchbend.ui.graph_viewer.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": False,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
            "loaders": [
                "django.template.loaders.app_directories.Loader",
            ],
        },
    },
]

STATIC_URL = "/static/"

# Maximum bytes the activation cache may occupy before eviction kicks in.
# Large generative models (StyleGAN, RAVE at high resolution) hold single
# activations of several hundred MB, and an activation that does not fit is not
# cached at all — which silently costs a full recomputation on every later
# request, since nothing remains to resume from. Raise it for those.
#
# Set through ``graph_viewer.run(cache_size="16GB")`` or the environment
# variable below; 0 disables caching entirely (always recompute).
import os as _os

ACTIVATION_CACHE_MAX_BYTES = int(
    _os.environ.get("TORCHBEND_ACTIVATION_CACHE_MAX_MB" * 1024 * 1024, 4 * 1024 * 1024 * 1024)
)  # 4 GB
