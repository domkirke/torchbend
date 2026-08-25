from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/methods/", views.api_methods, name="api_methods"),
    path("api/models/", views.api_models, name="api_models"),
    path("api/models/select/<str:name>/", views.api_select_model, name="api_select_model"),
    path("api/graph/<str:fn>/", views.api_graph, name="api_graph"),
    path("api/weights/<str:fn>/<str:node>/", views.api_weights, name="api_weights"),
    path("api/activate/<str:fn>/", views.api_activate, name="api_activate"),
    path("api/activate/<str:fn>/<str:node>/", views.api_activate_node, name="api_activate_node"),
    path("api/retrace/<str:fn>/", views.api_retrace, name="api_retrace"),
    path("api/eval_expr/<str:fn>/<str:node>/", views.api_eval_expr, name="api_eval_expr"),
    path("api/activate_audio/<str:fn>/<str:node>/", views.api_activate_audio, name="api_activate_audio"),
    path("api/activate_save/<str:fn>/<str:node>/", views.api_activate_save, name="api_activate_save"),
    path("api/node_source/<str:fn>/<str:node>/", views.api_node_source, name="api_node_source"),
    path("api/source/", views.api_source, name="api_source"),
    # bending API — fixed routes before the parameterised one
    path("api/bending/callbacks/", views.api_bending_callbacks, name="api_bending_callbacks"),
    path("api/bending/export/", views.api_bending_export, name="api_bending_export"),
    path("api/bending/config/", views.api_bending_config_export, name="api_bending_config_export"),
    path("api/bending/config/tbconfig/", views.api_bending_config_tbconfig, name="api_bending_config_tbconfig"),
    path("api/bending/config/tbconfig/import/", views.api_bending_config_tbconfig_import, name="api_bending_config_tbconfig_import"),
    path("api/bending/config/import/", views.api_bending_config_import, name="api_bending_config_import"),
    path("api/bending/reorder/", views.api_bending_reorder, name="api_bending_reorder"),
    path("api/bending/mode/", views.api_bending_mode, name="api_bending_mode"),
    path("api/bending/", views.api_bendings, name="api_bendings"),
    path("api/bending/<str:bid>/link/", views.api_bending_link, name="api_bending_link"),
    path("api/bending/<str:bid>/unlink/", views.api_bending_unlink, name="api_bending_unlink"),
    path("api/bending/<str:bid>/", views.api_bending_detail, name="api_bending_detail"),
    # BendingParameter API
    path("api/bending_params/", views.api_bending_params, name="api_bending_params"),
    path("api/bending_params/<str:name>/", views.api_bending_param_detail, name="api_bending_param_detail"),
    # Activation cache settings
    path("api/cache/", views.api_cache, name="api_cache"),
    # Node view selection
    path("api/views/<str:fn>/<str:node>/", views.api_views, name="api_views"),
    # Client UI state (pins, favs, tags, bookmarks) — sync only
    path("api/client-state/", views.api_client_state, name="api_client_state"),
    # Play mode
    path("play/", views.play, name="play"),
    path("api/play/devices/", views.api_play_devices, name="api_play_devices"),
    path("api/play/release/", views.api_play_release, name="api_play_release"),
    path("api/play/compile/", views.api_play_compile, name="api_play_compile"),
    path("api/play/macro/", views.api_play_macro, name="api_play_macro"),
    path("api/play/macros/", views.api_play_macros, name="api_play_macros"),
    path("api/play/join/", views.api_play_join, name="api_play_join"),
    path("api/play/run/", views.api_play_run, name="api_play_run"),
    path("api/play/run_audio/<str:idx>/", views.api_play_run_audio, name="api_play_run_audio"),
]
