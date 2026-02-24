# Gunicorn configuration for Render deployment
# This file overrides default settings to prevent timeouts during heavy AI model loading.

bind = "0.0.0.0:10000" # Default, will be overridden by $PORT env var in shell
workers = 1
timeout = 600
keepalive = 2
max_requests = 10
max_requests_jitter = 2
worker_class = 'sync' # Standard sync worker is best for memory-constrained free tier
preload_app = False   # KEEP FALSE to ensure lazy model loading works correctly
