"""Vercel serverless entrypoint for the existing FastAPI application."""
from app.main import app

# Vercel detects this ASGI application and routes rewritten requests to it.
