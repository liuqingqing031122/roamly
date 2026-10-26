"""FastAPI application entry point for the Roamly backend."""

from fastapi import FastAPI

app = FastAPI(title="Roamly API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the service is running."""
    return {"status": "ok"}
