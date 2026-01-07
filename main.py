"""Main entry point for Oz Vector Search."""

import uvicorn


def main():
    """Run the Oz API server."""
    uvicorn.run(
        "oz.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
