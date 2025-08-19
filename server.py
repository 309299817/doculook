import uvicorn

from doculook.cli.fast_api import app


def main():
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()


