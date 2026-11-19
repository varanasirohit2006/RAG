"""Run React and the Python API with one warmed process."""
import argparse
import os
import sys
from backend.files import PROJECT_ROOT


def main():
    sys.dont_write_bytecode = True
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args()
    if not (PROJECT_ROOT / 'frontend' / 'dist' / 'index.html').is_file():
        parser.error('Build React first: cd frontend; npm ci; npm run build')
    import uvicorn
    from backend.api.main import create_app
    uvicorn.run(create_app(), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
