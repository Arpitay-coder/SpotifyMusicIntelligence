"""
main.py
=======
Simple entry point: `python main.py` runs the Flask dev server.
(You can also run `python app.py` directly — both work.)
"""

from app import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
