import os

# find the version
version_file = os.path.join(
    os.path.dirname(__file__), "configs", "version", "default.yaml"
)
with open(version_file, "r") as f:
    version_text = f.read()
__version__ = version_text.split(":")[-1].strip()
