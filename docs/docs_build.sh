# Installs requirements and builds HTML version of PyTorch/XLA docs.
pip install -r requirements.txt
sphinx-build -b html source build