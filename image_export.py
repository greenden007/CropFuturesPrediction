import os

RESULTS_PATH = "training/results_all_models"
OUT_DIR = "images/"

os.makedirs(OUT_DIR, exist_ok=True)

for file in os.listdir(RESULTS_PATH):
    if file.endswith("curves.png"):
        os.rename(os.path.join(RESULTS_PATH, file), os.path.join(OUT_DIR, file))

