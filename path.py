import os

for root, dirs, files in os.walk(r"C:\Users\vidya singh"):
    if "Part_01" in dirs:
        print("\nDataset Found:")
        print(root)