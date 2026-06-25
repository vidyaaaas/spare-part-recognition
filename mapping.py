folder_to_code = {
    "Part_01": "P01-001",
    "Part_02": "P02-001",
    "Part_03": "P03-001",
    "Part_04": "P04-001",
    "Part_05": "P05-001",
    "Part_06": "P06-001"
}

def get_code(folder):
    if folder is None:
        return "Unknown"
    folder = folder.strip()
    return folder_to_code.get(folder, "Unknown")