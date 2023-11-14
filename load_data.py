def load_data(file_name,mode='r'):
    # This function will read the characters from the file
    with open(file_name, mode, encoding='utf-8') as f:
        text = f.read()

    return text

