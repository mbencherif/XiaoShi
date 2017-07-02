import os

class Config:
    word_dim = 128
    trans_seq_len = 50
    origin_seq_len = 16

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

class Path:
    root = root_path
    raw_input = os.path.join(root_path, "input", "raw")
    interim_input = os.path.join(root_path, "input", "interim")
    processed_input = os.path.join(root_path, "input", "processed")
    models = os.path.join(root_path, "models")
    logs = os.path.join(root_path, "logs")