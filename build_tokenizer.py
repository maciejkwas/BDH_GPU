import os
# Support both package and local runs
try:
    from .tokenizer import SimpleTokenizer
    from .config_utils import load_config
except ImportError:
    from tokenizer import SimpleTokenizer
    from config_utils import load_config


def main():
    cfg = load_config()
    tcfg = cfg["tokenizer"]
    files = tcfg["files"]
    vocab_size = int(tcfg.get("vocab_size", 2000))
    out = tcfg["out"]

    tk = SimpleTokenizer.train_from_files(files, vocab_size=vocab_size)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(tk.to_json_str())
    print("Saved tokenizer to", out)


if __name__ == "__main__":
    main()
