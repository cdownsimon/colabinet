"""GGUF split filenames, shared by local size checks and the standalone VM installer."""

import re


def gguf_files(path: str) -> list[str]:
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path)
    if not match:
        return [path]
    prefix, index, count = match.groups()
    if int(index) != 1 or int(count) < 1:
        raise ValueError("select the first shard (00001) of a split GGUF model")
    return [f"{prefix}-{i:05d}-of-{count}.gguf" for i in range(1, int(count) + 1)]
