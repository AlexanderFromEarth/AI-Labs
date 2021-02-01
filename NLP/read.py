from re import match
from os.path import splitext

from docx import Document


def read_text(path):
    text = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if match(r"^.*\*+$", line):
                yield "\n".join(text)
                text = []
            elif match(r"^$", line):
                pass
            else:
                text.append(line)
        yield "\n".join(text)


def read_docx(path):
    text = []
    for p in Document(path).paragraphs:
        if match(r"^.*\*+$", p.text):
            yield "\n".join(text)
            text = []
        elif match(r"^$", p.text):
            pass
        else:
            text.append(p.text)


def read_doc(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if match(r"^$", line):
                pass
            else:
                yield line


read_exts = {
    ".txt": read_text,
    ".docx": read_docx,
    ".doc": read_doc,
}


def read(path):
    return read_exts[splitext(path)[1]](path)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pprint import pprint

    parser = ArgumentParser()
    parser.add_argument("-i", dest="path", type=str, help="path to text")
    args = parser.parse_args()
    pprint(list(read(args.path)))
