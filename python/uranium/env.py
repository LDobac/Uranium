import os
import re
import sys
from pathlib import Path

ENV_FILE_KEY = "URANIUM_ENV_FILE"
BOM = b"\xef\xbb\xbf"
WHITESPACE = "".join(map(chr, [9, 10, 11, 12, 13, 32, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000]))
ESCAPES = {"\\": "\\", "'": "'", '"': '"', "$": "$", " ": " ", "n": "\n"}
KEY_START = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
KEY_REST = KEY_START | frozenset("0123456789.")
COMPLETE, ESCAPE, STRONG, STRONG_ESCAPE, WEAK, WEAK_ESCAPE, COMMENT, BLANK = range(8)
OPENERS = {"\\": ESCAPE, '"': WEAK, "'": STRONG}
NO_SUB, SUB, SUB_BRACED = range(3)


class ParseError(ValueError):
    pass


def _end_state(state: int, piece: str) -> tuple[int, int]:
    pos = 0
    for pos, c in enumerate(piece):
        if state == BLANK:
            if c == "#":
                return pos, COMMENT
            state = OPENERS.get(c, COMPLETE)
        elif state == COMPLETE:
            state = BLANK if c in WHITESPACE and c not in "\r\n" else OPENERS.get(c, COMPLETE)
        elif state == ESCAPE:
            state = COMPLETE
        elif state == WEAK:
            state = WEAK_ESCAPE if c == "\\" else COMPLETE if c == '"' else WEAK
        elif state == WEAK_ESCAPE:
            state = WEAK
        elif state == STRONG:
            state = STRONG_ESCAPE if c == "\\" else COMPLETE if c == "'" else STRONG
        else:
            state = STRONG
    return pos, state


def _lines(data: bytes):
    pieces = re.findall(rb"[^\n]*\n|[^\n]+", data.removeprefix(BOM))
    i = 0
    while i < len(pieces):
        first, buf, state = i + 1, "", COMPLETE
        while True:
            if i == len(pieces):
                raise ParseError("the file ends inside a quote, after a backslash or after a trailing blank without a newline", first)
            try:
                piece = pieces[i].decode("utf-8")
            except UnicodeDecodeError:
                raise ParseError("not UTF-8", i + 1) from None
            i += 1
            start = len(buf)
            buf += piece
            if buf.lstrip(WHITESPACE).startswith("#"):
                yield first, ""
                break
            pos, state = _end_state(state, piece)
            if state == COMPLETE:
                if buf.endswith("\n"):
                    buf = buf[:-1].removesuffix("\r")
                yield first, buf
                break
            if state == COMMENT:
                yield first, buf[:start + pos]
                break


def _key(line: str) -> tuple[str, str]:
    if not line or line[0] not in KEY_START:
        raise ParseError("a key must start with an ASCII letter or _")
    n = next((i for i, c in enumerate(line) if c not in KEY_REST), len(line))
    return line[:n], line[n:].lstrip(WHITESPACE)


def _substitute(name: list, subst: dict, environ, out: list) -> None:
    key = "".join(name)
    name.clear()
    v = environ.get(key)
    out.append(v if v is not None else subst.get(key) or "")


def _value(text: str, subst: dict, environ) -> str:
    strong = weak = escaped = ending = False
    mode, name, out = NO_SUB, [], []
    for c in text:
        if ending:
            if c in " \t":
                continue
            if c == "#":
                break
            raise ParseError("a blank inside an unquoted value (quote the value)")
        elif escaped:
            if c not in ESCAPES:
                raise ParseError(f"a backslash before {c!r}: outside single quotes only {' '.join(ESCAPES)!r} may follow one (put Windows paths in single quotes)")
            out.append(ESCAPES[c])
            escaped = False
        elif strong:
            if c == "'":
                strong = False
            else:
                out.append(c)
        elif mode != NO_SUB:
            if c.isalnum():
                name.append(c)
            elif mode == SUB_BRACED:
                if c == "}":
                    mode = NO_SUB
                    _substitute(name, subst, environ, out)
                else:
                    name.append(c)
            elif c == "{" and not name:
                mode = SUB_BRACED
            else:
                _substitute(name, subst, environ, out)
                if c != "$":
                    mode = NO_SUB
                    out.append(c)
        elif c == "$":
            mode = SUB
        elif weak:
            if c == '"':
                weak = False
            elif c == "\\":
                escaped = True
            else:
                out.append(c)
        elif c == "'":
            strong = True
        elif c == '"':
            weak = True
        elif c == "\\":
            escaped = True
        elif c in " \t":
            ending = True
        else:
            out.append(c)
    if mode == SUB_BRACED or strong or weak:
        raise ParseError("a quote or ${ is not closed (a $NAME right before the closing double quote swallows it: write ${NAME})")
    _substitute(name, subst, environ, out)
    return "".join(out)


def _pair(line: str, subst: dict, environ) -> tuple[str, str] | None:
    line = line.strip(WHITESPACE)
    if not line or line.startswith("#"):
        return None
    key, line = _key(line)
    if key == "export" and not line.startswith("="):
        key, line = _key(line)
    if not line.startswith("="):
        raise ParseError("no = after the key")
    line = line[1:].lstrip(WHITESPACE)
    if not line or line.startswith("#"):
        subst[key] = None
        return key, ""
    value = _value(line, subst, environ)
    subst[key] = value
    return key, value


def read(data: bytes, environ) -> tuple[int, str] | None:
    subst: dict = {}
    n = 0
    try:
        for n, line in _lines(data):
            pair = _pair(line, subst, environ)
            if pair is not None and pair[0] not in environ:
                environ[pair[0]] = pair[1]
    except ParseError as e:
        return (e.args[1] if len(e.args) > 1 else n), e.args[0]
    return None


def find(start: Path | None = None) -> Path | None:
    d = (start or Path.cwd()).resolve()
    for p in [d, *d.parents]:
        if (p / ".env").is_file():
            return p / ".env"
    return None


def load(path: Path | None = None) -> Path | None:
    if path is None and ENV_FILE_KEY in os.environ:
        path = Path(os.environ[ENV_FILE_KEY])
        if not path.is_file():
            return None
    path = path or find()
    if path is None:
        return None
    stop = read(path.read_bytes(), os.environ)
    if stop is not None:
        print(f"warning: failed to load {path}: line {stop[0]}: {stop[1]}; the lines after it are not read", file=sys.stderr, flush=True)
    return path


def env_choice(key: str, choices: tuple, default: str) -> str:
    v = os.environ.get(key, "") or default
    if v not in choices:
        raise ValueError(f"{key}={v!r}: expected one of {choices}")
    return v


M_TRIM_THRESHOLD = -1
M_MMAP_THRESHOLD = -3
M_ARENA_MAX = -8


def apply_malloc() -> int | None:
    raw = os.environ.get("URANIUM_MALLOC_MMAP_THRESHOLD", "")
    if not raw or not sys.platform.startswith("linux"):
        return None
    import ctypes

    n = int(raw)
    libc = ctypes.CDLL("libc.so.6")
    if not (libc.mallopt(M_MMAP_THRESHOLD, n) and libc.mallopt(M_TRIM_THRESHOLD, n)):
        raise RuntimeError(f"URANIUM_MALLOC_MMAP_THRESHOLD={n}: mallopt refused")
    return n


def apply_malloc_arenas() -> int | None:
    raw = os.environ.get("URANIUM_MALLOC_ARENA_MAX", "")
    if not raw or not sys.platform.startswith("linux"):
        return None
    import ctypes

    n = int(raw)
    if n < 1:
        raise ValueError(f"URANIUM_MALLOC_ARENA_MAX={n}: must be at least 1 (glibc ignores 0 and below)")
    if not ctypes.CDLL("libc.so.6").mallopt(M_ARENA_MAX, n):
        raise RuntimeError(f"URANIUM_MALLOC_ARENA_MAX={n}: mallopt refused")
    return n
