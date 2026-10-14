"""Generate decoder AR weights once; the server only reads the output."""
import argparse
import ctypes as C
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--body", choices=("q4_k", "q8_0"), default="q4_k")
    parser.add_argument("--library", default=str(
        Path(__file__).resolve().parents[1] / "build/native/libvime_qwen.so"))
    args = parser.parse_args()
    os.environ.pop("VIME_AR_WEIGHTS", None)
    os.environ["VIME_AR_BODY"] = args.body
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    library = C.CDLL(args.library)
    library.vime_load.argtypes = [C.c_char_p]
    library.vime_load.restype = C.c_void_p
    library.vime_model_free.argtypes = [C.c_void_p]
    library.vime_model_free.restype = None
    library.vime_write_ar_weights.argtypes = [C.c_void_p, C.c_char_p]
    library.vime_write_ar_weights.restype = C.c_int
    started = time.monotonic()
    context = library.vime_load(os.fsencode(args.model_dir))
    if not context:
        raise SystemExit("Could not load the original model")
    try:
        if library.vime_write_ar_weights(context, os.fsencode(args.output)) != 0:
            raise SystemExit("Could not generate AR weights; do not deploy this output")
    finally:
        library.vime_model_free(context)
    size = Path(args.output).stat().st_size
    print(f"Wrote {size / 1024**2:.1f} MiB in {time.monotonic() - started:.1f}s: {args.output}")


if __name__ == "__main__":
    main()
