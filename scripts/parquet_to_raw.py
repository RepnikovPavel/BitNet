#!/usr/bin/env python3
"""Convert a wikitext parquet file to plain text for llama-perplexity."""
import sys


def main():
    src, dst = sys.argv[1], sys.argv[2]
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is required: pip install pyarrow")
    table = pq.read_table(src)
    texts = table.column("text").to_pylist()
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))
    print(f"wrote {dst}: {len(texts)} rows")


if __name__ == "__main__":
    main()
