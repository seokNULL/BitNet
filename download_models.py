#!/usr/bin/env python3
"""Download (and optionally convert to GGUF) the 1-bit models supported by bitnet.cpp.

Examples:
    # Show all supported models
    python download_models.py --list

    # Download the official 2B model (pre-quantized GGUF, no conversion needed)
    python download_models.py --models BitNet-b1.58-2B-4T

    # Download several models and convert them to i2_s GGUF for benchmarking
    python download_models.py --models Falcon3-1B-Instruct-1.58bit Falcon3-3B-Instruct-1.58bit --convert

    # Download everything (large! see --list for sizes)
    python download_models.py --all --convert

"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("download_models")

# kind: "llm"        -> convert via utils/convert-hf-to-gguf-bitnet.py (--outtype i2_s)
#       "embedding"  -> convert via utils/convert-bitnet-embedding-to-gguf.py (i2_s)
#       "gguf"       -> repository already ships ggml-model-i2_s.gguf, nothing to convert
# size is the approximate download size.
MODELS = {
    "BitNet-b1.58-2B-4T": {
        "repo": "microsoft/BitNet-b1.58-2B-4T-gguf",
        "kind": "gguf",
        "size": "~1.2 GB",
        "note": "Official 2B model, pre-quantized i2_s GGUF",
    },
    "bitnet_b1_58-large": {
        "repo": "1bitLLM/bitnet_b1_58-large",
        "kind": "llm",
        "size": "~3 GB",
        "note": "Community 0.7B model (fp32 weights)",
    },
    "bitnet_b1_58-3B": {
        "repo": "1bitLLM/bitnet_b1_58-3B",
        "kind": "llm",
        "size": "~13 GB",
        "note": "Community 3.3B model (fp32 weights)",
    },
    "Llama3-8B-1.58-100B-tokens": {
        "repo": "HF1BitLLM/Llama3-8B-1.58-100B-tokens",
        "kind": "llm",
        "size": "~4 GB",
        "note": "Llama3-8B fine-tuned to 1.58 bit",
    },
    "Falcon3-1B-Instruct-1.58bit": {
        "repo": "tiiuae/Falcon3-1B-Instruct-1.58bit",
        "kind": "llm",
        "size": "~1 GB",
        "note": "Falcon3 family",
    },
    "Falcon3-3B-Instruct-1.58bit": {
        "repo": "tiiuae/Falcon3-3B-Instruct-1.58bit",
        "kind": "llm",
        "size": "~2 GB",
        "note": "Falcon3 family",
    },
    "Falcon3-3B-1.58bit": {
        "repo": "tiiuae/Falcon3-3B-1.58bit",
        "kind": "llm",
        "size": "~2 GB",
        "note": "Falcon3 family (base)",
    },
    "Falcon3-7B-Instruct-1.58bit": {
        "repo": "tiiuae/Falcon3-7B-Instruct-1.58bit",
        "kind": "llm",
        "size": "~3.5 GB",
        "note": "Falcon3 family",
    },
    "Falcon3-7B-1.58bit": {
        "repo": "tiiuae/Falcon3-7B-1.58bit",
        "kind": "llm",
        "size": "~3.5 GB",
        "note": "Falcon3 family (base)",
    },
    "Falcon3-10B-Instruct-1.58bit": {
        "repo": "tiiuae/Falcon3-10B-Instruct-1.58bit",
        "kind": "llm",
        "size": "~5 GB",
        "note": "Falcon3 family",
    },
    "Falcon3-10B-1.58bit": {
        "repo": "tiiuae/Falcon3-10B-1.58bit",
        "kind": "llm",
        "size": "~5 GB",
        "note": "Falcon3 family (base)",
    },
    "Falcon-E-1B-Instruct": {
        "repo": "tiiuae/Falcon-E-1B-Instruct",
        "kind": "llm",
        "size": "~2 GB",
        "note": "Falcon-Edge family",
    },
    "Falcon-E-1B-Base": {
        "repo": "tiiuae/Falcon-E-1B-Base",
        "kind": "llm",
        "size": "~2 GB",
        "note": "Falcon-Edge family",
    },
    "Falcon-E-3B-Instruct": {
        "repo": "tiiuae/Falcon-E-3B-Instruct",
        "kind": "llm",
        "size": "~4 GB",
        "note": "Falcon-Edge family",
    },
    "Falcon-E-3B-Base": {
        "repo": "tiiuae/Falcon-E-3B-Base",
        "kind": "llm",
        "size": "~4 GB",
        "note": "Falcon-Edge family",
    },
    "bitnet-embedding-0.6b": {
        "repo": "microsoft/bitnet-embedding-0.6b",
        "kind": "embedding",
        "size": "~1.5 GB",
        "note": "1-bit embedding model (see docs/bitnet-embeddings-i2s-guide.md)",
    },
    "bitnet-embedding-270m": {
        "repo": "microsoft/bitnet-embedding-270m",
        "kind": "embedding",
        "size": "~0.7 GB",
        "note": "1-bit embedding model (see docs/bitnet-embeddings-i2s-guide.md)",
    },
}


def hf_cli():
    """Return the HuggingFace download CLI available on this system."""
    for cli in ("huggingface-cli", "hf"):
        if shutil.which(cli):
            return cli
    logger.error("Neither 'huggingface-cli' nor 'hf' found. Install it with: pip install -U huggingface_hub")
    sys.exit(1)


def run(cmd, log_file=None):
    logger.info("$ %s", " ".join(cmd))
    if log_file:
        with open(log_file, "w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            logger.error("Command failed, see %s", log_file)
    else:
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            logger.error("Command failed with exit code %d", proc.returncode)
    return proc.returncode == 0


def resolve(name):
    """Accept either the short name or the full HF repo id."""
    if name in MODELS:
        return name
    for short, info in MODELS.items():
        if info["repo"].lower() == name.lower():
            return short
    logger.error("Unknown model '%s'. Use --list to see supported models.", name)
    sys.exit(1)


def download(short, models_dir):
    info = MODELS[short]
    local_dir = os.path.join(models_dir, short)
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s (%s) -> %s", info["repo"], info["size"], local_dir)
    return run([hf_cli(), "download", info["repo"], "--local-dir", local_dir])


def convert(short, models_dir, quant_embd=False):
    info = MODELS[short]
    local_dir = os.path.join(models_dir, short)
    i2s_path = os.path.join(local_dir, "ggml-model-i2_s.gguf")

    if info["kind"] == "gguf":
        logger.info("%s ships pre-quantized GGUF, nothing to convert.", short)
        return True
    if os.path.exists(i2s_path) and os.path.getsize(i2s_path) > 0:
        logger.info("%s: %s already exists, skipping conversion.", short, i2s_path)
        return True

    if info["kind"] == "embedding":
        return run([sys.executable, "utils/convert-bitnet-embedding-to-gguf.py",
                    local_dir, "--outtype", "i2_s", "--outfile", i2s_path])

    # llm: quantize to i2_s directly in the converter (the llama-quantize tool in
    # the current llama.cpp submodule no longer accepts the I2_S ftype)
    convert_cmd = [sys.executable, "utils/convert-hf-to-gguf-bitnet.py", local_dir, "--outtype", "i2_s"]
    if quant_embd:
        convert_cmd.append("--quant-embd")
    return run(convert_cmd)


def list_models():
    width = max(len(n) for n in MODELS)
    print(f"{'NAME':<{width}}  {'SIZE':>8}  {'KIND':<9}  REPO / NOTE")
    for name, info in MODELS.items():
        print(f"{name:<{width}}  {info['size']:>8}  {info['kind']:<9}  {info['repo']}  ({info['note']})")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", "-l", action="store_true", help="List supported models and exit")
    parser.add_argument("--models", "-m", nargs="+", default=[], metavar="NAME",
                        help="Model short names or HF repo ids to download (see --list)")
    parser.add_argument("--all", "-a", action="store_true", help="Download every supported model")
    parser.add_argument("--models-dir", "-d", default="models", help="Directory to store models (default: models)")
    parser.add_argument("--convert", "-c", action="store_true",
                        help="Convert downloaded models to i2_s GGUF (requires a prior setup_env.py build)")
    parser.add_argument("--quant-embd", action="store_true", help="Also quantize the embedding layer")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()

    if args.list:
        list_models()
        return

    targets = list(MODELS) if args.all else [resolve(m) for m in args.models]
    if not targets:
        logger.error("No models selected. Use --models NAME [NAME ...], --all, or --list.")
        sys.exit(1)

    failed = []
    for short in targets:
        ok = download(short, args.models_dir)
        if ok and args.convert:
            ok = convert(short, args.models_dir, quant_embd=args.quant_embd)
        if not ok:
            failed.append(short)

    print()
    for short in targets:
        status = "FAILED" if short in failed else "OK"
        print(f"  [{status}] {short} -> {os.path.join(args.models_dir, short)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
