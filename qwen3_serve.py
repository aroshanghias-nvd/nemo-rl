#!/usr/bin/env python
"""
qwen3_serve.py - Wrapper for vLLM OpenAI server for Qwen3-VL

This is a minimal wrapper that simply runs vLLM's OpenAI API server.
vLLM 0.12+ has native Qwen3VL support, so no patches are needed.

Official Qwen3-VL Thinking settings: seed=1234
"""

import sys
import runpy


def main():
    """Run vLLM's OpenAI-compatible API server."""
    # Inject seed=1234 for reproducibility (official Qwen3-VL setting)
    if "--seed" not in sys.argv:
        sys.argv.extend(["--seed", "1234"])
    
    # Run the vLLM OpenAI server
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
