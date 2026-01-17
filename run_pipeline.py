from __future__ import annotations

import argparse
import subprocess
import sys
import shutil
from pathlib import Path


def run_cmd(args: list[str]) -> None:
    print(f"[run_pipeline] cmd: {' '.join(args)}")
    subprocess.run(args, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", type=str, required=True, help="Input markdown file or folder")
    ap.add_argument("--stage0_path", type=str, default="stage_0_raw")
    ap.add_argument("--stage1_path", type=str, default="stage_1_clean")
    ap.add_argument("--stage2_path", type=str, default="stage_2_chunks")
    ap.add_argument("--prefer_stage1", action="store_true", help="Use stage_1_clean for chunking")
    ap.add_argument("--no_recursive", action="store_true", help="Do not recurse into subfolders")
    ap.add_argument("--emit_links", action="store_true")
    ap.add_argument("--clean_stage0", action="store_true", help="Delete stage_0_raw before copying")
    ap.add_argument("--clean_stage1", action="store_true", help="Delete stage_1_clean before cleaning")
    ap.add_argument("--clean_stage2", action="store_true", help="Delete stage_2_chunks before writing new chunks")
    ap.add_argument("--merge_chunks", action="store_true")
    ap.add_argument("--merged_jsonl", type=str, default="stage_2_chunks_merged.jsonl")
    ap.add_argument("--build_chroma", action="store_true")
    ap.add_argument("--chunks_jsonl", type=str, help="Explicit JSONL for stage 3 (overrides merged_jsonl)")
    ap.add_argument(
        "--chunks_path",
        type=str,
        help="Batch mode: build from all JSONL files under this directory",
    )
    ap.add_argument(
        "--chunks_pattern",
        type=str,
        default="*.jsonl",
        help="Pattern for --chunks_path (default=*.jsonl)",
    )
    ap.add_argument(
        "--chunks_recursive",
        action="store_true",
        help="Recurse when using --chunks_path",
    )
    ap.add_argument("--persist_path", type=str, default="stage_3_chroma")
    ap.add_argument("--collection", type=str, default="v1_chunks")
    ap.add_argument("--mode", type=str, choices=["rebuild", "append", "upsert"], default="upsert")
    ap.add_argument("--skip_unchanged", action="store_true", help="When upserting, skip chunks whose hash hasn't changed")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    python = sys.executable
    no_recursive_flag = ["--no_recursive"] if args.no_recursive else []
    dry_run_flag = ["--dry_run"] if args.dry_run else []

    if args.clean_stage0:
        stage0_path = Path(args.stage0_path).resolve()
        if stage0_path.exists():
            print(f"[run_pipeline] removing stage0_path: {stage0_path}")
            shutil.rmtree(stage0_path)

    run_cmd(
        [
            python,
            "00_copy_raw.py",
            "--input_path",
            args.input_path,
            "--stage0_path",
            args.stage0_path,
            *no_recursive_flag,
            *dry_run_flag,
        ]
    )

    if args.clean_stage1:
        stage1_path = Path(args.stage1_path).resolve()
        if stage1_path.exists():
            print(f"[run_pipeline] removing stage1_path: {stage1_path}")
            shutil.rmtree(stage1_path)

    run_cmd(
        [
            python,
            "01_clean.py",
            "--stage0_path",
            args.stage0_path,
            "--stage1_path",
            args.stage1_path,
            *no_recursive_flag,
            *dry_run_flag,
            *(["--emit_links"] if args.emit_links else []),
        ]
    )

    if args.clean_stage2:
        stage2_path = Path(args.stage2_path).resolve()
        if stage2_path.exists():
            print(f"[run_pipeline] removing stage2_path: {stage2_path}")
            shutil.rmtree(stage2_path)

    run_cmd(
        [
            python,
            "02_chunk.py",
            "--stage0_path",
            args.stage0_path,
            "--stage1_path",
            args.stage1_path,
            "--out_path",
            args.stage2_path,
            *no_recursive_flag,
            *dry_run_flag,
            *(["--prefer_stage1"] if args.prefer_stage1 else []),
        ]
    )

    merged_jsonl = None
    if args.merge_chunks:
        run_cmd(
            [
                python,
                "merge_chunks_jsonl.py",
                "--chunks_path",
                args.stage2_path,
                "--output_jsonl",
                args.merged_jsonl,
                *no_recursive_flag,
                *dry_run_flag,
            ]
        )
        merged_jsonl = args.merged_jsonl

    if args.build_chroma:
        if args.chunks_path:
            chunks_path = Path(args.chunks_path).resolve()
            if not chunks_path.exists():
                raise FileNotFoundError(f"Missing chunks_path: {chunks_path}")
            pattern = f"**/{args.chunks_pattern}" if args.chunks_recursive else args.chunks_pattern
            files = sorted([p for p in chunks_path.glob(pattern) if p.is_file()])
            if not files:
                raise FileNotFoundError(f"No JSONL files found in: {chunks_path}")
            for path in files:
                run_cmd(
                    [
                        python,
                        "03_chroma.py",
                        "--chunks_jsonl",
                        str(path),
                        "--persist_path",
                        args.persist_path,
                        "--collection",
                        args.collection,
                        *dry_run_flag,
                        "--mode",
                        args.mode,
                        *(["--skip_unchanged"] if args.skip_unchanged else []),
                    ]
                )
        else:
            chunks_input = args.chunks_jsonl or merged_jsonl or args.merged_jsonl
            if not args.dry_run and not Path(chunks_input).exists():
                raise FileNotFoundError(
                    "Missing chunks JSONL for stage 3. "
                    "Run with --merge_chunks or provide --chunks_jsonl."
                )
            run_cmd(
                [
                    python,
                    "03_chroma.py",
                    "--chunks_jsonl",
                    chunks_input,
                    "--persist_path",
                    args.persist_path,
                    "--collection",
                    args.collection,
                    *dry_run_flag,
                    "--mode",
                    args.mode,
                    *(["--skip_unchanged"] if args.skip_unchanged else []),
                ]
            )


if __name__ == "__main__":
    main()
