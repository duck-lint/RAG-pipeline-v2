from pathlib import Path
import argparse
import shutil

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_md", type=str, required=True, help="Path to ONE Obsidian .md note")
    ap.add_argument("--stage0_dir", type=str, default="stage_0_raw", help="Output folder for raw note copy")
    ap.add_argument("--dry_run", action="store_true", help="Print what would happen; do not copy")
    args = ap.parse_args()

    print("[stage_0] copy ONE markdown file → stage_0_raw/")
    print(f"[stage_0] args: {args}")

    src = Path(args.input_md).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Not found: {src}")
    if src.is_dir():
        raise IsADirectoryError(f"--input_md must be a file, got directory: {src}")

    dst_dir = Path(args.stage0_dir).expanduser().resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    print(f"[stage_0] src={src}")
    print(f"[stage_0] dst={dst}")

    # If user already put the file in stage_0_raw, don't crash.
    if src.resolve() == dst.resolve():
        print("[stage_0] NOTE: source is already inside stage_0_raw; nothing to copy.")
        return

    if args.dry_run:
        print("[stage_0] dry_run=True (no copy performed)")
        return

    shutil.copy2(src, dst)
    print("[stage_0] copied")


if __name__ == "__main__":
    main()
