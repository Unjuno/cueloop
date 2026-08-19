from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass(frozen=True)
class DeletionResult:
    path: str
    existed: bool
    removed: bool


def delete_session_data(
    project_root: Path,
    relative_targets: tuple[str, ...] = ("data", "logs", "artifacts", "recordings", "transcripts"),
    *,
    confirm: bool = False,
) -> list[DeletionResult]:
    """Delete only explicitly named, direct children of a resolved project root."""

    root = project_root.resolve()
    validated: list[tuple[Path, bool]] = []
    seen: set[str] = set()
    for target in relative_targets:
        candidate = Path(target)
        if (
            candidate.is_absolute()
            or len(candidate.parts) != 1
            or candidate.name in {"", ".", ".."}
            or candidate.name in seen
        ):
            raise ValueError("deletion targets must be unique named direct children of the project root")
        seen.add(candidate.name)
        target_path = root / candidate
        is_link = target_path.is_symlink()
        if not is_link and target_path.resolve().parent != root:
            raise ValueError("deletion target escapes project root")
        validated.append((target_path, is_link))
    if not confirm:
        return [
            DeletionResult(str(path), path.exists() or is_link, False)
            for path, is_link in validated
        ]
    results: list[DeletionResult] = []
    for target_path, is_link in validated:
        existed = target_path.exists() or is_link
        if existed:
            if is_link or target_path.is_file():
                target_path.unlink()
            elif target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                raise ValueError(f"unsupported deletion target: {target_path}")
        results.append(DeletionResult(str(target_path), existed, existed))
    return results
