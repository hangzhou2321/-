from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


class _Reader:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.index = 0

    def next(self) -> str:
        if self.index >= len(self.lines):
            raise ValueError("unexpected end of AC3D file")
        line = self.lines[self.index].strip()
        self.index += 1
        return line

    def peek(self) -> str:
        if self.index >= len(self.lines):
            return ""
        return self.lines[self.index].strip()


def _numbers(line: str) -> list[float]:
    return [float(value) for value in line.split()[1:]]


def _parse_object(
    reader: _Reader,
    parent_rotation: np.ndarray,
    parent_translation: np.ndarray,
    vertices: list[np.ndarray],
    faces: list[tuple[int, int, int]],
) -> None:
    header = reader.next()
    if not header.startswith("OBJECT "):
        raise ValueError(f"expected OBJECT, got {header!r}")

    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    local_vertices: list[np.ndarray] = []
    local_faces: list[list[int]] = []

    while True:
        line = reader.next()
        if not line:
            continue
        key = line.split(maxsplit=1)[0]
        if key == "loc":
            translation = np.asarray(_numbers(line), dtype=np.float64)
        elif key == "rot":
            rotation = np.asarray(_numbers(line), dtype=np.float64).reshape(3, 3)
        elif key == "numvert":
            count = int(line.split()[1])
            local_vertices = [
                np.asarray([float(value) for value in reader.next().split()], dtype=np.float64)
                for _ in range(count)
            ]
        elif key == "numsurf":
            count = int(line.split()[1])
            for _ in range(count):
                surface = reader.next()
                if not surface.startswith("SURF "):
                    raise ValueError(f"expected SURF, got {surface!r}")
                refs: list[int] = []
                while not refs:
                    field = reader.next()
                    if field.startswith("refs "):
                        ref_count = int(field.split()[1])
                        refs = [int(reader.next().split()[0]) for _ in range(ref_count)]
                if len(refs) >= 3:
                    local_faces.append(refs)
        elif key == "data":
            # FlightGear models use single-line metadata payloads here.
            reader.next()
        elif key == "kids":
            child_count = int(line.split()[1])
            object_rotation = parent_rotation @ rotation
            object_translation = parent_rotation @ translation + parent_translation
            offset = len(vertices)
            vertices.extend(
                object_rotation @ vertex + object_translation
                for vertex in local_vertices
            )
            for polygon in local_faces:
                first = offset + polygon[0]
                for index in range(1, len(polygon) - 1):
                    triangle = (first, offset + polygon[index], offset + polygon[index + 1])
                    if len(set(triangle)) == 3:
                        faces.append(triangle)
            for _ in range(child_count):
                _parse_object(
                    reader,
                    object_rotation,
                    object_translation,
                    vertices,
                    faces,
                )
            return


def convert(source: Path, output: Path) -> dict[str, object]:
    reader = _Reader(source.read_text(encoding="utf-8", errors="replace").splitlines())
    header = reader.next()
    if not header.startswith("AC3D"):
        raise ValueError(f"unsupported AC3D header {header!r}")
    while reader.peek().startswith("MATERIAL "):
        reader.next()

    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    _parse_object(
        reader,
        np.eye(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        vertices,
        faces,
    )
    if not vertices or not faces:
        raise ValueError("AC3D conversion produced no mesh")

    array = np.asarray(vertices, dtype=np.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"# Converted from {source.name}\n")
        for vertex in array:
            stream.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for a, b, c in faces:
            stream.write(f"f {a + 1} {b + 1} {c + 1}\n")

    return {
        "vertices": len(vertices),
        "triangles": len(faces),
        "minimum": array.min(axis=0).tolist(),
        "maximum": array.max(axis=0).tolist(),
        "extent": np.ptp(array, axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an AC3D polygon model to OBJ")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(convert(args.source.resolve(strict=True), args.output.resolve()))


if __name__ == "__main__":
    main()
