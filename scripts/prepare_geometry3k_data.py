"""Convert Geometry3K dataset to unified JSONL format for VLM CoT generation.

uvx --with tqdm python scripts/prepare_geometry3k_data.py
"""

import ast
import json
import random
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

SOURCE_DIR = Path("/lustre/fs1/portfolios/llmservice/users/jseppanen/data/geometry3k/train")
OUTPUT_PATH = Path("/lustre/fs1/portfolios/llmservice/users/jseppanen/data/geometry3k/geometry3k_train.jsonl")


QUESTION_PREFIX = (
    "Solve this geometry problem. Study the diagram and select the correct answer.\n\n"
)


def parse_logic_form(text: str):
    """Parse a logic form string into an AST node."""
    try:
        tree = ast.parse(text, mode="eval")
        return tree.body
    except SyntaxError:
        return None


SKIP_TOP_LEVEL = {
    "Angle",
    "Arc",
    "Circle",
    "Cirle",
    "Find",
    "Hexagon",
    "Kite",
    "Line",
    "Octagon",
    "Parallelogram",
    "Pentagon",
    "Point",
    "Polygon",
    "Quadrilateral",
    "Rectangle",
    "Rhombus",
    "Sector",
    "Shape",
    "Square",
    "Trapezoid",
    "Triangle",
}


def node_to_text(node, top_level: bool = False) -> str | None:
    """Convert an AST node to readable text."""
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = node_to_text(node.operand)
        return f"-{inner}" if inner else None
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Name):
        return None
    func = node.func.id
    if top_level and func in SKIP_TOP_LEVEL:
        return None
    args = [node_to_text(a) for a in node.args]
    if None in args:
        return None
    if any("$" in a for a in args):
        return None
    return format_call(func, args)


def format_call(func: str, args: list[str]) -> str | None:
    """Format a function call with its arguments into readable text."""
    if func == "Equals":
        if len(args) == 2:
            prefix = "angle " if args[0].startswith("∠") else ""
            return f"{prefix}{args[0]} = {args[1]}"
    elif func == "Line":
        if len(args) == 2:
            return f"{args[0]}{args[1]}"
    elif func == "Point":
        if len(args) == 1:
            return args[0]
    elif func == "Angle":
        return f"∠{''.join(args)}"
    elif func == "LengthOf":
        if len(args) == 1:
            return f"length of {args[0]}"
    elif func == "MeasureOf":
        if len(args) == 1:
            return args[0]
    elif func == "Perpendicular":
        if len(args) == 2:
            return f"{args[0]} ⟂ {args[1]}"
    elif func == "Parallel":
        if len(args) == 2:
            return f"{args[0]} ∥ {args[1]}"
    elif func == "PointLiesOnLine":
        if len(args) == 2:
            return f"point {args[0]} lies on segment {args[1]}"
    elif func == "PointLiesOnCircle":
        if len(args) == 2:
            return f"point {args[0]} lies on {args[1]}"
    elif func == "IntersectAt":
        if len(args) == 3:
            return f"{args[0]} and {args[1]} intersect at {args[2]}"
        elif len(args) == 2:
            return f"{args[0]} intersects {args[1]}"
    elif func == "Tangent":
        if len(args) == 2:
            return f"{args[0]} tangent to {args[1]}"
    elif func == "Congruent":
        if len(args) == 2:
            return f"{args[0]} ≅ {args[1]}"
    elif func == "Similar":
        if len(args) == 2:
            return f"{args[0]} ~ {args[1]}"
    elif func == "Triangle":
        if len(args) == 3:
            return f"△{args[0]}{args[1]}{args[2]}"
    elif func == "Parallelogram":
        if len(args) == 4:
            return f"▱{args[0]}{args[1]}{args[2]}{args[3]}"
    elif func == "Quadrilateral":
        if len(args) == 4:
            return f"quadrilateral {args[0]}{args[1]}{args[2]}{args[3]}"
    elif func == "Circle":
        if len(args) >= 1:
            return f"circle {args[0]}"
    elif func == "Cirle":
        if len(args) >= 1:
            return f"circle {args[0]}"
    elif func == "Arc":
        if len(args) == 2:
            return f"arc {args[0]}{args[1]}"
    elif func == "Minor":
        if len(args) == 1:
            return f"minor {args[0]}"
    elif func == "Major":
        if len(args) == 1:
            return f"major {args[0]}"
    elif func == "Polygon":
        return f"polygon {''.join(args)}"
    elif func == "Pentagon":
        return f"pentagon {''.join(args)}"
    elif func == "Hexagon":
        return f"hexagon {''.join(args)}"
    elif func == "Octagon":
        return f"octagon {''.join(args)}"
    elif func == "Kite":
        if len(args) == 4:
            return f"kite {args[0]}{args[1]}{args[2]}{args[3]}"
    elif func == "Sector":
        return f"sector {''.join(args)}"
    elif func == "Shape":
        return f"shape {''.join(args)}"
    elif func == "Trapezoid":
        if len(args) == 4:
            return f"trapezoid {args[0]}{args[1]}{args[2]}{args[3]}"
    elif func == "Rhombus":
        if len(args) == 4:
            return f"rhombus {args[0]}{args[1]}{args[2]}{args[3]}"
    elif func == "Rectangle":
        if len(args) == 4:
            return f"rectangle {args[0]}{args[1]}{args[2]}{args[3]}"
    elif func == "Square":
        if len(args) == 4:
            return f"square {args[0]}{args[1]}{args[2]}{args[3]}"
    elif func == "AreaOf":
        if len(args) == 1:
            return f"area of {args[0]}"
    elif func == "PerimeterOf":
        if len(args) == 1:
            return f"perimeter of {args[0]}"
    elif func == "HeightOf":
        if len(args) == 1:
            return f"height of {args[0]}"
    elif func == "RadiusOf":
        if len(args) == 1:
            return f"radius of {args[0]}"
    elif func == "DiameterOf":
        if len(args) == 1:
            return f"diameter of {args[0]}"
    elif func == "LegOf":
        if len(args) == 1:
            return f"leg of {args[0]}"
    elif func == "BaseOf":
        if len(args) == 1:
            return f"base of {args[0]}"
    elif func == "SideOf":
        if len(args) == 1:
            return f"side of {args[0]}"
    elif func == "DiagonalOf":
        if len(args) == 1:
            return f"diagonal of {args[0]}"
    elif func == "AltitudeOf":
        if len(args) == 1:
            return f"altitude of {args[0]}"
    elif func == "HypotenuseOf":
        if len(args) == 1:
            return f"hypotenuse of {args[0]}"
    elif func == "CosOf":
        if len(args) == 1:
            return f"cos({args[0]})"
    elif func == "SinOf":
        if len(args) == 1:
            return f"sin({args[0]})"
    elif func == "TanOf":
        if len(args) == 1:
            return f"tan({args[0]})"
    elif func == "Mul":
        if len(args) == 2:
            return f"{args[0]} × {args[1]}"
    elif func == "Add":
        if len(args) == 2:
            return f"{args[0]} + {args[1]}"
    elif func == "SumOf":
        if len(args) == 2:
            return f"{args[0]} + {args[1]}"
        elif len(args) > 2:
            return " + ".join(args)
    elif func == "Sub":
        if len(args) == 2:
            return f"{args[0]} - {args[1]}"
    elif func == "Div":
        if len(args) == 2:
            return f"{args[0]} / {args[1]}"
    elif func == "Sqrt":
        if len(args) == 1:
            return f"√{args[0]}"
    elif func == "Pow":
        if len(args) == 2:
            return f"{args[0]}^{args[1]}"
    elif func == "Half":
        if len(args) == 1:
            return f"{args[0]}/2"
    elif func == "HalfOf":
        if len(args) == 1:
            return f"half of {args[0]}"
    elif func == "RatioOf":
        if len(args) == 2:
            return f"ratio of {args[0]} to {args[1]}"
    elif func == "IsMedianOf":
        if len(args) == 2:
            return f"segment {args[0]} is median of {args[1]}"
    elif func == "IsMidpointOf":
        if len(args) == 2:
            return f"point {args[0]} is midpoint of segment {args[1]}"
    elif func == "IsCentroidOf":
        if len(args) == 2:
            return f"point {args[0]} is centroid of {args[1]}"
    elif func == "IsIncenterOf":
        if len(args) == 2:
            return f"point {args[0]} is incenter of {args[1]}"
    elif func == "IsCircumcenterOf":
        if len(args) == 2:
            return f"point {args[0]} is circumcenter of {args[1]}"
    elif func == "IsDiameterOf":
        if len(args) == 2:
            return f"segment {args[0]} is diameter of {args[1]}"
    elif func == "IsRadiusOf":
        if len(args) == 2:
            return f"segment {args[0]} is radius of {args[1]}"
    elif func == "IsChordOf":
        if len(args) == 2:
            return f"segment {args[0]} is chord of {args[1]}"
    elif func == "IsAltitudeOf":
        if len(args) == 2:
            return f"segment {args[0]} is altitude of {args[1]}"
    elif func == "IsPerpendicularBisectorOf":
        if len(args) == 2:
            return f"segment {args[0]} is perpendicular bisector of segment {args[1]}"
    elif func == "IsBaseOf":
        if len(args) == 2:
            return f"segment {args[0]} is base of {args[1]}"
    elif func == "IsLegOf":
        if len(args) == 2:
            return f"segment {args[0]} is leg of {args[1]}"
    elif func == "IsHypotenuseOf":
        if len(args) == 2:
            return f"segment {args[0]} is hypotenuse of {args[1]}"
    elif func == "IsDiagonalOf":
        if len(args) == 2:
            return f"segment {args[0]} is diagonal of {args[1]}"
    elif func == "IsSideOf":
        if len(args) == 2:
            return f"segment {args[0]} is side of {args[1]}"
    elif func == "IsMidsegmentOf":
        if len(args) == 2:
            return f"segment {args[0]} is midsegment of {args[1]}"
    elif func == "IsTangentOf":
        if len(args) == 2:
            return f"{args[0]} is tangent to {args[1]}"
    elif func == "IsSecantOf":
        if len(args) == 2:
            return f"{args[0]} is secant of {args[1]}"
    elif func == "Bisects":
        if len(args) == 2:
            return f"{args[0]} bisects {args[1]}"
    elif func == "BisectsAngle":
        if len(args) == 2:
            return f"{args[0]} bisects {args[1]}"
    elif func == "UseTheorem":
        if len(args) == 1:
            theorem = args[0].replace("_", " ")
            return f"use {theorem}"
    elif func == "InscribedIn":
        if len(args) == 2:
            return f"{args[0]} inscribed in {args[1]}"
    elif func == "CircumscribedAbout":
        if len(args) == 2:
            return f"{args[0]} circumscribed about {args[1]}"
    elif func == "CircumscribedTo":
        if len(args) == 2:
            return f"{args[0]} circumscribed to {args[1]}"
    elif func == "IsRightAngle":
        if len(args) == 1:
            return f"{args[0]} is a right angle"
    elif func == "RightAngle":
        if len(args) == 1:
            return f"{args[0]} is a right angle"
    elif func == "IsIsoscelesTriangle":
        if len(args) == 1:
            return f"{args[0]} is isosceles"
    elif func == "IsEquilateralTriangle":
        if len(args) == 1:
            return f"{args[0]} is equilateral"
    elif func == "Equilateral":
        if len(args) == 1:
            return f"{args[0]} is equilateral"
    elif func == "Isosceles":
        if len(args) == 1:
            return f"{args[0]} is isosceles"
    elif func == "IsRightTriangle":
        if len(args) == 1:
            return f"{args[0]} is a right triangle"
    elif func == "Right":
        if len(args) == 1:
            return f"{args[0]} is a right triangle"
    elif func == "IsSquare":
        if len(args) == 1:
            return f"{args[0]} is a square"
    elif func == "IsRectangle":
        if len(args) == 1:
            return f"{args[0]} is a rectangle"
    elif func == "IsRhombus":
        if len(args) == 1:
            return f"{args[0]} is a rhombus"
    elif func == "IsTrapezoid":
        if len(args) == 1:
            return f"{args[0]} is a trapezoid"
    elif func == "IsRegularPolygon":
        if len(args) == 1:
            return f"{args[0]} is a regular polygon"
    else:
        raise NotImplementedError(f"Unknown function: {func}({', '.join(args)})")


def format_logic_form(logic_form: str) -> str:
    """Format a logic form into readable text."""
    node = parse_logic_form(logic_form.strip())
    if node is None:
        return ""
    result = node_to_text(node, top_level=True)
    return result if result else ""


LIES_ON_PATTERN = re.compile(r"^point (\w+) lies on (.+)$")


def group_lies_on_hints(hints: list[str]) -> list[str]:
    """Group 'point X lies on Y' hints by target, then merge segments with same points."""
    segment_to_points: dict[str, set[str]] = {}
    other_hints = []

    for h in hints:
        match = LIES_ON_PATTERN.match(h)
        if match:
            point, target = match.groups()
            segment_to_points.setdefault(target, set()).add(point)
        else:
            other_hints.append(h)

    if not segment_to_points:
        return other_hints

    points_to_segments: dict[frozenset[str], list[str]] = {}
    for segment, points in segment_to_points.items():
        key = frozenset(points)
        points_to_segments.setdefault(key, []).append(segment)

    grouped_lines = []
    for points_set, targets in sorted(points_to_segments.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
        points_str = ", ".join(sorted(points_set))
        if len(points_set) == 1:
            point_part = f"point {points_str} lies"
        else:
            point_part = f"points {points_str} lie"

        segment_names = []
        circle_names = []
        for t in targets:
            if t.startswith("segment "):
                segment_names.append(t[8:])
            elif t.startswith("circle "):
                circle_names.append(t[7:])
            else:
                segment_names.append(t)

        target_parts = []
        if segment_names:
            names = ", ".join(sorted(segment_names))
            if len(segment_names) == 1:
                target_parts.append(f"segment {names}")
            else:
                target_parts.append(f"segments {names}")
        if circle_names:
            names = ", ".join(sorted(circle_names))
            if len(circle_names) == 1:
                target_parts.append(f"circle {names}")
            else:
                target_parts.append(f"circles {names}")

        grouped_lines.append(f"{point_part} on {' and '.join(target_parts)}")

    return other_hints + grouped_lines


def generate_hint(logic_form: dict) -> str:
    """Generate a hint from the logic form information."""
    hints = []

    text_forms = logic_form.get("text_logic_form", [])
    for lf in text_forms:
        formatted = format_logic_form(lf)
        if formatted.strip():
            hints.append(formatted)

    diagram_forms = logic_form.get("diagram_logic_form", [])
    for lf in diagram_forms:
        formatted = format_logic_form(lf)
        if formatted.strip():
            hints.append(formatted)

    if not hints:
        return ""

    hints = group_lies_on_hints(hints)
    hint_lines = [f"- {h}" for h in hints]
    return "\n".join(hint_lines)


def process_sample(args):
    """Process a single geometry3k sample."""
    sample_dir, sample_id = args

    data_path = sample_dir / "data.json"
    logic_form_path = sample_dir / "logic_form.json"
    image_path = sample_dir / "img_diagram.png"

    if not data_path.exists() or not image_path.exists():
        return None

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logic_form = {}
    if logic_form_path.exists():
        with open(logic_form_path, "r", encoding="utf-8") as f:
            logic_form = json.load(f)

    problem_text = data.get("annotat_text") or data.get("problem_text", "")
    problem_text = problem_text.strip()
    if problem_text and not problem_text.endswith((".", "?", "!")):
        problem_text += "."
    choices = data.get("choices", [])
    answer = data.get("answer", "")

    if choices:
        choice_str = "  ".join(
            f"({chr(ord('A') + i)}) {c}" for i, c in enumerate(choices)
        )
        question = f"{QUESTION_PREFIX}{problem_text}\n\n{choice_str}"
    else:
        question = f"{QUESTION_PREFIX}{problem_text}"

    hint = generate_hint(logic_form)

    sample = {
        "dataset": "geometry3k",
        "images": [str(image_path)],
        "question": question,
        "answer": answer,
        "hint": hint,
        "source_path": str(sample_dir),
        "id": sample_id,
        "problem_type_graph": data.get("problem_type_graph", []),
        "problem_type_goal": data.get("problem_type_goal", []),
    }
    return sample


def collect_sample_dirs():
    """Find all sample directories in the source directory."""
    sample_dirs = []
    for item in SOURCE_DIR.iterdir():
        if item.is_dir() and (item / "data.json").exists():
            sample_dirs.append(item)
    return sorted(sample_dirs, key=lambda x: int(x.name))


def main():
    sample_dirs = collect_sample_dirs()
    print(f"Found {len(sample_dirs)} samples")

    args_list = [(sample_dir, i + 1) for i, sample_dir in enumerate(sample_dirs)]

    with ProcessPoolExecutor() as executor:
        samples = list(
            tqdm(
                executor.map(process_sample, args_list),
                total=len(args_list),
                desc="Processing Geometry3K samples",
            )
        )

    samples = [s for s in samples if s is not None]

    random.seed(0)
    random.shuffle(samples)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
