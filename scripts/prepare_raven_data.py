"""Convert RAVEN dataset to unified JSONL format with PNG images.

srun -p cpu_interactive -A llmservice_fm_vision -N 1 \
    --job-name "nemo-rl-dev:interactive" \
    --exclusive \
    -t 04:00:00 \
    --pty \
    bash -l

uvx --with tqdm --with pillow --with numpy python scripts/prepare_raven_data.py
"""

import json
import math
import logging
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

RAVEN_ROOT = Path(
    "/lustre/fs1/portfolios/llmservice/users/jseppanen/data/RAVEN/RAVEN-10000"
)
OUTPUT_DIR = Path(
    "/lustre/fsw/portfolios/llmservice/users/jseppanen/data/RAVEN/prepared"
)
IMAGE_SIZE = 160
BASE_PADDING = 10
BASE_DIVIDER_HEIGHT = 20
BASE_FONT_SIZE = 24

QUESTION = "Look at the 3x3 puzzle on top and the 8 answer options on bottom. Which answer option (A–H) best completes the pattern?"

SHAPE_NAMES = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]
SIZE_NAMES = ["tiny", "small", "medium-small", "medium", "medium-large", "large"]
# 255, 224, 196, 168, 140, 112, 84, 56, 28, 0
COLOR_NAMES = [
    "white",
    "near-white",
    "very light gray",
    "light gray",
    "medium-light gray",
    "medium-dark gray",
    "dark gray",
    "very dark gray",
    "near-black",
    "black",
]
RULE_ATTR_MAP = {
    "Type": "shape",
    "Size": "size",
    "Color": "color",
    "Number": "count",
    "Position": "position",
    "Number/Position": "count",
}

POSITION_NAMES_2X2 = {
    (0.25, 0.25): "top-left",
    (0.25, 0.75): "top-right",
    (0.75, 0.25): "bottom-left",
    (0.75, 0.75): "bottom-right",
}

POSITION_NAMES_3X3 = {
    (0.16, 0.16): "top-left",
    (0.16, 0.5): "top",
    (0.16, 0.83): "top-right",
    (0.5, 0.16): "left",
    (0.5, 0.5): "center",
    (0.5, 0.83): "right",
    (0.83, 0.16): "bottom-left",
    (0.83, 0.5): "bottom",
    (0.83, 0.83): "bottom-right",
}

POSITION_NAMES_2X2_INNER = {
    (0.42, 0.42): "top-left",
    (0.42, 0.58): "top-right",
    (0.58, 0.42): "bottom-left",
    (0.58, 0.58): "bottom-right",
}

CONFIGS = [
    "center_single",
    "distribute_four",
    "distribute_nine",
    "left_center_single_right_center_single",
    "up_center_single_down_center_single",
    "in_center_single_out_center_single",
    "in_distribute_four_out_center_single",
]


def parse_rules_from_xml(xml_path: Path) -> dict:
    """Parse the XML file and return a nested dictionary of rules."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    first_panel = root.find("Panels/Panel")
    if first_panel is None:
        return {}

    components = first_panel.findall("Struct/Component")
    component_names = {}
    for comp in components:
        comp_id = comp.get("id")
        comp_name = comp.get("name", "Unknown")
        component_names[comp_id] = comp_name

    rules_elem = root.find("Rules")
    if rules_elem is None:
        return {}

    rules_dict = {}
    for rule_group in rules_elem.findall("Rule_Group"):
        group_id = rule_group.get("id", "0")
        comp_name = component_names.get(group_id, f"Component_{group_id}")

        component_rules = {}
        for rule in rule_group.findall("Rule"):
            attr = rule.get("attr", "unknown")
            name = rule.get("name", "unknown")
            if attr == "Number/Position":
                component_rules["count"] = name
                component_rules["position"] = name
                continue
            mapped_attr = RULE_ATTR_MAP.get(attr)
            if mapped_attr:
                component_rules[mapped_attr] = name
            else:
                component_rules[attr] = name

        if component_rules:
            rules_dict[comp_name] = component_rules

    return rules_dict


def extract_panel_attrs(
    xml_path: Path,
) -> tuple[list[list[dict]], list[list[dict]], list[str]]:
    """Extract entity attributes from all panels in the XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    panels = root.findall("Panels/Panel")

    puzzle_panels = []
    answer_panels = []
    component_names = []

    for i, panel in enumerate(panels):
        entities = []
        for comp in panel.findall(".//Component"):
            comp_name = comp.get("name", "Unknown")
            if i == 0:
                component_names.append(comp_name)
            layout = comp.find("Layout")
            position_indices = set()
            if layout is not None:
                for entity in layout.findall("Entity"):
                    bbox_str = entity.get("bbox", "[]")
                    try:
                        bbox = json.loads(bbox_str)
                        position_indices.add(
                            tuple(bbox) if isinstance(bbox, list) else bbox
                        )
                    except:
                        pass
                    entities.append(
                        {
                            "shape": int(entity.get("Type", 0)),
                            "size": int(entity.get("Size", 0)),
                            "color": int(entity.get("Color", 0)),
                            "component": comp_name,
                            "bbox": bbox if "bbox" in locals() else None,
                        }
                    )
        if i < 8:
            puzzle_panels.append(entities)
        else:
            answer_panels.append(entities)

    return puzzle_panels, answer_panels, component_names


def bbox_to_position_name(bbox) -> str:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 2:
        return str(bbox)
    y, x = round(bbox[0], 2), round(bbox[1], 2)
    for pos_dict in [POSITION_NAMES_2X2, POSITION_NAMES_3X3, POSITION_NAMES_2X2_INNER]:
        for (py, px), name in pos_dict.items():
            if abs(y - py) < 0.05 and abs(x - px) < 0.05:
                return name
    return f"({y:.2f},{x:.2f})"


def format_positions(bboxes: list) -> str:
    if not bboxes:
        return "none"
    names = sorted(set(bbox_to_position_name(b) for b in bboxes))
    if all(name in ["TL", "TR", "BL", "BR"] for name in names):
        return "{" + ",".join(names) + "}"
    elif all(name.isdigit() and 1 <= int(name) <= 9 for name in names):
        return "{" + ",".join(names) + "}"
    return "{" + ",".join(names) + "}"


def format_attr_value(attr: str, val) -> str:
    if isinstance(val, tuple):
        val = list(val)
    if isinstance(val, list):
        if not val:
            return "empty"
        if attr == "position":
            return format_positions(val)
        formatted = [format_attr_value(attr, v) for v in val]
        if len(set(formatted)) == 1:
            return formatted[0]
        return "(" + ", ".join(formatted) + ")"
    if attr == "shape":
        return SHAPE_NAMES[val]
    elif attr == "size":
        return SIZE_NAMES[val]
    elif attr == "color":
        return COLOR_NAMES[val]
    elif attr == "count":
        return str(val)
    elif attr == "position":
        return bbox_to_position_name(val)
    return str(val)


def extract_row_values(
    puzzle_panels: list, row_idx: int, attr: str, comp_filter
) -> list:
    row_indices = [row_idx * 3, row_idx * 3 + 1, row_idx * 3 + 2]
    panels = [puzzle_panels[i] if i < len(puzzle_panels) else [] for i in row_indices]
    if attr == "count":
        return [len([e for e in p if comp_filter(e)]) for p in panels]
    elif attr == "position":
        result = []
        for p in panels:
            filtered = [e for e in p if comp_filter(e)]
            bboxes = [e.get("bbox") for e in filtered if e.get("bbox") is not None]
            result.append(bboxes if bboxes else [])
        return result
    else:
        filtered = [[e for e in p if comp_filter(e)] for p in panels]
        return [[e.get(attr, 0) for e in p] for p in filtered]


def get_answer_attrs(answer_panel: list, comp_name: str, multi_component: bool) -> dict:
    if multi_component:
        filtered = [e for e in answer_panel if e.get("component") == comp_name]
    else:
        filtered = answer_panel
    if not filtered:
        return {}
    return {
        "count": len(filtered),
        "shapes": [e["shape"] for e in filtered],
        "sizes": [e["size"] for e in filtered],
        "colors": [e["color"] for e in filtered],
        "positions": [e.get("bbox") for e in filtered if e.get("bbox") is not None],
    }


def describe_entity(entity: dict) -> str:
    size_word = SIZE_NAMES[entity["size"]]
    color_word = COLOR_NAMES[entity["color"]]
    shape = SHAPE_NAMES[entity["shape"]]
    return f"{size_word} {color_word} {shape}"


def describe_panel(panel: list[dict]) -> str:
    if not panel:
        return "empty"
    if len(panel) == 1:
        return describe_entity(panel[0])
    shapes = [SHAPE_NAMES[e["shape"]] for e in panel]
    sizes = [SIZE_NAMES[e["size"]] for e in panel]
    unique_shapes = set(shapes)
    unique_sizes = set(sizes)
    size_str = list(unique_sizes)[0] if len(unique_sizes) == 1 else "mixed-size"
    if len(unique_shapes) == 1:
        return f"{len(panel)} {size_str} {list(unique_shapes)[0]}s"
    shape_counts = {}
    for s in shapes:
        shape_counts[s] = shape_counts.get(s, 0) + 1
    shape_parts = [
        f"{cnt} {shp}s" if cnt > 1 else f"1 {shp}"
        for shp, cnt in sorted(shape_counts.items())
    ]
    return f"{len(panel)} {size_str} objects: " + ", ".join(shape_parts)


def get_uniform_value(val_list):
    if isinstance(val_list, list) and val_list:
        if len(set(val_list)) == 1:
            return val_list[0]
    elif isinstance(val_list, int):
        return val_list
    return None


def rule_to_natural(
    rule_name: str, attr: str, row3_vals: list, row1_vals: list = None
) -> str:
    attr_plural = attr + "s"

    if rule_name == "Constant":
        if attr == "position":
            pos_strs = [format_attr_value("position", v) for v in row3_vals if v]
            if pos_strs and len(set(pos_strs)) == 1:
                return f"Wait, the positions are constant across each row! The missing cell should have objects at {pos_strs[0]}."
            return "The positions stay the same pattern within each row."
        row3_flat = []
        for v in row3_vals or []:
            if v is None:
                continue
            if isinstance(v, list):
                u = get_uniform_value(v)
                if u is not None:
                    row3_flat.append(u)
            else:
                row3_flat.append(v)
        row3_flat = [v for v in row3_flat if v is not None]
        if row3_flat and len(set(row3_flat)) == 1:
            target_val = row3_flat[0]
            target_str = format_attr_value(attr, target_val)
            return f"Wait, it looks like the {attr_plural} are the same in each row, so the missing cell should also be {target_str}."
        observed = []
        for row in (row1_vals or []), (row3_vals or []):
            for v in row:
                if isinstance(v, list):
                    norm = tuple(sorted(v))
                else:
                    norm = v
                observed.append(norm)
        observed = [o for o in observed if o is not None]
        if observed and len(set(observed)) == 1:
            first_val = format_attr_value(attr, observed[0])
            return f"Wait, the {attr_plural} are the same across columns, so the missing cell should match {first_val}."
        if observed:
            first_val = format_attr_value(attr, observed[0])
            return f"Hmm, {attr_plural} don't seem to follow any pattern."
        return f"It looks like {attr_plural} stay the same across columns."
    elif rule_name == "Progression":
        if attr == "position":
            pos_strs = [format_attr_value("position", v) for v in row3_vals if v]
            return f"I notice the positions shift systematically across columns (progression). Row 3 so far: {' → '.join(pos_strs)} → ?"
        flat_vals = [get_uniform_value(v) for v in row3_vals if v]
        flat_vals = [v for v in flat_vals if v is not None]
        if len(flat_vals) >= 2:
            diff = flat_vals[1] - flat_vals[0]
            if diff > 0:
                verb = "increase" if attr != "color" else "become darker"
                return f"This looks like a pattern where {attr_plural} {verb} from left to right."
            elif diff < 0:
                verb = "decrease" if attr != "color" else "become lighter"
                return f"This looks like a pattern where {attr_plural} {verb} from left to right."
        return (
            f"I notice that {attr_plural} change by a constant amount across columns."
        )
    elif rule_name == "Arithmetic":
        if attr == "position":
            p1 = format_attr_value("position", row3_vals[0]) if row3_vals else "?"
            p2 = format_attr_value("position", row3_vals[1]) if len(row3_vals) > 1 else "?"
            return f"Wait, positions follow an arithmetic rule! Col3 = union or difference of Col1 and Col2. Row 3: {p1} ∪/∖ {p2} = ?"
        flat_vals = [get_uniform_value(v) for v in row3_vals if v]
        flat_vals = [v for v in flat_vals if v is not None]
        if len(flat_vals) >= 2:
            c1, c2 = flat_vals[0], flat_vals[1]
            if attr == "color" or attr == "count":
                add_result = c1 + c2
                sub_result = c1 - c2
                add_formula = "col1 + col2"
                sub_formula = "col1 - col2"
                add_calc = f"{c1} + {c2}"
                sub_calc = f"{c1} - {c2}"
                max_val = 9
            else:
                add_result = c1 + c2 + 1
                sub_result = c1 - c2 - 1
                add_formula = "col1 + col2 + 1"
                sub_formula = "col1 - col2 - 1"
                add_calc = f"{c1} + {c2} + 1"
                sub_calc = f"{c1} - {c2} - 1"
                max_val = 5 if attr in ("shape", "size") else 9
            c1_str = format_attr_value(attr, c1)
            c2_str = format_attr_value(attr, c2)
            if 0 <= sub_result <= max_val and not (0 <= add_result <= max_val):
                result_str = format_attr_value(attr, sub_result)
                return f"Wait, there is a pattern here! In each row, {attr_plural} follow col3 = {sub_formula}. For row 3: col1={c1_str}, col2={c2_str}, so col3 = {sub_calc} = {sub_result} ({result_str})."
            elif 0 <= add_result <= max_val and not (0 <= sub_result <= max_val):
                result_str = format_attr_value(attr, add_result)
                return f"Wait, there is a pattern here! In each row, {attr_plural} follow col3 = {add_formula}. For row 3: col1={c1_str}, col2={c2_str}, so col3 = {add_calc} = {add_result} ({result_str})."
            else:
                add_str = (
                    format_attr_value(attr, add_result)
                    if 0 <= add_result <= max_val
                    else f"invalid({add_result})"
                )
                sub_str = (
                    format_attr_value(attr, sub_result)
                    if 0 <= sub_result <= max_val
                    else f"invalid({sub_result})"
                )
                return f"Wait, there is a pattern here! In each row, {attr_plural} follow an arithmetic rule. For row 3: col1={c1_str}, col2={c2_str}. This gives {add_formula}={add_str} or {sub_formula}={sub_str}."
        return f"Wait, the {attr_plural} in col3 follow an arithmetic pattern with col1 and col2 in each row."
    elif rule_name == "Distribute_Three":
        if attr == "position":
            pos_strs = [format_attr_value("position", v) for v in row3_vals if v]
            row1_pos_strs = [format_attr_value("position", v) for v in (row1_vals or []) if v]
            if row1_pos_strs:
                row1_set = set(row1_pos_strs)
                row3_set = set(pos_strs)
                missing = row1_set - row3_set
                if missing:
                    return f"Wait, there is a pattern here! Each row has the same 3 positions shuffled. Row 3 has {', '.join(pos_strs)}, so missing is {', '.join(missing)}."
            return "Wait, there is a pattern here! Each row contains the same three positions shuffled."
        row3_flat = [
            get_uniform_value(v) if isinstance(v, list) else v for v in row3_vals if v
        ]
        row3_flat = [v for v in row3_flat if v is not None]
        if row1_vals:
            row1_flat = [
                get_uniform_value(v) if isinstance(v, list) else v
                for v in row1_vals
                if v
            ]
            row1_flat = [v for v in row1_flat if v is not None]
            full_set = set(row1_flat)
            row3_set = set(row3_flat)
            missing_vals = full_set - row3_set
            if missing_vals:
                missing = list(missing_vals)[0]
                if attr == "count":
                    return f"Wait, the {attr_plural} follow a pattern! Each row contains the same three {attr_plural} shuffled. Row 3 has {row3_flat}, so the missing {attr} is {missing}."
                return f"Wait, the {attr_plural} follow a pattern! Each row contains the same three {attr_plural} shuffled. Row 3 has {[format_attr_value(attr, v) for v in row3_flat]}, so the missing {attr} is {format_attr_value(attr, missing)}."
        return f"Wait, the {attr_plural} follow a pattern! Each row contains the same three {attr_plural}, shuffled differently."
    return f"It looks like the {attr_plural} follow the {rule_name.lower()} rule."


def generate_gt_think(xml_path: Path, target: int, gt_rules: dict) -> str:
    puzzle_panels, answer_panels, component_names = extract_panel_attrs(xml_path)
    if not puzzle_panels or not answer_panels:
        return ""
    if target >= len(answer_panels):
        return ""

    correct_answer = answer_panels[target]
    multi_component = len(gt_rules) > 1
    answer_letter = chr(ord("A") + target)
    lines = []

    lines.append("<think>")
    lines.append(
        "Let me analyze this visual reasoning problem. I need to find the pattern in the 3x3 grid and determine what goes in the missing cell."
    )
    lines.append("")

    lines.append("First, let me examine the objects in the grid:")
    lines.append("")
    for row_idx in range(3):
        lines.append(f"Row {row_idx + 1}:")
        for col_idx in range(3):
            idx = row_idx * 3 + col_idx
            if row_idx == 2 and col_idx == 2:
                lines.append(
                    f"- Column {col_idx + 1}: ? (this is what we need to find)"
                )
            elif idx < len(puzzle_panels):
                panel = puzzle_panels[idx]
                lines.append(f"- Column {col_idx + 1}: {describe_panel(panel)}")
            else:
                lines.append(f"- Column {col_idx + 1}: empty")
    lines.append("")

    lines.append("Now let me look for patterns in each attribute:")
    lines.append("")

    attr_observations = {}
    constant_expectations = {}
    constant_holds = {}
    attr_used = {}
    for comp_name, rules in gt_rules.items():
        comp_filter = (
            lambda e, cn=comp_name: e.get("component") == cn
            if multi_component
            else True
        )
        for attr in ["shape", "size", "color", "count", "position"]:
            row1_vals = extract_row_values(puzzle_panels, 0, attr, comp_filter)
            row2_vals = extract_row_values(puzzle_panels, 1, attr, comp_filter)
            row3_vals = extract_row_values(puzzle_panels, 2, attr, comp_filter)[:2]
            attr_observations[(comp_name, attr)] = (row1_vals, row2_vals, row3_vals)

    for comp_name, rules in gt_rules.items():
        if multi_component:
            comp_prefix = {"In": "Inner", "Out": "Outer"}.get(comp_name, comp_name) + " "
        else:
            comp_prefix = ""

        for attr in ["shape", "size", "color", "count", "position"]:
            rule_name = rules.get(attr)
            if not rule_name:
                continue
            row1_vals, row2_vals, row3_vals = attr_observations.get(
                (comp_name, attr), ([], [], [])
            )
            if rule_name == "Constant":
                row3_flat = []
                for v in row3_vals:
                    if v is None:
                        continue
                    if isinstance(v, list):
                        if attr == "position":
                            norm = tuple(sorted(tuple(p) for p in v))
                        else:
                            u = get_uniform_value(v)
                            norm = u if u is not None else tuple(sorted(v))
                        if isinstance(norm, tuple) and len(norm) == 0:
                            continue
                    else:
                        norm = v
                    row3_flat.append(norm)
                if row3_flat and len(set(row3_flat)) == 1:
                    constant_expectations[(comp_name, attr)] = row3_flat[0]
                    constant_holds[(comp_name, attr)] = True
                else:
                    constant_expectations[(comp_name, attr)] = None
                    constant_holds[(comp_name, attr)] = False
                attr_used[(comp_name, attr)] = constant_holds[(comp_name, attr)]
            else:
                attr_used[(comp_name, attr)] = True

            formatted_r1 = [format_attr_value(attr, v) for v in row1_vals]
            formatted_r2 = [format_attr_value(attr, v) for v in row2_vals]
            formatted_r3 = [format_attr_value(attr, v) for v in row3_vals]

            lines.append(f"**{comp_prefix}{attr.capitalize()}:**")

            def strip_parens(s):
                s = s.strip()
                if s.startswith("(") and s.endswith(")"):
                    return s[1:-1]
                return s

            row1_str = " | ".join(strip_parens(v) for v in formatted_r1)
            row2_str = " | ".join(strip_parens(v) for v in formatted_r2)
            row3_str = " | ".join(strip_parens(v) for v in formatted_r3)

            lines.append(f"- Row 1: {row1_str}")
            lines.append(f"- Row 2: {row2_str}")
            lines.append(f"- Row 3: {row3_str} | ?")
            lines.append("")

            natural_rule = rule_to_natural(rule_name, attr, row3_vals, row1_vals)
            lines.append(natural_rule)
            lines.append("")

    lines.append("Let me look at all the answer options:")
    lines.append("")

    for i, answer_panel in enumerate(answer_panels):
        opt_letter = chr(ord("A") + i)
        if answer_panel:
            desc = describe_panel(answer_panel)
            lines.append(f"- {opt_letter}: {desc}")
        else:
            lines.append(f"- {opt_letter}: empty")

    lines.append("")
    lines.append("Based on the patterns I identified, the missing cell should have:")
    lines.append("")

    target_attrs = {}
    for comp_name, rules in gt_rules.items():
        answer_attrs = get_answer_attrs(correct_answer, comp_name, multi_component)
        if answer_attrs:
            expected_shape = constant_expectations.get((comp_name, "shape"))
            expected_size = constant_expectations.get((comp_name, "size"))
            expected_color = constant_expectations.get((comp_name, "color"))
            expected_count = constant_expectations.get((comp_name, "count"))
            expected_position = constant_expectations.get((comp_name, "position"))
            if expected_shape is not None:
                if isinstance(expected_shape, (list, tuple)):
                    answer_attrs["shapes"] = list(expected_shape)
                else:
                    answer_attrs["shapes"] = [expected_shape]
            if expected_size is not None:
                if isinstance(expected_size, (list, tuple)):
                    answer_attrs["sizes"] = list(expected_size)
                else:
                    answer_attrs["sizes"] = [expected_size]
            if expected_color is not None:
                if isinstance(expected_color, (list, tuple)):
                    answer_attrs["colors"] = list(expected_color)
                else:
                    answer_attrs["colors"] = [expected_color]
            if expected_count is not None:
                answer_attrs["count"] = expected_count
            if expected_position is not None:
                if isinstance(expected_position, (list, tuple)):
                    answer_attrs["positions"] = [list(p) for p in expected_position]
                else:
                    answer_attrs["positions"] = [expected_position]
            target_attrs[comp_name] = answer_attrs
            if multi_component:
                comp_prefix = {"In": "Inner", "Out": "Outer"}.get(comp_name, comp_name) + " "
            else:
                comp_prefix = ""
            shapes = answer_attrs.get("shapes", [])
            sizes = answer_attrs.get("sizes", [])
            colors = answer_attrs.get("colors", [])
            count_val = answer_attrs.get("count", 0)
            positions = answer_attrs.get("positions", [])
            disp_shapes = (
                [get_uniform_value(shapes)]
                if shapes and get_uniform_value(shapes) is not None
                else shapes
            )
            disp_sizes = (
                [get_uniform_value(sizes)]
                if sizes and get_uniform_value(sizes) is not None
                else sizes
            )
            disp_colors = (
                [get_uniform_value(colors)]
                if colors and get_uniform_value(colors) is not None
                else colors
            )
            use_shape = attr_used.get((comp_name, "shape"), False)
            use_size = attr_used.get((comp_name, "size"), False)
            use_color = attr_used.get((comp_name, "color"), False)
            use_count = attr_used.get((comp_name, "count"), False)
            use_position = attr_used.get((comp_name, "position"), False)
            if (
                use_shape
                and disp_shapes
                and any(isinstance(t, int) and t > 0 for t in disp_shapes)
            ):
                shapes_str = format_attr_value("shape", disp_shapes)
                lines.append(f"- {comp_prefix}Shape(s): {shapes_str}")
            if use_size and disp_sizes:
                sizes_str = format_attr_value("size", disp_sizes)
                lines.append(f"- {comp_prefix}Size(s): {sizes_str}")
            if use_color and disp_colors:
                colors_str = format_attr_value("color", disp_colors)
                lines.append(f"- {comp_prefix}Color(s): {colors_str}")
            if use_count and count_val > 0:
                lines.append(f"- {comp_prefix}Count: {count_val} object(s)")
            if use_position and positions:
                pos_str = format_positions(positions)
                lines.append(f"- {comp_prefix}Position(s): {pos_str}")

    lines.append("")
    lines.append("Now let me check each option against these requirements:")
    lines.append("")

    for i, answer_panel in enumerate(answer_panels):
        opt_letter = chr(ord("A") + i)
        if not answer_panel:
            lines.append(f"- {opt_letter}: empty, doesn't match")
            continue

        for comp_name, rules in gt_rules.items():
            expected = target_attrs.get(comp_name, {})
            if not expected:
                continue

            if multi_component:
                actual_entities = [
                    e for e in answer_panel if e.get("component") == comp_name
                ]
            else:
                actual_entities = answer_panel

            if not actual_entities:
                lines.append(f"- {opt_letter}: missing {comp_name} component")
                continue

            wrong_attrs = []
            correct_attrs = []

            use_shape = attr_used.get((comp_name, "shape"), False)
            use_size = attr_used.get((comp_name, "size"), False)
            use_color = attr_used.get((comp_name, "color"), False)
            use_count = attr_used.get((comp_name, "count"), False)
            use_position = attr_used.get((comp_name, "position"), False)

            exp_shapes = expected.get("shapes", []) if use_shape else []
            act_shapes = [e["shape"] for e in actual_entities]
            if exp_shapes:
                if sorted(exp_shapes) != sorted(act_shapes):
                    wrong_attrs.append(
                        f"shape is {format_attr_value('shape', act_shapes)} (not {format_attr_value('shape', exp_shapes)})"
                    )
                else:
                    correct_attrs.append("shape")

            exp_sizes = expected.get("sizes", []) if use_size else []
            act_sizes = [e["size"] for e in actual_entities]
            if exp_sizes:
                if sorted(exp_sizes) != sorted(act_sizes):
                    wrong_attrs.append(
                        f"size is {format_attr_value('size', act_sizes)} (not {format_attr_value('size', exp_sizes)})"
                    )
                else:
                    correct_attrs.append("size")

            exp_colors = expected.get("colors", []) if use_color else []
            act_colors = [e["color"] for e in actual_entities]
            if exp_colors:
                if sorted(exp_colors) != sorted(act_colors):
                    wrong_attrs.append(
                        f"color is {format_attr_value('color', act_colors)} (not {format_attr_value('color', exp_colors)})"
                    )
                else:
                    correct_attrs.append("color")

            exp_num = expected.get("count") if use_count else None
            act_num = len(actual_entities)
            if exp_num is not None:
                if act_num != exp_num:
                    wrong_attrs.append(f"has {act_num} object(s) (not {exp_num})")
                else:
                    correct_attrs.append("count")

            exp_positions = expected.get("positions", []) if use_position else []
            act_positions = [
                e.get("bbox") for e in actual_entities if e.get("bbox") is not None
            ]
            if exp_positions and act_positions:
                exp_pos_set = {
                    tuple(p) if isinstance(p, list) else p for p in exp_positions
                }
                act_pos_set = {
                    tuple(p) if isinstance(p, list) else p for p in act_positions
                }
                if exp_pos_set != act_pos_set:
                    wrong_attrs.append("wrong positions")
                else:
                    correct_attrs.append("position")

            if multi_component:
                comp_prefix = {"In": "Inner", "Out": "Outer"}.get(comp_name, comp_name) + " "
            else:
                comp_prefix = ""
            if wrong_attrs:
                lines.append(f"- {opt_letter}: {comp_prefix}{random.choice(wrong_attrs)}")
            else:
                lines.append(
                    f"- {opt_letter}: {comp_prefix}{', '.join(correct_attrs)}, all correct!"
                )

    lines.append("")
    lines.append(f"Therefore, the only answer that matches all the requirements is {answer_letter}.")
    lines.append("</think>")

    return "\n".join(lines)


def create_composite_image(images: np.ndarray, *, seed: int = None) -> Image.Image:
    """Create a single image showing the 3x3 problem grid and 8 answer options."""
    rng = random.Random(seed) if seed is not None else random
    padding = round(2 ** rng.uniform(-1, 1) * BASE_PADDING)
    divider_height = round(2 ** rng.uniform(-1, 1) * BASE_DIVIDER_HEIGHT)
    font_size = round(2 ** rng.uniform(-1, 1) * BASE_FONT_SIZE)
    label_height = round(font_size * 30 / 24)

    grid_width = 3 * IMAGE_SIZE + 4 * padding
    grid_height = 3 * IMAGE_SIZE + 4 * padding
    answer_cols = rng.randint(2, 5)
    answer_rows = math.ceil(8 / answer_cols)
    answers_width = answer_cols * IMAGE_SIZE + (answer_cols + 1) * padding
    answers_row_height = IMAGE_SIZE + padding + label_height
    answers_height = answer_rows * answers_row_height + (answer_rows + 1) * padding

    total_width = max(grid_width, answers_width)
    total_height = grid_height + divider_height + answers_height

    composite = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(composite)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except OSError:
        font = ImageFont.load_default()

    outline_tone = rng.randint(50, 200)
    outline_color = (outline_tone, outline_tone, outline_tone)
    grid_x_offset = (total_width - grid_width) // 2
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            x = grid_x_offset + padding + col * (IMAGE_SIZE + padding)
            y = padding + row * (IMAGE_SIZE + padding)

            if idx < 8:
                img = Image.fromarray(images[idx]).convert("RGB")
                composite.paste(img, (x, y))
                draw.rectangle(
                    [x - 1, y - 1, x + IMAGE_SIZE, y + IMAGE_SIZE],
                    outline=outline_color,
                    width=2,
                )
            else:
                bbox = draw.textbbox((0, 0), "?", font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = x + (IMAGE_SIZE - text_width) // 2
                text_y = y + (IMAGE_SIZE - text_height) // 2
                draw.text((text_x, text_y), "?", fill=(0, 0, 0), font=font)

    divider_y = grid_height + divider_height // 2
    divider_width = round(rng.uniform(0.0, 1.0) * (total_width - 2 * padding))
    divider_x = (total_width - divider_width) // 2
    draw.line(
        [(divider_x, divider_y), (divider_x + divider_width, divider_y)],
        fill=(200, 200, 200),
        width=2,
    )

    answers_x_offset = (total_width - answers_width) // 2
    answers_y_start = grid_height + divider_height
    labels = "ABCDEFGH"

    for i in range(8):
        row = i // answer_cols
        col = i % answer_cols
        x = answers_x_offset + padding + col * (IMAGE_SIZE + padding)
        y = answers_y_start + padding + row * answers_row_height

        img = Image.fromarray(images[8 + i]).convert("RGB")
        composite.paste(img, (x, y))
        draw.rectangle(
            [x - 1, y - 1, x + IMAGE_SIZE, y + IMAGE_SIZE],
            outline=outline_color,
            width=2,
        )

        label = labels[i]
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        label_x = x + (IMAGE_SIZE - text_width) // 2
        label_y = y + IMAGE_SIZE + 5
        draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

    scale = 2 ** rng.uniform(-1, 0)
    new_size = (round(total_width * scale), round(total_height * scale))
    resample = rng.choice(
        [
            Image.Resampling.NEAREST,
            Image.Resampling.BOX,
            Image.Resampling.BILINEAR,
            Image.Resampling.BICUBIC,
        ]
    )
    composite = composite.resize(new_size, resample=resample)
    return composite


def process_npz_file(args):
    """Process a single npz file and return sample dict."""
    npz_path, output_image_dir, sample_id, split = args

    data = np.load(npz_path)
    images = data["image"]
    target = int(data["target"])

    composite = create_composite_image(images, seed=sample_id)

    rel_path = npz_path.relative_to(RAVEN_ROOT)
    image_name = rel_path.with_suffix(".png").as_posix().replace("/", "_")
    image_path = output_image_dir / image_name
    composite.save(image_path)

    xml_path = npz_path.with_suffix(".xml")
    try:
        gt_rules = parse_rules_from_xml(xml_path) if xml_path.exists() else {}
        gt_think = (
            generate_gt_think(xml_path, target, gt_rules) if xml_path.exists() else ""
        )
    except Exception as e:
        logging.exception(f"Error generating CoT for {npz_path}: {e}")
        raise

    answer_letter = chr(ord("A") + target)
    config = npz_path.parent.name

    sample = {
        "dataset": f"raven-{config}",
        "images": [str(image_path)],
        "question": QUESTION,
        "answer": answer_letter,
        "gt_rules": gt_rules,
        "gt_think": gt_think,
        "source_path": str(npz_path),
        "id": sample_id,
    }
    return sample, split


def collect_npz_files(split: str):
    """Find all npz files for a given split in the RAVEN dataset."""
    npz_files = []
    for config in CONFIGS:
        config_dir = RAVEN_ROOT / config
        if not config_dir.exists():
            print(f"Warning: config directory not found: {config_dir}")
            continue
        for npz_file in sorted(config_dir.glob(f"*_{split}.npz")):
            npz_files.append(npz_file)
    return npz_files


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_dir = OUTPUT_DIR / "images"
    image_dir.mkdir(exist_ok=True)

    args_list = []
    current_id = 1
    for split in ["train", "val", "test"]:
        npz_files = sorted(collect_npz_files(split))
        print(f"Found {len(npz_files)} {split} npz files")
        for npz_path in npz_files:
            args_list.append((npz_path, image_dir, current_id, split))
            current_id += 1

    with ProcessPoolExecutor() as executor:
        processed = list(
            tqdm(
                executor.map(process_npz_file, args_list),
                total=len(args_list),
                desc="Processing RAVEN files",
            )
        )

    samples_by_split = {"train": [], "val": [], "test": []}
    for sample, split in processed:
        samples_by_split[split].append(sample)

    random.seed(0)
    random.shuffle(samples_by_split["train"])

    for split in ["train", "val", "test"]:
        output_path = OUTPUT_DIR / f"raven_{split}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples_by_split[split]:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Wrote {len(samples_by_split[split])} samples to {output_path}")


if __name__ == "__main__":
    main()
