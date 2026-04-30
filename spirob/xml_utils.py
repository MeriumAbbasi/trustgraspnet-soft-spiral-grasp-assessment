from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from .object_library import ObjectSpec, FAMILIES


# Exact 7-family labels
FAMILY_LABELS_7 = ["sphere", "cylinder", "capsule", "box", "flat", "branched", "irregular"]
FAMILY_TO_ID = {name: i for i, name in enumerate(FAMILY_LABELS_7)}

# Coarse 5-family labels
COARSE_SHAPE_LABELS_5 = ["round", "elongated", "square", "flat", "irregular"]
COARSE_SHAPE_TO_ID_5 = {name: i for i, name in enumerate(COARSE_SHAPE_LABELS_5)}

# Coarse 3-family labels
COARSE_SHAPE_LABELS_3 = ["round", "elongated", "square"]
COARSE_SHAPE_TO_ID_3 = {name: i for i, name in enumerate(COARSE_SHAPE_LABELS_3)}

# Exact-family -> coarse-5 mapping
EXACT_TO_COARSE_SHAPE_5 = {
    "sphere": "round",
    "cylinder": "elongated",
    "capsule": "elongated",
    "box": "square",
    "flat": "flat",
    "branched": "irregular",
    "irregular": "irregular",
}

# Exact-family -> coarse-3 mapping
# Edit this if you want different grouping behavior.
EXACT_TO_COARSE_SHAPE_3 = {
    "sphere": "round",
    "cylinder": "elongated",
    "capsule": "elongated",
    "box": "square",
    "flat": "square",
    "branched": "elongated",
    "irregular": "round",
}


def object_family_to_id(family: str) -> int:
    if family not in FAMILY_TO_ID:
        raise ValueError(f"Unknown exact family label: {family}")
    return FAMILY_TO_ID[family]


def object_family_to_coarse_shape_5(family: str) -> str:
    if family not in EXACT_TO_COARSE_SHAPE_5:
        raise ValueError(f"Unknown family for coarse5 mapping: {family}")
    return EXACT_TO_COARSE_SHAPE_5[family]


def object_family_to_coarse_shape_3(family: str) -> str:
    if family not in EXACT_TO_COARSE_SHAPE_3:
        raise ValueError(f"Unknown family for coarse3 mapping: {family}")
    return EXACT_TO_COARSE_SHAPE_3[family]


def coarse_shape_to_id_5(shape_name: str) -> int:
    if shape_name not in COARSE_SHAPE_TO_ID_5:
        raise ValueError(f"Unknown coarse5 shape label: {shape_name}")
    return COARSE_SHAPE_TO_ID_5[shape_name]


def coarse_shape_to_id_3(shape_name: str) -> int:
    if shape_name not in COARSE_SHAPE_TO_ID_3:
        raise ValueError(f"Unknown coarse3 shape label: {shape_name}")
    return COARSE_SHAPE_TO_ID_3[shape_name]


def get_family_label_info(mode: str) -> dict:
    if mode == "exact7":
        return {
            "num_families": 7,
            "family_labels": FAMILY_LABELS_7,
            "label_fn": object_family_to_id,
            "group_name_fn": lambda fam: fam,
            "label_maps": {label: i for i, label in enumerate(FAMILY_LABELS_7)},
        }

    if mode == "coarse5":
        return {
            "num_families": 5,
            "family_labels": COARSE_SHAPE_LABELS_5,
            "label_fn": lambda fam: coarse_shape_to_id_5(object_family_to_coarse_shape_5(fam)),
            "group_name_fn": object_family_to_coarse_shape_5,
            "label_maps": {label: i for i, label in enumerate(COARSE_SHAPE_LABELS_5)},
        }

    if mode == "coarse3":
        return {
            "num_families": 3,
            "family_labels": COARSE_SHAPE_LABELS_3,
            "label_fn": lambda fam: coarse_shape_to_id_3(object_family_to_coarse_shape_3(fam)),
            "group_name_fn": object_family_to_coarse_shape_3,
            "label_maps": {label: i for i, label in enumerate(COARSE_SHAPE_LABELS_3)},
        }

    raise ValueError(f"Unknown family label mode: {mode}")


def _numeric_suffix(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def parse_imu_layout(xml_path: str | Path) -> list[tuple[str, float]]:
    xml_path = Path(xml_path)
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    sites = []
    for site in root.findall(".//site"):
        name = site.attrib.get("name", "")
        if name.startswith("imu_") and name.endswith("_site"):
            suffix = name[len("imu_") : -len("_site")]
            pos = [float(v) for v in site.attrib.get("pos", "0 0 0").split()]
            sites.append((suffix, pos[2] if len(pos) >= 3 else 0.0))
    sites.sort(key=lambda x: _numeric_suffix(x[0]))
    return sites


def _clear_children(elem: ET.Element) -> None:
    for child in list(elem):
        elem.remove(child)


def _set_geom_from_spec(body: ET.Element, spec: ObjectSpec) -> None:
    ET.SubElement(body, "freejoint")

    if spec.family == "sphere":
        ET.SubElement(
            body,
            "geom",
            {
                "type": "sphere",
                "size": f"{spec.extras['radius']:.6f}",
                "rgba": "0.15 0.35 0.85 1",
                "density": "500",
                "friction": "0.8 0.08 0.02",
            },
        )
    elif spec.family == "cylinder":
        ET.SubElement(
            body,
            "geom",
            {
                "type": "cylinder",
                "euler": f"1.5707963 0 {spec.yaw:.6f}",
                "size": f"{spec.extras['radius']:.6f} {spec.extras['half_length']:.6f}",
                "rgba": "0.15 0.55 0.35 1",
                "density": "500",
                "friction": "0.8 0.08 0.02",
            },
        )
    elif spec.family == "capsule":
        ET.SubElement(
            body,
            "geom",
            {
                "type": "capsule",
                "euler": f"1.5707963 0 {spec.yaw:.6f}",
                "size": f"{spec.extras['radius']:.6f} {spec.extras['half_length']:.6f}",
                "rgba": "0.55 0.25 0.75 1",
                "density": "500",
                "friction": "0.8 0.08 0.02",
            },
        )
    elif spec.family in {"box", "flat"}:
        half_sizes = spec.extras["half_sizes"]
        ET.SubElement(
            body,
            "geom",
            {
                "type": "box",
                "euler": f"0 0 {spec.yaw:.6f}",
                "size": f"{half_sizes[0]:.6f} {half_sizes[1]:.6f} {half_sizes[2]:.6f}",
                "rgba": "0.75 0.45 0.15 1" if spec.family == "box" else "0.45 0.70 0.80 1",
                "density": "500",
                "friction": "0.8 0.08 0.02",
            },
        )
    elif spec.family == "branched":
        ET.SubElement(
            body,
            "geom",
            {
                "type": "capsule",
                "euler": f"1.5707963 0 {spec.yaw:.6f}",
                "size": f"{spec.extras['trunk_radius']:.6f} {spec.extras['trunk_half_length']:.6f}",
                "rgba": "0.75 0.25 0.25 1",
                "density": "500",
                "friction": "0.8 0.08 0.02",
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "type": "capsule",
                "pos": "0 0 0.0",
                "euler": f"0 0 {spec.yaw + 0.75:.6f}",
                "size": f"{spec.extras['arm_radius']:.6f} {spec.extras['arm_half_length']:.6f}",
                "rgba": "0.75 0.25 0.25 1",
                "density": "500",
                "friction": "0.8 0.08 0.02",
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "type": "capsule",
                "pos": "0 0 0.0",
                "euler": f"0 0 {spec.yaw - 0.75:.6f}",
                "size": f"{spec.extras['arm_radius']:.6f} {spec.extras['arm_half_length']:.6f}",
                "rgba": "0.75 0.25 0.25 1",
                "density": "500",
                "friction": "0.8 0.08 0.02",
            },
        )
    elif spec.family == "irregular":
        for blob in spec.extras["blobs"]:
            ET.SubElement(
                body,
                "geom",
                {
                    "type": "sphere",
                    "pos": f"{blob['pos'][0]:.6f} {blob['pos'][1]:.6f} {blob['pos'][2]:.6f}",
                    "size": f"{blob['radius']:.6f}",
                    "rgba": "0.50 0.50 0.50 1",
                    "density": "500",
                    "friction": "0.8 0.08 0.02",
                },
            )
    else:
        ET.SubElement(
            body,
            "geom",
            {
                "type": "sphere",
                "size": f"{spec.size:.6f}",
                "rgba": "0.15 0.35 0.85 1",
                "density": "500",
                "friction": "0.8 0.08 0.02",
            },
        )

    ET.SubElement(
        body,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": f"{spec.mass:.6f}",
            "diaginertia": f"{max(spec.mass * 1e-4, 1e-6):.8f} {max(spec.mass * 1e-4, 1e-6):.8f} {max(spec.mass * 1e-4, 1e-6):.8f}",
        },
    )


def patch_xml_with_object(base_xml: str | Path, spec: ObjectSpec) -> str:
    base_xml = Path(base_xml)
    tree = ET.parse(str(base_xml))
    root = tree.getroot()
    target = root.find(".//body[@name='O1']")
    if target is None:
        raise ValueError("Body named 'O1' not found in base XML.")

    target.attrib["pos"] = f"{spec.pose_x:.6f} 0 {spec.pose_z:.6f}"
    target.attrib["euler"] = f"0 0 {spec.yaw:.6f}"
    _clear_children(target)
    _set_geom_from_spec(target, spec)

    return ET.tostring(root, encoding="unicode")


def write_patched_xml(base_xml: str | Path, spec: ObjectSpec) -> Path:
    base_xml = Path(base_xml)
    xml_text = patch_xml_with_object(base_xml, spec)
    fd, tmp_path = tempfile.mkstemp(prefix="spirob_obj_", suffix=".xml", dir=str(base_xml.parent))
    Path(tmp_path).write_text(xml_text, encoding="utf-8")
    return Path(tmp_path)