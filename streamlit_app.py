# streamlit_app.py
# Streamlit Cloud app to build an IC FMEDA from a draw.io block diagram + user rules + LLM reasoning
# Author: ChatGPT
# Notes:
# - Add a requirements.txt with: streamlit, pandas, openpyxl, xlsxwriter, pyyaml, openai
# - Set your OpenAI API key in Streamlit Secrets: OPENAI_API_KEY
# - If no API key is set, the app still works in "deterministic baseline" mode (no LLM).

import base64
import io
import json
import re
import sys
import zlib
import html
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Optional YAML support
try:
    import yaml  # type: ignore
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

# Optional OpenAI support
try:
    from openai import OpenAI
    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False

st.set_page_config(page_title="IC FMEDA Builder (draw.io + rules + LLM)", layout="wide")

# ----------------------------- Utilities -----------------------------

def decode_drawio_diagram(diagram_elem: ET.Element) -> Optional[ET.Element]:
    """Return an mxGraphModel element given a <diagram> element.
    Handles plain XML or compressed/encoded content.
    """
    # Case 1: diagram contains an mxGraphModel child directly
    mx_models = diagram_elem.findall(".//mxGraphModel")
    if mx_models:
        return mx_models[0]

    # Case 2: diagram.text contains compressed XML (base64 deflate)
    text = (diagram_elem.text or "").strip()
    if not text:
        return None
    try:
        raw = base64.b64decode(text)
        # draw.io uses "raw deflate" (no zlib header) often; try both
        try:
            xml_bytes = zlib.decompress(raw, -zlib.MAX_WBITS)
        except zlib.error:
            xml_bytes = zlib.decompress(raw)
        inner = ET.fromstring(xml_bytes)
        if inner.tag == "mxGraphModel":
            return inner
        # Or nested
        found = inner.find(".//mxGraphModel")
        return found
    except Exception:
        return None


def parse_drawio(file_bytes: bytes) -> Tuple[List[Dict], List[Dict]]:
    """Parse a .drawio file and extract nodes (vertices) and edges.
    Returns: (nodes, edges)
    nodes: {id, label, style, x, y, w, h}
    edges: {id, source, target, label}
    """
    try:
        root = ET.fromstring(file_bytes)
    except ET.ParseError as e:
        # Some draw.io files are saved as XML string in a wrapper; try a decode fallback
        raise RuntimeError(f"Failed to parse draw.io XML: {e}")

    # Locate diagrams
    diagrams = root.findall(".//diagram")
    if not diagrams:
        # Maybe the root is already mxGraphModel
        if root.tag == "mxGraphModel":
            model = root
        else:
            raise RuntimeError("No <diagram> found in draw.io file.")
    else:
        # Use the first diagram by default
        model = decode_drawio_diagram(diagrams[0])
        if model is None:
            raise RuntimeError("Could not decode the <diagram> content into mxGraphModel.")

    # Now parse cells
    cells = model.findall(".//mxCell")
    nodes: List[Dict] = []
    edges: List[Dict] = []
    for c in cells:
        attrs = c.attrib
        value = html.unescape(attrs.get("value", "")).strip()
        # remove simple HTML tags
        value = re.sub(r"<[^>]+>", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        style = attrs.get("style", "")
        geom = c.find("./mxGeometry")
        x = float(geom.attrib.get("x", "nan")) if geom is not None and "x" in geom.attrib else None
        y = float(geom.attrib.get("y", "nan")) if geom is not None and "y" in geom.attrib else None
        w = float(geom.attrib.get("width", "nan")) if geom is not None and "width" in geom.attrib else None
        h = float(geom.attrib.get("height", "nan")) if geom is not None and "height" in geom.attrib else None

        if attrs.get("edge") == "1":
            edges.append({
                "id": attrs.get("id", ""),
                "source": attrs.get("source", ""),
                "target": attrs.get("target", ""),
                "label": value,
            })
        elif attrs.get("vertex") == "1":
            nodes.append({
                "id": attrs.get("id", ""),
                "label": value,
                "style": style,
                "x": x, "y": y, "w": w, "h": h,
            })
    return nodes, edges

# Simple classifier
PORT_KEYWORDS = {
    "VIN","BAT","SYS","SW","TS","NTC","ILIM","PROG","I2C","SDA","SCL","PG","STAT","LDO","DCDC","USB","VREF","GND","AGND","PGND","DSS","LPO","EN","CE","CHG","OVP","UVLO","OV","UV","THERM","VBUS","VBAT","PMID"
}

KEYWORD_TYPES = {
    "buck": "buck converter",
    "boost": "boost converter",
    "dcdc": "dc-dc converter",
    "ldo": "ldo regulator",
    "charger": "battery charger",
    "charge": "battery charger",
    "pmic": "pmic",
    "mosfet": "mosfet",
    "hs-fet": "mosfet high-side",
    "ls-fet": "mosfet low-side",
    "sense": "sense / measurement",
    "ts": "temperature sense",
    "ntc": "temperature sense",
    "i2c": "i2c interface",
    "adc": "adc",
    "usb": "usb port",
    "sys": "system rail",
    "vin": "input rail",
    "bat": "battery rail",
    "sw": "switch node",
    "ovp": "protection",
    "scp": "protection",
    "ocp": "protection",
    "otp": "protection",
    "ilimit": "current limit",
    "ilim": "current limit",
    "prog": "programming pin",
    "stat": "status pin",
    "pg": "power good",
    "gnd": "ground",
}


def classify_node(label: str, style: str) -> str:
    lbl = (label or "").strip()
    u = lbl.upper()
    if u in PORT_KEYWORDS or (len(u) <= 5 and u.isalpha() and u == u.upper()):
        return "port"
    for k, v in KEYWORD_TYPES.items():
        if k.upper() in u:
            return v
    if "shape=mxgraph.electrical" in (style or ""):
        return "electrical"
    return "functional block"


@dataclass
class Diagram:
    nodes: List[Dict]
    edges: List[Dict]

    def id_to_node(self) -> Dict[str, Dict]:
        return {n["id"]: n for n in self.nodes}

    def typed_nodes(self) -> List[Dict]:
        out = []
        for n in self.nodes:
            t = classify_node(n.get("label",""), n.get("style",""))
            out.append({**n, "type": t})
        return out

    def adjacency(self) -> Dict[str, List[str]]:
        adj = {}
        for e in self.edges:
            s, t = e.get("source"), e.get("target")
            if s and t:
                adj.setdefault(s, []).append(t)
        return adj


DEFAULT_RULES_YAML = """
version: 1
matchers:
  # Each matcher assigns a type and default failure modes based on label keywords
  - name: buck
    when:
      any_label_contains: ["buck", "dcdc"]
    set_type: buck converter
    failure_modes:
      - name: SW short to GND
        local_effect: Converter cannot regulate; input stress
      - name: SW short to VIN
        local_effect: SYS/BAT overvoltage risk; catastrophic stress
      - name: HS-FET short (on)
        local_effect: Uncontrolled current; overvoltage/overcharge
      - name: LS-FET short (on)
        local_effect: Short to GND; input overcurrent
      - name: Inductor open
        local_effect: No energy transfer; SYS drops

  - name: ldo
    when:
      any_label_contains: ["ldo"]
    set_type: ldo regulator
    failure_modes:
      - name: Pass transistor short
        local_effect: Output ≈ input; overvoltage at load
      - name: Pass transistor open
        local_effect: No output; undervoltage at load

  - name: charger
    when:
      any_label_contains: ["charg", "chg"]
    set_type: battery charger
    failure_modes:
      - name: Charge FET short
        local_effect: Overcharge; battery damage
      - name: Charge FET open
        local_effect: No charge

  - name: hs_fet
    when:
      any_label_contains: ["hs-fet"]
    set_type: mosfet high-side
    failure_modes:
      - name: Short D-S
        local_effect: VIN→SYS; overvoltage

  - name: ls_fet
    when:
      any_label_contains: ["ls-fet"]
    set_type: mosfet low-side
    failure_modes:
      - name: Short D-S
        local_effect: SYS→GND; undervoltage

propagation_hints:
  # Optional patterns to help LLM follow risk paths
  - from_type: buck converter
    to_labels_regex: "SYS|BAT"
    failure_mode_names: ["SW short to VIN", "HS-FET short (on)"]
    inferred_system_effect: SYS/BAT overvoltage
  - from_type: ldo regulator
    to_labels_regex: "SYS|BAT"
    failure_mode_names: ["Pass transistor short"]
    inferred_system_effect: Downstream overvoltage
"""

# ----------------------------- Sidebar UI -----------------------------

st.sidebar.title("Inputs")
file = st.sidebar.file_uploader("Upload draw.io block diagram", type=["drawio", "xml"])  
rules_upload = st.sidebar.file_uploader("Upload rules (YAML or JSON)", type=["yml","yaml","json"], help="You can also edit rules below.")

st.sidebar.markdown("---")
goals_text = st.sidebar.text_area(
    "Safety Goals (one per line)",
    value=(
        "Prevent SYS overvoltage > 5.5 V\n"
        "Prevent BAT overcharge > 4.25 V\n"
        "Limit charge current to ≤ 2.0 A"
    ),
    height=100,
)

st.sidebar.markdown("---")
llm_enabled = st.sidebar.toggle("Use LLM reasoning", value=True)
model = st.sidebar.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-5"], index=0)
max_rows = st.sidebar.slider("Max FMEDA rows from LLM", min_value=50, max_value=1000, value=300, step=50)

st.sidebar.markdown("---")
if HAVE_OPENAI:
    st.sidebar.caption("OpenAI client available.")
else:
    st.sidebar.warning("OpenAI package not found. Install 'openai' to enable LLM mode.")

st.sidebar.info("Set API key in Secrets: OPENAI_API_KEY")

# ----------------------------- Rules Editor -----------------------------

st.title("IC FMEDA Builder — draw.io + rules + LLM")

st.subheader("1) Rules")

if "rules_text" not in st.session_state:
    st.session_state["rules_text"] = DEFAULT_RULES_YAML

if rules_upload is not None:
    try:
        st.session_state["rules_text"] = rules_upload.read().decode("utf-8")
    except Exception as e:
        st.error(f"Failed to read uploaded rules: {e}")

rules_text = st.text_area("Rules (edit here)", value=st.session_state["rules_text"], height=280)

col_r1, col_r2 = st.columns([1,1])
with col_r1:
    st.download_button("Download rules file", data=rules_text, file_name="rules.yaml")
with col_r2:
    st.session_state["rules_text"] = rules_text
    st.success("Rules are ready.")

# Parse rules

def load_rules(text: str) -> Dict:
    text = text.strip()
    if not text:
        return {}
    # JSON first
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)
    # YAML second (if available)
    if HAVE_YAML:
        return yaml.safe_load(text)
    # Fallback
    raise RuntimeError("Rules not JSON and PyYAML not installed.")

try:
    RULES = load_rules(rules_text) or {}
    st.caption("Rules parsed OK.")
except Exception as e:
    RULES = {}
    st.error(f"Rules parse error: {e}")

# ----------------------------- Diagram -----------------------------

st.subheader("2) Block diagram (.drawio)")

if file is None:
    st.info("Upload a .drawio file to proceed.")
    st.stop()

file_bytes = file.read()
try:
    nodes, edges = parse_drawio(file_bytes)
except Exception as e:
    st.error(f"Failed to parse draw.io: {e}")
    st.stop()

diagram = Diagram(nodes=nodes, edges=edges)

typed_nodes = diagram.typed_nodes()
node_df = pd.DataFrame([{k: n.get(k) for k in ["id","label","type","x","y"]} for n in typed_nodes])
edge_df = pd.DataFrame(edges)

st.write(f"**Nodes:** {len(typed_nodes)} | **Edges:** {len(edges)}")
st.dataframe(node_df, use_container_width=True, hide_index=True)
with st.expander("Show edges"):
    st.dataframe(edge_df, use_container_width=True, hide_index=True)

# ----------------------------- Deterministic Baseline FMEDA -----------------------------

st.subheader("3) Deterministic FMEDA (baseline from rules)")

# Build baseline using matchers' failure_modes

def match_type_by_rules(label: str, initial_type: str) -> str:
    if not RULES:
        return initial_type
    matchers = (RULES or {}).get("matchers", [])
    lb = (label or "").lower()
    chosen = initial_type
    for m in matchers:
        when = (m or {}).get("when", {})
        any_contains = [s.lower() for s in (when.get("any_label_contains") or [])]
        any_equals = [s.lower() for s in (when.get("any_label_equals") or [])]
        cond1 = any(any(s in lb for s in any_contains)) if any_contains else False
        cond2 = (lb in any_equals) if any_equals else False
        if cond1 or cond2:
            chosen = m.get("set_type", chosen)
    return chosen


def baseline_fmeda_rows(typed_nodes: List[Dict], rules: Dict, limit_modes: int = 5) -> List[Dict]:
    rows: List[Dict] = []
    matchers = (rules or {}).get("matchers", [])
    # Build a quick index of failure modes per assigned type from matchers
    type_to_modes: Dict[str, List[Dict]] = {}
    for m in matchers:
        set_type = m.get("set_type")
        fms = m.get("failure_modes") or []
        if set_type and fms:
            type_to_modes.setdefault(set_type, []).extend(fms)

    rid = 1
    for n in typed_nodes:
        label = n.get("label") or "(unnamed)"
        initial_type = n.get("type") or "functional block"
        use_type = match_type_by_rules(label, initial_type)
        modes = type_to_modes.get(use_type, [])[:limit_modes]
        if not modes:
            # generic fallback
            modes = [
                {"name":"Open circuit", "local_effect":"Loss of function"},
                {"name":"Short circuit", "local_effect":"Unintended current path"}
            ]
        for m in modes:
            rows.append({
                "ID": rid,
                "Block": label,
                "Type": use_type,
                "Failure Mode": m.get("name",""),
                "Local Effect": m.get("local_effect",""),
                "System Effect": "",
                "Severity (S)": "",
                "Occurrence (FIT)": "",
                "Detection": "",
                "Diagnostic Coverage (%)": "",
                "Safety Mechanism": "",
                "Comments": "",
            })
            rid += 1
    return rows

baseline_rows = baseline_fmeda_rows(typed_nodes, RULES)
baseline_df = pd.DataFrame(baseline_rows)
st.dataframe(baseline_df, use_container_width=True, hide_index=True)

col_dl1, col_dl2, col_dl3 = st.columns(3)
with col_dl1:
    st.download_button("Download baseline CSV", baseline_df.to_csv(index=False).encode("utf-8"), file_name="FMEDA_baseline.csv")
with col_dl2:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        baseline_df.to_excel(writer, index=False, sheet_name="FMEDA")
        ws = writer.sheets["FMEDA"]
        for i, col in enumerate(baseline_df.columns):
            ws.set_column(i, i, min(45, max(12, int(baseline_df[col].astype(str).str.len().quantile(0.9)) + 4)))
    st.download_button("Download baseline XLSX", buf.getvalue(), file_name="FMEDA_baseline.xlsx")
with col_dl3:
    st.download_button("Download baseline JSON", baseline_df.to_json(orient="records", force_ascii=False, indent=2), file_name="FMEDA_baseline.json")

# ----------------------------- LLM Reasoning -----------------------------

st.subheader("4) LLM reasoning: apply rules, follow propagation, check safety goals")

if not llm_enabled:
    st.info("LLM reasoning disabled. Enable it in the sidebar to generate enriched FMEDA and propagation explanations.")
else:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not HAVE_OPENAI or not api_key:
        st.warning("OpenAI not configured. Showing only baseline.")
    else:
        client = OpenAI(api_key=api_key)
        goals = [g.strip() for g in (goals_text or "").splitlines() if g.strip()]

        # Compact diagram JSON for the prompt
        compact_nodes = [{
            "id": n["id"],
            "label": n.get("label",""),
            "type": match_type_by_rules(n.get("label",""), n.get("type","functional block"))
        } for n in typed_nodes]
        compact_edges = [{"s": e.get("source"), "t": e.get("target"), "label": e.get("label","")} for e in edges]

        # Prompt construction
        system_prompt = (
            "You are a functional safety engineer expert in IC PMIC/charger FMEDA. "
            "You must reason over a given block diagram graph (nodes+edges), apply user rules, "
            "follow failure propagation, identify potential violations of provided safety goals, "
            "explain the propagation, and output an FMEDA table in strict JSON."
        )

        user_payload = {
            "diagram": {"nodes": compact_nodes, "edges": compact_edges},
            "rules": RULES,
            "safety_goals": goals,
            "limits": {"max_rows": max_rows}
        }

        output_schema = {
            "type": "object",
            "properties": {
                "fmeda": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Block": {"type":"string"},
                            "Type": {"type":"string"},
                            "Failure Mode": {"type":"string"},
                            "Local Effect": {"type":"string"},
                            "System Effect": {"type":"string"},
                            "Detection": {"type":"string"},
                            "Diagnostic Coverage (%)": {"type":"string"},
                            "Safety Mechanism": {"type":"string"},
                            "Comments": {"type":"string"}
                        },
                        "required": ["Block","Type","Failure Mode","Local Effect","System Effect"]
                    }
                },
                "propagations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "failure_mode": {"type":"string"},
                            "path": {"type":"string"},
                            "goal": {"type":"string"},
                            "why_violates_goal": {"type":"string"}
                        },
                        "required": ["failure_mode","path","goal","why_violates_goal"]
                    }
                }
            },
            "required": ["fmeda","propagations"]
        }

        prompt = (
            "Return ONLY JSON that validates against this JSON Schema. "
            "Focus FMEDA rows on blocks and failure modes that can plausibly affect the given safety goals. "
            "Prefer concrete, concise ‘System Effect’, specify which rail/signal is impacted and direction (OV/UV/OC). "
            "Use Detection/Safety Mechanism fields based on rules or standard PMIC features when obvious; otherwise leave empty. "
            "Do not exceed max_rows. "
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"JSON_SCHEMA:\n{json.dumps(output_schema)}\n\nPAYLOAD:\n{json.dumps(user_payload)}\n\nINSTRUCTIONS:\n{prompt}"
            )}
        ]

        try:
            with st.spinner("Running LLM analysis..."):
                resp = client.chat.completions.create(model=model, messages=messages)
            content = resp.choices[0].message.content or ""
            # try to extract JSON
            m = re.search(r"\{[\s\S]*\}\s*$", content)
            json_text = m.group(0) if m else content
            data = json.loads(json_text)
        except Exception as e:
            st.error(f"LLM failed: {e}")
            data = None

        if data:
            f_df = pd.DataFrame(data.get("fmeda", []))
            p_df = pd.DataFrame(data.get("propagations", []))

            if not f_df.empty:
                # Add IDs and standard columns if missing
                f_df.insert(0, "ID", range(1, len(f_df) + 1))
                for col in ["Severity (S)", "Occurrence (FIT)"]:
                    if col not in f_df.columns:
                        f_df[col] = ""
                # Reorder
                preferred = ["ID","Block","Type","Failure Mode","Local Effect","System Effect","Severity (S)","Occurrence (FIT)","Detection","Diagnostic Coverage (%)","Safety Mechanism","Comments"]
                cols = [c for c in preferred if c in f_df.columns] + [c for c in f_df.columns if c not in preferred]
                f_df = f_df[cols]
                st.markdown("### 5) FMEDA (LLM-enriched)")
                st.dataframe(f_df, use_container_width=True, hide_index=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.download_button("Download LLM CSV", f_df.to_csv(index=False).encode("utf-8"), file_name="FMEDA_LLM.csv")
                with c2:
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                        f_df.to_excel(writer, index=False, sheet_name="FMEDA")
                        ws = writer.sheets["FMEDA"]
                        for i, col in enumerate(f_df.columns):
                            ws.set_column(i, i, min(45, max(12, int(f_df[col].astype(str).str.len().quantile(0.9)) + 4)))
                    st.download_button("Download LLM XLSX", buf.getvalue(), file_name="FMEDA_LLM.xlsx")
                with c3:
                    st.download_button("Download LLM JSON", f_df.to_json(orient="records", force_ascii=False, indent=2), file_name="FMEDA_LLM.json")
            else:
                st.info("LLM returned no FMEDA rows.")

            st.markdown("### 6) Propagation explanations")
            if p_df.empty:
                st.info("No propagation explanations returned.")
            else:
                st.dataframe(p_df, use_container_width=True, hide_index=True)
        else:
            st.info("Using baseline only (LLM output unavailable).")

# ----------------------------- Footer -----------------------------

st.markdown("---")
st.caption("This tool parses a draw.io IC block diagram, applies user-defined rules, optionally uses an LLM to follow failure propagation against safety goals, and outputs an FMEDA table.")
