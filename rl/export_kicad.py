import os
import math

def format_mm(x):
    return f"{x:.3f}"

def generate_footprints(positions):
    footprint_strs = []
    for i, (x, y) in enumerate(positions):
        footprint_strs.append(f"""
  (module component_{i} (layer F.Cu) (at {format_mm(x)} {format_mm(y)})
    (fp_text reference "C{i}" (at 0 0) (layer F.SilkS))
    (fp_circle (center 0 0) (end 1 0) (layer F.Cu))
  )
        """)
    return "\n".join(footprint_strs)

def generate_tracks_from_graph(positions, graph):
    track_strs = []
    if "edges" in graph:
        for edge in graph["edges"]:
            try:
                source_node = edge["source"]
                target_node = edge["target"]

                source_index = -1
                target_index = -1

                # Try to extract index based on node ID
                if isinstance(source_node, str) and source_node.startswith("n"):
                    source_index = int(source_node[1:])
                elif isinstance(source_node, int):
                    source_index = source_node
                else:
                    print(f"Warning: Unexpected source node type: {source_node}")

                if isinstance(target_node, str) and target_node.startswith("n"):
                    target_index = int(target_node[1:])
                elif isinstance(target_node, int):
                    target_index = target_node
                else:
                    print(f"Warning: Unexpected target node type: {target_node}")

                if 0 <= source_index < len(positions) and 0 <= target_index < len(positions):
                    x1, y1 = positions[source_index]
                    x2, y2 = positions[target_index]
                    track_strs.append(f"""
  (segment (start {format_mm(x1)} {format_mm(y1)}) (end {format_mm(x2)} {format_mm(y2)}) (width 0.25) (layer F.Cu))
        """)
                else:
                    print(f"Warning: Invalid index for edge: {edge} (source: {source_index}, target: {target_index}, num_positions: {len(positions)})")

            except (ValueError, KeyError) as e:
                print(f"Error processing edge: {edge} - {e}")
    return "\n".join(track_strs)

def export_to_kicad(component_positions, graph, filepath):
    directory = os.path.dirname(filepath)
    if directory:  # Only create directories if the path has a directory component
        os.makedirs(directory, exist_ok=True)

    with open(filepath, "w") as f:
        f.write(f"""
(kicad_pcb (version 4) (host pcbnew 4.0.7)
  (general)
  (paper "A4")
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
  )

  {generate_footprints(component_positions)}
  {generate_tracks_from_graph(component_positions, graph)}

  (gr_line (start 0 0) (end 100 0) (layer Edge.Cuts) (width 0.15))
  (gr_line (start 100 0) (end 100 100) (layer Edge.Cuts) (width 0.15))
  (gr_line (start 100 100) (end 0 100) (layer Edge.Cuts) (width 0.15))
  (gr_line (start 0 100) (end 0 0) (layer Edge.Cuts) (width 0.15))
)
        """)
    print(f"✅ Exported layout to {filepath}")