import re
import json
import networkx as nx
from pathlib import Path

def parse_kicad_pcb(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    G = nx.Graph()

    # Step 1: Extract net definitions
    net_defs = re.findall(r'\(net\s+(\d+)\s+"?([^"\)]+)"?\)', content)
    net_id_to_name = {net_id: name.strip() for net_id, name in net_defs}
    print(f"Net definitions extracted: {net_id_to_name}")  # Debug print

    # Step 2: Extract all module blocks (components)
    module_blocks = re.findall(r'\(module\s.*?\(at\s+([\d.]+)\s+([\d.]+).*?\(fp_text reference\s+(\w+).*?\)(.*?)\)\s*\)', content, re.DOTALL)

    net_to_components = {}

    for x, y, ref, module_body in module_blocks:
        G.add_node(ref, x=float(x), y=float(y), type="component")

        # Debugging module extraction
        print(f"Module {ref} at ({x}, {y}):")
        print(f"Module body: {module_body[:100]}...")  # Print first 100 chars of module body for debugging

        # Extract all pad net references inside this module (improved regex)
        pad_nets = re.findall(r'\(pad\s+[^\)]*?\(net\s+(\d+)\)', module_body)
        print(f"Pad nets for {ref}: {pad_nets}")  # Debug print for pad nets

        for net_id in pad_nets:
            net_name = net_id_to_name.get(net_id, f"net_{net_id}")
            if net_name not in net_to_components:
                net_to_components[net_name] = set()
            net_to_components[net_name].add(ref)

    # Debugging the net to components mapping
    print(f"Net to components mapping: {net_to_components}")

    # Step 3: Connect all components on the same net
    for net, comps in net_to_components.items():
        comps = list(set(comps))  # Remove duplicates
        print(f"Net {net} connects components: {comps}")  # Debug print for edges
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                # Add edge between components connected by the same net
                G.add_edge(comps[i], comps[j], net=net)

    return G

def save_graph_to_json(graph, out_path):
    data = {
        "nodes": [{"id": n, **graph.nodes[n]} for n in graph.nodes()],
        "edges": [{"source": u, "target": v, **graph.edges[u, v]} for u, v in graph.edges()]
    }
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)


# Example usage
if __name__ == "__main__":
    pcb_path = Path("C:/Users/Kalu Okechukwu/Desktop/project_root/pcb_projects/6_Axis_Robot_Controller/RoProp.kicad_pcb")
    output_path = Path("pcb_graph.json")

    graph = parse_kicad_pcb(pcb_path)
    save_graph_to_json(graph, output_path)
    print(f"✅ Graph built with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
