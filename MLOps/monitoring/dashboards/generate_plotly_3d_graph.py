import os
import ast
import networkx as nx
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
import json # For potential structured data in tooltips or customdata

# --- Configuration ---
ROOT_DIRS_TO_SCAN = ["FinRobot/finrl/", "FinNLP/fingpt/", "MLOps/"]
OUTPUT_HTML_FILE = "analysis/interactive_plotly_graph.html"
ERROR_LOG_FILE = "analysis/error.md"
ERROR_RESOLUTION_PLAN_DIR = "analysis/errors/"
BACKLOG_FILE = "log/backlog.md"

# Specific tooltip for data_processor.py as requested
DATA_PROCESSOR_PY_PATH = os.path.normpath("FinRobot/finrl/meta/data_processor.py")
DATA_PROCESSOR_TOOLTIP = """This file defines a high-level DataProcessor class for the FinRL library, aimed at fetching, preprocessing, and feature engineering financial data from various sources like Yahoo Finance and WRDS.

Key functionalities include:
- Downloading stock data (OHLCV, adjusted close).
- Cleaning data (handling NaNs, infinite values).
- Feature engineering:
    - Calculating technical indicators (MACD, RSI, CCI, ADX).
    - Adding turbulence index.
- Handling different date ranges and time intervals.
- Supporting multiple stock tickers.

The class is designed to be flexible and provide a standardized way to prepare data for reinforcement learning trading agents within the FinRL framework. It emphasizes ease of use for users wanting to quickly get started with financial data for RL.
"""

# --- Helper Functions ---

def get_formatted_timestamp():
    """Generates a formatted timestamp string."""
    # Adapted from helpers/get_time_id.py
    # Using a more precise timestamp format as per the example in coding-strategy.md
    # Ensure timezone is consistent if script runs in different environments
    try:
        # Attempt to use a specific timezone if available and relevant, e.g., UTC for consistency
        # For local execution, system's local time might be fine, but UTC is often preferred for logs
        # The original get_time_id.py used utc_plus_2. For general logging, UTC is safer.
        # Let's stick to UTC for broader applicability unless specific local time is needed.
        # If utc_plus_2 is critical, it can be defined as:
        # utc_plus_2 = timezone(timedelta(hours=2))
        # timestamp_obj = datetime.now(utc_plus_2)
        timestamp_obj = datetime.now(timezone.utc)
        # Format for filenames (no colons, no spaces)
        filename_timestamp = timestamp_obj.strftime('%Y-%m-%d_%H%M%S')
        # Format for logging (more readable)
        log_timestamp = timestamp_obj.strftime('%Y-%m-%d %H:%M:%S.%f %Z') # Added timezone info
        return filename_timestamp, log_timestamp
    except Exception as e:
        # Fallback to simpler timestamp if timezone handling fails for some reason
        now = datetime.now()
        return now.strftime('%Y-%m-%d_%H%M%S'), now.strftime('%Y-%m-%d %H:%M:%S.%f')

def log_error(message, exception_obj=None):
    """Logs an error to the console and to the error log file."""
    _, log_ts = get_formatted_timestamp()
    full_message = f"## Error at {log_ts}\n\n{message}\n"
    if exception_obj:
        full_message += f"\n**Exception:**\n```\n{type(exception_obj).__name__}: {str(exception_obj)}\n```\n"
    
    print(f"ERROR: {message}" + (f" Details: {str(exception_obj)}" if exception_obj else ""))
    
    # Ensure analysis directory exists
    os.makedirs(os.path.dirname(ERROR_LOG_FILE), exist_ok=True)
    
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(full_message + "\n---\n")

    # Create resolution plan file
    filename_ts, _ = get_formatted_timestamp()
    resolution_filename = f"{filename_ts}_error_resolution_plotly_graph.md"
    resolution_filepath = os.path.join(ERROR_RESOLUTION_PLAN_DIR, resolution_filename)
    
    os.makedirs(ERROR_RESOLUTION_PLAN_DIR, exist_ok=True)
    with open(resolution_filepath, "w", encoding="utf-8") as f:
        f.write(f"# Error Resolution Plan: Plotly Graph Generation ({log_ts})\n\n")
        f.write(f"**Error Logged:**\n{message}\n\n")
        if exception_obj:
            f.write(f"**Exception Details:**\n```\n{type(exception_obj).__name__}: {str(exception_obj)}\n```\n\n")
        f.write("## Affected Component(s):\n- Plotly 3D graph generation script.\n\n")
        f.write("## Initial Analysis:\n- [Provide initial thoughts on the cause of the error]\n\n")
        f.write("## Proposed Resolution Steps:\n1. [Step 1 to investigate/fix]\n2. [Step 2 to investigate/fix]\n...\n\n")
        f.write("## Verification:\n- [How to verify the fix]\n\n")
        f.write(f"*(This plan follows guidance from `analysis/debugging.md`)*\n")
    print(f"Created error resolution plan: {resolution_filepath}")


def append_to_backlog(summary_message):
    """Appends a message to the backlog file with a timestamp."""
    _, log_ts = get_formatted_timestamp()
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(BACKLOG_FILE), exist_ok=True)
    
    with open(BACKLOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n---\n**Entry: {log_ts}**\n\n{summary_message}\n")
    print(f"Appended to backlog: {BACKLOG_FILE}")


# --- Core Logic Placeholder ---
def discover_python_files(root_dirs):
    """Recursively finds all Python files (.py) in the given root directories."""
    py_files = []
    print(f"Starting file discovery in: {root_dirs}")
    for root_dir in root_dirs:
        abs_root_dir = os.path.abspath(root_dir)
        if not os.path.isdir(abs_root_dir):
            print(f"Warning: Root directory '{root_dir}' (resolved to '{abs_root_dir}') does not exist or is not a directory. Skipping.")
            continue
        
        print(f"Scanning directory: {abs_root_dir}")
        for dirpath, _, filenames in os.walk(abs_root_dir):
            for filename in filenames:
                if filename.endswith(".py"):
                    full_path = os.path.join(dirpath, filename)
                    # Store relative paths to the initial CWD for consistency if needed,
                    # or absolute paths. For this project, relative to workspace is good.
                    # os.path.relpath(full_path, os.getcwd()) might be useful if paths need to be relative to script execution
                    # However, storing absolute paths or paths relative to a known base (like workspace) is often more robust.
                    # For now, let's use normalized absolute paths.
                    py_files.append(os.path.normpath(full_path))
    
    print(f"Discovered {len(py_files)} Python files.")
    return py_files

def analyze_python_file(file_path):
    """
    Analyzes a single Python file to extract module docstring, classes, functions, and imports.
    Returns a dictionary with the extracted information, or None if parsing fails.
    """
    print(f"Analyzing file: {file_path}")
    analysis_data = {
        "path": file_path,
        "module_docstring": None, # Initialize as None
        "classes": [],
        "functions": [],
        "imports": []
    }
    try:
        with open(file_path, "r", encoding="utf-8", errors='ignore') as source_file:
            source_code = source_file.read()
        
        tree = ast.parse(source_code, filename=file_path)
        
        # Extract module docstring
        analysis_data["module_docstring"] = ast.get_docstring(tree)

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    analysis_data["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module if node.module else "" # Handle "from . import X"
                for alias in node.names:
                    analysis_data["imports"].append(f"{module_name}.{alias.name}")
            elif isinstance(node, ast.FunctionDef):
                # Top-level functions
                func_docstring = ast.get_docstring(node)
                analysis_data["functions"].append({
                    "name": node.name,
                    "docstring": func_docstring
                })
            elif isinstance(node, ast.ClassDef):
                class_docstring = ast.get_docstring(node)
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef): # Methods
                        methods.append(item.name)
                
                parent_classes = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        parent_classes.append(base.id)
                    elif isinstance(base, ast.Attribute): # e.g., parent.ClassName
                        # Attempt to reconstruct the full parent name, might be complex
                        parent_name_parts = []
                        curr = base
                        while isinstance(curr, ast.Attribute):
                            parent_name_parts.insert(0, curr.attr)
                            curr = curr.value
                        if isinstance(curr, ast.Name):
                            parent_name_parts.insert(0, curr.id)
                        parent_classes.append(".".join(parent_name_parts))
                    # Add more complex base class name extractions if needed (e.g., Subscript)

                analysis_data["classes"].append({
                    "name": node.name,
                    "docstring": class_docstring,
                    "methods": methods,
                    "parents": parent_classes
                })
        return analysis_data
    except SyntaxError as e:
        log_error(f"Syntax error parsing {file_path}. Skipping file.", e)
        return None # Indicate failure to parse
    except FileNotFoundError:
        log_error(f"File not found during analysis: {file_path}. Skipping file.", None)
        return None
    except Exception as e:
        log_error(f"Unexpected error analyzing {file_path}. Skipping file.", e)
        return None

def build_graph(analyzed_files_data):
    """Builds a NetworkX graph from the analyzed file data."""
    G = nx.Graph()
    print(f"Building NetworkX graph from {len(analyzed_files_data)} analyzed files.")

    # Keep track of added directory nodes to avoid duplicates and connect files to them
    added_dirs = set()

    # Normalize DATA_PROCESSOR_PY_PATH once for comparison
    # The paths in analyzed_data are already normalized absolute paths from discover_python_files
    # and analyze_python_file.
    normalized_data_processor_path = DATA_PROCESSOR_PY_PATH # Already normalized in config

    for file_data in analyzed_files_data:
        file_path = file_data["path"] # This is an absolute, normalized path
        file_node_id = f"file:{file_path}"
        
        # Add directory nodes and edges
        current_dir = os.path.dirname(file_path)
        parent_dir_node_id = None
        # Create nodes for each part of the directory structure
        # e.g., for /a/b/c.py, create nodes for /a, /a/b and edges /a -> /a/b, /a/b -> file:/a/b/c.py
        # This needs to be relative to the workspace or a common root for clarity.
        # For simplicity, let's use the full path for directory nodes for now.
        
        # Add nodes for parent directories and connect them
        # We want to show the hierarchy from the ROOT_DIRS_TO_SCAN
        # Let's find which root_dir this file_path belongs to
        common_ancestor_dir = None
        for r_dir in ROOT_DIRS_TO_SCAN:
            abs_r_dir = os.path.abspath(r_dir)
            if file_path.startswith(abs_r_dir):
                common_ancestor_dir = abs_r_dir
                break
        
        # Create directory nodes from the common ancestor up to the file's directory
        if common_ancestor_dir:
            relative_file_dir = os.path.dirname(os.path.relpath(file_path, common_ancestor_dir))
            # If file is directly in common_ancestor_dir, relative_file_dir will be empty
            # The path segments will be from the common_ancestor_dir
            path_parts = [common_ancestor_dir] + ([p for p in relative_file_dir.split(os.sep) if p] if relative_file_dir else [])
        else: # Should not happen if discover_python_files works correctly
            path_parts = os.path.dirname(file_path).split(os.sep)
            if not os.path.isabs(file_path): # if paths are relative, make them absolute for splitting
                 path_parts = os.path.abspath(file_path).split(os.sep)


        last_dir_node_id = None
        current_path_cumulative = ""
        for i, part in enumerate(path_parts):
            if i == 0 and os.path.isabs(part): # For absolute paths like C:\ or /
                 current_path_cumulative = part
            else:
                 current_path_cumulative = os.path.join(current_path_cumulative, part)
            
            dir_node_id = f"dir:{os.path.normpath(current_path_cumulative)}"
            if dir_node_id not in G:
                G.add_node(dir_node_id, type="directory", label=os.path.basename(current_path_cumulative) or current_path_cumulative, tooltip=f"Directory: {os.path.normpath(current_path_cumulative)}")
            
            if last_dir_node_id and last_dir_node_id != dir_node_id : # Avoid self-loops if part is empty or root
                G.add_edge(last_dir_node_id, dir_node_id, type="contains_dir")
            last_dir_node_id = dir_node_id

        # Connect the final directory to the file
        if last_dir_node_id:
            G.add_edge(last_dir_node_id, file_node_id, type="contains_file")


        # File node
        module_doc = file_data["module_docstring"] if file_data["module_docstring"] else "No module docstring."
        class_names = [c["name"] for c in file_data["classes"]]
        func_names = [f["name"] for f in file_data["functions"]]
        
        file_tooltip = ""
        # Compare normalized absolute paths
        if file_path == normalized_data_processor_path:
            file_tooltip = DATA_PROCESSOR_TOOLTIP
        else:
            file_tooltip = f"File: {file_path}\n{module_doc}\n\nContains Classes:\n{', '.join(class_names) or 'None'}\n\nContains Functions:\n{', '.join(func_names) or 'None'}"
        
        G.add_node(file_node_id, type="file", label=os.path.basename(file_path), tooltip=file_tooltip, full_path=file_path)

        # Class nodes and edges
        for cls_data in file_data["classes"]:
            class_name = cls_data["name"]
            class_node_id = f"class:{file_path}:{class_name}"
            class_doc = cls_data["docstring"] if cls_data["docstring"] else "No class docstring."
            method_names = ", ".join(cls_data["methods"]) or "None"
            parent_names = ", ".join(cls_data["parents"]) or "None"
            class_tooltip = f"Class: {class_name}\nParents: {parent_names}\n{class_doc}\n\nMethods:\n{method_names}"
            
            G.add_node(class_node_id, type="class", label=class_name, tooltip=class_tooltip)
            G.add_edge(file_node_id, class_node_id, type="defines_class")

            # Inheritance edges
            for parent_name in cls_data["parents"]:
                # This is a simplification: assumes parent class is defined in one of the scanned files
                # A more robust solution would require resolving parent_name to its actual node_id
                # For now, we create a potential target node_id. If it exists, an edge is made.
                # This requires iterating all files first to know all class_node_ids, or a two-pass approach.
                # Let's try to find the parent class node.
                # This is tricky because parent_name might be "module.ParentClass" or just "ParentClass"
                # We need a way to map parent_name to a unique class_node_id.
                # For now, let's assume parent_name is the simple name and it's unique or we connect to any match.
                # This part needs refinement for accurate cross-file inheritance.
                # A simple approach: search for class nodes with label == parent_name
                
                # Simplistic approach: if a class with parent_name exists anywhere, link to it.
                # This could be ambiguous if multiple classes have the same name.
                # A better way: resolve imports to find the file of the parent class.
                # For now, let's defer complex inheritance linking or make it very basic.
                # Let's assume parent_name is a simple name for now.
                # We'll add these edges in a second pass after all class nodes are created.
                pass # Defer inheritance edges to a second pass

        # Function nodes and edges
        for func_data in file_data["functions"]:
            func_name = func_data["name"]
            func_node_id = f"func:{file_path}:{func_name}"
            func_doc = func_data["docstring"] if func_data["docstring"] else "No function docstring."
            func_tooltip = f"Function: {func_name}\n{func_doc}"
            
            G.add_node(func_node_id, type="function", label=func_name, tooltip=func_tooltip)
            G.add_edge(file_node_id, func_node_id, type="defines_function")

    # Second pass for inheritance and import edges
    # Create a map of class name to list of (file_path, class_node_id) for resolving inheritance
    class_locations = {}
    for file_data in analyzed_files_data:
        file_path = file_data["path"]
        for cls_data in file_data["classes"]:
            class_name = cls_data["name"]
            class_node_id = f"class:{file_path}:{class_name}"
            if class_name not in class_locations:
                class_locations[class_name] = []
            class_locations[class_name].append({"file_path": file_path, "node_id": class_node_id})

    for file_data in analyzed_files_data:
        file_path = file_data["path"]
        # Inheritance
        for cls_data in file_data["classes"]:
            child_class_node_id = f"class:{file_path}:{cls_data['name']}"
            for parent_name_full in cls_data["parents"]:
                # parent_name_full could be "module.Parent" or just "Parent"
                # Try to resolve parent_name_full
                # 1. Check if parent_name_full is like "module.Class" and module is an import
                # 2. Check if parent_name_full (if simple name) is a class in the same file
                # 3. Check if parent_name_full (if simple name) is a class in an imported file
                # 4. Check if parent_name_full (if simple name) is any known class (less precise)

                parent_simple_name = parent_name_full.split('.')[-1] # Get 'Parent' from 'module.Parent'

                # Option A: Link to any class with that simple name (less precise)
                if parent_simple_name in class_locations:
                    for parent_candidate in class_locations[parent_simple_name]:
                        # This links to ALL classes with that name. Might be too broad.
                        # A better approach would be to check imports to narrow down.
                        # For now, let's make one edge to the first found candidate for simplicity.
                        # Or, if we want to be more precise, we'd need to resolve imports.
                        # Let's try to find parent in same file first.
                        parent_in_same_file_id = f"class:{file_path}:{parent_simple_name}"
                        if G.has_node(parent_in_same_file_id):
                             G.add_edge(child_class_node_id, parent_in_same_file_id, type="inherits_from")
                             break # Found in same file
                        else:
                            # If not in same file, link to the first one found (could be from another file)
                            # This is a simplification.
                            if class_locations[parent_simple_name]:
                                G.add_edge(child_class_node_id, class_locations[parent_simple_name][0]["node_id"], type="inherits_from")
                                break # Linked to one candidate
        
        # Basic Import-based dependencies (file to file)
        # This is a very simplified version. Real import resolution is complex.
        # It assumes an import "module_x" refers to "module_x.py" or a directory "module_x".
        current_file_node_id = f"file:{file_path}"
        for imp_statement in file_data["imports"]:
            # imp_statement can be "module", "package.module", ".local_module"
            # Try to find a file node that matches this import
            # This is highly heuristic.
            # e.g. import FinRL.finrl.meta.preprocessor.yahoodownloader
            # We need to map this to a file path.
            
            # Simplification: if 'module_name' is imported, and 'module_name.py' exists as a file node, connect.
            # Or if 'package.module' is imported, and 'package/module.py' exists.
            
            # Let's try to match the end of the import string with file paths.
            # Example: import A.B.C -> look for files like .../A/B/C.py
            # Example: from A.B import C -> look for .../A/B.py (if C is in B) or .../A/B/C.py
            
            # This is very basic: if an import name (or part of it) matches a file name base.
            potential_target_module_parts = imp_statement.split('.')
            
            # Try to find a file node corresponding to this import
            for target_file_data in analyzed_files_data:
                target_file_path = target_file_data["path"]
                target_file_basename_no_ext = os.path.splitext(os.path.basename(target_file_path))[0]
                
                # Scenario 1: import module -> module.py
                if len(potential_target_module_parts) == 1 and potential_target_module_parts[0] == target_file_basename_no_ext:
                    target_file_node_id = f"file:{target_file_path}"
                    if current_file_node_id != target_file_node_id:
                         G.add_edge(current_file_node_id, target_file_node_id, type="imports")
                    break # Found a potential match

                # Scenario 2: import package.module -> package/module.py
                # Check if target_file_path ends with package/module.py
                # e.g., imp_statement = "FinRL.finrl.meta.preprocessor"
                # target_file_path = ".../FinRobot/finrl/meta/preprocessor.py"
                # This requires careful path manipulation and matching.
                
                # A simpler check: if the import string (normalized) is part of the target file path (normalized)
                normalized_import_path_like = os.path.join(*potential_target_module_parts) # e.g., "A/B/C"
                
                # Check if ".../A/B/C.py" or ".../A/B/C/__init__.py" exists
                if target_file_path.endswith(normalized_import_path_like + ".py") or \
                   target_file_path.endswith(os.path.join(normalized_import_path_like, "__init__.py")):
                    target_file_node_id = f"file:{target_file_path}"
                    if current_file_node_id != target_file_node_id:
                        G.add_edge(current_file_node_id, target_file_node_id, type="imports")
                    # Don't break here, an import could resolve to multiple files if ambiguous
                    # but for simplicity, one edge is fine for now.

    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def generate_plotly_figure(G, layout):
    """Generates a Plotly Figure object for the 3D graph with search/select functionality."""
    if not G.nodes:
        print("Graph has no nodes. Returning empty figure.")
        return go.Figure()

    print(f"Generating Plotly figure for graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    node_indices = {node_id: i for i, node_id in enumerate(G.nodes())}

    # Node positions
    node_x = [layout[node][0] for node in G.nodes()]
    node_y = [layout[node][1] for node in G.nodes()]
    node_z = [layout[node][2] for node in G.nodes()]

    # Node hover text and other properties
    node_text = []
    initial_node_colors = []
    initial_node_sizes = []
    node_symbols = []
    
    file_nodes_for_dropdown = [] # Store (label, node_id, index)
    function_nodes_for_dropdown = [] # Store (label, node_id, index)


    # Define colors, sizes, symbols for different node types
    type_map = {
        "directory": {"color": "rgba(100, 100, 200, 0.8)", "size": 15, "symbol": "circle"},
        "file":      {"color": "rgba(100, 200, 100, 0.8)", "size": 12, "symbol": "square"},
        "class":     {"color": "rgba(200, 100, 100, 0.8)", "size": 10, "symbol": "diamond"},
        "function":  {"color": "rgba(200, 200, 100, 0.8)", "size": 8,  "symbol": "cross"}
    }
    default_type_attrs = {"color": "rgba(150, 150, 150, 0.7)", "size": 7, "symbol": "circle-open"}
    highlight_color = "rgba(255, 255, 0, 1)" # Bright yellow for highlight
    highlight_size_increase = 5


    for i, (node_id, attrs) in enumerate(G.nodes(data=True)):
        label = attrs.get("label", str(node_id).split(':')[-1])
        node_text.append(attrs.get("tooltip", label))
        node_type = attrs.get("type", "unknown")
        
        type_attrs = type_map.get(node_type, default_type_attrs)
        initial_node_colors.append(type_attrs["color"])
        initial_node_sizes.append(type_attrs["size"])
        node_symbols.append(type_attrs["symbol"])

        if node_type == "file":
            file_nodes_for_dropdown.append({"label": label, "id": node_id, "idx": i, "pos": layout[node_id]})
        elif node_type == "function":
            # For functions, label might be just func_name. Prepend file for uniqueness if needed.
            # full_path_prefix = node_id.split(':')[1] # e.g. path from func:path:name
            # unique_label = f"{os.path.basename(full_path_prefix)}::{label}"
            function_nodes_for_dropdown.append({"label": label, "id": node_id, "idx": i, "pos": layout[node_id]})

    # Sort dropdown items alphabetically by label
    file_nodes_for_dropdown.sort(key=lambda x: x["label"])
    function_nodes_for_dropdown.sort(key=lambda x: x["label"])


    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        text=[G.nodes[node].get('label', str(node).split(':')[-1]) for node in G.nodes()],
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=initial_node_colors, # Use initial colors
            size=initial_node_sizes,   # Use initial sizes
            symbol=node_symbols,
            opacity=0.8,
            line=dict(width=0.5, color='DarkSlateGrey')
        ),
        textposition='top center',
        textfont=dict(size=8, color='white')
    )

    # Edge positions
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        x0, y0, z0 = layout[edge[0]]
        x1, y1, z1 = layout[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(width=0.7, color='#aaa'),
        hoverinfo='none'
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])

    # --- Create Updatemenus for Search/Select ---
    updatemenus = []
    
    # Helper to create camera arguments for focusing on a node
    def get_camera_args(node_pos, eye_distance_factor=1.5):
        # Center camera on the node, eye position slightly away
        # Adjust eye_distance_factor for how far the "zoom out" is
        # A smaller factor means closer zoom.
        # We need to ensure the eye is not *at* the node_pos.
        # A simple offset along a vector or fixed offset can work.
        # Let's try to position the eye such that the node is in the center of view.
        # The 'center' of the camera view in Plotly scene is (0,0,0) by default if not specified.
        # We want the selected node to be the center.
        # And the eye to be at a reasonable distance looking at it.
        # For simplicity, let's set the eye relative to the node's position,
        # looking towards the origin (0,0,0) if the node is far, or slightly offset if near origin.
        
        # Target the node itself as the center of the view
        cam_center_x, cam_center_y, cam_center_z = node_pos
        
        # Position the eye slightly offset from the node, looking at it.
        # A simple strategy: offset along each axis.
        # The distance of the eye from the center affects zoom.
        # Let's make the eye position relative to the node's position, but further out.
        # If node_pos is (nx, ny, nz), eye could be (nx+d, ny+d, nz+d)
        # The 'up' vector also matters for orientation. Default is z-up.
        
        # A more robust way to set eye for "zoom":
        # Eye position should be node_pos + some_vector.
        # The camera 'center' should be node_pos.
        # Let's try setting camera 'center' to node_pos and 'eye' to a point slightly further out.
        # The default 'up' vector is {x:0, y:0, z:1}.
        
        # Calculate a view vector (e.g., from a standard viewpoint towards the node)
        # and then place the eye along that vector.
        # Or, more simply, set the scene.camera.center to the node's coordinates.
        # And set scene.camera.eye to a position that gives a good view of that center.
        # Let's try: eye is node_pos + a small offset in x,y,z for a slight perspective.
        # The default eye is (1.25, 1.25, 1.25) looking at (0,0,0).
        # If we set center to (node_x, node_y, node_z),
        # eye could be (node_x + offset, node_y + offset, node_z + offset)
        # The 'eye_distance_factor' here is not directly used in this simplified approach.
        # We'll set the camera center to the node, and eye to a fixed offset from it.
        
        offset = 0.5 # Adjust this for zoom level
        return dict(
            center=dict(x=node_pos[0], y=node_pos[1], z=node_pos[2]),
            eye=dict(x=node_pos[0] + offset, y=node_pos[1] + offset, z=node_pos[2] + offset),
            up=dict(x=0, y=0, z=1) # Standard Z-up
        )

    # Dropdown for Files
    file_buttons = [dict(label="Select File...", method="skip")] # Placeholder
    for f_node in file_nodes_for_dropdown:
        # Create new marker colors/sizes for highlighting
        new_marker_colors = list(initial_node_colors) # Make a copy
        new_marker_sizes = list(initial_node_sizes)
        new_marker_colors[f_node["idx"]] = highlight_color
        new_marker_sizes[f_node["idx"]] = initial_node_sizes[f_node["idx"]] + highlight_size_increase
        
        file_buttons.append(dict(
            label=f_node["label"],
            method="update",
            args=[{"marker.color": [new_marker_colors], "marker.size": [new_marker_sizes]}, # Update trace
                  {"scene.camera": get_camera_args(f_node["pos"])}]  # Update layout (camera)
        ))
    
    # Dropdown for Functions
    func_buttons = [dict(label="Select Function...", method="skip")] # Placeholder
    for func_node in function_nodes_for_dropdown:
        new_marker_colors = list(initial_node_colors)
        new_marker_sizes = list(initial_node_sizes)
        new_marker_colors[func_node["idx"]] = highlight_color
        new_marker_sizes[func_node["idx"]] = initial_node_sizes[func_node["idx"]] + highlight_size_increase
        
        func_buttons.append(dict(
            label=func_node["label"],
            method="update",
            args=[{"marker.color": [new_marker_colors], "marker.size": [new_marker_sizes]},
                  {"scene.camera": get_camera_args(func_node["pos"])}]
        ))

    # Button to reset view and highlights
    reset_button_args = [
        {"marker.color": [initial_node_colors], "marker.size": [initial_node_sizes]}, # Reset trace
        {"scene.camera": dict(eye=dict(x=1.2, y=1.2, z=1.2), center=dict(x=0,y=0,z=0), up=dict(x=0,y=0,z=1))} # Reset camera
    ]


    if file_nodes_for_dropdown: # Only add if there are files
        updatemenus.append(dict(
            buttons=file_buttons,
            direction="down",
            showactive=True,
            x=0.12, # Position to the left
            xanchor="left",
            y=0.95,
            yanchor="top",
            font=dict(color="white"),
            bgcolor="rgba(50,50,50,0.8)",
            bordercolor="white"
        ))
    
    if function_nodes_for_dropdown: # Only add if there are functions
        updatemenus.append(dict(
            buttons=func_buttons,
            direction="down",
            showactive=True,
            x=0.37, # Position next to file dropdown
            xanchor="left",
            y=0.95,
            yanchor="top",
            font=dict(color="white"),
            bgcolor="rgba(50,50,50,0.8)",
            bordercolor="white"
        ))

    # Add a general reset button
    updatemenus.append(dict(
        type="buttons",
        direction="right",
        buttons=[dict(label="Reset View", method="update", args=reset_button_args)],
        showactive=False,
        x=0.62, # Position further right
        xanchor="left",
        y=0.95,
        yanchor="top",
        font=dict(color="white"),
        bgcolor="rgba(80,80,80,0.8)",
        bordercolor="white"
    ))


    fig.update_layout(
        updatemenus=updatemenus,
        title=dict(
            text='Interactive 3D Code Structure Graph',
            x=0.5,
            font=dict(color='white')
        ),
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title='', color='white'),
            yaxis=dict(showbackground=False, showticklabels=False, title='', color='white'),
            zaxis=dict(showbackground=False, showticklabels=False, title='', color='white'),
            bgcolor='rgb(30, 30, 30)',
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2), center=dict(x=0,y=0,z=0), up=dict(x=0,y=0,z=1)) # Ensure center is also defined for reset
        ),
        margin=dict(l=10, r=10, b=10, t=40, pad=4), # Added pad
        paper_bgcolor='rgb(30,30,30)',
        font=dict(color='white'),
        annotations=[
            dict(
                showarrow=False,
                text="Use dropdowns to find & highlight files/functions. Reset view if needed.",
                xref="paper",
                yref="paper",
                x=0.005,
                y=0.005,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=10, color="lightgrey")
            )
        ]
    )
    
    print("Plotly figure generated with search/select dropdowns.")
    return fig

def main():
    """Main function to generate the interactive 3D Plotly graph."""
    print("Starting 3D interactive graph generation...")
    
    # 1. File Discovery
    python_files = discover_python_files(ROOT_DIRS_TO_SCAN)
    if not python_files:
        log_error("No Python files found in the specified directories. Cannot proceed.", None)
        append_to_backlog("Graph generation aborted: No Python files found in specified directories.")
        print("Aborting: No Python files found.")
        return # Exit if no files
    
    # 2. Content Analysis
    analyzed_data = []
    skipped_files_count = 0
    for py_file in python_files:
        data = analyze_python_file(py_file)
        if data:
            analyzed_data.append(data)
        else:
            # Error/warning already logged by analyze_python_file
            skipped_files_count += 1
            
    if not analyzed_data and python_files: # All files failed to parse
        log_error("All Python files failed to be analyzed. Cannot proceed with graph generation.", None)
        append_to_backlog(f"Graph generation aborted: All {len(python_files)} Python files failed analysis.")
        print("Aborting: All Python files failed analysis.")
        return

    # 3. Graph Construction
    graph = build_graph(analyzed_data)
    
    # 4. 3D Layout Calculation
    pos_3d = {}
    if graph.number_of_nodes() > 0:
        print(f"Calculating 3D layout for {graph.number_of_nodes()} nodes... (this may take a while for large graphs)")
        try:
            # Iterations: 50-100. Start with fewer for speed, increase for better spread.
            # k: optimal distance between nodes. Can be tuned.
            # seed: for reproducibility
            # Adjust k based on graph size to prevent too dense or too sparse layouts.
            k_val = 0.5 / (graph.number_of_nodes()**0.5) if graph.number_of_nodes() > 1 else 0.5
            # Ensure k_val is not too small or zero for very large graphs, which might cause issues.
            k_val = max(k_val, 0.01) # Set a minimum k_val
            
            pos_3d = nx.spring_layout(graph, dim=3, iterations=60, k=k_val, seed=42)
            print("3D layout calculation complete.")
        except Exception as e:
            log_error("Failed to compute NetworkX spring_layout", e)
            append_to_backlog(f"Graph generation warning: Failed to compute 3D layout. Nodes might be clumped. Error: {type(e).__name__}")
            # Fallback: random layout if spring_layout fails, though less ideal
            # Create a simple, somewhat spread out random layout as a fallback
            print("Falling back to a simple random layout.")
            for i, node in enumerate(graph.nodes()):
                pos_3d[node] = (random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1))
    else:
        print("Warning: Graph is empty, no layout to compute.")

    # 5. Interactive 3D Graph Generation using Plotly
    plotly_fig = go.Figure() # Initialize to empty figure
    if graph.number_of_nodes() > 0 and pos_3d:
        plotly_fig = generate_plotly_figure(graph, pos_3d)
    elif graph.number_of_nodes() > 0 and not pos_3d:
        print("Warning: Layout computation failed or resulted in empty layout, but graph has nodes. Plotly figure might be malformed or empty.")
    else: # graph is empty
        print("Warning: Graph is empty, generating an empty Plotly figure.")

    # 6. Output
    completion_message = ""
    try:
        os.makedirs(os.path.dirname(OUTPUT_HTML_FILE), exist_ok=True)
        plotly_fig.write_html(OUTPUT_HTML_FILE)
        print(f"Successfully saved interactive graph to: {OUTPUT_HTML_FILE}")
        
        summary = f"Interactive Plotly 3D graph generated and saved to `{OUTPUT_HTML_FILE}`."
        if skipped_files_count > 0:
            summary += f" {skipped_files_count} file(s) were skipped due to parsing errors (see `{ERROR_LOG_FILE}`)."
        
        if not python_files:
             summary = f"Graph generation attempted. No Python files found. Empty graph saved to `{OUTPUT_HTML_FILE}`."
        elif not graph.number_of_nodes() and python_files: # Files found, but graph is empty (e.g. all failed analysis or no structures)
             summary += " However, the graph is empty as no valid code structures could be extracted or all files failed analysis."
        elif graph.number_of_nodes() > 0 and not pos_3d: # Graph has nodes, but layout failed
            summary += " Layout computation failed; nodes may be poorly positioned."


        completion_message = summary
        
    except Exception as e:
        log_error(f"Failed to save Plotly graph HTML to {OUTPUT_HTML_FILE}", e)
        completion_message = f"Attempted to generate Plotly 3D graph. Errors occurred during saving. Check `{ERROR_LOG_FILE}`. Output might be incomplete or missing at `{OUTPUT_HTML_FILE}`."
        if skipped_files_count > 0:
            completion_message += f" Additionally, {skipped_files_count} file(s) were skipped during analysis."

    # 8. Logging to Backlog
    if completion_message:
        append_to_backlog(completion_message)
    else:
        append_to_backlog("Plotly graph generation script finished with an undetermined state. No specific completion message generated.")
    
    print("Interactive 3D Plotly graph generation process complete.")
    if skipped_files_count > 0:
        print(f"Note: {skipped_files_count} Python files were skipped due to parsing errors. Check '{ERROR_LOG_FILE}' for details.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Catch-all for unhandled exceptions in main flow
        log_error("An unhandled error occurred during script execution", e)
        print(f"CRITICAL ERROR: Script terminated. Check {ERROR_LOG_FILE} and resolution plans in {ERROR_RESOLUTION_PLAN_DIR}.")
        # Optionally, append a critical failure message to backlog as well
        append_to_backlog(f"CRITICAL FAILURE during Plotly graph generation. Unhandled exception: {type(e).__name__}. See {ERROR_LOG_FILE}.")