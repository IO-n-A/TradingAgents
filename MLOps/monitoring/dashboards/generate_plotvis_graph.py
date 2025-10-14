import os
import ast
import datetime
import sys
import traceback

# Attempt to import pyvis and handle if not found
try:
    from pyvis.network import Network
except ImportError:
    print("Error: The 'pyvis' library is not installed. Please install it by running: pip install pyvis")
    print("SCRIPT_REPORT: Pyvis not installed. Graph generation aborted.")
    sys.exit(1)

# --- Configuration ---
ROOT_DIRS = ["FinRobot/finrl/", "FinNLP/fingpt/", "MLOps/", "core/"]
OUTPUT_HTML_PATH = "analysis/interactive_plotvis_graph.html"
ERROR_LOG_PATH = "analysis/error.md"
ERROR_RESOLUTION_DIR = "analysis/errors/"
# Normalize path for consistent comparison
SPECIAL_FILE_TOOLTIP_PATH = "FinRobot/finrl/meta/data_processor.py".replace("\\", "/")

# User-provided tooltip for the special file
SPECIAL_FILE_TOOLTIP_CONTENT = """
**File Path:** FinRobot/finrl/meta/data_processor.py
**Type:** Data Processor Module

**Description:**
This module is a crucial part of the FinRL library, specifically designed for preprocessing and preparing financial data for reinforcement learning environments. It acts as a bridge between raw financial data sources and the structured data required by RL agents.

**Key Responsibilities:**
*   **Data Fetching:** Interfaces with various data sources (e.g., Yahoo Finance, Alpaca, local CSV files) to download historical market data (OHLCV - Open, High, Low, Close, Volume) and potentially other relevant financial indicators.
*   **Data Cleaning:** Handles missing values (e.g., imputation, removal), corrects inconsistencies, and ensures data integrity.
*   **Feature Engineering:**
    *   Calculates a wide array of technical indicators (e.g., Moving Averages, RSI, MACD, Bollinger Bands) that can serve as features for the RL agent.
    *   May incorporate other features like VIX (volatility index) or sentiment scores if available.
    *   Allows for customization of features to be included.
*   **Data Splitting:** Divides the dataset into training, validation, and testing periods to prevent lookahead bias and ensure robust model evaluation.
*   **State Space Construction:** Transforms the processed data into the format expected by the RL environment's state representation. This often involves creating a sliding window of historical data for each time step.
*   **Handling Multiple Tickers:** Capable of processing data for multiple financial instruments (stocks, cryptocurrencies, etc.) simultaneously.
*   **User Customization:** Provides parameters to customize data processing steps, such as the date range, list of tickers, technical indicators to use, and data cleaning methods.

**Core Classes/Functions (Illustrative - actual names might vary):**
*   `DataProcessor`: Main class orchestrating the data processing pipeline.
    *   `download_data()`: Fetches data.
    *   `clean_data()`: Cleans the raw data.
    *   `add_technical_indicator()`: Adds specified technical indicators.
    *   `add_turbulence()`: (If applicable) Adds market turbulence features.
    *   `df_to_array()`: Converts pandas DataFrame to NumPy array for the environment.
*   Utility functions for specific calculations or data transformations.

**Dependencies:**
*   `pandas` for data manipulation.
*   `numpy` for numerical operations.
*   `stockstats` or similar libraries for technical indicator calculation.
*   APIs for data sources (e.g., `yfinance`, `alpaca_trade_api`).

**Workflow:**
1.  Initialize `DataProcessor` with configuration (tickers, date range, indicators).
2.  Call `download_data()` to fetch raw data.
3.  Call `clean_data()` to preprocess.
4.  Call `add_technical_indicator()` to enrich data.
5.  (Optional) Add other features like VIX or turbulence.
6.  Split data into train/test sets.
7.  Prepare data for environment consumption (e.g., convert to arrays).

**Significance:**
The quality and relevance of the data processed by this module directly impact the performance of the RL trading agents. Well-engineered features and clean data are fundamental for successful financial machine learning applications. It abstracts away the complexities of data handling, allowing users to focus on agent design and strategy development.
"""

# --- Helper Functions for AST Parsing ---
def get_module_docstring(node):
    return ast.get_docstring(node) if node else None

def get_class_details(class_node):
    name = class_node.name
    docstring = ast.get_docstring(class_node)
    methods = []
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef):
            methods.append(item.name)
    parents = [p.id for p in class_node.bases if isinstance(p, ast.Name)]
    # Could also try to capture ast.Attribute for more complex parent names like "module.ParentClass"
    for p_node in class_node.bases:
        if isinstance(p_node, ast.Attribute):
            # Attempt to reconstruct full parent name (e.g., "module.Parent")
            parent_parts = []
            curr = p_node
            while isinstance(curr, ast.Attribute):
                parent_parts.insert(0, curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                parent_parts.insert(0, curr.id)
            parents.append(".".join(parent_parts))
        elif isinstance(p_node, ast.Subscript): # e.g. Generic[T]
            if isinstance(p_node.value, ast.Name):
                parents.append(p_node.value.id + "[]") # Simplified representation
    return {"name": name, "docstring": docstring, "methods": methods, "parents": list(set(parents))}


def get_function_details(func_node, source_lines=None):
    name = func_node.name
    docstring = ast.get_docstring(func_node)
    comments = []
    if source_lines and func_node.lineno > 0:
        # func_node.lineno is 1-indexed
        # We want to look at lines *before* the function definition
        for i in range(func_node.lineno - 2, -1, -1):
            line = source_lines[i].strip()
            if line.startswith("#"):
                comments.insert(0, line[1:].strip()) # Add to the beginning to keep order
            elif not line: # Stop at an empty line
                break
            else: # Stop at a non-comment, non-empty line
                break
    return {"name": name, "docstring": docstring, "comments": comments if comments else None}

def get_imports(node):
    imports = []
    for item in node.body:
        if isinstance(item, ast.Import):
            for alias in item.names:
                imports.append({"name": alias.name, "asname": alias.asname})
        elif isinstance(item, ast.ImportFrom):
            module_name = ""
            if item.module: # item.module can be None for "from . import X"
                module_name = item.module
            
            # Handle relative imports like "from . import foo" or "from ..bar import baz"
            # item.level > 0 indicates relative import
            # For simplicity, we'll prepend dots to module_name
            if item.level > 0:
                module_name = "." * item.level + module_name

            for alias in item.names:
                imports.append({"name": f" {module_name}.{alias.name}", "asname": alias.asname, "is_from_import": True, "from_module": module_name, "imported_name": alias.name})
    return imports

def parse_python_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors='ignore') as source_file:
            source_code = source_file.read()
        if not source_code.strip():
            return None # Skip empty or effectively empty files
        tree = ast.parse(source_code, filename=file_path)
        source_lines = source_code.splitlines() # For comment extraction

        module_docstring = get_module_docstring(tree)
        classes = []
        functions = []
        
        for node_item in tree.body:
            if isinstance(node_item, ast.ClassDef):
                classes.append(get_class_details(node_item))
            elif isinstance(node_item, ast.FunctionDef):
                functions.append(get_function_details(node_item, source_lines))
        
        imports_list = get_imports(tree)

        return {
            "file_path": file_path.replace("\\", "/"), # Ensure path is normalized
            "module_docstring": module_docstring,
            "classes": classes,
            "functions": functions,
            "imports": imports_list
        }
    except SyntaxError as e:
        raise # Re-raise to be caught by the main loop's specific handling
    except Exception as e:
        raise


# --- Helper Functions for Pyvis Graph ---
def add_nodes_and_edges(net, file_data_list, special_file_path_norm, special_tooltip_content_html):
    DIR_COLOR = "#FFC300"    # Vivid Gold
    FILE_COLOR = "#4682B4"   # SteelBlue
    CLASS_COLOR = "#50C878"  # EmeraldGreen
    FUNC_COLOR = "#FF7F50"   # Coral
    
    root_dir_map = {name.strip('/\\'): i for i, name in enumerate(ROOT_DIRS)}
    all_defined_classes_map = {} # "module/path.py/ClassName" -> node_id
    all_file_nodes = set()

    # Pass 1: Add all directory, file, class, and function nodes
    for data in file_data_list:
        file_path_norm = data["file_path"]
        all_file_nodes.add(file_path_norm)

        file_group = None
        normalized_dir_name_for_group = ""
        for root_name, group_idx in root_dir_map.items():
            if file_path_norm.startswith(root_name):
                file_group = group_idx
                normalized_dir_name_for_group = root_name # Use the root as the main group identifier
                break
        
        # Directory Nodes (one for each unique directory segment)
        current_path_segments = file_path_norm.split('/')
        path_accumulator = ""
        for i, segment in enumerate(current_path_segments[:-1]): # Up to the parent directory of the file
            path_accumulator = f" {path_accumulator}{segment}/" if path_accumulator else f" {segment}/"
            dir_node_id = path_accumulator.rstrip('/')
            
            # Determine group for this directory node
            dir_group = None
            for root_name_grp, group_idx_grp in root_dir_map.items():
                if dir_node_id.startswith(root_name_grp):
                    dir_group = group_idx_grp
                    break

            if dir_node_id and dir_node_id not in net.node_ids:
                label_for_dir = dir_node_id # Use full path for directory label
                net.add_node(dir_node_id, label=label_for_dir, title=f" Directory: {dir_node_id}", 
                             color=DIR_COLOR, group=dir_group, shape='box', size=20)
            # Link to parent directory if not a root dir itself
            if i > 0:
                parent_dir_node_id = "/".join(current_path_segments[:i]).rstrip('/')
                if parent_dir_node_id and parent_dir_node_id in net.node_ids and dir_node_id != parent_dir_node_id:
                     # Check if edge already exists to avoid duplicates if processing out of order
                    edge_exists = any(e['from'] == parent_dir_node_id and e['to'] == dir_node_id for e in net.edges)
                    if not edge_exists:
                        net.add_edge(parent_dir_node_id, dir_node_id, title="contains_dir", color="#CCCCCC", value=0.1)


        # File Node
        file_node_id = file_path_norm
        file_label = os.path.basename(file_path_norm)
        
        is_special_file = (file_path_norm == special_file_path_norm)
        is_non_python = data.get("is_non_python_file", False)

        if is_special_file:
            file_tooltip_html = special_tooltip_content_html
        elif is_non_python:
            file_tooltip_html = f"File: {file_path_norm}<i>(Non-Python file)</i>"
        else: # It's a Python file (and not the special one)
            file_tooltip_html = f"This is a File Node representing {file_path_norm}."
            if data.get("module_docstring"):
                doc_html = data['module_docstring'][:300].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '')
                file_tooltip_html += f"\n Module Docstring:{doc_html}..."
            cls_names = [c['name'] for c in data.get('classes', [])]
            func_names = [f['name'] for f in data.get('functions', [])]
            if cls_names:
                file_tooltip_html += "\n Classes: " + ", ".join(cls_names) 
            if func_names:
                file_tooltip_html += "\n Functions: " + ", ".join(func_names) 

        net.add_node(file_node_id, label=file_label, title=file_tooltip_html, color=FILE_COLOR, group=file_group, shape='ellipse', size=15)
        parent_dir_of_file = os.path.dirname(file_path_norm)
        if parent_dir_of_file and parent_dir_of_file in net.node_ids:
            net.add_edge(parent_dir_of_file, file_node_id, title="contains_file", color="#DDDDDD", value=0.2)

        # Class Nodes
        for cls in data["classes"]:
            class_node_id = f" {file_path_norm}/{cls['name']}" # Unique ID for class
            all_defined_classes_map[class_node_id] = class_node_id
            
            class_tooltip_html = f"This is a Class Node for {cls['name']} (defined in {file_label})."
            if cls.get("docstring"):
                cls_doc_html = cls['docstring'][:250].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '')
                class_tooltip_html += f"\n Docstring:{cls_doc_html}..."
            if cls.get("methods"):
                class_tooltip_html += "\n  Methods: " + ", ".join(cls['methods']) 
            if cls["parents"]:
                class_tooltip_html += "\n  Inherits from: " + ", ".join(cls['parents'])
            
            net.add_node(class_node_id, label=cls['name'], title=class_tooltip_html, color=CLASS_COLOR, group=file_group, shape='diamond', size=10)
            net.add_edge(file_node_id, class_node_id, title="defines_class", value=0.3)

        # Function Nodes
        for func in data["functions"]:
            func_node_id = f" {file_path_norm}/{func['name']}" # Unique ID for function
            func_tooltip_html = f"This is a Function Node for {func['name']} (within {file_label})."
            if func.get("comments"):
                # Escape each comment line individually, then join with 
                escaped_comments = [c.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;') for c in func['comments']]
                func_comment_html = "".join(escaped_comments)
                func_tooltip_html += f"\n Preceding Comments:{func_comment_html}"
            if func.get("docstring"):
                func_doc_html = func['docstring'][:250].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '')
                func_tooltip_html += f"\n Docstring:{func_doc_html}..."
            
            net.add_node(func_node_id, label=func['name'], title=func_tooltip_html, color=FUNC_COLOR, group=file_group, shape='dot', size=8)
            net.add_edge(file_node_id, func_node_id, title=" defines_function", value=0.3)

    # Pass 2: Add inheritance and import edges
    for data in file_data_list:
        if data.get("is_non_python_file", False): # Skip non-python files for these edges
            continue
        file_path_norm = data["file_path"]
        
        # Inheritance Edges
        for cls in data["classes"]:
            child_class_node_id = f" {file_path_norm}/{cls['name']}"
            for parent_name_full in cls["parents"]: # parent_name_full could be "ClassName" or "module.ClassName"
                # Try to find parent class node
                # Scenario 1: Parent in the same file
                potential_parent_id_same_file = f" {file_path_norm}/{parent_name_full}"
                if potential_parent_id_same_file in all_defined_classes_map:
                    net.add_edge(child_class_node_id, all_defined_classes_map[potential_parent_id_same_file], title=" inherits_from", arrows="to", color="purple", value=2)
                    continue

                # Scenario 2: Parent name is fully qualified (e.g. some.module.ClassName)
                # This requires resolving 'some.module' to a file path.
                # For simplicity, we'll search all_defined_classes_map for a key ending with /parent_name_full
                # or /parent_name_full.split('.')[-1] if parent_name_full contains dots.
                
                simple_parent_name = parent_name_full.split('.')[-1] # Get the actual class name part
                found_parent = False
                for defined_class_path_key in all_defined_classes_map.keys():
                    if defined_class_path_key.endswith(f" /{simple_parent_name}"):
                         # Check if the module path also matches somewhat
                        module_path_of_defined_class = os.path.dirname(defined_class_path_key).replace("/",".")
                        if parent_name_full.startswith(module_path_of_defined_class) or \
                           parent_name_full == simple_parent_name: # direct name match
                            net.add_edge(child_class_node_id, all_defined_classes_map[defined_class_path_key], title=f" inherits_from ({parent_name_full})", arrows="to", color="purple", value=2)
                            found_parent = True
                            break
                # if not found_parent:
                #     print(f" Debug: Parent class '{parent_name_full}' for '{child_class_node_id}' not robustly resolved.")


        # Import Edges (Simplified)
        current_file_node_id = file_path_norm
        for imp_data in data["imports"]:
            imported_name_full = imp_data["name"] # e.g., "os", "sys", "module.submodule.item", ".local_module.item"
            
            target_file_node_id = None

            # Attempt to resolve import to a project file
            # Heuristic: convert module path to file path
            # e.g. FinRL.finrl.meta.data_processor -> FinRobot/finrl/meta/data_processor.py
            
            import_path_parts = []
            if imp_data.get("is_from_import"):
                # from module.sub import item -> module/sub.py (target file)
                # from .local import item -> current_dir/local.py
                base_module_for_from = imp_data["from_module"]
                if base_module_for_from.startswith('.'): # Relative import
                    level = base_module_for_from.count('.')
                    current_dir_parts = os.path.dirname(file_path_norm).split('/')
                    # For 'from .mod import X', target is current_dir/mod.py
                    # For 'from ..mod import X', target is parent_dir/mod.py
                    if level == 1 and len(base_module_for_from) > 1: # from .actual_module import ...
                        effective_module_path = "/".join(current_dir_parts) + "/" + base_module_for_from[1:]
                    elif level > 1 and len(base_module_for_from) > level: # from ..actual_module import ...
                         effective_module_path = "/".join(current_dir_parts[:-(level-1)]) + "/" + base_module_for_from[level:]
                    elif len(base_module_for_from) == level : # from . import X or from .. import X (importing __init__.py implicitly)
                         effective_module_path = "/".join(current_dir_parts[:-(level-1)])
                    else: # Should not happen with valid Python
                        effective_module_path = base_module_for_from # Fallback

                    import_path_parts = effective_module_path.split('/')

                else: # Absolute import from a module
                    import_path_parts = base_module_for_from.split('.')
            else: # Direct import: import module.submodule
                import_path_parts = imported_name_full.split('.')

            # Try to find a matching file or directory (__init__.py)
            for i in range(len(import_path_parts), 0, -1):
                potential_module_as_file = "/".join(import_path_parts[:i]) + ".py"
                potential_module_as_dir_init = "/".join(import_path_parts[:i]) + "/__init__.py"
                
                # Check against all known file nodes, preferring longer matches
                found_match = None
                for proj_file_node in all_file_nodes:
                    if proj_file_node.endswith(potential_module_as_file):
                        found_match = proj_file_node
                        break
                    if proj_file_node.endswith(potential_module_as_dir_init):
                        found_match = proj_file_node
                        break
                if found_match:
                    target_file_node_id = found_match
                    break
            
            if target_file_node_id and target_file_node_id != current_file_node_id:
                if target_file_node_id in net.node_ids:
                    net.add_edge(current_file_node_id, target_file_node_id, title=f" imports ({imported_name_full})", 
                                 arrows="to", value=0.5, color="#A9A9A9", dashes=True, length=300)


# --- Main Script Logic ---
def main():
    all_file_data = []
    skipped_files_log = []

    # 1. File Discovery & 2. Content Analysis
    print("Starting file discovery and parsing...")
    for root_dir_name in ROOT_DIRS:
        abs_root_dir = os.path.abspath(root_dir_name)
        if not os.path.isdir(abs_root_dir):
            print(f" Warning: Root directory '{root_dir_name}' (abs: {abs_root_dir}) not found. Skipping.")
            continue

        for root, _, files in os.walk(root_dir_name):
            for file_name in files: # Renamed 'file' to 'file_name' to avoid conflict with built-in
                raw_file_path = os.path.join(root, file_name)
                file_path_norm = raw_file_path.replace("\\", "/") # Normalize path once

                if file_name.endswith(".py"):
                    try:
                        parsed_data = parse_python_file(raw_file_path) 
                        if parsed_data:
                            all_file_data.append(parsed_data)
                        else:
                            skipped_files_log.append(f" Skipped (empty or non-Python AST content): {file_path_norm}")
                    except Exception as e:
                        warning_msg = f" WARNING: Could not parse {file_path_norm}: {type(e).__name__} - {e}"
                        print(warning_msg)
                        skipped_files_log.append(warning_msg)
                else: # For non-Python files
                    all_file_data.append({
                        "file_path": file_path_norm,
                        "module_docstring": "Non-Python file", 
                        "classes": [],
                        "functions": [],
                        "imports": [],
                        "is_non_python_file": True
                    })
    
    if not all_file_data:
        print("No files found or parsed. Aborting graph generation.")
        print("SCRIPT_REPORT: No files processed. Graph not generated.")
        return skipped_files_log, False


    print(f" Processed {len(all_file_data)} files (Python and other types).")
    print("Generating interactive graph...")
    # 3. Interactive Graph Generation
    net = Network(height="1000px", width="100%", directed=True, notebook=False, 
                  cdn_resources='remote', 
                  select_menu=False,  # Changed from True
                  filter_menu=False,  # Changed from True
                  bgcolor="#222222", font_color="white")
    
    # Configure physics for better initial layout, can be tweaked by user
    # The string passed to set_options must be a valid JSON object.
    # Removed "var options = " and comments.
    net.set_options("""
    {
      "nodes": {
        "font": {
          "size": 12
        }
      },
      "edges": {
        "smooth": {
          "type": "continuous",
          "forceDirection": "none",
          "roundness": 0.2
        },
        "arrows": {
          "to": { "enabled": true, "scaleFactor": 0.7 }
        },
        "color": {
            "inherit": false
        }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08,
          "damping": 0.4,
          "avoidOverlap": 0.5
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based",
        "timestep": 0.5
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "navigationButtons": true,
        "keyboard": true
      },
      "manipulation": {
        "enabled": false
      }
    }
    """)


    try:
        special_tooltip_html = SPECIAL_FILE_TOOLTIP_CONTENT.replace("\n", "")
        add_nodes_and_edges(net, all_file_data, SPECIAL_FILE_TOOLTIP_PATH, special_tooltip_html)

        # Generate the graph HTML to a string
        graph_html_content = net.generate_html(name=OUTPUT_HTML_PATH, notebook=False) # Use generate_html to get string

        # Custom HTML and JavaScript for search functionality
        custom_html_js = '''
        <div style="position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 1000; background-color: #333; padding: 10px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.5); display: flex; flex-direction: column; align-items: center;">
            <div>
                <label for="nodeFilterInput" style="color: white; margin-right: 5px;">Filter Nodes:</label>
                <input type="text" id="nodeFilterInput" onkeyup="filterNodeOptions()" placeholder="Type to filter..." style="padding: 5px; width: 180px; border-radius: 3px; border: 1px solid #555; background-color: #444; color: white; margin-right: 5px;">
            </div>
            <div style="margin-top: 5px;">
                <label for="nodeSelector" style="color: white; margin-right: 5px; display: none;">Select Node:</label> <!-- Hidden label, filter acts as main interaction -->
                <select id="nodeSelector" style="padding: 5px; width: 250px; border-radius: 3px; border: 1px solid #555; background-color: #444; color: white; max-height: 150px; overflow-y: auto;"></select>
                <button onclick="focusSelectedNode()" style="padding: 5px 10px; margin-left: 5px; border-radius: 3px; border: none; background-color: #5C8EBC; color: white; cursor: pointer;">Zoom to Selected</button>
            </div>
            <p id="selectionStatus" style="color: #aaa; font-size: 0.9em; margin-top: 5px; text-align: center;"></p>
        </div>

        <script type="text/javascript">
            var network = null; // Ensure network is accessible
            var allNodesCache = []; // Cache for all nodes to repopulate dropdown

            function filterNodeOptions() {
                var input, filter, select, options, i, txtValue;
                input = document.getElementById('nodeFilterInput');
                filter = input.value.toLowerCase();
                select = document.getElementById('nodeSelector');
                options = select.getElementsByTagName('option');
                var visibleOptionsCount = 0;

                for (i = 0; i < options.length; i++) {
                    txtValue = options[i].textContent || options[i].innerText;
                    if (txtValue.toLowerCase().indexOf(filter) > -1) {
                        options[i].style.display = "";
                        visibleOptionsCount++;
                    } else {
                        options[i].style.display = "none";
                    }
                }
            }

            function populateNodeSelector() {
                var selectElement = document.getElementById('nodeSelector');
                var statusElement = document.getElementById('selectionStatus');
                selectElement.innerHTML = ''; 
                allNodesCache = []; 

                if (network && network.body && network.body.data && network.body.data.nodes) {
                    var nodesRaw = network.body.data.nodes.get({
                        fields: ['id', 'label', 'title'],
                        orderBy: 'label'
                    });
                    
                    if (nodesRaw.length === 0) {
                        statusElement.textContent = "No nodes available to select.";
                        return;
                    }

                    allNodesCache = nodesRaw.map(function(node) {
                        var displayLabel = node.label || node.id;
                        if (String(node.id).includes("/")) {
                            displayLabel = String(node.id);
                        }
                        return { id: node.id, text: displayLabel };
                    });
                    
                    allNodesCache.forEach(function(nodeData) {
                         var option = document.createElement('option');
                         option.value = nodeData.id;
                         option.textContent = nodeData.text;
                         selectElement.appendChild(option);
                    });

                    statusElement.textContent = "Filter or select a node, then click 'Zoom to Selected'.";
                    filterNodeOptions(); 
                } else {
                    statusElement.textContent = "Network not ready or no nodes found for selector.";
                    console.error("Pyvis network object or its internal data (nodes) not found/ready for populating selector. Current network state:", network);
                }
            }

            function focusSelectedNode() {
                var selectElement = document.getElementById('nodeSelector');
                var selectedNodeId = selectElement.value;
                var statusElement = document.getElementById('selectionStatus');
                console.log("Focus selected node triggered. Selected ID: '" + selectedNodeId + "'");

                if (!selectedNodeId) {
                    statusElement.textContent = "No node selected from the dropdown.";
                    console.log("No node ID selected.");
                    return;
                }

                if (network && network.body && network.body.data && network.body.data.nodes) {
                    var nodeExists = network.body.data.nodes.get(selectedNodeId);
                    if (nodeExists) {
                        statusElement.textContent = "Focusing on: " + selectedNodeId;
                        console.log("Focusing on node:", selectedNodeId, "with scale 5.0");
                        network.focus(selectedNodeId, {
                            scale: 5.0, 
                            offset: {x:0, y:0},
                            animation: {
                                duration: 1000,
                                easingFunction: "easeInOutQuad"
                            }
                        });
                        network.selectNodes([selectedNodeId]);
                    } else {
                        statusElement.textContent = "Node ID '" + selectedNodeId + "' not found in the network.";
                        console.warn("Node ID '" + selectedNodeId + "' not found for focusing.");
                    }
                } else {
                    statusElement.textContent = "Network not initialized or ready for focus operation.";
                    console.error("Pyvis network object not found or not ready for focusing. Current network state:", network);
                }
            }
            
            function setupNetworkInteractions() {
                console.log("Attempting to set up network interactions. Current network object:", network);
                if (!(typeof network !== 'undefined' && network && network.body && network.body.data && network.body.data.nodes)) {
                    console.warn("Network not fully ready for interaction setup.");
                    return false; // Indicate setup failed
                }

                try {
                    populateNodeSelector();
                    
                    network.on("click", function (params) {
                        console.log("Node click event:", params);
                        if (params.nodes && params.nodes.length > 0) {
                            var nodeId = params.nodes[0];
                            console.log("Clicked node ID: " + nodeId + ". Zooming with scale 5.0.");
                            document.getElementById('selectionStatus').textContent = "Clicked & Zoomed: " + nodeId;
                            network.focus(nodeId, {
                                scale: 5.0,
                                animation: {
                                    duration: 1000,
                                    easingFunction: "easeInOutQuad"
                                }
                            });
                            var selector = document.getElementById('nodeSelector');
                            var filterInput = document.getElementById('nodeFilterInput');
                            if(selector) {
                                selector.value = nodeId; 
                                if (filterInput) filterInput.value = ''; 
                                filterNodeOptions(); 
                                for(var i=0; i < selector.options.length; i++){
                                    if(selector.options[i].value === nodeId){
                                        selector.options[i].style.display = "";
                                        break;
                                    }
                                }
                            }
                        } else if (params.edges && params.edges.length > 0) {
                             console.log("Clicked edge: " + params.edges[0]);
                        } else {
                            console.log("Clicked on empty space.");
                        }
                    });
                    console.log("Successfully set up network interactions and node selector.");
                    return true; // Indicate setup succeeded
                } catch (e) {
                    console.error("Error during network interaction setup:", e);
                    var statusElement = document.getElementById('selectionStatus');
                    if(statusElement) statusElement.textContent = "Error setting up network interactions.";
                    return false; // Indicate setup failed
                }
            }

            document.addEventListener('DOMContentLoaded', function() {
                console.log("DOMContentLoaded event fired.");
                if (setupNetworkInteractions()) {
                    // Successfully setup on DOMContentLoaded
                    return;
                }

                console.log("Network object not immediately available or fully initialized after DOMContentLoaded. Starting polling mechanism.");
                var checkNetworkInterval = setInterval(function() {
                    console.log("Polling for network object. typeof network:", typeof network, "network:", network);
                    if (setupNetworkInteractions()) {
                        clearInterval(checkNetworkInterval); 
                    }
                }, 1000); // Check every 1 second

                setTimeout(function() {
                    clearInterval(checkNetworkInterval); 
                    // Final check after timeout
                    if (!(typeof network !== 'undefined' && network && network.body && network.body.data && network.body.data.nodes)) {
                        console.log("Stopped checking for network object after 30s timeout.");
                        var statusElement = document.getElementById('selectionStatus');
                        if (statusElement) {
                            if (typeof network === 'undefined' || !network) {
                                statusElement.textContent = "Failed to initialize node selector: Pyvis Network object not found.";
                                console.error("CRITICAL: Pyvis Network object ('network') was not found after 30s.");
                            } else {
                                statusElement.textContent = "Network object found, but not fully initialized for selector after 30s.";
                                console.warn("WARN: Pyvis Network object ('network') was found, but not fully initialized (e.g., network.body.data.nodes missing) after 30s. Current network state:", network);
                            }
                        }
                    } else {
                         console.log("Network became available within the 30s timeout (checked by final timeout function).");
                    }
                }, 30000); // Stop after 30 seconds
            });
        </script>
        '''

        # New method: Insert before the *last* </body> tag
        idx = graph_html_content.rfind("</body>")
        if idx != -1:
            graph_html_content = graph_html_content[:idx] + custom_html_js + "\n" + graph_html_content[idx:]
        else: # Fallback if no body tag (should not happen for valid HTML)
            graph_html_content += "\n" + custom_html_js


        # 4. Output
        os.makedirs(os.path.dirname(OUTPUT_HTML_PATH), exist_ok=True)
        # net.save_graph(OUTPUT_HTML_PATH) # Original save
        with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
            f.write(graph_html_content)
            
        print(f" Successfully generated interactive graph with search: {OUTPUT_HTML_PATH}")
        
        if skipped_files_log:
            print("\n--- Files Skipped or Parsing Issues ---")
            for log_entry in skipped_files_log:
                print(log_entry)
            print("------------------------------------")
        
        return skipped_files_log, False # False indicates no graph generation error

    except Exception as e_graph:
        # 5. Error Handling for graph generation
        print(f" CRITICAL: Error during graph generation: {e_graph}")
        detailed_traceback = traceback.format_exc()
        print(detailed_traceback)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f" )
        handle = timestamp.replace(":", "").replace("-", "").replace(".", "").replace(" ", "_") # FS-safe handle

        # Ensure error log directory exists
        os.makedirs(os.path.dirname(ERROR_LOG_PATH), exist_ok=True)
        
        error_message_for_md = f" ## Interactive Graph Generation Error - {timestamp}\n\n"
        error_message_for_md += f" **Error:**\n```\n{str(e_graph)}\n```\n\n"
        error_message_for_md += f" **Traceback:**\n```\n{detailed_traceback}\n```\n"

        # Append to analysis/error.md
        try:
            with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f_error_log:
                f_error_log.write("\n" + error_message_for_md)
        except Exception as e_io_log:
            print(f" Failed to write to error log {ERROR_LOG_PATH}: {e_io_log}")


        # Create error resolution plan
        os.makedirs(ERROR_RESOLUTION_DIR, exist_ok=True)
        resolution_plan_filename = f" {handle}_error_resolution_interactive_graph.md"
        resolution_plan_path = os.path.join(ERROR_RESOLUTION_DIR, resolution_plan_filename)
        
        plan_content = f"""# Error Resolution Plan: Interactive Graph Generation
**Timestamp:** {timestamp}
**Handle (Error Log):** {handle} (Refer to [`../../analysis/error.md`](../../analysis/error.md) for full error details)

## 1. Bug Description
An error occurred during the generation of the interactive architecture graph.
**Error Summary:** {str(e_graph)}
*(Full details including traceback are logged in [`../../analysis/error.md`](../../analysis/error.md))*

## 2. Affected Component(s)
Interactive architecture graph generation script (`generate_plotvis_graph.py`).

## 3. Hypothesized Cause(s)
*   Issue with `pyvis` library processing specific node/edge data.
*   Unexpected data structure from file parsing results.
*   Problematic logic in `add_nodes_and_edges` function.
*   Resource limitations (memory, etc.) if the graph is extremely large.
*   Incorrect HTML/JS options for pyvis.

## 4. Debugging Steps to be Taken
1.  Review the full traceback in [`../../analysis/error.md`](../../analysis/error.md).
2.  Examine the `all_file_data` structure being passed to `add_nodes_and_edges`, especially for files processed just before the error.
3.  Step through the `add_nodes_and_edges` function with a debugger, focusing on the problematic data.
4.  Test with a smaller subset of Python files (e.g., one directory, or even one file) to isolate the issue.
5.  Check `pyvis` library documentation and open issues for similar error reports.
6.  Simplify node/edge properties or pyvis options to see if a specific feature is causing the problem.

## 5. Proposed Solution(s)
*   (To be determined after debugging)

## 6. Test Cases to Verify Fix
*   Successful generation of `{OUTPUT_HTML_PATH}` without errors.
*   Graph correctly displays nodes and edges for a sample set of files.
*   Tooltips are functional and display correct information.
*   Graph interactivity (zoom, pan, drag) works as expected.
"""
        try:
            with open(resolution_plan_path, "w", encoding="utf-8") as f_plan:
                f_plan.write(plan_content)
            print(f" Created error resolution plan: {resolution_plan_path}")
            
            # Append link to plan in analysis/error.md
            link_to_plan_md = f" \n**Error Resolution Plan:** [`./errors/{resolution_plan_filename}`](./errors/{resolution_plan_filename})\n---\n"
            with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f_error_log:
                f_error_log.write(link_to_plan_md)
            print(f" Appended link to resolution plan in {ERROR_LOG_PATH}")

        except Exception as e_io_plan:
            print(f" Failed to write error resolution plan or update log with link: {e_io_plan}")

        return skipped_files_log, True # True indicates graph generation error

if __name__ == "__main__":
    skipped_files, graph_error_occurred = main()

    if graph_error_occurred:
        print("\nSCRIPT_REPORT: Graph generation failed. Error details logged.")
    else:
        print("\nSCRIPT_REPORT: Graph generation successful.")

    if skipped_files:
        print("SCRIPT_REPORT: Some files were skipped or had parsing issues:")
        for f_log_entry in skipped_files:
            print(f" SCRIPT_REPORT_SKIPPED: {f_log_entry}")
    else:
        print("SCRIPT_REPORT: No files were skipped during parsing.")