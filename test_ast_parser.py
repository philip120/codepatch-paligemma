import sys
import os
from pprint import pprint

# Add the parent directory of 'ast_parser' to the Python path
# This allows us to import the matlab_parser module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ast_parser.matlab_parser import MatlabParser
from ast_parser.matlab_lexer import MatlabLexer

def rebuild_node_to_text(node):
    """
    Recursively rebuilds a readable text representation from an AST node.
    This is the core of the patch extraction logic.
    """
    if not isinstance(node, tuple):
        return str(node).strip("'")
    
    # Handle leaf nodes that are tuples, e.g. ('A',)
    if len(node) == 1 and isinstance(node[0], str):
        return node[0].strip("'")

    node_type, children = node

    # --- Handle specific node types for better reconstruction ---
    if node_type == 'unary - oper':
        return f"-{rebuild_node_to_text(children[0])}"

    if node_type == 'func_call/array_idxing':
        try:
            # The 'expr' part can be complex (e.g., an assignment)
            # We need to handle it carefully to get just the function name
            expr_node = children[0]
            if expr_node[0] == 'expr' and expr_node[1][0][0] == 'assign':
                 func_name = expr_node[1][0][1][1] # Dig deep for the name
            else:
                 func_name = rebuild_node_to_text(expr_node)
            
            args_node = children[1]
            rebuilt_args = [rebuild_node_to_text(arg) for arg in args_node[1]]
            args_string = ", ".join(rebuilt_args)
            return f"{func_name}({args_string})"
        except (ValueError, IndexError, TypeError):
            return "[parse error: func_call]"

    if node_type == 'assign':
        try:
            var_name = rebuild_node_to_text(children[0])
            value_node = children[1]
            
            # Special case for function calls on the right side of assignment
            if isinstance(value_node, tuple) and value_node[0] == 'func_call/array_idxing':
                func_call_str = rebuild_node_to_text(value_node)
                # The function name is already part of the rebuilt string
                return f"{var_name} = {func_call_str}"

            value = rebuild_node_to_text(value_node)
            return f"{var_name} = {value}"
        except (ValueError, IndexError):
            return "[parse error: assign]"

    if node_type == 'range ':
        try:
            parts = [rebuild_node_to_text(c) for c in children]
            return ":".join(parts)
        except (ValueError, IndexError):
            return "[parse error: range]"

    if node_type == 'command':
        return rebuild_node_to_text(children[0]).rstrip(';')

    # --- Generic fallback for other node types ---
    if isinstance(children, tuple):
        # Join children, but filter out None from failed rebuilds
        parts = [rebuild_node_to_text(child) for child in children]
        return "".join(p for p in parts if p)
    else:
        return str(children)

def extract_semantic_patches(ast_root):
    """
    Walks the AST and extracts a list of semantically meaningful patches.
    It focuses on 'statement' nodes as they represent a single line or action.
    """
    patches = []
    if not isinstance(ast_root, tuple) or ast_root[0] != 'code_block':
        return patches

    # The root 'code_block' contains a tuple of 'statement' nodes
    statements = ast_root[1]
    for stmt_node in statements:
        # Rebuild the full text of the statement
        rebuilt_statement = rebuild_node_to_text(stmt_node).strip()
        # Only add non-empty statements to our patch list
        if rebuilt_statement:
            patches.append(rebuilt_statement)
    return patches


def main():
    """
    Parses a sample MATLAB code string and prints the resulting AST.
    """
    # Sample MATLAB code, similar to what we might find in the dataset
    matlab_code = """
    % Define system parameters
    A = -2;
    B = 1;
    C = 1;
    D = 0;

    % Create state-space model
    sys = ss(A, B, C, D);

    % Define time vector and input signal
    t = 0:0.01:5;
    u = ones(size(t));

    % Simulate the system response
    y = lsim(sys, u, t);

    % Plot the results
    figure;
    plot(t, y);
    title('Step Response of First-Order State-Space System');
    xlabel('Time (s)');
    ylabel('Output');
    grid on;
    """

    print("--- Parsing MATLAB Code ---")
    lexer = MatlabLexer()
    parser = MatlabParser()
    # First, use the lexer to tokenize the code string
    tokens = lexer.tokenize(matlab_code)
    # Then, parse the stream of tokens to get the AST
    ast = parser.parse(tokens)

    print("\n--- Generated Abstract Syntax Tree (AST) ---")
    pprint(ast)

    print("\n--- Extracting Semantic Patches from AST ---")
    patches = extract_semantic_patches(ast)
    print("Found patches:")
    pprint(patches)

    print("\n--- Test Complete ---")


if __name__ == "__main__":
    main() 