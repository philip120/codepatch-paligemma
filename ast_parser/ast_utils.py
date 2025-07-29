from sly.lex import Lexer
from sly.yacc import Parser


def reconstruct_code_from_ast(node):
    """
    Recursively reconstructs a line of code from an AST node.
    """
    if isinstance(node, tuple):
        node_type = node[0]
        children = node[1:]

        # Simple tokens
        if len(children) == 0 and isinstance(node_type, str):
            return node_type

        # Recursive reconstruction
        if len(children) > 0 and isinstance(children[0], (list, tuple)):
            reconstructed_children = [reconstruct_code_from_ast(child) for child in children[0]]
        else:
            reconstructed_children = [reconstruct_code_from_ast(child) for child in children]

        # Handling different node types based on MATLAB syntax
        if 'oper' in node_type:
            op = node_type.split('"')[1] if '"' in node_type else node_type.split(' ')[1]
            if "unary" in node_type:
                return f"{op}{reconstructed_children[0]}"
            return f"{reconstructed_children[0]} {op} {reconstructed_children[1]}"
        elif node_type == 'assign':
            return f"{reconstructed_children[0]} = {reconstructed_children[1]}"
        elif node_type == 'func_call/array_idxing':
            func_name = reconstructed_children[0]
            args = reconstructed_children[1]
            return f"{func_name}({args})"
        elif node_type == 'args':
            return ", ".join(reconstructed_children)
        elif node_type == 'statement':
            return f"{reconstructed_children[0]};"
        elif node_type == 'code_block':
            return "\n".join(reconstructed_children)
        elif node_type in ['expr', 'bracket']:
            return f"({reconstructed_children[0]})" if node_type == 'bracket' else reconstructed_children[0]

    # Base case for terminals like variable names or numbers
    elif isinstance(node, (str, int, float)):
        return str(node)

    return ""


def get_semantic_patches(matlab_code: str, parser: Parser, lexer: Lexer) -> list[str]:
    """
    Parses MATLAB code and extracts a list of semantic statements from the AST.
    """
    if not matlab_code.strip():
        return []

    # Parse the code to get the AST
    ast = parser.parse(lexer.tokenize(matlab_code))
    
    patches = []
    if ast and ast[0] == 'code_block':
        statements = ast[1]
        for stmt_node in statements:
            # Reconstruct each statement and add it as a patch
            reconstructed = reconstruct_code_from_ast(stmt_node)
            if reconstructed:
                patches.append(reconstructed.replace(" ;", ";")) # Clean up spacing before semicolon
                
    return patches 