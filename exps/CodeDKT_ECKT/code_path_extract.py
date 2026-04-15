import os
import ast
import json
import sys
import time

from anytree import Node
from anytree.search import findall_by_attr
from anytree.walker import Walker
import random
import javalang

def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'  # node.pop()
    elif isinstance(node, javalang.ast.Node):
        token = node.__class__.__name__

    return token

def get_children(root):
    if isinstance(root, javalang.ast.Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))

def get_trees(current_node, parent_node, order):
    
    token, children = get_token(current_node), get_children(current_node)
    node = Node([order,token], parent=parent_node, order=order)

    for child_order in range(len(children)):
        get_trees(children[child_order], node, order+str(int(child_order)+1))

def get_path_length(path):
    """Calculating path length.
    Input:
    path: list. Containing full walk path.

    Return:
    int. Length of the path.
    """
    
    return len(path)

def get_path_width(raw_path):
    """Calculating path width.
    Input:
    raw_path: tuple. Containing upstream, parent, downstream of the path.

    Return:
    int. Width of the path.
    """
    
    return abs(int(raw_path[0][-1].order)-int(raw_path[2][0].order))

def hashing_path(path, hash_table):
    """Calculating path width.
    Input:
    raw_path: tuple. Containing upstream, parent, downstream of the path.

    Return:
    str. Hash of the path.
    """
    
    if path not in hash_table:
        hash = random.getrandbits(128)
        hash_table[path] = str(hash)
        return str(hash)
    else:
        return hash_table[path]
    
def get_node_rank(node_name, max_depth):
    """Calculating node rank for leaf nodes.
    Input:
    node_name: list. where the first element is the string order of the node, second element is actual name.
    max_depth: int. the max depth of the code.

    Return:
    list. updated node name list.
    """
    while len(node_name[0]) < max_depth:
        node_name[0] += "0"
    return [int(node_name[0]),node_name[1]]


def extracting_path(java_code, max_length, max_width, hash_path, hashing_table):
    """Extracting paths for a given json code.
    Input:
    json_code: json object. The json object of a snap program to be extracted.
    max_length: int. Max length of the path to be restained.
    max_width: int. Max width of the path to be restained.
    hash_path: boolean. if true, MD5 hashed path will be returned to save space.
    hashing_table: Dict. Hashing table for path.

    Return:
    walk_paths: list of AST paths from the json code.
    """
    
    # Initialize head node of the code.
    head = Node(["1",get_token(java_code)])
    
    # Recursively construct AST tree.
    
    for child_order in range(len(get_children(java_code))):

        get_trees(get_children(java_code)[child_order], head, "1"+str(int(child_order)+1))
    
    # Getting leaf nodes.
    leaf_nodes = findall_by_attr(head, name="is_leaf", value=True)
    
    # Getting max depth.
    max_depth = max([len(node.name[0]) for node in leaf_nodes])
    
    # Node rank modification.
    for leaf in leaf_nodes:
        leaf.name = get_node_rank(leaf.name,max_depth)
    
    walker = Walker()
    text_paths = []
    
    # Walk from leaf to target
    for leaf_index in range(len(leaf_nodes)-1):
        for target_index in range(leaf_index+1, len(leaf_nodes)):
            raw_path = walker.walk(leaf_nodes[leaf_index], leaf_nodes[target_index])
            
            # Combining up and down streams
            walk_path = [n.name[1] for n in list(raw_path[0])]+[raw_path[1].name[1]]+[n.name[1] for n in list(raw_path[2])]
            text_path = "@".join(walk_path)
            
            # Only keeping satisfying paths.
            if get_path_length(walk_path) <= max_length and get_path_width(raw_path) <= max_width:
                if not hash_path:
                # If not hash path, then output original text path.
                    text_paths.append(walk_path[0]+","+text_path+","+walk_path[-1])
                else:
                # If hash, then output hashed path.
                # first and last of walk_path shoud be hashed because it can have ',' when the parent of the node is "Literal"
                    text_paths.append(hashing_path(walk_path[0], hashing_table) +
                                      ","+hashing_path(text_path, hashing_table) +
                                      ","+hashing_path(walk_path[-1], hashing_table))
    
    return text_paths


def program_parser(func):
    tokens = javalang.tokenizer.tokenize(func)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree

module_path = os.path.abspath("../..")
if module_path not in sys.path:
    sys.path.append(module_path)

import src.make_logger as make_logger
from src.misc import *

def path_extract(main_df, code_path_length, code_path_width, save_dir, logger, logger_name, log_dir):
    logger.info("path_extract.start")
    st = time.time()
    def extract_process(assignment_id):
        assignment_st = time.time()

        assignment_df = main_df[main_df["AssignmentID"] == assignment_id]
        mp_logger = make_logger.make(logger_name+f"_{assignment_id}", time_filename=False, save_dir=log_dir)
        mp_logger.info(f"path_extract. assignment:{assignment_id}. start")

        save_path = os.path.join(save_dir, f"labeled_paths_{assignment_id}.tsv")
        if os.path.isfile(save_path):
            mp_logger.info(f"path_extract. already don:{save_path}")
            return
        
        parsed_code = []
        for c in list(assignment_df['Code']):
            try:
                parsed = program_parser(c)
            except:
                parsed = "Uncompilable"
            parsed_code.append(parsed)

        hashing_table = {}

        AST_paths = []
        for java_code in parsed_code:
            ast_path = extracting_path(java_code, max_length=code_path_length, max_width=code_path_width, hash_path=True, hashing_table=hashing_table)
            AST_paths.append(ast_path)

        # Storing the raw paths
        assignment_df["RawASTPath"] = ["@".join(A) for A in AST_paths]
        
        assignment_df.to_csv(save_path, sep="\t", header=True)

        mp_logger.info(f"path_extract. assignment:{assignment_id}. end. elapse:{format_elapse_from(assignment_st)}")

    import multiprocessing

    assignment_list = sorted(list(main_df["AssignmentID"].unique()))

    process_list = []
    for assignment in assignment_list:
        process = multiprocessing.Process(target=extract_process, args=[assignment])
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()

    logger.info(f"path_extract.end. elapse:{format_elapse_from(st)}")
