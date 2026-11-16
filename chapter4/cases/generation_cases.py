import ast
import gc
from pathlib import Path
import torch
PARDEN=Path(__file__).parent/"references/PARDEN"

def official_functions():
    names={'flush_memory','templatize','templatize_for_repeat','generate_mini_batch','get_windowed_bleu','get_tpr_fpr'}
    tree=ast.parse((PARDEN/'utils.py').read_text())
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    namespace={'torch':torch,'gc':gc}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(PARDEN/'utils.py'),'exec'),namespace)
    return namespace
