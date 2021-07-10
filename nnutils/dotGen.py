# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from graphviz import Digraph
from hiddenlayer.graph import Graph, Node
from typing import *
from .layer_info import LayerInfo

def make_dot(var, params: dict = None, module_info_dict: Mapping[int, LayerInfo] = None):
    """ Produces Graphviz representation of PyTorch graph
    Blue nodes are the Variables that require grad, orange ones are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    module_list_input = []
    if module_info_dict is not None:
        for info in module_info_dict.values():
            module_list_input += info.inputs_list

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    # origin
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    hl_g = Graph()
    seen = set()

    def size_to_str(size):
        return '(' + ', '.join(['%d' % v for v in size]) + ')'

    def add_nodes(var, module_name=""):
        if var not in seen:
            # saved tensors
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                hl_n = Node(uid=str(id(var)), name=None, output_shape=var.size(), op="")
                hl_g.add_node(hl_n)

            # AccumulateGrad has variable attribution pointed to leaf tensor
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                # Replace grad_accumulator by `.variable` using it's name and shape
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
                hl_n = Node(uid=str(id(var)), name=node_name, output_shape=u.size(), op="")
                hl_g.add_node(hl_n)
            # Backward object
            else:
                no_module_path = False
                if module_info_dict is not None:
                    # module_name = "{}\n".format(str(module_info_dict[id(var)])) if id(var) in module_info_dict.keys() else module_name
                    if id(var) in module_info_dict.keys():
                        module_name = "{}\n".format(str(module_info_dict[id(var)]))

                    if id(var) not in module_info_dict.keys() and var in module_list_input:
                        no_module_path = True

                ndname = str(type(var).__name__)
                if no_module_path:
                    ndname = ndname.replace('Backward', '')
                else:
                    ndname = module_name + ndname.replace('Backward', '')
                dot.node(str(id(var)), ndname)
                hl_n = Node(uid=str(id(var)), name=ndname, op="")
                hl_g.add_node(hl_n)
                # dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    # None, Backward object, AccumulateGrad
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        hl_g.add_edge_by_id(str(id(u[0])), str(id(var)))
                        add_nodes(u[0], module_name)
            # For custom autograd functions
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    hl_g.add_edge_by_id(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    resize_graph(dot)
    return dot, hl_g


def graph(y, params=None, module_info_dict: Mapping[int, LayerInfo] = None):
    dot, hl_g = make_dot(y, params, module_info_dict)

    return dot, hl_g


# ----------------------------------------------------------------
# from torchviz: https://github.com/szagoruyko/pytorchviz
def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)
# ----------------------------------------------------------------

def draw(nnname, x, y, model, module_info_dict: Mapping[int, LayerInfo] = None) -> Tuple[Digraph, Graph, str]:
    params = dict(list(model.named_parameters()) + [('x{}'.format(p_i), p) for p_i, p in enumerate(x)])
    if isinstance(y, dict):
        for k, v in y.items():
            if v.grad_fn:
                outputname = './/outputs//torch//' + nnname + '_' + k
                g, hl_g = graph(v, params=params, module_info_dict=module_info_dict)
    elif 'ssd_mo' in nnname:
        yname = ('scores', 'boxes')
        for v, name in zip(y, yname):
            outputname = './/outputs//torch//' + nnname + '_' + name
            g, hl_g = graph(v, params=params, module_info_dict=module_info_dict)
    elif 'ssd_r' in nnname:
        yname = ('boxes', 'label', 'scores')
        for v, name in zip(y, yname):
            if v[0].grad_fn:
                outputname = './/outputs//torch//' + nnname + '_' + name
                g, hl_g = graph(v[0], params=params, module_info_dict=module_info_dict)
    elif 'crnn' == nnname:
        print()  # try: except CalledProcessError:
    else:  # general case, plot using the first output
        # selection operation introduces two selection layer
        v = y[0]
        if isinstance(v[0], torch.Tensor):
            if v[0].grad_fn:
                outputname = './/outputs//torch//' + nnname
                # try:
                g, hl_g = graph(v[0], params=params, module_info_dict=module_info_dict)
                # except:
                #     print('Failed to generate model Graph')
    return g, hl_g, outputname
