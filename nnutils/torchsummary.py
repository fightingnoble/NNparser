""" torchsummary.py """

from .model_statistics import *
from torch.utils.hooks import RemovableHandle
from hiddenlayer.graph import Graph, Node
from graphviz import Digraph
from .dotGen import draw
from typing import Tuple
# Some modules do the computation themselves using parameters
# or the parameters of children. Treat these as layers.
LAYER_MODULES = (torch.nn.MultiheadAttention,)
INPUT_SIZE_TYPE = Sequence[Union[int, Sequence[Any], torch.Size]]


def summary(
        model: nn.Module,
        input_data: Union[torch.Tensor, torch.Size, Sequence[torch.Tensor], INPUT_SIZE_TYPE],
        *args: Any,
        batch_dim: int = 0,
        branching: int = 1,  # 0: no branch,1:branch line,2: branch
        col_names: Sequence[str] = (
                # add column to the table
                "coming",
                "going",
                "input_size",
                "output_size",
                "kernel_size",
                "stride_size",
                "pad_size",
                "num_in",
                "num_out",
                "num_params",
                "gemm",
                "vect",
                "acti",
                # add backprop
                "gemmB",
                "vectB",
                "actiB",),
        col_width: int = 25,
        depth: int = 3,
        device: Optional[torch.device] = None,
        dtypes: Optional[List[torch.dtype]] = None,
        verbose: int = 1,
        ucfg: {},  # user config: name, bs, bpe
        # ? ucfg: Optional[Dict[str,(str,int)]] = {}, # user config: name, bs, bpe
        **kwargs: Any
) -> Tuple[ModelStatistics, Digraph, Graph, str]:
    """
    Summarize the given PyTorch model. Summarized information includes:
        # ? Layer names
        1) output shape,
        2) kernel shape,
        3) number of parameters
        4) number of operations (Mult-Adds)
    Arguments:
        model (nn.Module): PyTorch model to summarize
        input_data (Sequence of Sizes or Tensors):
            Example input tensor of the model (dtypes inferred from model input).
            - OR -
            Shape of input data as a List/Tuple/torch.Size (dtypes must match model
            input, default is FloatTensors).
        batch_dim (int): batch_dimension of input data
        branching (bool): Whether to use the branching layout for the printed output.
        col_names (Sequence[str]): specify which columns to show in the output.
            Currently supported:
            ('output_size', 'num_params', 'kernel_size', 'mult_adds')
        col_width (int): width of each column
        depth (int): number of nested layers to traverse (e.g. Sequentials)
        device (torch.Device): Uses this torch device for model and input_data.
            Defaults to torch.cuda.is_available().
        dtypes (List[torch.dtype]): for multiple inputs, specify the size of both inputs, and
            also specify the types of each parameter here.
        verbose (int):
            0 (quiet): No output
            1 (default): Print model summary
            2 (verbose): Show weight and bias layers in full detail
        args, kwargs: Other arguments used in `model.forward` function.
    """
    assert verbose in (0, 1, 2)
    summary_list = []  # type: List[LayerInfo]
    hooks = []  # type: List[RemovableHandle]
    idx = {}  # type: Dict[int, int]

    # here recursively calls apply_hooks
    apply_hooks(model, model, depth, summary_list, hooks, idx, batch_dim)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # single real input
    if isinstance(input_data, torch.Tensor):
        input_size = get_correct_input_sizes(input_data.size())
        x = [input_data.to(device)]
    # ====================== input with args : ([x],[args])
    elif isinstance(input_data, tuple):
        # x list
        inputx = input_data[0]
        x = []
        if not isinstance(inputx, list):
            input_size = get_correct_input_sizes(inputx.size())
            x = [inputx.to(device)]
        else:  # mutiple input in list
            # if all(isinstance(data, torch.Tensor) for data in inputx):  # real input
            #     # scanning all elements in the tuple
            #     input_sizes = [data.size() for data in inputx]  # type: ignore
            #     input_size = get_correct_input_sizes(input_sizes)
            #     x = [data.to(device) for data in inputx]
            # else:  # input shape: not used in this version
            #     for item in inputx:
            #         if dtypes is None:
            #             dtypes = [torch.float] * len(item)
            #             input_size = get_correct_input_sizes(item)
            #             x.append(get_input_tensor(input_size, batch_dim, dtypes, device))
            x = input_data_to(inputx, device)
        # args list
        if len(input_data) > 1:
            inputarg = input_data[1]
            if isinstance(inputarg, (tuple, list)):
                for arg in inputarg:
                    if isinstance(arg, list):
                        tmp = [data.to(device) for data in arg]
                        x.append(tmp)
                    else:
                        x.append(arg.to(device))
            else:
                x.append(inputarg.to(device))

    else:
        raise TypeError(
            "Input type is not recognized. Please ensure input_data is valid.\n"
            "For multiple inputs to the network, ensure input_data passed in is "
            "a sequence of tensors or a list of tuple sizes. If you are having trouble here, "
            "please submit a GitHub issue."
        )

    args = tuple([t.to(device) if torch.is_tensor(t) else t for t in args])
    kwargs = {k: kwargs[k].to(device) if torch.is_tensor(kwargs[k]) else k for k in kwargs}

    try:
        # with torch.no_grad():
        #     _ = model.to(device)(*x, *args, **kwargs)
        y = model.to(device)(*x, *args, **kwargs)
    except Exception:
        print(
            "Failed to run torchsummary, printing sizes of executed layers: {}".format(summary_list)
        )
        raise
    finally:
        for hook in hooks:
            hook.remove()

    module_input_list = []
    module_output_list = []
    for info in summary_list:
        module_input_list += [str(id(i)) for i in info.inputs_list]
        module_output_list += [str(id(o)) for o in info.outputs_list]

    # print('module_input_list:',module_input_list,'\n')
    # print('module_output_list',module_output_list, '\n')

    module_info_dict = {}
    for current_info in summary_list:
        for o in current_info.outputs_list:
            if id(o) not in module_info_dict.keys():
                module_info_dict[id(o)] = current_info
            else:
                orig_model = module_info_dict[id(o)].module
                submodules = [m for m in orig_model.modules() if m is not orig_model]
                if current_info.module in submodules and submodules:
                        module_info_dict[id(o)] = current_info
    # print(module_info_dict)

    if ucfg['draw_graph']:
        g, hl_g, outputname = draw(ucfg['nnname'], x, y, model, module_info_dict)
    else:
        g, hl_g, outputname = [None]*3

    # info.outputs_list
    no_module_path_mapping_oppose = {}
    for current_info in summary_list:
        for o in current_info.outputs_list:

            # current_info.going_list = [str(layer_info) for layer_info in summary_list if o in layer_info.inputs_list]
            # for layer_info in summary_list:
            #     if o in layer_info.inputs_list:
            #         current_info.going_list.append(str(layer_info))
            # elif id(o) in no_module_path_mapping:
            #     current_info.going_list.append(no_module_path_mapping[id(o)])
            #     no_module_path_mapping_oppose[id(o)] = str(current_info)
            def find_going(node):
                outgoing_l = hl_g.outgoing(node)
                if outgoing_l:
                    for outgoing_t in outgoing_l:
                        if outgoing_t.id in module_output_list:
                            current_info.going_list.append(outgoing_t.name.split('\n')[0])
                        else:
                            find_going(outgoing_t)
            find_going(hl_g[str(id(o))])

            def find_coming(node):
                income_l = hl_g.incoming(node)
                if income_l:
                    for income_t in income_l:
                        if income_t.id in module_output_list:
                            current_info.coming_list.append(income_t.name.split('\n')[0])
                        else:
                            find_coming(income_t)
            find_coming(hl_g[str(id(o))])

        # for i in current_info.inputs_list:
        #
        #     # current_info.coming_list = [str(layer_info) for layer_info in summary_list if i in layer_info.outputs_list]
        #
        #     def find_coming(i):
        #         income_l = hl_g.incoming(i)
        #         if income_l:
        #             for income_t in income_l:
        #                 if income_t.id in module_output_list:
        #                     current_info.coming_list.append(income_t.name.split('\n')[0])
        #                 else:
        #                     find_coming(income_t)
        #     find_coming(hl_g[str(id(i))])

            # for layer_info in summary_list:
            #     if i in layer_info.outputs_list:
            #         current_info.coming_list.append(str(layer_info))
                # elif id(i) in layer_info.going_list:
                #     current_info.coming_list.append()



    # 1,...,depth
    for current_depth in range(1,depth+1):
        current_depth_list = [info for info in summary_list if info.depth==current_depth]
        for current_info in current_depth_list:
            print("layer:", str(current_info),
                  "inputs:", current_info.inputs_list,
                  "outputs:", current_info.outputs_list,
                  "come in:",current_info.coming_list,
                  "go to:",current_info.going_list, '\n')

    formatting = FormattingOptions(branching, depth, verbose, col_names, col_width)
    formatting.set_layer_name_width(summary_list)
    results = ModelStatistics(summary_list, input_size, formatting, ucfg)
    return results, g, hl_g, outputname


def input_data_to(
        inputs: CORRECTED_INPUT_SIZE_TYPE,
        device: torch.device,
):
    """ Get input_tensor with batch size 2 for use in model.forward() """
    def f(inputs):
        if isinstance(inputs, (list, tuple)):
            return [f(i) for i in inputs]
        elif isinstance(inputs, tuple):
            return (f(i) for i in inputs)
        elif isinstance(inputs, dict):
            return {k: f(v) for k, v in inputs.items()}
        elif isinstance(inputs, torch.Tensor):
            return inputs.to(device)
        else:
            return inputs
    return f(inputs)

def get_input_tensor(
        input_size: CORRECTED_INPUT_SIZE_TYPE,
        batch_dim: int,
        dtypes: List[torch.dtype],
        device: torch.device,
) -> List[torch.Tensor]:
    """ Get input_tensor with batch size 2 for use in model.forward() """
    x = []
    for size, dtype in zip(input_size, dtypes):
        # list for args, tuple for inputs in DLRM, 0619
        if isinstance(size, list):
            # for list of tensors:input in [(,)]
            if isinstance(size[0], tuple):
                tmp = []
                for si in size[0]:
                    if isinstance(si, tuple):  # if a high-dim tensor, int/float random?
                        input_tensor = torch.rand(*si)  # to do: integer?
                    else:
                        input_tensor = torch.randint(0, 1, (1, si))
                        input_tensor = input_tensor[0]  # 1d tensor
                    result = input_tensor.to(device).type(torch.long)
                    tmp.append(result)
                x.append(tmp)
                continue
            else:
                input_tensor = torch.randint(0, 1, size)  # for DLRM only
                result = input_tensor.to(device).type(torch.long)
                x.append(result)
                continue
        # add batch_size of 2 for BatchNorm
        if isinstance(size, (tuple)):
            # Case: input_tensor is a list of dimensions
            input_tensor = torch.rand(*size)
            if size[0] == 1:
                input_tensor = input_tensor[0]
            input_tensor = input_tensor.unsqueeze(dim=batch_dim)
            input_tensor = torch.cat([input_tensor] * 2, dim=batch_dim)
            result = input_tensor.to(device).type(dtype)
            if isinstance(result, torch.Tensor):
                x.append(result)
    return x


def get_correct_input_sizes(input_size: INPUT_SIZE_TYPE) -> CORRECTED_INPUT_SIZE_TYPE:
    """ Convert input_size to the correct form, which is a list of tuples.
    Also handles multiple inputs to the network. """

    def flatten(nested_array: INPUT_SIZE_TYPE) -> Generator:
        """ Flattens a nested array. """
        for item in nested_array:
            if isinstance(item, (list, tuple)):
                yield from flatten(item)
            else:
                yield item

    # assert input_size is not None
    # assert all(size > 0 for size in flatten(input_size)), "Negative size found in input_data."

    if isinstance(input_size, list) and isinstance(input_size[0], int):
        return [tuple(input_size)]
    if isinstance(input_size, list):
        return input_size
    if isinstance(input_size, tuple) and isinstance(input_size[0], tuple):
        return list(input_size)
    return [input_size]


# batch size is 1 regardless of user input, it will only take effect when generating xlsx file
def apply_hooks(
        module: nn.Module,
        orig_model: nn.Module,
        depth: int,
        summary_list: List[LayerInfo],
        hooks: List[RemovableHandle],
        idx: Dict[int, int],
        batch_dim: int,
        curr_depth: int = 0,
) -> None:
    '''
    If input_data is provided, recursively adds hooks to all layers of the model.
    Else, fills summary_list with layer info without computing a forward pass through the network.
    '''

    # Fallback is used if the layer's hook is never called, in Module Lists, for example.
    def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
        """ Create a LayerInfo object to aggregate information about that layer. """
        idx[curr_depth] = idx.get(curr_depth, 0) + 1
        # ! info here extract LayerInfo from that layer
        info = LayerInfo(module, curr_depth, idx[curr_depth])
        info.calculate_input_size(inputs, batch_dim)
        info.calculate_io_source(inputs, outputs)
        del inputs
        info.calculate_output_size(outputs, batch_dim)
        info.calculate_num_params()
        # * pass the info to external summary_list (in summary func scope)
        info.check_recursive(summary_list)
        summary_list.append(info)  # contains info of is_recursive in that layer

    submodules = [m for m in module.modules() if m is not orig_model]

    # module is not orig_model OR
    # module is LAYER_MODULES OR
    # submodules is empty (all modules are orig_model)
    if module != orig_model or isinstance(module, LAYER_MODULES) or not submodules:
        hooks.append(module.register_forward_hook(hook))

    if curr_depth <= depth:
        for child in module.children():
            apply_hooks(
                child, orig_model, depth, summary_list, hooks, idx, batch_dim,
                curr_depth + 1,
            )
