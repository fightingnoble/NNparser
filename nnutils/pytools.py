# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:47:13 2020
@author: LL

Updated on Mon Aug 31 20:09:31 2020
@Stephen Qian
"""

import torch
from nnutils.torchsummary import summary
# from torch.autograd import Variable
import nnutils.formattable as ft
import nnutils.dotGen as dg
from torchvision import models
import pandas as pd

def modelLst(ucfg, draw_graph=True):
    ''' ucfg: user's Config for the table output: nnname, BS, BPE '''

    nnname = ucfg['nnname']
    c, h, w = ucfg['channel'], ucfg['height'], ucfg['width']
    # produce config list of models per layer of the given nn model name
    isconv = True
    depth = 4
    col_names_noconv = (
        # add column to the table
        "coming",
        "going",
        "input_size",
        "output_size",
        "num_in",
        "num_out",
        "num_params",
        "gemm",
        "vect",
        "acti",
        # add backprop
        "gemmB",
        "vectB",
        "actiB",)

    # do NOT draw densenet201 or higher as it would take tremendous amount of time
    # densenet1xx are all allowed, although densenet169 would take about 3 hours
    if nnname.startswith('densenet'):
        draw_graph = False

    if nnname == 'newmodel':
        import sys
        sys.path.append("..")
        from newmodel import pymodel
        x, model, depth, isconv, y = pymodel()
        ucfg['draw_graph'] = draw_graph
        if isconv:
            result, g, hl_g, outputname = summary(model, x, depth=depth, branching=2, verbose=1, ucfg=ucfg)
        else:
            result, g, hl_g, outputname = summary(model, x, col_names=col_names_noconv, depth=depth, branching=2, verbose=1, ucfg=ucfg)
        sys.path.remove("..")
        g.render(outputname, cleanup=True, format='pdf')
        ms = str(result)
        return ms, depth, isconv, g, hl_g

    # vision models in torchvision
    if hasattr(models, nnname):
        model = getattr(models, nnname)()
        # draw dropout layer
        # training model instead of evaluation
        x = torch.rand(1, c, h, w)
        ucfg['draw_graph'] = draw_graph
        result, g, hl_g, outputname = summary(model, x, depth=depth, branching=2, verbose=1, ucfg=ucfg)
        g.render(outputname, cleanup=True, format='pdf')
        ms = str(result)
        return ms, depth, isconv, g, hl_g

    # TODO add YOLO v3

    if nnname == 'maskrcnn':
        # TODO add MUltiScaleRoIAlign size
        depth = 6
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=False)
        model.eval()
        x = [torch.rand(3, 800, 800)]
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(True)
        keys = ['boxes', 'labels', 'masks', 'image_id', 'area', 'iscrowd']
        val = [torch.FloatTensor([[100.,100.,400.,400.]]), torch.LongTensor(1), torch.ByteTensor(3, 800, 800),
               torch.LongTensor(1), torch.rand(1), torch.ByteTensor(1)]
        targets = [dict(zip(keys, val))]
        ucfg['draw_graph'] = draw_graph
        result, g, hl_g, outputname = summary(model, ([x, targets],), depth=depth, branching=2, verbose=1, ucfg=ucfg)

        # model.eval()
        # x = [torch.rand(1,3, 800, 800)]
        # result, g, hl_g, outputname = summary(model, (x,), depth=depth, branching=2, verbose=1, ucfg=ucfg)

        g.render(outputname, cleanup=True, format='pdf')
        ms = str(result)
        return ms, depth, isconv, g, hl_g

    if nnname == 'dlrm':
        depth = 2
        from torchmodels.dlrm.dlrm_s_pytorch import DLRM_Net
        import numpy as np
        # Setting for Criteo Kaggle Display Advertisement Challenge
        m_spa = 16
        ln_emb = np.array(
            [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683, 8351593, 3194, 27, 14992, 5461306, 10,
             5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572])
        ln_bot = np.array([13, 512, 256, 64, 16])
        ln_top = np.array([367, 512, 256, 1])
        model = DLRM_Net(m_spa, ln_emb, ln_bot, ln_top,
                         arch_interaction_op="dot",
                         sigmoid_top=ln_top.size - 2,
                         qr_operation=None,
                         qr_collisions=None,
                         qr_threshold=None,
                         md_threshold=None,
                         )
        x = torch.rand(2, ln_bot[0])  # dual samples
        lS_i = [torch.Tensor([0, 1, 2]).to(torch.long)] * len(ln_emb)  # numof indices >=1, but < ln_emb[i]
        lS_o = torch.Tensor([[0, 2]] * len(ln_emb)).to(torch.long)
        y = model(x, lS_o, lS_i)
        inst = (x, [lS_o, lS_i])
        col_names = col_names_noconv
        ucfg['draw_graph'] = draw_graph
        result, g, hl_g, outputname = summary(model, inst, col_names=col_names, depth=depth, branching=2, verbose=1, ucfg=ucfg)
        g.render(outputname, cleanup=True, format='pdf')
        ms = str(result)
        return ms, depth, isconv, g, hl_g

    if nnname == 'bert-base-cased':
        from transformers import AutoModel  # using Huggingface's version
        model = AutoModel.from_pretrained(nnname)
        # psudeo input
        inst = torch.randint(100, 2000, (1, 7))
        depth = 2
        col_names = col_names_noconv
        ucfg['draw_graph'] = draw_graph
        result, g, hl_g, outputname = summary(model, inst, col_names=col_names, depth=depth, branching=2, verbose=1, ucfg=ucfg)
        g.render(outputname, cleanup=True, format='pdf')
        ms = str(result)
        return ms, depth, isconv, g, hl_g

    if nnname == 'mymodel':
        depth = 2
        ## ===== To add a customized model ====
        # model cfgs
        N, D_in, H, D_out = 64, 1000, 100, 10
        # Create random input Tensors
        x = torch.randn(N, D_in)

        # define the NN model using pytorch operators.
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
        )

        ## ===== end of your codes  ======

        ucfg['draw_graph'] = draw_graph
        result, g, hl_g, outputname = summary(model, x, depth=depth, branching=2, verbose=1, ucfg=ucfg)
        g.render(outputname, cleanup=True, format='pdf')
        ms = str(result)
        return ms, depth, isconv, g, hl_g

    if nnname == 'lstm':
        depth = 2
        from torchmodels.lstm import LSTMNet
        x = torch.rand(2, 1).to(torch.long)
        model = LSTMNet()
        # add column to the table
        col_names = ("coming", "going",
                     "input_size", "output_size", "num_in", "num_out",
                     "num_params", "gemm", "vect", "acti", 'gemmB', 'vectB', 'actiB')
        ucfg['draw_graph'] = draw_graph
        result, g, hl_g, outputname = summary(model, x, col_names=col_names, depth=depth, branching=2, verbose=1, ucfg=ucfg)
        g.render(outputname, cleanup=True, format='pdf')
        ms = str(result)
        return ms, depth, isconv, g, hl_g

    if nnname == 'gru':
        depth = 2
        from torchmodels.gru import GRUNet
        x = torch.rand(2, 1).to(torch.long)
        model = GRUNet()
        # add column to the table
        col_names = ("coming", "going",
                     "input_size", "output_size", "num_in", "num_out",
                     "num_params", "gemm", "vect", "acti", 'gemmB', 'vectB', 'actiB')
        ucfg['draw_graph'] = draw_graph
        result, g, hl_g, outputname = summary(model, x, col_names=col_names, depth=depth, branching=2, verbose=1, ucfg=ucfg)
        g.render(outputname, cleanup=True, format='pdf')
        ms = str(result)
        return ms, depth, isconv, g, hl_g

    if nnname == 'ssd_mobilenet':
        depth = 3
        branching = 2
        from torchmodels.ssd.ssd_mobilenet_v1 import create_mobilenetv1_ssd
        model = create_mobilenetv1_ssd(91)
        model.eval()
        x = torch.rand(1, 3, 300, 300)
        ucfg['draw_graph'] = draw_graph
        result, g, hl_g, outputname = summary(model, (x,), depth=depth, branching=branching, verbose=1, ucfg=ucfg)
        g.render(outputname, cleanup=True, format='pdf')
        ms = str(result)
        if branching == 0:
            depth = 0
        return ms, depth, isconv, g, hl_g

    if nnname == 'ssd_r34':
        depth = 6
        branching = 2
        from torchmodels.ssd.ssd_r34 import SSD_R34
        model = SSD_R34()
        # model.eval()
        x = torch.rand(1, 3, 1200, 1200)
        ucfg['draw_graph'] = draw_graph
        result, g, hl_g, outputname = summary(model, (x,), depth=depth, branching=branching, verbose=1, ucfg=ucfg)
        g.render(outputname, cleanup=True, format='pdf')
        ms = str(result)
        return ms, depth, isconv, g, hl_g

    if nnname == 'gnmt':
        depth = 4
        col_names = col_names_noconv
        from torchmodels.seq2seq.models.gnmt import GNMT
        model_config = {'hidden_size': 1024,
                        'num_layers': 4,
                        'dropout': 0.2, 'batch_first': True,
                        'share_embedding': True}
        model = GNMT(vocab_size=2048, **model_config)
        model.eval()
        seqlen = 1
        batch = 2
        x = torch.rand(batch, seqlen).to(torch.long)
        srclen = torch.ones(batch).to(torch.long) * seqlen
        y = model(x, srclen, x)
        ucfg['draw_graph'] = draw_graph
        result, g, hl_g, outputname = summary(model, ([x, srclen, x],), col_names=col_names, depth=depth, branching=2, verbose=1, ucfg=ucfg)
        g.render(outputname, cleanup=True, format='pdf')
        ms = str(result)
        return ms, depth, isconv, g, hl_g

    if nnname == 'crnn':
        depth = 6
        from torchmodels.crnn import CRNN
        model = CRNN(32, 1, 37, 256)
        x = torch.rand(1, 1, 32, 100)
        model.eval()
        ucfg['draw_graph'] = draw_graph
        result, g, hl_g, outputname = summary(model, (x,), depth=depth, branching=2, verbose=1, ucfg=ucfg)
        g.render(outputname, cleanup=True, format='pdf')
        ms = str(result)
        return ms, depth, isconv, g, hl_g


# table gen
def tableGen(ms, depth, isconv):
    # produce table text list, and a summary header0 (merged header)
    layer = max(depth, 1)
    header0 = 'Layer Hierarchy,' * layer
    header = ''
    for i in range(layer):
        header += 'L{},'.format(i)
    # add column to the table
    header0 += 'Connection,' * 4
    header += 'input0, input1, output0, output1, '

    header += 'Channel, Height, Width,' * 2
    header0 += 'Input Dimension,' * 3 + 'Output Dimension,' * 3
    if isconv:
        header += 'Height, Width,'  # kernel
        header += 'X, Y,' * 2  # stride and padding
        header0 += 'Kernel,' * 2 + 'Stride,' * 2 + 'Padding,' * 2
    # else: # FC style networks
    header += 'Input, Output, Weight,'  # of parameters
    header += 'GEMM, ElemWise, Activation,' * 2 + '\n'
    header0 += 'Size of Parameters,' * 3 + 'Forward Ops,' * 3 + 'Backward Ops,' * 3 + '\n'
    return header0 + header + ms

def tableExport(ms, nnname, backward):
    ms = ms.split('\n')[:-1]  # remove the last row--None
    paralist = []
    for row in ms:
        lst=row.split(',')
        for i in range(len(lst)):
            lst[i] = int(lst[i]) if lst[i].strip().isnumeric() else lst[i].strip()
        paralist.append(lst)

    # MultiIndex columns
    headers = list(zip(*paralist[:2]))

    df = pd.DataFrame(paralist[2:], columns=pd.MultiIndex.from_tuples(headers))

    if not backward:
        df.drop('Backward Ops', axis=1, level=0, inplace=True)

    df.drop(df.columns[[-1]], axis=1, inplace = True) # remove last column
    paraout = './/outputs//torch//'+nnname+'.xlsx'
    df.to_excel(paraout, sheet_name='Details')

    # add summary sheet and formatting
    ft.SumAndFormat(paraout, df)

