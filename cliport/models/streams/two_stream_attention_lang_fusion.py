from cliport.models.core.clip import tokenize
import numpy as np
import torch
import torch.nn.functional as F
import pdb

from cliport.models.core.attention import Attention
import cliport.models as models
import cliport.models.core.fusion as fusion
# from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import re


class TwoStreamAttentionLangFusion(Attention):
    """Two Stream Language-Conditioned Attention (a.k.a Pick) module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        # pdb.set_trace()
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)

        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def attend(self, x, l):
        x1 = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, l)
        x = self.fusion(x1, x2)
        return x

    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        # pdb.set_trace()
        in_data = np.pad(inp_img, self.padding, mode='constant')
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)  # [B 6 W H]
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)

        # Forward pass.
        # pdb.set_trace()
        logits = []
        for x in in_tens:
            lgts = self.attend(x, lang_goal) # [B 1 W H]
            logits.append(lgts)
        logits = torch.cat(logits, dim=0)
        # pdb.set_trace()

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output


class TwoStreamAttentionLangFusionLat(TwoStreamAttentionLangFusion):
    """Language-Conditioned Attention (a.k.a Pick) module with lateral connections."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        self.keywords = {'on', 'in', 'to', 'into', 'from'}
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def attend(self, x, l):
        x1, lat = self.attn_stream_one(x)
        # pdb.set_trace()
        # sentence = re.sub(r'^\S+\s*', '', l)
        # pattern = r'\b(' + '|'.join(self.keywords) + r')\b'
        # match = re.search(pattern, sentence)
        # if match:
        #     start_index = match.start()
        #     l = sentence[:start_index].strip()

        x2 = self.attn_stream_two(x, lat, l)  # into cliport models
        x = self.fusion(x1, x2)
        return x
    
# It is also in the front part of context.
# QA_input = {
#     'question': 'What should I pick up?',
#     'context': 'push the pile of blue blocks into the red square'
# }
# QA_input = {
#     'question': 'Where should I place the object?',
#     'context': l
# }
# QA_input = {
#     'question': 'Do you know the preposition?',
#     'context': l
# }
# QA_input = {
#     'question': 'Where to place?',
#     'context': l
# }
# put the yellow block on the green and gray blocks
# put all the pepsi next box objects in the brown box
# pack all the brown and gray blocks into the brown box
# push the pile of blue blocks into the red square

 # model_name = "deepset/roberta-base-squad2"
        # nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
        # with torch.no_grad():
        #     QA_input = {
        #         'question': 'What should I pick up? It is also in the front part of context.',
        #         'context': l
        #     }
        #     answer = self.nlp(QA_input)
        #     l = answer['answer']
        # model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # pdb.set_trace()