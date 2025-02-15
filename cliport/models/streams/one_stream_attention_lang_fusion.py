"""Attention module."""

import pdb
import cliport.models as models
from cliport.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusion
import re


class OneStreamAttentionLangFusion(TwoStreamAttentionLangFusion):
    """Attention (a.k.a Pick) module with language features fused at the bottleneck."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        self.keywords = {'on', 'in', 'to', 'into', 'from'}
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        print(f"Attn FCN: {stream_one_fcn}")

    def attend(self, x, l):
        # pdb.set_trace()
        sentence = re.sub(r'^\S+\s*', '', l)
        pattern = r'\b(' + '|'.join(self.keywords) + r')\b'
        match = re.search(pattern, sentence)
        if match:
            start_index = match.start()
            l = sentence[:start_index].strip()
        x = self.attn_stream_one(x, l)
        return x