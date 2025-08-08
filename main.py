
from dispatcher import Dispatcher
from llm_main import LLMRouter
import sys
import os
import json
from infer import greedy_decode, MiniTransformer
import torch
import sentencepiece as spm

class Classifier:
    def __init__(self):
        # Load model and tokenizer once for efficiency
        self.tokenizer_path = 'tokenizer/tokenizer.model'
        self.model_path = 'saved/best_model.pt'
        self.max_len = 512
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.tokenizer_path)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        vocab_size = self.sp.GetPieceSize()
        self.model = MiniTransformer(
            vocab_size=vocab_size,
            d_model=checkpoint['model_state'][list(checkpoint['model_state'].keys())[0]].shape[1],
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            max_len=self.max_len,
            pad_idx=self.sp.PieceToId('[PAD]')
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])

    def classify(self, raw_query):
        src_ids = [self.sp.PieceToId('[BOS]')] + self.sp.EncodeAsIds(raw_query) + [self.sp.PieceToId('[EOS]')]
        src_ids = src_ids[:self.max_len]
        out_ids = greedy_decode(self.model, src_ids, self.sp, self.max_len, self.device)
        tokens = [self.sp.IdToPiece(i) for i in out_ids[1:]]  # exclude BOS
        text = self.sp.DecodePieces(tokens).strip()
        KNOWN_FUNCTIONS = {"Asaoka_data", "reporter_Asaoka", "plot_combi_S", "SM_overview"}
        # Try JSON first
        try:
            result = json.loads(text)
            func_name = result.get('function', None)
            if func_name:
                print(f"[DEBUG] Classifier model output (JSON): {func_name}")
                return func_name, func_name
        except Exception:
            pass
        # Fallback: treat plain text as function name if it matches known functions
        if text in KNOWN_FUNCTIONS:
            print(f"[DEBUG] Classifier model output (plain text): {text}")
            return text, text
        print(f"[ERROR] Classifier model failed to parse output: {text}")
        return None, None

if __name__ == "__main__":
    router = LLMRouter(max_turns=3)
    classifier = Classifier()
    disp = Dispatcher(classifier, router)

    while True:
        raw = input(">> ")
        print(f"[DEBUG] User input: {raw}")
        func_name, func = disp.classify(raw)
        print(f"[DEBUG] Classifier output: func_name={func_name}, func={func}")
        if func_name:
            print(f"[DEBUG] Gathering parameters for function: {func_name}")
            params = disp.gather_params(raw, func_name)
            print(f"[DEBUG] Gathered params: {params}")
            out = disp.run_function(func_name, params) if params else None
            print(f"[DEBUG] Function output: {out}")
        else:
            print(f"[DEBUG] No function matched. Sending raw query to LLM.")
            params, out = {}, None
        print(f"[DEBUG] Dispatching to LLMRouter: func_name={func_name}, params={params}, out={out}")
        disp.build_and_send(raw, func_name, params, out)
