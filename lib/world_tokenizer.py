"""Trie-based byte-level токенизатор RWKV World (порт из rwkv_v7_demo.py)."""

class RWKV_WORLD_TOKENIZER:
    def __init__(self, file_name):
        self.idx2token = {}
        sorted_tokens = []
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted_tokens += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]

        for i in reversed(range(len(sorted_tokens))):
            s = sorted_tokens[i]
            if len(s) >= 2:
                s0 = int(s[0]); s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encode_bytes(self, src: bytes):
        src_len = len(src)
        tokens = []
        i = 0
        while i < src_len:
            s = src[i:i + 1]
            if i < src_len - 1:
                s1 = int(src[i + 1]); s0 = int(src[i])
                if s1 in self.good[s0]:
                    sss = src[i:i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except StopIteration:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens

    def decode_bytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encode_bytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decode_bytes(tokens).decode('utf-8', errors='replace')
