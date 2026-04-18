from .tree import Node, AdaptiveHuffmanTree


class AdaptiveHuffmanDecoder(AdaptiveHuffmanTree):

    # Initializes the decoder.
    def __init__(self):
        super().__init__()

    # Converts binary data back into a character.
    def bits_to_char(self, bits: str) -> str:
        return chr(int(bits, 2))

    # Decodes a compressed bitstring back into text.
    def decode(self, bitstring: str) -> str:
        if not isinstance(bitstring, str):
            raise TypeError("decode(bitstring) expects a string of bits.")

        if any(bit not in {"0", "1"} for bit in bitstring):
            raise ValueError("bitstring must contain only '0' and '1'.")

        result: list[str] = []
        i = 0
        current = self.root

        while True:
            if current.is_leaf():
                if current is self.nyt:
                    if i == len(bitstring):
                        break

                    if i + 8 > len(bitstring):
                        raise ValueError("Incomplete bitstring: missing bits for new symbol.")

                    ch_bits = bitstring[i:i + 8]
                    i += 8
                    ch = self.bits_to_char(ch_bits)
                    result.append(ch)

                    old_nyt = self.nyt
                    new_nyt = Node(
                        weight=0,
                        symbol=None,
                        parent=old_nyt,
                        order=old_nyt.order - 2,
                    )
                    new_symbol = Node(
                        weight=0,
                        symbol=ch,
                        parent=old_nyt,
                        order=old_nyt.order - 1,
                    )

                    old_nyt.left = new_nyt
                    old_nyt.right = new_symbol
                    old_nyt.symbol = None

                    self.nyt = new_nyt
                    self.symbol_nodes[ch] = new_symbol

                    self.update_tree(new_symbol)
                    current = self.root
                else:
                    result.append(current.symbol)
                    self.update_tree(current)
                    current = self.root
            else:
                if i >= len(bitstring):
                    break

                bit = bitstring[i]
                i += 1

                if bit == "0":
                    current = current.left
                else:
                    current = current.right

        return "".join(result)


# Decompresses encoded data and returns the original text.
def decompress(compressed_data: str) -> str:
    decoder = AdaptiveHuffmanDecoder()
    return decoder.decode(compressed_data)