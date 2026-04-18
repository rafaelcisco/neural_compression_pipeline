from .tree import Node, AdaptiveHuffmanTree


class AdaptiveHuffmanEncoder(AdaptiveHuffmanTree):

    # Initializes the encoder.
    def __init__(self):
        super().__init__()

    # Converts a character into binary form.
    def char_to_bits(self, ch: str) -> str:
        return format(ord(ch), "08b")

    # Builds the code for a node based on its path in the tree.
    def get_code(self, node: Node) -> str:
        bits = []
        current = node

        while current.parent is not None:
            if current.parent.left is current:
                bits.append("0")
            else:
                bits.append("1")
            current = current.parent

        return "".join(reversed(bits))

    # Encodes the full text into a compressed bitstring.
    def encode(self, text: str) -> str:
        encoded_bits: list[str] = []

        for ch in text:
            if ch in self.symbol_nodes:
                node = self.symbol_nodes[ch]
                encoded_bits.append(self.get_code(node))
                self.update_tree(node)
            else:
                encoded_bits.append(self.get_code(self.nyt))
                encoded_bits.append(self.char_to_bits(ch))

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

        return "".join(encoded_bits)


# Compresses text and returns the encoded data with basic metrics.
def compress(text: str) -> dict:
    if not isinstance(text, str):
        raise TypeError("compress(text) expects a string.")

    encoder = AdaptiveHuffmanEncoder()
    compressed_data = encoder.encode(text)

    original_bits = len(text) * 8
    compressed_bits = len(compressed_data)

    compression_ratio = (
        compressed_bits / original_bits if original_bits > 0 else 0.0
    )

    return {
        "compressed_data": compressed_data,
        "original_text": text,
        "original_bits": original_bits,
        "compressed_bits": compressed_bits,
        "compression_ratio": compression_ratio,
    }