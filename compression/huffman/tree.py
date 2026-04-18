from dataclasses import dataclass
from typing import Optional


@dataclass
class Node:
    weight: int
    symbol: Optional[str] = None
    parent: Optional["Node"] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    order: int = 0

    # Checks whether the current node is a leaf node.
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class AdaptiveHuffmanTree:

    # Initializes the shared tree structure.
    def __init__(self):
        self.max_order = 512
        self.root = Node(weight=0, symbol=None, order=self.max_order)
        self.nyt = self.root
        self.symbol_nodes: dict[str, Node] = {}

    # Traverses the tree and stores each node in a list.
    def collect_nodes(self, node: Optional[Node], nodes: list[Node]) -> None:
        if node is None:
            return
        nodes.append(node)
        self.collect_nodes(node.left, nodes)
        self.collect_nodes(node.right, nodes)

    # Finds a node with the same weight and highest order.
    def find_highest_order_same_weight(self, node: Node) -> Node:
        nodes: list[Node] = []
        self.collect_nodes(self.root, nodes)

        candidate = node
        for other in nodes:
            if (
                other.weight == node.weight
                and other.order > candidate.order
                and other is not node
                and other.parent is not node
                and node.parent is not other
            ):
                candidate = other

        return candidate

    # Swaps two nodes in the tree.
    def swap_nodes(self, a: Node, b: Node) -> None:
        if a is b or a.parent is None or b.parent is None:
            return

        a_parent = a.parent
        b_parent = b.parent

        if a_parent.left is a:
            a_parent.left = b
        else:
            a_parent.right = b

        if b_parent.left is b:
            b_parent.left = a
        else:
            b_parent.right = a

        a.parent, b.parent = b_parent, a_parent
        a.order, b.order = b.order, a.order

    # Updates the tree after a symbol is processed.
    def update_tree(self, node: Node) -> None:
        current = node

        while current is not None:
            highest = self.find_highest_order_same_weight(current)
            if highest is not current and highest is not current.parent:
                self.swap_nodes(current, highest)

            current.weight += 1
            current = current.parent