"""
llst.py

Implements a double linked list with support for arbitrary pop and insert operations.
"""

from typing import Any, List, Optional, Tuple


class Node:
    __slots__ = ("next", "prev", "item")

    def __init__(self, next, prev, item):
        self.next = next
        self.prev = prev
        self.item = item

    def __repr__(self):
        return "Node(%r, %r, %r)" % (self.next, self.prev, self.item)


class LinkedList:
    def __init__(self):
        self.node = []  # The nodes(next, prev, item)
        self.free = []  # Positions that are free in node
        self.head = -1  # first node, -1 if empty
        self.tail = -1  # last node, -1 if emnpty

    @classmethod
    def from_list(cls, ls: List[Any]):
        ll = cls()
        if ls:
            ll.node = [
                Node(next, prev, item)
                for (next, prev, item) in zip(
                    range(1, len(ls) + 1), range(-1, len(ls)), ls
                )
            ]
            ll.node[-1].next = -1
            ll.head = 0
            ll.tail = len(ls) - 1
        return ll

    def _getfreeidx(self) -> int:
        if self.free:
            return self.free.pop()
        self.node.append(None)
        return len(self.node) - 1

    def __len__(self):
        return len(self.node) - len(self.free)

    def append(self, item) -> int:
        idx = self._getfreeidx()
        if len(self) == 1:
            self.head = idx
            self.tail = idx
            self.node[idx] = Node(-1, -1, item)
        else:
            self.node[idx] = Node(-1, self.tail, item)
            if self.tail != -1:
                self.node[self.tail].next = idx
            self.tail = idx
        return idx

    def push(self, idx: int, item: Any) -> int:
        """
        Insert a new node with item after node at idx
        """
        node = self.node[idx]
        idx = self._getfreeidx()

        if node.next != -1:
            self.node[node.next].prev = idx

        self.node[idx] = Node(node.next, idx, item)
        node.next = idx

        if idx == self.tail:
            self.tail = idx
        return idx

    def pop(self, idx: int) -> Node:
        """
        Removes node at idx
        """
        self.free.append(idx)
        node = self.node[idx]

        if node.next != -1:
            self.node[node.next].prev = node.prev

        if node.prev != -1:
            self.node[node.prev].next = node.next

        if idx == self.head:
            self.head = node.next

        if idx == self.tail:
            self.tail = node.prev
        self.node[idx] = None
        return node

    def pair_starting_at(self, idx: int) -> Optional[Tuple[Any, Any]]:
        left = self.node[idx]
        if left.next != -1:
            right = self.node[left.next]
            return (left.item, right.item), idx

    def pair_ending_at(self, idx: int) -> Optional[Tuple[Any, Any]]:
        right = self.node[idx]
        if right.prev != -1:
            left = self.node[right.prev]
            return (left.item, right.item), right.prev

    def to_list(self) -> List[Any]:
        output = []
        idx = self.head
        while idx != -1:
            node = self.node[idx]
            output.append(node.item)
            idx = node.next
        return output
