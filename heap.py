"""
heap.py

Max heap implementation with operations to decrease the key/priority for some item.
This code follows the implementation from Robert Sedgwick and Kevin Wayne avaialble at
https://algs4.cs.princeton.edu/code/edu/princeton/cs/algs4/Heap.java.html. For information
about sink and swim methods, see https://algs4.cs.princeton.edu/24pq
"""


class Heap:
    def __init__(self):
        self.pq = []  # Keys
        self.item = []  # Items
        # for item i, qp[i] = j, where item[j] = i, pq[j] = key(i)
        self.qp = {}

    def __len__(self):
        return len(self.pq)

    def less(self, i, j):
        return self.pq[i] < self.pq[j]

    def push(self, key, item):
        if item in self.qp:
            raise ValueError(f"{item=} already inserted")
        self.pq.append(key)
        self.item.append(item)
        self.qp[item] = len(self) - 1
        self.swim(len(self) - 1)

    def pop(self):
        if len(self) == 0:
            raise IndexError("Trying to pop from empty heap.")
        item = self.item[0]
        self.exch(0, len(self) - 1)
        self.qp.pop(item)
        item = self.item.pop()
        key = self.pq.pop()
        self.sink(0)
        return key, item

    def decrease(self, decrease, item):
        idx = self.qp.get(item, None)
        if idx is not None:
            self.pq[idx] -= decrease
            self.sink(idx)

    def swim(self, k):
        while k > 0 and self.less((k - 1) // 2, k):
            self.exch(k, (k - 1) // 2)
            k = (k - 1) // 2

    def sink(self, k):
        while (j := 2 * k + 1) < len(self):
            if j < len(self) - 1 and self.less(j, j + 1):
                j += 1
            if not self.less(k, j):
                break
            self.exch(k, j)
            k = j

    def exch(self, i, j):
        self.qp[self.item[i]], self.qp[self.item[j]] = (
            self.qp[self.item[j]],
            self.qp[self.item[i]],
        )
        self.pq[i], self.pq[j] = self.pq[j], self.pq[i]
        self.item[i], self.item[j] = self.item[j], self.item[i]
