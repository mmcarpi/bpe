import random
import unittest
from heap import Heap


def isHeap(arr, i, n):
    if i >= int((n - 1) / 2):
        return True

    if (
        (arr[i] >= arr[2 * i + 1])
        and (arr[i] >= arr[2 * i + 2])
        and isHeap(arr, 2 * i + 1, n)
        and isHeap(arr, i * 2 + 2, n)
    ):
        return True


class TestHeap(unittest.TestCase):
    def build_heap(self, n, key=None):
        key = key or (lambda x: x)
        heap = Heap()
        for i in range(n):
            heap.push(key(i), i)
        return heap

    def heap_to_list(self, heap):
        output = []
        while heap:
            output.append(heap.pop()[1])
        return output

    def test_insert(self):
        n = 200
        heap = self.build_heap(n)
        self.assertTrue(isHeap(heap.pq, 0, len(heap)))
        self.assertEqual(n, len(heap))

    def test_insert_pop_1(self):
        n = 200
        heap = self.build_heap(n)
        self.assertEqual(heap.pop(), (n-1, n-1))
        self.assertTrue(isHeap(heap.pq, 0, len(heap)))
        output = self.heap_to_list(heap)

        self.assertEqual(n-1, len(output))
        self.assertEqual(output, list(reversed(range(n-1))))

    def test_insert_pop_2(self):
        n = 200
        heap = self.build_heap(n)

        for i in range(100):
            self.assertEqual(heap.pop(), (n-i-1, n-i-1))
            self.assertTrue(isHeap(heap.pq, 0, len(heap)))

        for i in range(heap.pq[0]+1, heap.pq[0]+51):
            heap.push(-i if i % 2 else i, i)
            self.assertTrue(isHeap(heap.pq, 0, len(heap)))
            heap.pop()
            self.assertTrue(isHeap(heap.pq, 0, len(heap)))

    def test_decrease_1(self):
        heap = self.build_heap(10)
        heap.decrease(1.5, 9)
        heap.pop()
        heap.decrease(1, 9)
        heap.pop()
        heap.decrease(2.5, 4)
        heap.pop()
        heap.decrease(5.5, 6)

        self.assertTrue(isHeap(heap.pq, 0, len(heap.pq)))
