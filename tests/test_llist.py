import unittest
from llst import LinkedList


class TestList(unittest.TestCase):
    def test_build_llist(self):
        s = list(range(200))
        ll = LinkedList.from_list(s)

        self.assertEqual(s, ll.to_list())

    def test_pop(self):
        s = list(range(200))
        ll = LinkedList.from_list(s)

        ll.pop(2)
        ll.pop(3)
        s.pop(2)
        s.pop(2)
        self.assertEqual(s, ll.to_list())

    def test_pop_append(self):
        s = list(range(200))
        ll = LinkedList.from_list(s)

        ll.pop(2)
        ll.pop(3)
        idx = ll.push(1, 2)
        ll.push(idx, 3)

        self.assertEqual(s, ll.to_list())

        ll.pop(40)
        ll.pop(50)
        ll.pop(39)

        idx = ll.push(38, 39)
        idx = ll.push(idx, 40)
        idx = ll.push(49, 50)

        self.assertEqual(s, ll.to_list())
