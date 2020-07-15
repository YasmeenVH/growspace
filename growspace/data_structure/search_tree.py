
class Node(object):
    def __init__(self, data = None):
        self.data = data
        self.left = None
        self.right = None

class BST(object):
    def __init__(self):
        self.root = None

    def insert(self,data):
        if self.root is None:
            self.root = Node(data)
        else:
            self._insert(data, self.root)

    def _insert(self, data, cur_node):
        if data < cur_node.data:
            if cur_node.left is None:
                cur_node.left = Node(data)
            else:
                self._insert(data, cur_node.left)
        elif data >= cur_node.data:       # allow for duplicates since we may have same X values
            if cur_node.right is None:
                cur_node.right = Node(data)
            else:
                self._insert(data, cur_node.right)
        else:
            print('The X value does not fit in the tree')

    def find(self, data):
        if self.root:
            is_found = self._find(data, self.root)
            if is_found:
                return True
            return False
        # this is when tree is empty
        else:
            return None

    def _find(self, data, cur_node):
        if data > cur_node.data and cur_node.right:
            return self._find(data, cur_node.right)
        elif data < cur_node.data and cur_node.left:
            return self._find(data, cur_node.left)
        if data == cur_node.data:
            return True

    def key_range(self, data, r1, r2):
        keys = []
        if data is None:
            return

        if r1 < data.data:
            self.key_range(data.left, r1, r2)

        if r1 <= data.data and r2 >= data.data:
            keys.append(data)
            return keys

        if r2 > data.data:
            self.key_range(data.right, r1, r2)

    



