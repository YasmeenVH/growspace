
class Node(object):
    def __init__(self, value=None):
        self.value = value
        self.left = None
        self.right = None

class BST(object):
    def __init__(self):
        self.root = None

    def insert(self, data):
        if self.root is None:
            self.root = Node(data)
        else:
            self._insert(data, self.root)

    def _insert(self, data, cur_node):
        if data[0] < cur_node.value[0]:
            if cur_node.left is None:
                cur_node.left = Node(data)
            else:
                self._insert(data, cur_node.left)
        elif data[0] >= cur_node.value[0]:       # allow for duplicates since we may have same X values
            if cur_node.right is None:
                cur_node.right = Node(data)
            else:
                self._insert(data, cur_node.right)
        else:
            print('The X value does not fit in the tree')

 # no need for this now

    def find(self, data):
        if self.root:
            is_found = self._find(data, self.root)
            if is_found:
                return True
            return False
        # this is when tree is empty
        else:
            return None

# no need for this now

    def _find(self, data, cur_node):
        if data > cur_node.data and cur_node.right:
            return self._find(data, cur_node.right)
        elif data < cur_node.data and cur_node.left:
            return self._find(data, cur_node.left)
        if data == cur_node.data:
            return True

    def key_range(self, cur_node, r1, r2):
        keys = []
        if cur_node is None:
            return

        if r1 < cur_node.value[0]:
            self.key_range(cur_node.left, r1, r2)

        if r1 <= cur_node.value[0] <= r2:
            keys.append(cur_node.value)
            return keys

        if r2 > cur_node.value[0]:
            self.key_range(cur_node.right, r1, r2)

## no need for the recursive _keyrange function
    def _key_range(self, cur_node, r1,r2):
        keys = []

        if cur_node is None:
            return
        if r1 < cur_node.value[0]:
            self._key_range(cur_node.left, r1, r2)

        if r1 <= cur_node.value[0] <= r2:
            keys.append(cur_node.value)
            return keys,

        if r2 > cur_node.value[0]:
            self._key_range(cur_node.right, r1, r2)






