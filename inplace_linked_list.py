# Template for linked list node class

class LinkedListNode:
    # __init__ will be used to make a LinkedListNode type object.
    def __init__(self, data, next=None):
        self.data = data
        self.next = next


# Template for the linked list
class LinkedList:
    # __init__ will be used to make a LinkedList type object.
    def __init__(self):
        self.head = None

    # insert_node_at_head method will insert a LinkedListNode at
    # head of a linked list.
    def insert_node_at_head(self, node):
        if self.head:
            node.next = self.head
            self.head = node
        else:
            self.head = node

    # create_linked_list method will create the linked list using the
    # given integer array with the help of InsertAthead method.
    def create_linked_list(self, lst):
        for x in reversed(lst):
            new_node = LinkedListNode(x)
            self.insert_node_at_head(new_node)

    # __str__(self) method will display the elements of linked list.
    def __str__(self):
        result = ""
        temp = self.head
        while temp:
            result += str(temp.data)
            temp = temp.next
            if temp:
                result += ", "
        result += ""
        return result


def reverse(head):
    prev = nxt = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    head = prev
    return head


def traverse_linked_list(head):
    current, nxt = head, None
    while current:
      nxt = current.next
      current = nxt


def reverse_between(head, left, right):
    cur = head
    left_node, prev_node = None, None
    next_node = None
    counter = 1
    while cur:
        if counter == left:
            left_node = cur
            if prev_node:
                prev_node.next = None
            break
        prev_node = cur
        cur = cur.next
        counter += 1
    while cur:
        if counter == right:
            next_node = cur.next
            cur.next = None
            break
        cur = cur.next
        counter += 1
    reversed_nodes_head = reverse(left_node)
    if prev_node:
        prev_node.next = reversed_nodes_head
    else:
        prev_node = reversed_nodes_head
        head = prev_node
    while prev_node:
        if prev_node.next is None:
            prev_node.next = next_node
            break
        prev_node = prev_node.next

    return head


def get_middle_of_list(slow, fast):
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    return slow


def reorder_list(head):
    if not head:
        return head

        # find the middle of linked list
        # in 1->2->3->4->5->6 find 4
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        # reverse the second part of the list
    # convert 1->2->3->4->5->6 into 1->2->3 and 6->5->4
    # reverse the second half in-place
    prev, curr = None, slow
    while curr:
        curr.next, prev, curr = prev, curr, curr.next

        # merge two sorted linked lists
    # merge 1->2->3 and 6->5->4 into 1->6->2->5->3->4
    first, second = head, prev
    while second.next:
        first.next, first = second, first.next
        second.next, second = first, second.next

    return head


def swap_nodes(head, k):
    cur = head
    count = 1
    front = end = None
    while cur:
        if end:
            end = end.next
        if count == k:
            front = cur
            end = head
        cur = cur.next
        count += 1
    front.data, end.data = end.data, front.data
    return head


def reverse_even_length_groups(head):
    prev = head  # Node immediately before the current group
    l = 2  # The head doesn't need to be reversed since it's a group of one node, so starts with length 2
    while prev.next:
        node = prev
        n = 0
        for i in range(l):
            if not node.next:
                break
            n += 1
            node = node.next
        if n % 2:  # odd length
            prev = node
        else:      # even length
            reverse = node.next
            curr = prev.next
            for j in range(n):
                curr_next = curr.next
                curr.next = reverse
                reverse = curr
                curr = curr_next
            prev_next = prev.next
            prev.next = node
            prev = prev_next
        l += 1
    return head


def swap_pairs(head):
    cur = head
    if not cur.next:
        return head
    nxt = cur.next
    while cur and nxt:
        tmp = cur.data
        cur.data = nxt.data
        nxt.data = tmp
        cur = nxt.next
        if cur:
            nxt = cur.next
    return head


if __name__ == "__main__":
    a = LinkedListNode(1)
    b = LinkedListNode(2)
    c = LinkedListNode(3)
    d = LinkedListNode(4)
    e = LinkedListNode(5)
    f = LinkedListNode(6)
    a.next = b
    b.next = c
    c.next = d
    d.next = e
    e.next = None
    f.next = None
    # print(reverse(a))
    # print(reverse_between(a, 1, 6))
    # print(reverse_between(a, 3, 5))
    # print(reorder_list(a))
    # print(swap_nodes(a, 1))
    print(swap_pairs(a))
