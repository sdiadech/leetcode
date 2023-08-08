from typing import List


class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        mapper = {'}': '{', ']': '[', ')': '('}
        for c in s:
            if c not in mapper:
                stack.append(c)
                continue
            if not stack or stack[-1] != mapper[c]:
                return False
            stack.pop()
        return not stack

    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for c in tokens:
            if c == "+":
                stack.append(int(stack.pop()) + int(stack.pop()))
            elif c == "-":
                a, b = int(stack.pop()), int(stack.pop())
                stack.append(b - a)
            elif c == "*":
                stack.append(int(stack.pop()) * int(stack.pop()))
            elif c == "/":
                a, b = int(stack.pop()), int(stack.pop())
                stack.append(int(b / a))
            else:
                stack.append(c)
        return int(stack[0])

    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res = [0] * len(temperatures)
        stack = []

        for i, v in enumerate(temperatures):
            while stack and v > stack[-1][0]:
                val, prev_index = stack.pop()
                res[prev_index] = i - prev_index
            stack.append((v, i))
        print(res)
        return res

    def removeDuplicates(self, s: str) -> str:
        stack = []
        for c in s:
            if stack and c == stack[-1]:
                stack.pop()
                continue
            stack.append(c)
        return ''.join(stack)

    def minRemoveToMakeValid(self, s: str) -> str:
        stack = []
        s_list = list(s)

        for i, val in enumerate(s):
            if len(stack) > 0 and stack[-1][0] == '(' and val == ')':
                stack.pop()

            elif val == '(' or val == ')':
                stack.append([val, i])

        for p in stack:
            s_list[p[1]] = ""

        result = ''.join(s_list)

        return result


class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if self.min_stack:
            self.min_stack.append(val if val < self.min_stack[-1] else self.min_stack[-1])
        else:
            self.min_stack.append(val)

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]


class Stack:
    def __init__(self):
        self.stack_list = []

    def is_empty(self):
        return len(self.stack_list) == 0

    def top(self):
        if self.is_empty():
            return None
        return self.stack_list[-1]

    def size(self):
        return len(self.stack_list)

    def push(self, value):
        self.stack_list.append(value)

    def pop(self):
        if self.is_empty():
            return None
        return self.stack_list.pop()


class MyQueue(object):

    # constructor to initialize two stacks
    def __init__(self):
        self.stack1 = Stack()
        self.stack2 = Stack()

    def push(self, x):
        while not self.stack1.is_empty():
            self.stack2.push(self.stack1.pop())
        self.stack1.push(x)

        while not self.stack2.is_empty():
            self.stack1.push(self.stack2.pop())

    def pop(self):
        return self.stack1.pop()

    def peek(self):
        return self.stack1.top()

    def empty(self):
        return self.stack1.is_empty()


if __name__ == "__main__":
    s = Solution()
    # s.isValid("()[]{}")
    # s.dailyTemperatures([73,74,75,71,69,72,76,73])
    # print(s.removeDuplicates('acc'))
    print(s.minRemoveToMakeValid("ab)ca(so)(sc(s)("))
