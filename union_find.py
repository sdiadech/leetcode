from typing import List


class UnionFind:
    # Constructor
    def __init__(self, n):
        self.parent = []
        self.rank = rank = [1] * (n + 1)
        for i in range(n + 1):
            self.parent.append(i)

    # Function to find which subset a particular element belongs to
    def find_parent(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find_parent(self.parent[x])
        return self.parent[x]

    # Function to join two subsets into a single subset
    def union(self, v1, v2):
        p1, p2 = self.find_parent(v1), self.find_parent(v2)
        if p1 == p2:
            return False
        elif self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] = self.rank[p1] + self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] = self.rank[p1] + self.rank[p2]
        return True


class UnionFind:
    def __init__(self):
        self.f = {}

    def findParent(self, x):
        y = self.f.get(x, x)
        if x != y:
            y = self.f[x] = self.findParent(y)
        return y

    def union(self, x, y):
        self.f[self.findParent(x)] = self.findParent(y)


class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        dsu = UnionFind()
        for a, b in edges:
            dsu.union(a, b)
        return len(set(dsu.findParent(x) for x in range(n)))


class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        graph = UnionFind(len(edges))
        for v1, v2 in edges:
            if not graph.union(v1, v2):
                print([v1, v2])
                return [v1, v2]


if __name__ == "__main__":
    s = Solution()
    s.findRedundantConnection([[1,2],[1,3],[2,3]])
