from collections import defaultdict
from collections import deque


def undirected_path(edges, node_A, node_B):
    adj = defaultdict(list)
    visited = set()

    for n1, n2 in edges:
        adj[n1].append(n2)
        adj[n2].append(n1)

    def dfs(src, dst):
        if src == dst:
            return True
        if src in visited:
            return False

        visited.add(src)
        for n in adj[src]:
            if dfs(n, dst):
                return True

    return dfs(node_A, node_B)


def largest_component(graph):
    visited = set()
    largest_component = 0

    def dfs(graph, node):
        if node in visited:
            return 0

        size = 1
        visited.add(node)
        for n in graph[node]:
            size += dfs(graph, n)
        return size

    for neighbord in graph:
        size_component = dfs(graph, neighbord)
        if size_component > largest_component:
            largest_component = size_component
    return largest_component


def shortest_path(edges, node_A, node_B):
    adj_list = defaultdict(list)
    for n1, n2 in edges:
        adj_list[n1].append(n2)
        adj_list[n2].append(n1)

    q = deque()

    visisted = set()
    q.append((node_A, 0))
    visisted.add(node_A)
    while q:
        node, dist = q.popleft()
        if node == node_B:
            return dist
        visisted.add(node)
        for neighbord in adj_list[node]:
            if neighbord not in visisted:
                visisted.add(neighbord)
                q.append((neighbord, dist + 1))
    return -1


def island_count(grid):
    visited = set()
    count = 0

    def dfs(grid, r, c):
        if (r, c) in visited:
            print('visited')
            return False
        if (r not in range(len(grid))
                or c not in range(len(grid[0]))):
            print('range')
            return False
        if grid[r][c] == 'W':
            print('w')
            return False
        visited.add((r, c))
        dfs(grid, r + 1, c)
        dfs(grid, r - 1, c)
        dfs(grid, r, c + 1)
        dfs(grid, r, c - 1)
        return True

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if dfs(grid, r, c):
                count += 1
                print(count)
    return count


def minimum_island(grid):
    visited = set()
    min_size = float('inf')

    def dfs(grid, r, c):
        if (
                r not in range(len(grid))
                or c not in range(len(grid[0]))
        ):
            return 0
        if (r, c) in visited:
            return 0
        if grid[r][c] == 'W':
            return 0

        size = 1
        visited.add((r, c))
        size += dfs(grid, r + 1, c)
        size += dfs(grid, r - 1, c)
        size += dfs(grid, r, c + 1)
        size += dfs(grid, r, c - 1)
        return size

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            size = dfs(grid, r, c)
            if size > 0 and size < min_size:
                min_size = size
    return min_size


if __name__ == "__main__":
    edges = [
        ('i', 'j'),
        ('k', 'i'),
        ('m', 'k'),
        ('k', 'l'),
        ('o', 'n')
    ]

    print(undirected_path(edges, 'j', 'm'))  # -> True
    print(largest_component({
        0: [8, 1, 5],
        1: [0],
        5: [0, 8],
        8: [0, 5],
        2: [3, 4],
        3: [2, 4],
        4: [3, 2]
    }))  # -> 4

    edges = [
        ['w', 'x'],
        ['x', 'y'],
        ['z', 'y'],
        ['z', 'v'],
        ['w', 'v']
    ]

    print(shortest_path(edges, 'w', 'z'))  # -> 2

    grid = [
        ['W', 'L', 'W', 'W', 'W'],
        ['W', 'L', 'W', 'W', 'W'],
        ['W', 'W', 'W', 'L', 'W'],
        ['W', 'W', 'L', 'L', 'W'],
        ['L', 'W', 'W', 'L', 'L'],
        ['L', 'L', 'W', 'W', 'W'],
    ]

    island_count(grid)  # -> 3
    grid = [
        ['W', 'L', 'W', 'W', 'W'],
        ['W', 'L', 'W', 'W', 'W'],
        ['W', 'W', 'W', 'L', 'W'],
        ['W', 'W', 'L', 'L', 'W'],
        ['L', 'W', 'W', 'L', 'L'],
        ['L', 'L', 'W', 'W', 'W'],
    ]

    minimum_island(grid)  # -> 2
