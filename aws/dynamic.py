# https://www.youtube.com/watch?v=oBt53YbR9Kk&t=2251s

def sum_possible(amount, numbers, memo={}):
    if amount == 0:
        return True
    if amount < 0:
        return False
    if amount in memo:
        return memo[amount]
    for num in numbers:
        remain = amount - num
        if sum_possible(remain, numbers):
            memo[amount] = True
            return True
    memo[amount] = False
    return False


def sum_val(val):
    if val == 1:
        return 1
    return val + sum_val(val - 1)


def count_paths(grid):
    def _count_paths(grid, r, c, memo):
        pos = (r, c)
        if pos in memo:
            return memo[pos]
        # Validate outbound
        if r == len(grid) or c == len(grid[0]) or grid[r][c] == 'X':
            return 0

        if r == len(grid) - 1 and c == len(grid[0]) - 1:
            return 1
        memo[pos] = _count_paths(grid, r + 1, c, memo) + _count_paths(grid, r, c + 1, memo)
        return memo[pos]

    return _count_paths(grid, 0, 0, {})


def count_paths_tabulation(grid):
    if not grid or not grid[0] or grid[0][0] == 'X':
        return 0
    if len(grid) == 1 and len(grid[0]) == 1 and grid[0][0] == 'X':
        return 0
    m = len(grid)
    n = len(grid[0])
    table = [[0] * n for _ in range(m)]
    table[0][0] = 1
    for i in range(m):
        for j in range(n):
            if i + 1 < m and grid[i][j] == 'O':
                table[i + 1][j] += table[i][j]
            if j + 1 < n and grid[i][j] == 'O':
                table[i][j + 1] += table[i][j]
            else:
                continue
    print(table[m - 1][n - 1])
    return table[m - 1][n - 1]


def max_path_sum(grid):
    if not grid or not grid[0]:
        return 0

    def _max_path_sum(grid, r, c, memo):
        path = (r, c)

        if path in memo:
            return memo[path]
        if r == len(grid) or c == len(grid[0]):
            return float("-inf")
        if r == len(grid) - 1 and c == len(grid[0]) - 1:
            return grid[r][c]
        down = _max_path_sum(grid, r + 1, c, memo)
        right = _max_path_sum(grid, r, c + 1, memo)
        memo[path] = grid[r][c] + max(down, right)
        return memo[path]

    return _max_path_sum(grid, 0, 0, {})

def min_change(amount, coins):
    ans = _min_change(amount, coins, {})
    if ans == float('inf'):
        return -1
    return ans


def _min_change(amount, coins, memo):
    if amount in memo:
        return memo[amount]

    if amount == 0:
        return 0

    if amount < 0:
        return float('inf')

    min_val = float('inf')
    for coin in coins:
        remain = amount - coin
        num_coins = 1 + _min_change(remain, coins, memo)
        if num_coins < min_val:
            min_val = num_coins
    memo[amount] = min_val
    return memo[amount]


def _non_adjacent_sum(nums, i, memo):
    if i in memo:
        return memo[i]
    if i >= len(nums):
        return 0

    include = nums[i] + _non_adjacent_sum(nums, i + 2, memo)
    exclude = _non_adjacent_sum(nums, i + 1, memo)
    memo[i] = max(include, exclude)
    return memo[i]


def non_adjacent_sum(nums):
    return _non_adjacent_sum(nums, 0, {})


def counting_change(amount, coins):
    return _counting_change(amount, coins, 0, {})


def _counting_change(amount, coins, i, memo):
    key = (amount, i)
    if key in memo:
        return memo[key]
    if amount == 0:
        return 1
    if i == len(coins):
        return 0
    coin = coins[i]
    total = 0

    for qty in range(0, amount // coin + 1):
        remain = amount - (qty * coin)
        total += _counting_change(remain, coins, i + 1, memo)
    memo[key] = total
    return memo[key]


def _array_stepper(numbers, i, memo):
    if i in memo:
        return memo[i]

    if i >= len(numbers) - 1:
        return True

    max_step = numbers[i]
    for step in range(1, max_step + 1):
        if _array_stepper(numbers, i + step, memo):
            memo[i] = True
            return True
    memo[i] = False
    return memo[i]


def array_stepper(numbers):
    return _array_stepper(numbers, 0, {})


def _max_palin_subsequence(s, i, j, memo):
    key = (i, j)
    if key in memo:
        return memo[key]
    if i == j:
        return 1
    if i > j:
        return 0
    if s[i] == s[j]:
        memo[key] = 2 + _max_palin_subsequence(s, i + 1, j - 1, memo)
    else:
        memo[key] = max(_max_palin_subsequence(s, i + 1, j, memo),
                        _max_palin_subsequence(s, i, j - 1, memo)
                        )
    return memo[key]


def max_palin_subsequence(string):
    return _max_palin_subsequence(string, 0, len(string) - 1, {})


if __name__ == "__main__":
    # grid = [
    #   ["O", "O", "X"],
    #   ["O", "O", "O"],
    #   ["O", "O", "O"],
    # ]
    # grid = [
    #     ["O", "O", "O"],
    #     ["O", "X", "X"],
    #     ["O", "O", "O"],
    # ]
    grid = [
        ["O", "O", "X", "O", "O", "O"],
        ["O", "O", "O", "O", "O", "X"],
        ["X", "O", "O", "O", "O", "O"],
        ["X", "X", "X", "O", "O", "O"],
        ["O", "O", "O", "O", "O", "O"],
    ]
    count_paths_tabulation(grid) # -> 5
    # print(sum_possible(8, [5, 12, 4])) # -> True, 4 + 4
    # print(sum_possible(103, [6, 20, 1]))  # -> True
    # print(sum_possible(2017, [4, 2, 10]))  # -> False
    # print(sum_possible(271, [10, 8, 265, 24]))  # -> False
    # sum_val(5)
    grid = [
        [1, 3, 12],
        [5, 1, 1],
        [3, 6, 1],
    ]
    # grid = [
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9],
    # ]
    # print(max_path_sum(grid))  # -> 18
    # min_change(8, [1, 5, 4, 12])  # -> 2, because 4+4 is the minimum coins possible
    # min_change(271, [10, 8, 265, 24])  # -> -1
    min_change(11, [1, 2, 5])  # -> 3

    # nums = [2, 4, 5, 12, 7]
    # non_adjacent_sum(nums)  # -> 16
    # counting_change(4, [1, 2, 3])  # -> 4
    # counting_change(512, [1, 5, 10, 25])  # -> 20119
    array_stepper([2, 4, 2, 0, 0, 1])  # -> True
    array_stepper([2, 3, 2, 0, 0, 1])  # -> False
    max_palin_subsequence("luwxult")  # -> 5
    max_palin_subsequence("xyzaxxzy")  # -> 6
