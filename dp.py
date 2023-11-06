from typing import List


def count_coin_combinations(coins, target_sum):
    # Initialize a table to store the number of ways to make each amount from 0 to target_sum.
    dp = [0] * (target_sum + 1)

    # There is one way to make amount 0 (by not selecting any coin).
    dp[0] = 1

    # Loop through each coin in the list.
    for coin in coins:
        # Update the dp table for each possible sum from the current coin value to target_sum.
        for i in range(coin, target_sum + 1):
            dp[i] += dp[i - coin]

    # The dp[target_sum] now contains the number of ways to make the target_sum.
    return dp[target_sum]


class Solution:
    def _min_path_sum(self, grid, m, n, memo):
        path = (m, n)
        if path in memo:
            return memo[path]
        if m == len(grid) - 1 and n == len(grid[0]) - 1 :
            return grid[m][n]
        if m == len(grid) or n == len(grid[0]):
            return float("inf")
        down = self._min_path_sum(grid, m + 1, n, memo)
        right = self._min_path_sum(grid, m, n + 1, memo)
        memo[path] = grid[m][n] + min(down, right)
        return memo[path]

    def minPathSum(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        return self._min_path_sum(grid, 0, 0, {})

    def _minmum_total(self, triangle, r, c, memo):
        path = (r, c)
        if path in memo:
            return memo[path]
        if r == len(triangle) - 1:
            return triangle[r][c]
        if r == len(triangle):
            return float("inf")
        down = self._minmum_total(triangle, r + 1, c, memo)
        right = self._minmum_total(triangle, r + 1, c + 1, memo)
        memo[path] = triangle[r][c] + min(down, right)
        return memo[path]

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if not triangle:
            return 0
        return self._minmum_total(triangle, 0, 0, {})

    def _minCostClimbingStairs(self, cost, i, memo):
        if i in memo:
            return memo[i]
        if i < 0:
            return 0  # You have reached the top.

        one_step = self._minCostClimbingStairs(cost, i - 1, memo)
        two_steps = self._minCostClimbingStairs(cost, i - 2, memo)
        memo[i] = cost[i] + min(one_step, two_steps)
        return memo[i]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        return min(self._minCostClimbingStairs(cost, n - 1, {}), self._minCostClimbingStairs(cost, n - 2, {}))

    def _find_max_form(self, s, m, n, i, memo):
        path = (m, n, i)
        if path in memo:
            return memo[path]
        if i == len(s):
            return 0
        if m < 0 or n < 0:
            return 0
        m_count = s[i].count("0")
        n_count = s[i].count("1")

        # Check if the current string can be included in the subset
        if m - m_count >= 0 and n - n_count >= 0:
            # Choose to include the current string or not
            included = 1 + self._find_max_form(s, m - m_count, n - n_count, i + 1, memo)
            not_included = self._find_max_form(s, m, n, i + 1, memo)
            memo[path] = max(included, not_included)
        else:
            # If the current string can't be included, skip it
            memo[path] = self._find_max_form(s, m, n, i + 1, memo)

        return memo[path]

    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        return self._find_max_form(strs, m, n, 0, {})

    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0
        for a in range(1, amount + 1):
            for c in coins:
                if a - c >= 0:
                    dp[a] = min(dp[a], 1 + dp[a - c])
        return dp[amount] if dp[amount] != amount + 1 else -1

    def _min_falling_path_sum(self, matrix, r, c, memo):
        path = (r, c)
        if path in memo:
            return memo[path]
        if r == len(matrix) or c < 0 or c >= len(matrix[0]):
            return float("inf")
        if r == len(matrix) - 1:
            return matrix[r][c]
        below = self._min_falling_path_sum(matrix, r + 1, c, memo)
        left = self._min_falling_path_sum(matrix, r + 1, c - 1, memo)
        right = self._min_falling_path_sum(matrix, r + 1, c + 1, memo)
        memo[path] = matrix[r][c] + min(left, right, below)
        return memo[path]

    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        n = len(matrix[0])  # Width of the matrix
        min_sum = float("inf")
        # Iterate through all possible starting points in the first row
        for col in range(n):
            min_sum = min(min_sum, self._min_falling_path_sum(matrix, 0, col, {}))
        return min_sum


if __name__ == "__main__":
    s = Solution()
    print(s.coinChange(coins = [1,2,5], amount = 11))
    print(count_coin_combinations([1,2,3], 4))
    s.minPathSum([[1,3,1],[1,5,1],[4,2,1]])
