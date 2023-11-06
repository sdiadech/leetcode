from collections import deque
from typing import List


class Solution:
    def combinationSum(self, candidates, target):
        output = []

        def dfs(i, cur, total):
            # Base cases
            if total == target:
                output.append(cur[:])
                return
            if i >= len(candidates) or total > target:
                return
            cur.append(candidates[i])
            dfs(i, cur, total + candidates[i])
            cur.pop()
            dfs(i + 1, cur, total)

        dfs(0, [], 0)

        return output

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        output = []
        candidates.sort()

        def dfs(start, cur, target):
            # Base cases
            if target == 0:
                output.append(cur[:])
                return
            if target <= 0:
                return
            prev = -1
            for i in range(start, len(candidates)):
                if candidates[i] == prev:
                    continue
                cur.append(candidates[i])
                dfs(i + 1, cur, target - candidates[i])
                cur.pop()
                prev = candidates[i]

        dfs(0, [], target)
        return output

    def permute(self, nums: List[int]) -> List[List[int]]:
        permutations = []
        if len(nums) == 1:
            return [nums[:]]
        for i in range(len(nums)):
            n = nums.pop(0)  # pop first element
            perms = self.permute(nums)  # [[2,3], [3,2]]
            for p in perms:
                p.append(n)  # [2,3,1], [3,2,1]
            permutations.extend(perms)
            nums.append(n)  # return to nums all items
        return permutations

    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = []
        subsets = []

        def dfs(i):
            if i >= len(nums):
                result.append(subsets[:])
                return

            #  add item [1]
            subsets.append(nums[i])
            dfs(i + 1)
            #  not add []
            subsets.pop()
            dfs(i + 1)

        dfs(0)
        print(result)
        return result

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        subsets = []
        #  Sort nums to skip duplicates
        nums.sort()

        def dfs(i):
            if i >= len(nums):
                res.append(subsets[:])
                return
            #  add item [1]
            subsets.append(nums[i])
            dfs(i + 1)
            #  not add []
            subsets.pop()
            # skip duplicates
            while i + 1 < len(nums) and nums[i] == nums[i + 1]:
                i += 1
            dfs(i + 1)

        dfs(0)
        print(res)
        return res

    def exist(self, board: List[List[str]], word: str) -> bool:
        rows, cols = len(board), len(board[0])
        path = set()

        def dfs(r, c, i):
            if i == len(word):
                return True
            if (
                    r < 0 or c < 0
                    or r >= rows
                    or c >= cols
                    or (r, c) in path
                    or board[r][c] != word[i]
            ): return False
            path.add((r, c))
            res = (
                dfs(r + 1, c, i + 1)
                or dfs(r - 1, c, i + 1)
                or dfs(r, c + 1, i + 1)
                or dfs(r, c - 1, i + 1)
            )
            path.remove((r, c))
            return res

        for r in range(rows):
            for c in range(cols):
                if dfs(r, c, 0):
                    return True
        return False


def subset_bfs(nums):
    subsets = []
    queue = deque()
    queue.append([])  # Start with an empty subset.

    while queue:
        current_subset = queue.popleft()
        subsets.append(current_subset)

        for num in nums:
            if not current_subset or num > current_subset[-1]:
                new_subset = current_subset + [num]
                queue.append(new_subset)

    return subsets


def subset_dfs(nums):
    result = []
    subsets = []

    def dfs(i):
        if i >= len(nums):
            result.append(subsets[:])  # Append a copy of the current subset
            return
        subsets.append(nums[i])
        dfs(i + 1)
        subsets.pop()
        dfs(i + 1)
        return subsets
    dfs(0)

    return result



if __name__ == "__main__":
    s = Solution()
    # o = s.combinationSum(candidates = [2,3,6,7], target = 7)
    # s.permute(nums = [1,2,3])
    # s.subsets(nums=[1, 2, 3])
    s.subsetsWithDup(nums=[1, 2, 2])
    # print(subset_bfs([1, 5, 3]))
    print(subset_dfs([1, 5, 3]))
