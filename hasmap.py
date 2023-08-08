from collections import defaultdict
from typing import List


class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        counter = {}
        for char in s:
            if counter.get(char):
                counter[char] += 1
            else:
                counter[char] = 1
        for char in t:
            if counter.get(char):
                counter[char] -= 1
        if sum(counter.values()) > 0:
            return False
        return True

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        ht = {}
        for i, value in enumerate(nums):
            diff = target - value
            if diff in ht:
                return [ht[diff], i]
            ht[value] = i

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hm = defaultdict(list)
        for s in strs:
            count = [0] * 26  # a...z
            for c in s:
                count[ord(c) - ord("a")] += 1
            hm[tuple(count)].append(s)
        return list(hm.values())

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = defaultdict(set)
        cols = defaultdict(set)
        squares = defaultdict(set)
        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue
                elif (
                        board[r][c] in rows[r]
                        or board[r][c] in cols[c]
                        or board[r][c] in squares[(r // 3, c // 3)]):
                    return False
                else:
                    cols[c].add(board[r][c])
                    rows[r].add(board[r][c])
                    squares[r // 3, c // 3].add(board[r][c])
        return True


if __name__ == "__main__":
    s = Solution()
    # print(s.isAnagram(s = "anagram", t = "nagaram"))
    # print(s.twoSum(nums = [2,7,11,15], target = 9))
    # print(s.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
    board = [
        ["8", "3", ".", ".", "7", ".", ".", ".", "."]
        , ["6", ".", ".", "1", "9", "5", ".", ".", "."]
        , [".", "9", "8", ".", ".", ".", ".", "6", "."]
        , ["8", ".", ".", ".", "6", ".", ".", ".", "3"]
        , ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
        , ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
        , [".", "6", ".", ".", ".", ".", "2", "8", "."]
        , [".", ".", ".", "4", "1", "9", ".", ".", "5"]
        , [".", ".", ".", ".", "8", ".", ".", "7", "9"]
    ]
    print(s.isValidSudoku(board))
