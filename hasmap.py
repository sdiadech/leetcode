from collections import Counter
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

    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        hash_map = {}

        for current in nums2:
            while stack and current > stack[-1]:
                hash_map[stack.pop()] = current
            stack.append(current)

        while stack:
            hash_map[stack.pop()] = -1

        ans = []

        for num in nums1:
            ans.append(hash_map[num])

        return ans

    def isIsomorphic(self, s: str, t: str) -> bool:
        mapper1 = {}
        mapper2 = {}
        for i in range(len(s)):
            c1 = s[i]
            c2 = t[i]
            if c1 in mapper1 and mapper1[c1] != c2:
                return False
            if c2 in mapper2 and mapper2[c2] != c1:
                return False
            mapper1[c1] = c2
            mapper2[c2] = c1
        return True

    def longestPalindrome(self, s: str) -> int:
        counter = Counter(s)
        # Initialize variables to track the length of the longest palindrome
        odd_flag = False
        pal_string = 0
        # Iterate through the character frequencies
        for count in counter.values():
            if count % 2 == 0:
                pal_string += count  # Even frequency contributes to palindrome length
            else:
                pal_string += count - 1  # Odd frequency contributes all but one character
                odd_flag = True
        # Add 1 to the palindrome length if an odd frequency character was found
        if odd_flag:
            pal_string += 1
        return pal_string

    def canPermutePalindrome(self, s: str) -> bool:
        counter = Counter(s)
        odd_counter = 0
        # Iterate through the character frequencies
        for count in counter.values():
            if count % 2:
                odd_counter += 1  # Counter odd frequency
        # If odd counter is <= 0 - > palindrome
        if odd_counter <= 1:
            return True
        return False

    def findAnagrams(self, s: str, p: str) -> List[int]:
        indexes = []
        len_s = len(s)
        len_p = len(p)
        if len_p > len_s:
            return []
        map_p = Counter(p)
        map_s = Counter(s[:len_p])
        i = 0
        while i <= len_s - len_p:
            if map_s == map_p:
                indexes.append(i)

            i += 1
            map_s = Counter(s[i:len_p + i])
        return indexes


class MyHashMap:

    def __init__(self):
        self.size = 2048
        self.bucket = [[]] * self.size

    def put(self, key: int, value: int) -> None:
        bucket, index = self.get_index(key)
        if index < 0:
            bucket.append([key, value])
        else:
            bucket[index] = [key, value]

    def get(self, key: int) -> int:
        bucket, index = self.get_index(key)
        if index < 0:
            return -1
        else:
            return bucket[index][1]

    def remove(self, key: int) -> None:
        bucket, index = self.get_index(key)
        if index < 0:
            return
        else:
            bucket.remove(bucket[index])

    def get_index(self, key):
        index = key % self.size
        bucket = self.bucket[index]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                return bucket, i
        return bucket, -1


class Logger:

    def __init__(self):
        self.logger = {}
        self.sleep = 10

    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        if message in self.logger and timestamp - self.logger[message] < self.sleep:
            return False

        else:
            self.logger[message] = timestamp
            return True


if __name__ == "__main__":
    s = Solution()
    s.nextGreaterElement([1, 2, 3], [1, 2, 3, 4])
    # print(s.isAnagram(s = "anagram", t = "nagaram"))
    # print(s.twoSum(nums = [2,7,11,15], target = 9))
    # print(s.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
    # board = [
    #     ["8", "3", ".", ".", "7", ".", ".", ".", "."]
    #     , ["6", ".", ".", "1", "9", "5", ".", ".", "."]
    #     , [".", "9", "8", ".", ".", ".", ".", "6", "."]
    #     , ["8", ".", ".", ".", "6", ".", ".", ".", "3"]
    #     , ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
    #     , ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
    #     , [".", "6", ".", ".", ".", ".", "2", "8", "."]
    #     , [".", ".", ".", "4", "1", "9", ".", ".", "5"]
    #     , [".", ".", ".", ".", "8", ".", ".", "7", "9"]
    # ]
    # print(s.isValidSudoku(board))
    # "MyHashMap", "put", "put", "get", "get", "put", "get", "remove", "get"
    # [[], [1, 1], [2, 2], [1], [3], [2, 1], [2], [2], [2]]
    # m = MyHashMap()
    # r = m.put(1, 1)
    # r = m.put(2, 2)
    # r = m.get(1)
    # r = m.get(3)
    # r = m.put(2, 1)
    # r = m.get(2)
    # r = m.remove(2)
    # r = m.get(2)
    s.findAnagrams(s = "abab", p = "ab")

