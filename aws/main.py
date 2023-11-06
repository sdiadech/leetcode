from collections import defaultdict
from collections import deque
from typing import List

"""
Amazon | OA 2019 | Optimal Utilization
https://leetcode.com/discuss/interview-question/373202/amazon-oa-2019-optimal-utilization

AWS Online Assessment (OA) SDE-III
https://leetcode.com/discuss/interview-question/2255506/AWS-Online-Assessment-(OA)-SDE-III

Amazon | AWS SDE I ONSITE | Sep 29th
https://leetcode.com/discuss/interview-question/2640456/Amazon-or-AWS-SDE-I-ONSITE-or-Sep-29th

Given an array, find the sum of count of distinct elements in all subarrays.
https://leetcode.com/discuss/interview-question/4010393/array-oa-problem-subarray/2053877

"""
# Definition for singly-linked list.
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        ht = {}
        for i, item in enumerate(nums):
            diff = target - item
            if diff in ht:
                return [ht[diff], i]
            ht[item] = i


    def minimumSwaps(self, nums: List[int]) -> int:
        n = len(nums) - 1
        lowest, largest, lowest_idx, largest_idx = float('inf'), float('-inf'), 0, 0
        for i, item in enumerate(nums):
            if item < lowest:
                lowest = item
                lowest_idx = i
            if item >= largest:
                largest = item
                largest_idx = i

        res = lowest_idx + (n - largest_idx)
        return res - 1 if lowest_idx > largest_idx else res

    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        cur = dummy
        carry = 0
        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0
            val = v1 + v2 + carry
            carry = val // 10
            val = val % 10
            cur.next = ListNode(val)
            cur = cur.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return dummy.next

    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        str_log = []
        nums_log = []
        for line in logs:
            id_, line_items = line.split(" ", 1)
            (str_log if line_items[0].isalpha() else nums_log).append((line_items, id_))
        str_log.sort()
        res = [f"{id_} {line_items}" for line_items, id_ in str_log + nums_log]
        return res

    def slowestKey(self, releaseTimes: List[int], keysPressed: str) -> str:
        start = 0
        max_duration = 0
        char = keysPressed[0]
        for i in range(len(keysPressed)):
            actual_duration = releaseTimes[i] - start
            if actual_duration > max_duration:
                char = keysPressed[i]
                max_duration = actual_duration
            if actual_duration == max_duration:
                char = max(char, keysPressed[i])
            start = releaseTimes[i]
        return char

    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0

        max_length = 0
        char_set = set()
        left = 0

        for right in range(len(s)):
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1

            char_set.add(s[right])
            max_length = max(max_length, right - left + 1)

        return max_length

    #  Frequency of The Most Frequent Element
    def maxFrequency(self, nums: List[int], k: int) -> int:
        nums.sort()
        result = 0
        total = 0
        l, r = 0, 0
        while r < len(nums):
            total += nums[r]
            while nums[r] * (r - l + 1) > total + k:
                l += 1
                total -= nums[l]
            result = max(result, r - l + 1)
            r += 1
        return result

    def two_sum_sorted(self, nums, total):
        l = 0
        r = len(nums) - 1
        while l < r:
            s = nums[l] + nums[r]
            if s == total:
                return True
            elif s < total:
                l += 1
            else:
                r -= 1
        return False

    #  2262. Total Appeal of A String
    def appealSum(self, s: str) -> int:
        n = len(s)
        last = defaultdict(lambda: -1)
        res = 0
        for i in range(n):
            res += (i - last[s[i]]) * (n - i)
            last[s[i]] = i
        return res

    #  2825. Make String a Subsequence Using Cyclic Increments
    # def canMakeSubsequence(self, str1: str, str2: str) -> bool:

    def validPathdfs(self, edges: List[List[int]], source: int, destination: int) -> bool:
        adj_list = defaultdict(list)

        for n1, n2 in edges:
            adj_list[n1].append(n2)
            adj_list[n2].append(n1)

        def dfs(src, dst, seen):
            if src == dst:
                return True
            seen.add(src)
            for node in adj_list[n1]:
                if dfs(node, dst, seen):
                    return True
            return False
        return dfs(source, destination)

    def validPathbfs(self, edges: List[List[int]], source: int, destination: int) -> bool:
        adj_list = defaultdict(list)

        for n1, n2 in edges:
            adj_list[n1].append(n2)
            adj_list[n2].append(n1)
        q = deque()
        q.append(source)
        while q:
            node = q.popleft()
            if node == destination:
                return False
            for node in adj_list[node]:
                q.append(node)
        return False


def max_subarray_sum(arr):
    max_ending_here = max_so_far = arr[0]

    for num in arr[1:]:
        max_ending_here = max(num, max_ending_here + num)
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far

if __name__ == "__main__":
    s = Solution()
    print(s.twoSum([2,7,11,15], 9))
    s.minimumSwaps(nums = [5, 6, 2, 3, 4, 1])
    s.reorderLogFiles(logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"])
    s.slowestKey(releaseTimes = [12,23,36,46,62], keysPressed = "spuda")
    s.lengthOfLongestSubstring(s = "abcabcbb")
    s.maxFrequency(nums = [1,2,4], k = 5)
    s.two_sum_sorted([2, 7, 9, 11], 9)

    arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    result = max_subarray_sum(arr)
