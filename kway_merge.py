import heapq
from collections import Counter
from heapq import *
from typing import List


class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p1 = m - 1
        p2 = n - 1
        x = 0
        for p in range(n + m - 1, -1, -1):
            if p2 < 0:
                break
            x += 1
            if p1 >= 0 and nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1
            else:
                nums1[p] = nums2[p2]
                p2 -= 1

    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        min_heap = []
        value = 0
        if not matrix:
            return value
        if_empty = True
        for lst in matrix:
            if len(lst) > 0:
                if_empty = False
        if if_empty:
            return value
        # Push the first element of each list to the min_heap
        for i, lst in enumerate(matrix):
            if lst:
                heappush(min_heap, (lst[0], i, 0))  # (value, list index, element index)

        while min_heap and k > 0:
            value, list_index, element_index = heappop(min_heap)
            k -= 1

            # Move to the next element of the list if available
            if element_index + 1 < len(matrix[list_index]):
                heappush(min_heap, (matrix[list_index][element_index + 1], list_index, element_index + 1))
        # If k is greater than the total number of elements in the input lists,
        # return the greatest element from all the lists
        if k > 0:
            max_element = max(element for lst in matrix for element in lst)
            return max_element

        return value

    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        min_heap = []
        result = []
        # Push sum of pairs with first element in nums2 and each element in nums1 to the min_heap
        for i, val in enumerate(nums1):
            heappush(min_heap, (sum([nums2[0], val]), i, 0))  # (sum, nums1 index, nums2 index)

        while min_heap and k > 0:
            value, nums1_index, nums2_index = heappop(min_heap)
            result.append([nums1[nums1_index], nums2[nums2_index]])

            # increment the index for 2nd list, as we've
            # compared all possible pairs with the 1st index of list2
            next_element = nums2_index + 1
            if len(nums2) > next_element:
                # if next element is available for list2 then add it to the heap
                heappush(min_heap, (sum([nums2[next_element], nums1[nums1_index]]), nums1_index, next_element))
            k -= 1

        return result

    def kth_smallest_element(self, matrix, k):
        # Your code will replace the return statement placeholder below
        min_heap = []
        for i, lst in enumerate(matrix):
            if lst:
                heappush(min_heap, (lst[0], i, 0))
            else:
                pass

        while min_heap and k > 0:
            val, matrix_index, list_index = heappop(min_heap)
            k -= 1
            next_element = list_index + 1
            if next_element < len(matrix[matrix_index]):
                heappush(min_heap, (matrix[matrix_index][next_element], matrix_index, next_element))

        return val

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        freq_map = Counter(nums)
        min_heap = [(count, num) for num, count in freq_map.items()]
        heapq.heapify(min_heap)
        print(min_heap)
        while len(min_heap) > k:
            heapq.heappop(min_heap)
        return [i[1] for i in min_heap]


if __name__ == "__main__":
    s = Solution()
    # s.merge([1,2,3,0,0,0], 3, [4,5,6], 3)
    # s.kthSmallest([[2,6,8],[3,7,10],[5,8,11]], 5)
    # s.kthSmallest([[], [], []], 5)
    # s.kSmallestPairs([1,1,2], [1,2,3], 3)
    s.topKFrequent(nums = [1,1,1,2,2,3], k = 2)
