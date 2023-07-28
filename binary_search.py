import random
from typing import List


class RandomPickWithWeight:

    def __init__(self, weights):
        self.sum_weights = []
        self.total_sum = 0
        for weight in weights:
            self.total_sum += weight
            self.sum_weights.append(self.total_sum)

    def pick_index(self):
        target = random.randint(1, self.total_sum)
        # index = bisect_left(self.sum_weights, target)
        # Return the index corresponding to the random number
        # return index

        start = 1
        end = len(self.sum_weights)
        # Perform binary search to find the first value higher than the target
        while start < end:
            mid = start + (end - start) // 2
            if target > self.sum_weights[mid]:
                start = mid + 1
            else:
                end = mid

        # Return the index
        return start


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        start = 0
        end = len(nums) - 1
        mid = (end + start) // 2
        while start <= end:
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                start = mid + 1
                mid = (end + start) // 2
            else:
                end = mid - 1
                mid = (end + start) // 2
        return -1

    def binary_search_rotated(self, nums: List[int], target: int) -> int:
        start = 0
        end = len(nums) - 1
        while start <= end:
            mid = start + (end - start) // 2
            if nums[mid] == target:
                return mid
            if nums[start] <= nums[mid]:
                if nums[start] <= target < nums[mid]:
                    end = mid - 1
                else:
                    start = mid + 1
            else:
                if nums[mid] < target <= nums[end]:
                    start = mid + 1
                else:
                    end = mid - 1
        return -1

    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        left, right = 0, len(arr) - k

        while left < right:
            mid = left + (right - left) // 2

            if x - arr[mid] > arr[mid + k] - x:
                left = mid + 1
            else:
                right = mid

        return arr[left:left + k]

    def singleNonDuplicate(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left != right:
            mid = left + (right - left) // 2
            if mid % 2 != 0:
                mid -= 1
            if nums[mid] == nums[mid + 1]:
                left = mid + 2
            else:
                right = mid
        return nums[left]

    def search2(self, nums: List[int], target: int) -> bool:
        start = 0
        end = len(nums) - 1
        while start <= end:
            while start < end and nums[start] == nums[start + 1]:
                start += 1
            while start < end and nums[end] == nums[end - 1]:
                end -= 1
            mid = start + (end - start) // 2
            if nums[mid] == target:
                return True
            if nums[start] <= nums[mid]:
                if nums[start] <= target <= nums[mid]:
                    end = mid - 1
                else:
                    start = mid + 1
            else:
                if nums[mid] < target <= nums[end]:
                    start = mid + 1
                else:
                    end = mid - 1
        return False

    # def firstBadVersion(self, n: int) -> int:
    #     first = 1
    #     last = n
    #     calls = 0
    #     # while first < last:
    #     #     mid = first + (last - first) // 2
    #     #     if is_bad_version(mid):
    #     #         last = mid
    #     #     else:
    #     #         first = mid + 1
    #     #     calls += 1
    #     return first, calls


if __name__ == "__main__":
    s = Solution()
    # s.search([-1,0,3,5,9,12], target=9)
    # s.binary_search_rotated([3, 1], target=1)
    # s.findClosestElements([1, 2, 3, 4, 5, 6, 7], k=5, x=7)
    # s.singleNonDuplicate([1,1,2,3,3,4,4,8,8])
    s.search2([1, 0, 1, 1, 1], target=0)
