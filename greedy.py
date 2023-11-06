from typing import List


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        target = len(nums) - 1
        for i in range(len(nums) - 2, -1, -1):
            if target <= i + nums[i]:
                target = i
        if target == 0:
            return True
        return False

    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(cost) > sum(gas):
            return -1

        current_gas = 0
        starting_index = 0

        for i in range(len(gas)):

            current_gas += (gas[i] - cost[i])

            if current_gas < 0:
                current_gas = 0
                starting_index = i + 1

        return starting_index

    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = nums[0]

        max_actual = 0
        for n in nums:
            max_actual += n
            max_sum = max(max_sum, max_actual)
            if max_actual < 0:
                max_actual = 0
        return max_sum

    def numRescueBoats(self, people: List[int], limit: int) -> int:
        people.sort()
        left = 0
        right = len(people) - 1
        boat_counter = 0
        while left <= right:
            if people[left] + people[right] <= limit:
                left += 1
            right -= 1
            boat_counter += 1
        return boat_counter

    def jump(self, nums: List[int]) -> int:
        res = 0
        l = r = 0

        while r < len(nums) - 1:
            farthest = 0
            for i in range(l, r + 1):
                farthest = max(farthest, i + nums[i])
            l = r + 1
            r = farthest
            res += 1
        return res


if __name__ == "__main__":
    s = Solution()
    # s.canJump([2,3,1,1,4])
    s.jump(nums = [3, 2, 1, 1, 4])
