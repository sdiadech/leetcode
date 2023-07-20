import heapq
from collections import Counter
from typing import List


class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.top_k_heap = nums
        heapq.heapify(self.top_k_heap)
        while len(self.top_k_heap) > k:
            heapq.heappop(self.top_k_heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.top_k_heap, val)
        if len(self.top_k_heap) > self.k:
            heapq.heappop(self.top_k_heap)
        return self.top_k_heap[0]

    # returns kth largest element from heap
    def return_Kth_largest(self):
        return self.top_k_heap[0]


class Solution:
    def reorganizeString(self, s: str) -> str:
        reordered_str = ''
        freq_map = Counter(s)
        # Create a max heap of tuples (frequency, character)
        max_heap = [(-freq, char) for char, freq in freq_map.items()]
        heapq.heapify(max_heap)
        prev_counter = 0
        prev_char = None

        while len(max_heap) > 0:
            counter, char = heapq.heappop(max_heap)
            reordered_str += char

            if prev_counter < 0:
                heapq.heappush(max_heap, (prev_counter, prev_char))

            # Decrement the frequency of the current character and store it
            prev_counter, prev_char = counter + 1, char
        if len(reordered_str) != len(s):
            return ""

        return reordered_str

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        result = []
        # Define a custom distance function to calculate the distance from the origin (0, 0)
        def distance(point):
            return point[0] ** 2 + point[1] ** 2

        # Create a min heap of tuples (distance, point) based on the custom distance function
        min_heap = [(distance(point), point) for point in points]
        heapq.heapify(min_heap)

        # Extract the k closest points from the min heap
        for _ in range(k):
            _, point = heapq.heappop(min_heap)
            result.append(point)

        return result

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        freq_map = Counter(nums)

        min_heap = [(counter, num) for num, counter in freq_map.items()]
        heapq.heapify(min_heap)
        while len(min_heap) > k:
            heapq.heappop(min_heap)
        return [i[1] for i in min_heap]

    def findKthLargest(self, nums: List[int], k: int) -> int:
        min_heap = []
        for num in nums[:k]:
            heapq.heappush(min_heap, num)
        for num in nums[k:]:
            if num > min_heap[0]:
                heapq.heappop(min_heap)
                heapq.heappush(min_heap, num)
        return min_heap[0]

    def kthSmallest(self, root, k: int) -> int:
        def inorder_traversal(root):
            res = []
            if root:
                res = inorder_traversal(root.left)
                res.append(root.data)
                res += inorder_traversal(root.right)
            return res

        max_heap = []
        nodes = inorder_traversal(root)
        i = 0
        while i < len(nodes):
            heapq.heappush(max_heap, -nodes[i])
            if len(max_heap) > k:
                heapq.heappop(max_heap)
            i += 1
        return abs(max_heap[0])


if __name__ == "__main__":
    # Your KthLargest object will be instantiated and called as such:
    # obj = KthLargest(3, [4, 5, 8, 2])
    # param_1 = obj.add(3)
    # param_2 = obj.add(5)
    s = Solution()
    # s.reorganizeString("abb")
    # s.kClosest([[3,3],[5,-1],[-2,4]], k = 2)
    # print(s.topKFrequent([1,1,1,2,2,3], k = 2))
    s.findKthLargest([3,2,3,1,2,4,5,5,6], k = 4)
