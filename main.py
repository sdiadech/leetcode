import re
from typing import List


# Template for the linked list
from typing import Optional


class LinkedList:
    # __init__ will be used to make a LinkedList type object.
    def __init__(self):
        self.head = None

    # insert_node_at_head method will insert a LinkedListNode at head
    # of a linked list.
    def insert_node_at_head(self, node):
        if self.head:
            node.next = self.head
            self.head = node
        else:
            self.head = node

    # create_linked_list method will create the linked list using the
    # given integer array with the help of InsertAthead method.
    def create_linked_list(self, lst):
        for x in reversed(lst):
            new_node = LinkedListNode(x)
            self.insert_node_at_head(new_node)

    # returns the number of nodes in the linked list
    def get_length(self, head):
        temp = head
        length = 0
        while (temp):
            length += 1
            temp = temp.next
        return length

    # returns the node at the specified position(index) of the linked list
    def get_node(self, head, pos):
        if pos != -1:
            p = 0
            ptr = head
            while p < pos:
                ptr = ptr.next
                p += 1
            return ptr

    # __str__(self) method will display the elements of linked list.
    def __str__(self):
        result = ""
        temp = self.head
        while temp:
            result += str(temp.data)
            temp = temp.next
            if temp:
                result += ", "
        result += ""
        return result


class LinkedListNode:
    # __init__ will be used to make a LinkedListNode type object.
    def __init__(self, data, next=None):
        self.data = data
        self.next = next


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 0
        j = len(nums) - 1
        while i < j:
            if nums[i] == nums[i + 1]:
                del nums[i + 1]
                j -= 1
            else:
                i += 1
        return len(nums)

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        ht = {}
        for i, value in enumerate(nums):
            diff = target - value
            if diff in ht:
                return [ht[diff], i]
            ht[value] = i

    def runningSum(self, nums: List[int]) -> List[int]:
        for i in range(len(nums)):
            if i == 0:
                continue
            else:
                nums[i] += nums[i - 1]
        return nums

    def maximumWealth(self, accounts: List[List[int]]) -> int:
        max_wealth_ever = 0
        for customers in accounts:
            max_wealth = 0
            for money in customers:
                max_wealth += money
            max_wealth_ever = max(max_wealth, max_wealth_ever)
        return max_wealth_ever

    def numberOfSteps(self, num: int) -> int:
        steps = 0
        while num > 0:
            if num % 2 == 0:
                num = num / 2
            else:
                num -= 1
            steps += 1
        return steps

    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        ht = dict()
        for letter in magazine:
            if ht.get(letter):
                ht[letter] += 1
            else:
                ht[letter] = 1
        for letter in ransomNote:
            cur_count = ht.get(letter, 0)
            if cur_count == 0:
                return False
            ht[letter] -= 1
        return True

    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        max_consecutive_ever = 0
        max_consecutive = 0
        for i in nums:
            if i == 1:
                max_consecutive += 1
            else:
                max_consecutive = 0
            max_consecutive_ever = max(max_consecutive_ever, max_consecutive)
        return max_consecutive_ever

    def countDigit(self, n):
        count = 0
        while n != 0:
            n //= 10
            count += 1
        return count

    def findNumbers(self, nums: List[int]) -> int:
        def countDigit(n):
            count = 0
            while n != 0:
                n //= 10
                count += 1
            return count

        max_even = 0
        for i in nums:
            counter_digit = countDigit(i)
            if counter_digit % 2 == 0:
                max_even += 1
        return max_even

    def sortedSquares(self, nums: List[int]) -> List[int]:
        squares_list = [i for i in nums]
        low_pointer = 0
        high_pointer = len(nums) - 1
        index = len(nums) - 1
        while index >= 0:
            if abs(nums[low_pointer]) >= abs(nums[high_pointer]):
                squares_list[index] = nums[low_pointer] ** 2
                low_pointer += 1
            else:
                squares_list[index] = nums[high_pointer] ** 2
                high_pointer -= 1
            index -= 1
        return squares_list

    def duplicateZeros1(self, arr: List[int]) -> None:
        queue = []
        for i in range(len(arr)):
            if arr[i] == 0:
                queue.append(0)
                queue.append(0)
            else:
                queue.append(arr[i])
            arr[i] = queue.pop(0)
        print(arr)

    def duplicateZeros(self, arr: List[int]) -> None:
        """
        Do not return anything, modify arr in-place instead.
        """

        possible_dups = 0
        length_ = len(arr) - 1

        # Find the number of zeros to be duplicated
        for left in range(length_ + 1):

            # Stop when left points beyond the last element in the original list
            # which would be part of the modified list
            if left > length_ - possible_dups:
                break

            # Count the zeros
            if arr[left] == 0:
                # Edge case: This zero can't be duplicated. We have no more space,
                # as left is pointing to the last element which could be included
                if left == length_ - possible_dups:
                    arr[length_] = 0  # For this zero we just copy it without duplication.
                    length_ -= 1
                    break
                possible_dups += 1

        # Start backwards from the last element which would be part of new list.
        last = length_ - possible_dups

        # Copy zero twice, and non zero once.
        for i in range(last, -1, -1):
            if arr[i] == 0:
                arr[i + possible_dups] = 0
                possible_dups -= 1
                arr[i + possible_dups] = 0
            else:
                arr[i + possible_dups] = arr[i]

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p1 = m - 1
        p2 = n - 1
        p3 = m + n - 1
        for i in range(p3, -1, -1):
            if p2 < 0:
                break
            if p1 >= 0 and nums1[p1] >= nums2[p2]:
                nums1[i] = nums1[p1]
                p1 -= 1
            else:
                nums1[i] = nums2[p2]
                p2 -= 1
        print(nums1)

    def removeElement(self, nums: List[int], val: int) -> int:
        # 0, 1, 2, 2, 3, 0, 4, 2
        i = 0
        for item in nums:
            if item != val:
                nums[i] = item
                i += 1
        print(nums)
        return i

    def checkIfExist(self, arr: List[int]) -> bool:
        hash_table = {}
        for i in range(len(arr)):
            hash_table[arr[i]] = arr[:i]

        for key, val in hash_table.items():
            if 2 * key in val:
                return True
            elif key % 2 == 0 and key // 2 in val:
                return True

    def validMountainArray(self, arr: List[int]) -> bool:
        up = False
        down = False
        len_arr = len(arr)

        for i in range(1, len_arr):
            if not down and arr[i - 1] < arr[i]:
                up = True
                continue
            elif up and arr[i - 1] > arr[i]:
                down = True
                continue
            else:
                return False
        if up and down:
            return True
        return False

    def replaceElements(self, arr: List[int]) -> List[int]:
        if len(arr) <= 1:
            return [-1]
        right_max = arr[-1]
        for i in range(len(arr) - 2, -1, -1):
            if arr[i] > right_max:
                temp = arr[i]
                arr[i] = right_max
                right_max = temp
            else:
                arr[i] = right_max

        arr[-1] = -1
        print(arr)

    def moveZeroes(self, nums: List[int]) -> None:
        i = 0
        j = 1
        len_nums = len(nums)
        while j < len_nums:
            if nums[i] == 0 and nums[j] != 0:
               nums[i] = nums[j]
               nums[j] = 0
               i += 1
               j += 1
            elif nums[i] != 0 and nums[j] == 0:
                i += 1
            else:
                j += 1
        print(nums)

    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        start = 0
        end = len(nums) - 1
        while start < end:
            if nums[start] % 2 == 0:
                start += 1
            elif nums[end] % 2 != 0:  # to avoid not necessary swap if the end is already odd
                end -= 1
            else:
                nums[start], nums[end] = nums[end], nums[start]
        print(nums)
        return nums

    def heightChecker(self, heights: List[int]) -> int:
        correct_heights = sorted(heights)
        incorrect_heights = 0
        for i in range(len(heights)):
            if correct_heights[i] != heights[i]:
                incorrect_heights += 1
        print(incorrect_heights)
        return incorrect_heights

    def findMaxConsecutiveOnes2(self, nums: List[int]) -> int:
        left = 0
        right = 0
        max_consecutive_ones = 0
        num_zero = 0
        while right < len(nums):
            if nums[right] == 0:
                num_zero += 1
            while num_zero == 2:
                if nums[left] == 0:
                    num_zero -= 1
                left += 1
            max_consecutive_ones = max(max_consecutive_ones, right - left + 1)
            right += 1
        return max_consecutive_ones

    def thirdMax(self, nums: List[int]) -> int:
        nums_set = set(nums)
        sorted_nums = sorted(nums_set, reverse=True)
        if len(sorted_nums) >= 3:
            return sorted_nums[2]
        else:
            return sorted_nums[0]

    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        for i in range(len(nums)):
            temp = abs(nums[i]) - 1
            if nums[temp] > 0:
                nums[temp] *= -1

        res = []
        for i, n in enumerate(nums):
            if n > 0:
                res.append(i + 1)

        return res

    def pivotIndex(self, nums: List[int]) -> int:
        suml = 0
        sumr = sum(nums)
        for i in range(len(nums)):
            sumr -= nums[i]
            if suml == sumr:
                return i
            suml += nums[i]
        return -1

    def dominantIndex(self, nums: List[int]) -> int:
        max_item = max(nums)
        max_item_index = nums.index(max_item)
        nums.remove(max_item)
        if 2 * max(nums) <= max_item:
            return max_item_index
        return -1

    def plusOne(self, digits: List[int]) -> List[int]:
        digits[-1] += 1
        for i in reversed(range(1, len(digits))):
            if digits[i] != 10:
                break
            digits[i] = 0
            digits[i - 1] += 1
        if digits[0] == 10:
            digits[0] = 1
            digits.append(0)
        return digits

    def plusOne2(self, digits: List[int]) -> List[int]:
        n = len(digits)

        # move along the input array starting from the end
        for i in range(n):
            idx = n - 1 - i
            # set all the nines at the end of array to zeros
            if digits[idx] == 9:
                digits[idx] = 0
            # here we have the rightmost not-nine
            else:
                # increase this rightmost not-nine by 1
                digits[idx] += 1
                # and the job is done
                return digits

        # we're here because all the digits are nines
        return [1] + digits

    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        # Step 1
        result = []
        # Step 2
        m, n = len(mat), len(mat[0])
        i = j = direction = 0

        # Step 3
        for _ in range(m * n):
            result.append(mat[i][j])

            if direction == 0:  # Up-right direction
                if j == n - 1:
                    direction = 1  # Change direction to down-left
                    i += 1
                elif i == 0:
                    direction = 1  # Change direction to down-left
                    j += 1
                else:
                    i -= 1
                    j += 1
            else:  # Down-left direction
                if i == m - 1:
                    direction = 0  # Change direction to up-right
                    j += 1
                elif j == 0:
                    direction = 0  # Change direction to up-right
                    i += 1
                else:
                    i += 1
                    j -= 1
        return result

    def findDiagonalOrder2(self, mat: List[List[int]]) -> List[int]:
        m, n = len(mat), len(mat[0])
        res = []
        turn = 0
        for k in range(m + n - 1):
            for i in (
            range(min(m - 1, k), max(k - n, -1), -1) if not k % 2 else range(max(k - n + 1, 0), min(m, k + 1))):
                res.append(mat[i][k - i])
        return res

    def findDiagonalOrder3(self, mat: List[List[int]]) -> List[int]:
        m = len(mat)
        n = len(mat[0])
        i = j = direction = 0
        result = []
        for _ in range(m * n):
            result.append(mat[i][j])
            if direction == 0:
                if j == n - 1:
                    direction = 1
                    i += 1
                elif i == 0:
                    # change direction
                    direction = 1
                    j += 1
                else:
                    i -= 1
                    j += 1
            else:
                if j == 0:
                    direction = 0
                    i += 1
                elif i == m - 1:
                    direction = 0
                    j += 1
                else:
                    i += 1
                    j -= 1
        return result

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        result = []
        rows, columns = len(matrix), len(matrix[0])
        up = left = 0
        right = columns - 1
        down = rows - 1

        while len(result) < rows * columns:
            # Traverse from left to right.
            for col in range(left, right + 1):
                result.append(matrix[up][col])

            # Traverse downwards.
            for row in range(up + 1, down + 1):
                result.append(matrix[row][right])

            # Make sure we are now on a different row.
            if up != down:
                # Traverse from right to left.
                for col in range(right - 1, left - 1, -1):
                    result.append(matrix[down][col])

            # Make sure we are now on a different column.
            if left != right:
                # Traverse upwards.
                for row in range(down - 1, up, -1):
                    result.append(matrix[row][left])

            left += 1
            right -= 1
            up += 1
            down -= 1

        return result

    def spiralOrder2(self, matrix: List[List[int]]) -> List[int]:
        result = []
        rows, columns = len(matrix), len(matrix[0])
        left = up = 0
        down = rows - 1
        right = columns - 1
        while len(result) < rows * columns:
            for i in range(left, right + 1):
                result.append(matrix[up][i])
            for j in range(up + 1, down + 1):
                result.append(matrix[j][right])
            if up != down:
                for i in range(right - 1, left - 1, -1):
                    result.append(matrix[down][i])
            if left != right:
                for j in range(down - 1, up, -1):
                    result.append(matrix[j][left])
            left += 1
            up += 1
            right -= 1
            down -= 1
        return result

    def generate(self, numRows: int) -> List[List[int]]:
        out = []
        if numRows == 1:
            return [[1]]
        row = 1
        while row <= numRows:
            if row == 1:
                out.append([1])
            elif row == 2:
                out.append([1, 1])
            else:
                j = 0
                next_row = [1]
                last_row = out[-1]
                while j < len(out[-1]) - 1:
                    next_row.append(sum((last_row[j], last_row[j + 1])))
                    j += 1
                next_row.append(1)
                out.append(next_row)
            row += 1

        return out

    def longestCommonPrefix(self, strs: List[str]) -> str:
        longest_prefix = ""
        i = 1
        while i <= len(strs[0]):
            item = 0
            current_prefix = strs[0][:i]
            while item < len(strs):
                if current_prefix != strs[item][:i]:
                    return longest_prefix
                item += 1
            longest_prefix = current_prefix
            i += 1
        return longest_prefix

    def climbStairs(self, n: int) -> int:
        """
        F(1) = 1
        F(2) = 2
        F(3) = F(2) + F(1) = 2 + 1 = 3
        F(4) = F(3) + F(2) = 3 + 2 = 5
        F(5) = F(4) + F(3) = 5 + 3 = 8
        """
        if n == 1:
            return 1
        if n == 2:
            return 2
        prev1 = 1
        prev2 = 2
        for i in range(3, n + 1):
            steps = prev1 + prev2
            prev1 = prev2
            prev2 = steps
        return steps

    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0

        left = 1
        right = x

        while left <= right:
            mid = left + (right - left) // 2
            if mid * mid <= x < (mid + 1) * (mid + 1):
                return mid
            elif x < mid * mid:
                right = mid - 1
            else:
                left = mid + 1

    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        i = 0
        j = len(s) - 1
        while i < j:
            temp = s[i]
            s[i] = s[j]
            s[j] = temp
            i += 1
            j -= 1
        print(s)

    def mergeTwoLists(self, list1, list2):
        new_list_head = new_list = ListNode()
        while list1 and list2:
            if list1.val < list2.val:
                new_list.next = list1
                list1 = list1.next
            else:
                new_list.next = list2
                list2 = list2.next
            new_list = new_list.next
        if list1:
            new_list.next = list1
        if list2:
            new_list.next = list2
        return new_list_head.next

    def backspaceCompare(self, s: str, t: str) -> bool:
        stack_s = []
        stack_t = []
        for i in s:
            if i == '#':
                if stack_s:
                    stack_s.pop()
            else:
                stack_s.append(i)
        for j in t:
            if j == '#':
                if stack_t:
                    stack_t.pop()
            else:
                stack_t.append(j)
        return ''.join(stack_s) == ''.join(stack_t)

    def isSubsequence(self, s: str, t: str) -> bool:
        i = 0  # Pointer for string s
        j = 0  # Pointer for string t

        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1

        return i == len(s)

    def twoSum_mid(self, numbers: List[int], target: int):
        i = 0
        j = len(numbers) - 1
        while i < j:
            s = numbers[i] + numbers[j]
            if s == target:
                return [i + 1, j + 1]
            elif s < target:
                i += 1
            else:
                j -= 1

    def reverseStr(self, s: str, k: int) -> str:
        res = list(s)
        i = 0
        n = len(s)
        while i < n:
            if i + k > n:
                res[i:] = res[i:][::-1]
            else:
                res[i:i + k] = res[i:i + k][::-1]
            i += 2 * k
        return ''.join(res)

    def reverseStr_(self, s: str, k: int) -> str:
        res = ""

        for i in range(0, len(s), 2 * k):
            res += s[i:i + (2 * k)][:k][::-1] + s[i:i + (2 * k)][k:]

        return res

    def reverseWords(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        def revert(i, j):
            while i < j:
                temp = s[i]
                s[i] = s[j]
                s[j] = temp
                i += 1
                j -= 1
        revert(0, len(s) - 1)
        i = 0
        j = 0
        while i < len(s):
            if s[i] == " ":
                revert(j, i - 1)
                j = i
                j += 1
            if i == len(s) - 1:
                revert(j, i)
            i += 1
        print(s)

    def reverseWords151(self, s: str) -> str:
        sl = s.strip()[::-1].split(" ")
        res = ""
        for i in range(len(sl)):
            if not sl[i]:
                continue
            if i == len(sl):
                res += sl[i][::-1]
            else:
                res += sl[i][::-1] + " "
        return res

    def reverseWords151_2(self, s: str) -> str:
        return ' '.join(s.split()[::-1])

    def rotate(self, nums, k):
        n = len(nums)
        k %= n  # Normalize k to be within the range of array length

        def reverse(nums, start, end):
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1

        # Reverse the entire array
        reverse(nums, 0, n - 1)
        # Reverse the first k elements
        reverse(nums, 0, k - 1)
        # Reverse the remaining n - k elements
        reverse(nums, k, n - 1)
        print(nums)

    def maxArea(self, height: List[int]) -> int:
        i = 0
        j = len(height) - 1
        max_area = 0
        while i < j:
            area_height = min(height[i], height[j]) * (j - i)
            if area_height > max_area:
                max_area = area_height
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return max_area

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix = [1] * n
        for i in range(1, n):
            prefix[i] = nums[i - 1] * prefix[i - 1]
        suffix = 1
        for i in range(n - 1, -1, -1):
            prefix[i] *= suffix
            suffix *= nums[i]
        return prefix

    def find_sum_of_three(self, nums, target):
        sorted_nums = sorted(nums)
        for i in range(len(sorted_nums) - 2):
            low = i + 1
            high = len(sorted_nums) - 1
            while low < high:
                actual_sum = sum([sorted_nums[i], sorted_nums[low], sorted_nums[high]])
                if actual_sum == target:
                    return True
                elif actual_sum < target:
                    low += 1
                else:
                    high -= 1
        return False

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result = []
        target = 0
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            low = i + 1
            high = len(nums) - 1
            while low < high:
                actual_sum = sum([nums[i] + nums[low] + nums[high]])
                if actual_sum == target:
                    sub_result = [nums[i], nums[low], nums[high]]
                    result.append(sub_result)
                    while low < high and nums[low] == nums[low + 1]:
                        low += 1
                        # Skip duplicates for the third element
                    while low < high and nums[high] == nums[high - 1]:
                        high -= 1
                    low += 1
                    high -= 1
                elif actual_sum < target:
                    low += 1
                else:
                    high -= 1
        return result

    def reverse_words(self, sentence: str):
        # remove leading, trailing and multiple spaces
        sentence = re.sub(' +', ' ', sentence.strip())
        def revert(rev, i, j):
            while i < j:
                rev[i], rev[j] = rev[j], rev[i]
                i += 1
                j -= 1

        sentence_list = list(sentence.strip())
        sentence_len = len(sentence_list)

        revert(sentence_list, 0, sentence_len - 1)
        start = end = 0
        while start < sentence_len:
            while end < sentence_len - 1 and sentence_list[end] != " ":
                end += 1
            if sentence_list[end] == " ":
                revert(sentence_list, start, end - 1)
                start = end + 1
                end += 1
            if end == sentence_len - 1:
                revert(sentence_list, start, end)
                break
        return ''.join(sentence_list)

    def is_palindrome2(self, s):
        def is_palindrome(s, i, j):
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True
        i = 0
        j = len(s) - 1
        while i < j:
            if s[i] != s[j]:
                return is_palindrome(s, i + 1, j) or is_palindrome(s, i, j - 1)
            i += 1
            j -= 1
        return True

    def isHappy(self, n: int) -> bool:
        def square_sum_digits(num):
            sum_of_squares = 0
            while num > 0:
                digit = num % 10
                sum_of_squares += digit ** 2
                num //= 10
            return sum_of_squares

        slow = n
        fast = n
        while True:
            slow = square_sum_digits(slow)  # Move slow pointer by one step
            fast = square_sum_digits(square_sum_digits(fast))  # Move fast pointer by two steps

            if fast == 1:
                return True  # Number is a happy number
            if slow == fast:
                return False  # Detected a cycle, not a happy number

    def hasCycle(self, head: Optional[LinkedListNode]) -> bool:
        if head is None:
            return False
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

    def middleNode(self, head: Optional[LinkedListNode]) -> Optional[LinkedListNode]:
        mid_pointer = head
        end_pointer = head
        while end_pointer and end_pointer.next:
            mid_pointer = mid_pointer.next
            end_pointer = end_pointer.next.next
        return mid_pointer

    def circularArrayLoop(self, nums: List[int]) -> bool:
        # A function to calculate the next step
        def next_step(pointer, value, size):
            result = (pointer + value) % size
            if result < 0:
                result += size
            return result

        # A function to detect a cycle doesn't exist
        def is_not_cycle(nums, prev_direction, pointer):

            # Set current direction to True if current element is positive, set False otherwise.
            curr_direction = nums[pointer] >= 0
            # If current direction and previous direction is different or moving a pointer takes back to the same value,
            # then the cycle is not possible, we return True, otherwise return False.
            if (prev_direction != curr_direction) or (abs(nums[pointer] % len(nums)) == 0):
                return True
            else:
                return False
        size = len(nums)
        for i in range(size):
            # Set slow and fast pointer at current index value.
            slow, fast = i, i

            # Set true in 'forward' if element is positive, set false otherwise.
            forward = nums[i] > 0

            while True:
                # Move slow pointer to one step.
                slow = next_step(slow, nums[slow], size)
                # If cycle is not possible, break the loop and start from next element.
                if is_not_cycle(nums, forward, slow):
                    break

                # First move of fast pointer.
                fast = next_step(fast, nums[fast], size)
                # If cycle is not possible, break the loop and start from next element.
                if is_not_cycle(nums, forward, fast):
                    break
                # Second move of fast pointer.
                fast = next_step(fast, nums[fast], size)
                # If cycle is not possible, break the loop and start from next element.
                if is_not_cycle(nums, forward, fast):
                    break

                # At any point, if fast and slow pointers meet each other,
                # it indicate that loop has been found, return True.
                if slow == fast:
                    return True
        return False


if __name__ == "__main__":
    s = Solution()
    # a = s.removeDuplicates([1, 1, 2])

    # prices = [7, 1, 5, 3, 6, 4]
    # nums = [3, 2, 4]
    # target = 6
    # b = s.twoSum(nums, target)
    # c = s.runningSum([1, 2, 3, 4])
    # accounts = [[1, 2, 3], [3, 2, 1]]
    # d = s.maximumWealth(accounts)
    # d1 = s.numberOfSteps(123)
    # f = s.canConstruct(ransomNote="aa", magazine="aab")
    # e = s.findMaxConsecutiveOnes([1, 1, 0, 1, 1, 1])
    # g = s.findNumbers([12, 345, 2, 6, 7896])
    # h = s.sortedSquares([-7, -3, 2, 3, 11])
    # s.duplicateZeros1([1, 0, 2, 3, 0, 4, 5, 0])
    # s.merge(nums1=[1, 2, 3, 0, 0, 0], m=3, nums2=[2, 5, 6], n=3)
    # print(s.removeElement(nums=[0, 1, 2, 2, 3, 0, 4, 2], val=2))
    # print(s.removeElement(nums=[3,2,2,3], val=3))
    # s.checkIfExist([10,2,5,3])
    # s.validMountainArray([1,3,2])
    # s.replaceElements([17,18,5,4,6,1])
    # s.moveZeroes([0])
    # s.sortArrayByParity([1,0,3])
    # s.heightChecker([1,1,4,2,1,3])
    # print(s.thirdMax([3,2,1]))
    # s.findDisappearedNumbers([1,1])
    # print(s.pivotIndex([-1,-1,0,1,1,0]))
    # print(s.dominantIndex([1,2,3,4]))
    # print(s.findDiagonalOrder([[1,2],[3,4]]))
    # print(s.spiralOrder2([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))
    # print(s.generate(5))
    # t1 = ["dog", "racecar", "car"]
    # t2 = ["flower","flow","flight"]
    # print(s.longestCommonPrefix(t1))
    # print(s.climbStairs(4))
    # print(s.reverseString(["H","a","n","n","a"]))
    # print(s.backspaceCompare("a##c", "#a#c"))
    # print(s.isSubsequence("aaaaaa", "bbaaaa"))
    # print(s.twoSum_mid([2,7,11,15], 9))
    # print(s.reverseStr("abcdefg", 2))
    # print(s.reverseWords151_2("a good   example"))
    # print(s.rotate(nums=[1,2,3,4,5,6,7], k=3))
    # print(s.maxArea(height = [2,3,4,5,18,17,6]))
    # print(s.productExceptSelf([1,2,3,4]))
    # print(s.threeSum([-1,0,1,2,-1,-4]))
    # print(s.reverse_words("Hello     World"))
    # print(s.is_palindrome2("abbababb"))
    # print(s.isHappy(4))
    print(s.hasCycle([2,4,6,8,10]))
