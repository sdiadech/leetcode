from typing import List


class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        subset = []

        def backtrack(index):
            if index >= len(nums):
                res.append(subset[:])  # Add the current subset to the result
                return

            subset.append(nums[index])  # include nums[index] | left decision tree
            backtrack(index + 1)  # Recursively explore the next element
            subset.pop()  # not include nums[index] | right decision tree
            backtrack(index + 1)  # Recursively explore the next element

        backtrack(0)
        return res

    def permute(self, nums: List[int]) -> List[List[int]]:
        permutations = []
        if len(nums) == 1:
            return [nums[:]]
        for i in range(len(nums)):
            n = nums.pop(0)
            perms = self.permute(nums)
            for perm in perms:
                perm.append(n)
            permutations.extend(perms)
            nums.append(n)
        return permutations

    def permute_word(self, word):
        if len(word) == 0:
            return ['']

        permutations = []
        for i in range(len(word)):
            prefix = word[i]
            rest = word[:i] + word[i + 1:]
            print(f"1: {prefix}--{rest}")
            for perm in self.permute_word(rest):
                print(f"2: {prefix + perm}")
                permutations.append(prefix + perm)
            print(f"3: {permutations}")

        return permutations

    def letterCombinations(self, digits: str) -> List[str]:
        result = []
        #  Mapping the digits to their corresponding letters
        digits_mapping = {
            "1": [""],
            "2": ["a", "b", "c"],
            "3": ["d", "e", "f"],
            "4": ["g", "h", "i"],
            "5": ["j", "k", "l"],
            "6": ["m", "n", "o"],
            "7": ["p", "q", "r", "s"],
            "8": ["t", "u", "v"],
            "9": ["w", "x", "y", "z"]}

        def backtrack(index, cur_str):
            if len(cur_str) == len(digits):
                result.append(cur_str)
                return
            for c in digits_mapping[digits[index]]:
                backtrack(index + 1, cur_str + c)

        # Base
        if digits:
            backtrack(0, "")
        return result


def generate_combinations(n):
    def backtrack(s, open_counter, close_counter):
        if len(s) == 2 * n:
            combinations.append(s)
        if open_counter < n:
            backtrack(s + '(', open_counter + 1, close_counter)
        if close_counter < open_counter:
            backtrack(s + ')', open_counter, close_counter + 1)

    combinations = []
    backtrack('', 0, 0)

    return combinations


def get_k_sum_subsets(set_of_integers, target_sum):
    def backtrack(start, target, path):
        if target == 0:
            subsets.append(path[:])
            return

        for i in range(start, len(set_of_integers)):
            if set_of_integers[i] <= target:
                path.append(set_of_integers[i])
                backtrack(i + 1, target - set_of_integers[i], path)
                path.pop()

    subsets = []
    backtrack(0, target_sum, [])
    return subsets


if __name__ == "__main__":
    s = Solution()
    # s.subsets([2, 5, 7])
    # s.permute([1,2,3])
    # s.permute_word("abc")
    print(s.letterCombinations("23"))
