from typing import List


class Codec:
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        encoded = ""
        for word in strs:
            len_word = len(word)
            encoded += f"{len_word}#{word}"
        print(encoded)
        return encoded

    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings.
        """
        decoded = []
        i = 0
        while i < len(s):
            j = i
            while s[j] != '#':
                j += 1
            len_word = int(s[i:j])
            decoded.append(s[j + 1: j + 1 + len_word])
            i = j + 1 + len_word
        return decoded


class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        counter = {}
        for num in nums:
            if counter.get(num):
                counter[num] += 1
            else:
                counter[num] = 1
        if max(counter.values()) > 1:
            return True
        return False

    def containsDuplicate2(self, nums: List[int]) -> bool:
        nums.sort()
        for i in range(len(nums) - 1):
            if nums[i] == nums[i + 1]:
                return True
        return False

    def containsDuplicate3(self, nums: List[int]) -> bool:
        hashset = set()
        for i in nums:
            if i in hashset:
                return True
            hashset.add(i)
        return False


if __name__ == "__main__":
    s = Solution()
    # print(s.containsDuplicate2([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))
    c = Codec()
    s1 = c.encode(["I", "am", "ironman"])
    c.decode(s1)
