import heapq
from typing import List


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        output = []
        output.append(intervals[0])
        for interval in intervals[1:]:
            if output[-1][0] <= interval[0] <= output[-1][-1]:
                output[-1][-1] = max(output[-1][-1], interval[-1])
            else:
                output.append(interval)
        return output

    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        res = []
        i = 0
        n = len(intervals)
        # Add intervals before newInterval that end before newInterval starts
        while i < n and intervals[i][-1] < newInterval[0]:
            res.append(intervals[i])
            i += 1
        # Merge overlapping intervals with newInterval
        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(intervals[i][0], newInterval[0])
            newInterval[1] = max(intervals[i][-1], newInterval[-1])
            i += 1

        res.append(newInterval)  # Add the merged newInterval

        # Add remaining intervals after newInterval
        while i < n:
            res.append(intervals[i])
            i += 1
        return res

    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        if not secondList:
            return []
        res = []
        i = j = 0
        n1 = len(firstList)
        n2 = len(secondList)
        while i < n1 and j < n2:
            start = max(firstList[i][0], secondList[j][0])
            end = min(firstList[i][-1], secondList[j][-1])
            if start <= end:
                res.append([start, end])
            # Move the pointer of the list that has the interval ending first
            if firstList[i][1] < secondList[j][1]:
                i += 1
            else:
                j += 1
        return res

    def leastInterval(self, tasks: List[str], n: int) -> int:
        frequencies = {}

        # store the frequency of each task
        for t in tasks:
            frequencies[t] = frequencies.get(t, 0) + 1

        # sort the frequencies
        frequencies = dict(sorted(frequencies.items(), key=lambda x: x[1]))

        # store the max frequency
        max_freq = frequencies.popitem()[1]

        # compute the maximum possible idle time
        idle_time = (max_freq - 1) * n

        # iterate over the freqencies array and update the idle time
        while frequencies and idle_time > 0:
            idle_time -= min(max_freq - 1, frequencies.popitem()[1])
        idle_time = max(0, idle_time)

        # return the total time, which is the sum of the busy time and idle time
        return len(tasks) + idle_time

    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        intervals.sort()
        i = 0
        j = 1
        n = len(intervals)
        while j < n:
            if intervals[i][1] > intervals[j][0]:
                return False
            else:
                i += 1
                j += 1
        return True

    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0

            # Sort the intervals based on start times
        intervals.sort(key=lambda x: x[0])

        # Initialize a min heap to store the end times of ongoing meetings
        rooms = []
        heapq.heappush(rooms, intervals[0][1])  # Add the end time of the first meeting

        # Iterate through the remaining intervals
        for i in range(1, len(intervals)):
            # If the start time of the current meeting is later than the earliest end time,
            # a new room is required, so add the end time to the heap
            if intervals[i][0] >= rooms[0]:
                heapq.heappop(rooms)
            heapq.heappush(rooms, intervals[i][1])

        # The size of the heap represents the minimum number of meeting rooms required
        return len(rooms)


if __name__ == "__main__":
    s = Solution()
    # print(s.merge([[1, 5], [3, 7], [4, 6]]))
    # print(s.insert([[1,3], [6, 9]], [2,5]))
    # print(s.intervalIntersection([[0,2],[5,10],[13,23],[24,25]], [[1,5],[8,12],[15,24],[25,26]]))
    # print(s.intervalIntersection([[2, 6], [7, 9], [10, 13], [14, 19], [20, 24]], [[1, 4], [6, 8], [15, 18]]))
    # print(s.leastInterval(["A","A","A","B","B","B"], n = 2))
    # print(s.canAttendMeetings([[19,20],[1,10],[5,14]]))
    print(s.minMeetingRooms([[1,2],[3,4],[5,6]]))

