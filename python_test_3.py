
# ------------------------------------------------------------------- Array block ------------------------------------------------------------------------------

'''
Remove duplicates from SORTED array
EPI 5.5
'''

arr = [2,3,5,5,7,11,11,11,13]

write_index = 1
i = 1

while(i < len(arr)):
    if arr[i] != arr[i-1]: # if arr[i] != arr[i-2]: # variation -> each elem should not occur more than twice
        arr[write_index] = arr[i]
        write_index += 1

    i += 1

print(arr)

j = len(arr) - 1

while(j >= write_index):
    arr.pop()
    j -= 1

print(arr)

'''
121 Best Time to Buy and Sell stock
https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
EPI 5.6 Buy and sell stocks once

'''

prices = [310,315,275,295,260,270,290,230,255,250]
min_buy_price = [] # On the curr day what is the min price that you could have purchased the stocks
curr_min = prices[0]


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0

        min_purchase_list = [prices[0]]
        max_profit = 0

        for i in range(1, len(prices)):
            price = prices[i]

            if price < min_purchase_list[-1]:
                min_purchase_list.append(price)
            else:
                min_purchase_list.append(min_purchase_list[-1])

        for ind, price in enumerate(prices):
            profit = price - min_purchase_list[ind]

            if profit > max_profit:
                max_profit = profit

        return max_profit



'''
15 3 Sum
https://leetcode.com/problems/3sum/
'''

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # [-1,-1,0,1,2,4]
        nums.sort()
        i = 0
        op_list = []
        op_set = set()

        while (i < len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                i += 1

            j = i + 1
            k = len(nums) - 1

            while (j < k):
                curr_sum = nums[i] + nums[j] + nums[k]

                if curr_sum == 0:
                    tup = (nums[i], nums[j], nums[k])
                    if tup in op_set:
                        pass
                    else:
                        op_list.append([nums[i], nums[j], nums[k]])
                        op_set.add((nums[i], nums[j], nums[k]))
                    j += 1
                    k -= 1

                elif curr_sum < 0:
                    j += 1

                else:
                    k -= 1

            i += 1

        return op_list


# time O(n ^ 2)
# space O(n)


'''
152. Maximum Product Subarray 
https://leetcode.com/problems/maximum-product-subarray/

Given an integer array nums, find the contiguous subarray within an array (containing at least one number) which has the largest product.
'''

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_so_far = nums[0]
        min_so_far = nums[0]
        max_prod = nums[0]

        for i in range(1, len(nums)):
            # print('\n', i, '\n', max_so_far, '\n', min_so_far)
            curr_num = nums[i]
            temp_max = max(curr_num, curr_num * min_so_far, curr_num * max_so_far)
            min_so_far = min(curr_num, curr_num * min_so_far, curr_num * max_so_far)
            max_so_far = temp_max
            max_prod = max(max_prod, max_so_far)

        return max_prod

# time O(n)
# space O(n)


'''
713. Subarray Product Less Than K 
https://leetcode.com/problems/subarray-product-less-than-k/

Your are given an array of positive integers nums.

Count and print the number of (contiguous) subarrays where the product of all the elements in the subarray is less than k.
'''

class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        i = 0
        j = 1
        res = 0
        curr_prod = nums[i]

        if curr_prod < k:
            res += 1

        while (j < len(nums)):
            num_at_j = nums[j]
            curr_prod *= num_at_j

            if num_at_j < k:
                res += 1

            while (curr_prod >= k and i < j):
                curr_prod //= nums[i]
                i += 1

            res += (j - i)
            j += 1

        return res

# time O(n)
# space O(1)


'''
325. Maximum Size Subarray Sum Equals k 
https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/

Given an array nums and a target value k, find the maximum length of a subarray that sums to k. If there isn't one, return 0 instead.
'''

class Solution:
    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        running_sums_dict = {0: 0}
        running_sum = 0
        max_subarr_len = 0

        for ind, num in enumerate(nums):
            running_sum += num
            diff = running_sum - k

            if diff in running_sums_dict:
                curr_sub_arr_len = (ind - running_sums_dict[diff]) + 1
                max_subarr_len = max(max_subarr_len, curr_sub_arr_len)

            if not running_sum in running_sums_dict:
                running_sums_dict[running_sum] = ind + 1

        return max_subarr_len

# time O(n)
# space O(n)

'''
209. Minimum Size Subarray Sum 
https://leetcode.com/problems/minimum-size-subarray-sum/
Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. 
If there isn't one, return 0 instead.
'''

class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        if not nums:
            return 0

        i = 0
        j = 1
        cum_sum = nums[0]
        min_len_subarr = float('+inf')

        if cum_sum >= s: return 1

        while (j < len(nums)):
            num = nums[j]
            cum_sum += num

            while (cum_sum >= s and i <= j):
                curr_subarr_len = j - i + 1
                min_len_subarr = min(min_len_subarr, curr_subarr_len)
                cum_sum -= nums[i]
                i += 1

            j += 1

        if min_len_subarr == float('+inf'):
            return 0
        else:
            return min_len_subarr

# time O(n)
# space O(1)

'''
53. Maximum subarray
https://leetcode.com/problems/maximum-subarray/
[34, -50, 42, 14, -5, 86]

Given this input array, the output should be 137. The contiguous subarray with the
largest sum is [42, 14, -5, 86].

Your solution should run in linear time.
'''

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        curr_sum = nums[0]
        max_sum = curr_sum

        for ind, num in enumerate(nums):
            if ind == 0:
                continue

            if curr_sum + num < num:
                curr_sum = num
            else:
                curr_sum += num

            max_sum = max(curr_sum, max_sum)

        return max_sum

# time O(n)
# space O(1)

'''
Pimco Onine assessment
Farmer harvest. Circular farm is divided into equal poritions os size len(profit). K is the num of segments he can choose to reap. He also can reap the segments opposite (in the circle) to each of those 
k segments. Profit of choosing this k segments is profit of the k segments + profit of the segments opposite to those k segments. Compute max profit
Refer Pimco_farmer_1.jpg and pimco_farmer_2.jpg
'''

def maxProfit(k, profit):
    profit_len = len(profit)
    opposite_segment_array_difference = profit_len // 2
    curr_segment_start_index = 0
    curr_segment_end_index = k
    curr_partition_profit = 0

    for segment_index in range(curr_segment_start_index, curr_segment_end_index):
        curr_partition_profit += profit[segment_index]
        opposite_segment_index = (segment_index + opposite_segment_array_difference) % profit_len
        curr_partition_profit += profit[opposite_segment_index]

    max_profit = curr_partition_profit
    curr_start_segment_opp_ind = (curr_segment_start_index + opposite_segment_array_difference) % profit_len
    curr_partition_profit -= profit[curr_segment_start_index] + profit[curr_start_segment_opp_ind]
    curr_segment_start_index += 1

    while (curr_segment_start_index + opposite_segment_array_difference < profit_len):
        profit_end_index = profit[curr_segment_end_index]
        end_index_opp_ind = (curr_segment_end_index + opposite_segment_array_difference) % profit_len
        profit_end_index_opp_ind = profit[end_index_opp_ind]

        curr_partition_profit += profit_end_index + profit_end_index_opp_ind
        max_profit = max(max_profit, curr_partition_profit)

        curr_start_segment_opp_ind = (curr_segment_start_index + opposite_segment_array_difference) % profit_len
        curr_partition_profit -= profit[curr_segment_start_index] + profit[curr_start_segment_opp_ind]

        curr_segment_start_index += 1
        curr_segment_end_index += 1

    return max_profit


def maxProfit(k, profit): #approach 2
    profit_len = len(profit)
    opposite_segment_array_difference = profit_len // 2
    curr_segment_start_index = 0
    curr_segment_end_index = k
    max_profit = float('-inf')

    while (curr_segment_start_index + opposite_segment_array_difference < profit_len):
        curr_partition_profit = 0

        for segment_index in range(curr_segment_start_index, curr_segment_end_index):
            curr_partition_profit += profit[segment_index]
            opposite_segment_index = (segment_index + opposite_segment_array_difference) % profit_len
            curr_partition_profit += profit[opposite_segment_index]

        max_profit = max(max_profit, curr_partition_profit)
        curr_segment_start_index += 1
        curr_segment_end_index += 1

    return max_profit

print(maxProfit(2,[-2,5,3,-1,-8,7,6,1]))
print(maxProfit(1, [-3,-6,3,6]))
print(maxProfit(1, [3,-5]))
print(maxProfit(2, [1,5,1,3,7,-3]))


'''
881. Boats to Save People
https://leetcode.com/problems/boats-to-save-people/
'''


class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        num_boats = 0
        people.sort()
        i = 0
        j = len(people) - 1

        while (i <= j):
            if i == j:
                num_boats += 1
                break

            elif people[i] + people[j] <= limit:
                num_boats += 1
                i += 1
                j -= 1

            else:
                num_boats += 1
                j -= 1

        return num_boats


# time O(n log n)
# space O(1)

'''
Paint the ceiling - Roblox
https://leetcode.com/discuss/interview-question/853151/Roblox-New-Grad-Online-Assessment-Questions
'''


def paint(n, s0, k, b, m, a):
    s = [0] * n
    s[0] = s0
    count = 0
    for i in range(1, len(s)):
        s[i] = ((k * s[i - 1] + b) % m) + 1 + s[i - 1]  # to create the rest of the array
    # print(s)

    i = 0;
    j = len(s) - 1

    while i < j:

        if s[i] * s[j] <= a:
            # check if s[0]*s[len - 1] <= a if so
            # then s[0]*s[1],s[0]*s[2]...s[0]*s[n-1] are all
            # less than a so add all that and increment by 1
            count += (j - i) * 2
            i += 1
        else:
            # decrement  j by 1 if above statement is not true
            j -= 1

    for i in s:
        if i * i <= a:
            # count all the possibilities where a number multiplied by
            # itself is also less than or equal to a
            count += 1

    print(count)


n, s0, k, b, m, a = map(int, input().split())
paint(n, s0, k, b, m, a)

'''
Cutting metal surplus - Roblox
https://leetcode.com/discuss/interview-question/849752/Roblox-SWE-Intern-OA-(Summer-2021)
https://leetcode.com/discuss/interview-question/428244/Cutting-Metal-Audible-Online-Assesment
'''


def maxProfit(cost_per_cut, sale_price, lengths):
    print(cost_per_cut, ' ', sale_price, lengths)
    max_profit = float('-inf')
    max_len_rod = max(lengths)
    best_len = None

    for uniform_length in range(1, max_len_rod + 1):
        num_uniform_rods = 0
        num_cuts = 0

        for curr_rod_len in lengths:
            if uniform_length > curr_rod_len:
                continue
            num_uniform_rods += curr_rod_len // uniform_length

            if curr_rod_len == uniform_length:
                pass
            elif curr_rod_len % uniform_length == 0:
                num_cuts += (curr_rod_len // uniform_length) - 1
            else:
                num_cuts += (curr_rod_len // uniform_length)

            print(num_cuts)

        curr_profit = (num_uniform_rods * uniform_length * sale_price) - (num_cuts * cost_per_cut)

        if curr_profit > max_profit:
            best_len = uniform_length

        max_profit = max(max_profit, curr_profit)

    return max_profit, best_len

print(maxProfit(25, 1, [20, 40, 21]))


# Practice problem - keep odd first, even next in integer array
arr = [1,2,3,4,5,6]
# approach 1
i = j = 0

while(j < len(arr)):
    if arr[j] % 2 != 1:
        arr[i], arr[j] = arr[j], arr[i]
        i += 1
        j += 1
    else:
        j += 1

print(arr)

# approach 2 - DO NOT FOLLOW THIS APPROACH
i = 0
j = len(arr) - 1
num_swaps = 0
# odd first even nexr
while(i < j):
    if arr[i] % 2 == 0 and arr[j] % 2 == 1: # e o
        arr[i], arr[j] = arr[j], arr[i]
        i += 1
        j -= 1
        num_swaps += 1

    elif arr[i] % 2 == 1 and arr[j] % 2 == 0: # o e
        i += 1
        j -= 1

    elif arr[i] % 2 == 1 and arr[j] % 2 == 1: # o o
        j -= 1

    elif arr[i] % 2 == 0 and arr[j] % 2 == 0: # e e
        i += 1

print(arr)


'''
1 2 3 4

[1] [1, 2, 2, 4,], [1, 2, 4, 3, 6, 9] [1, 2, 4, 3, 6, 9, 4, 8, 12, 16]

1, 3, 6, 10
'''

'''
647. Palindromic Substrings
https://leetcode.com/problems/palindromic-substrings/
'''


class Solution:
    def countSubstrings(self, s: str) -> int:
        num_pals = 0
        len_s = len(s)

        def helper(st, en):
            nonlocal num_pals

            while (st > -1 and en < len_s and s[st] == s[en]):
                num_pals += 1
                st -= 1
                en += 1

        for i in range(0, len_s):
            num_pals += 1
            helper(i - 1, i + 1)
            helper(i - 1, i)

        return num_pals


# time O(n ^ 2)
# space O(1)


'''
347 Top K Frequent Elements
https://leetcode.com/problems/top-k-frequent-elements/submissions/
'''

from collections import Counter, defaultdict

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        num_freq_map = Counter(nums)
        freq_num_map = defaultdict(list)
        orig_k = k
        nums = [] # this is the freq_array referred to in leet_code_backup1.py

        for key, num_occurances in num_freq_map.items():
            freq_num_map[num_occurances].append(key)
            nums.append(num_occurances)

        k = len(nums) - k

        def rearrange(st_ind, end_ind):
            pivot_ind = random.randint(st_ind, end_ind)
            pivot = nums[pivot_ind]
            nums[pivot_ind], nums[end_ind] = nums[end_ind], nums[pivot_ind]
            i = j = st_ind

            while (j < end_ind):
                if nums[j] < pivot:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1

                j += 1

            nums[end_ind], nums[i] = nums[i], nums[end_ind]
            return i

        def quicksort(st_ind, end_ind): # Quick select
            if st_ind > end_ind:
                return

            pivot_ind = rearrange(st_ind, end_ind)

            if k == pivot_ind:
                return

            elif k < pivot_ind:
                quicksort(st_ind, pivot_ind - 1)

            else:
                quicksort(pivot_ind + 1, end_ind)

        quicksort(0, len(nums) - 1)
        kth_elem = nums[k]
        op_list = []

        for key in freq_num_map:
            if key >= kth_elem:
                op_list.extend(freq_num_map[key])

        return op_list

# time avg case O(n) worst case O(n ^ 2)
# space O(n)

'''
Daily Interview Pro - Microsoft
You are given an array of integers in an arbitrary order. Return whether or not it is possible to make the array non-decreasing by modifying at most
1 element to any value.

We define an array is non-decreasing if array[i] <= array[i + 1] holds for every i (1 <= i < n).

Example:

[13, 4, 7] should return true, since we can modify 13 to any value 4 or less, to make it non-decreasing.

[13, 4, 1] however, should return false, since there is no way to modify just one element to make the array non-decreasing.

[13,4,7] - True
[13,4,1] - False
[5,1,3,2,5] - False

'''

self.num_changes_made = 0
def non_dec_poss(arr):
    i = 0
    j = 1

    while(j < len(arr)):
        if arr[j] >= arr[i]:
            i += 1
            j += 1

        else:
            if self.num_changes_made == 1:
                return False
            else:
                self.num_changes_made += 1
                temp_arr = list(arr)
                temp_arr[j] = temp_arr[i]
                is_change_1_poss = non_dec_poss(temp_arr) # change val at j, make a recursive call

                arr[i] = arr[j]
                is_change_2_poss = non_dec_poss(temp_arr) # change val at i, make a recursive call

                return is_change_1_poss or is_change_2_poss

    return True


'''
271 Encode and Decode Strings
https://leetcode.com/problems/encode-and-decode-strings/solution/
'''


class Codec:
    def encode(self, strs: [str]) -> str:
        """Encodes a list of strings to a single string.
        """
        encoded_str = ''

        for word in strs:
            if not word:
                encoded_str += '0000'
                continue

            len_word = len(word)

            if len_word < 10:
                encoded_str += '000' + str(len_word) + word

            elif len_word < 100:
                encoded_str += '00' + str(len_word) + word

            elif len_word < 1000:
                encoded_str += '0' + str(len_word) + word

            else:
                encoded_str += str(len_word) + word

        return encoded_str.strip('\t')

    def decode(self, s: str) -> [str]:
        """Decodes a single string to a list of strings.
        """
        op_list = []
        i = 0

        while (i < len(s)):
            len_next_word = int(s[i:i + 4])
            i += 4
            word = s[i:i + len_next_word]
            i += len_next_word
            op_list.append(word)

        return op_list


# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(strs))

# time O(n * k) n is the num of words and k is the len of the longest word
# space O(n * k)

'''
8. String to Integer (atoi)
https://leetcode.com/problems/string-to-integer-atoi/
'''


class Solution:
    def myAtoi(self, s: str) -> int:
        s = s.strip()

        if len(s) > 1 and s[0] in ['+', '-'] and s[1] in ['+', '-']:
            return 0

        s = s.strip('+')

        if s == '':
            return 0

        mul_factor = 1

        if s[0] == '-':
            mul_factor = -1
            s = s[1:]

        i = 0
        res = 0

        while (i < len(s)):
            num_at_i = s[i]

            if num_at_i.isnumeric() == False:
                break

            res = ((res * 10) + int(num_at_i))
            i += 1

            if res > (2 ** 31) and mul_factor == -1:
                return -(2 ** 31)

            if res > (2 ** 31 - 1) and mul_factor == 1:
                return (2 ** 31 - 1)

        return mul_factor * res


# time O(n)
# space O(1)


'''
75. Sort Colors
https://leetcode.com/problems/sort-colors/
'''


class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # approach 1 worst case 2 pass
        i = j = 0
        nums_len = len(nums)

        while (i < nums_len):

            if nums[i] == 0:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j += 1

            else:
                i += 1

        i = j

        while (i < nums_len):
            if nums[i] == 1:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j += 1

            else:
                i += 1

        # approach 2 one pass (time and space same as approach 1)
        i = j = 0
        k = len(nums) - 1

        while (i <= k):  # ******************** Notice the condition. its 'k' not len_nums ********************

            if nums[i] == 0:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j += 1

            elif nums[i] == 2:
                nums[i], nums[k] = nums[k], nums[i]
                k -= 1  # ******************** Notice you are only decr 'k' and not incrementing i ********************

            else:
                i += 1

        return nums


# time O(n)
# space o(1)
# ------------------------------------------------------------------- End of Array block ------------------------------------------------------------------------------





















# ------------------------------------------------------------- Binary search block ---------------------------------------------------------------------------------------------
'''
153 Find Minimum in Rotated Sorted Array
https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).

Find the minimum element.

You may assume no duplicate exists in the array.
'''

class Solution:
    def findMin(self, nums: List[int]) -> int:
        i = 0
        j = len(nums) - 1

        if nums[j] > nums[i]:
            return nums[i]

        while (i < j):
            mid = (i + j) // 2

            if nums[mid] > nums[j]:
                i = mid + 1
            else:
                j = mid # this is just "mid" and NOT "mid - 1". Your mid elem may be your ans. If you do mid - 1, you are skipping mid elem all together

        return nums[j]

# time O(log n)
# space O(1)

'''
33 Search in Rotated Sorted Array
https://leetcode.com/problems/search-in-rotated-sorted-array/

Given an integer array nums sorted in ascending order, and an integer target.

Suppose that nums is rotated at some pivot unknown to you beforehand (i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

You should search for target in nums and if you found return its index, otherwise return -1
'''


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        lo = 0
        hi = len(nums) - 1

        while (lo < hi):
            mid = (lo + hi) // 2

            if nums[mid] > nums[hi]:
                lo = mid + 1
            else:
                hi = mid

        rotation_point = lo

        if target > nums[-1]:
            lo = 0
            hi = rotation_point - 1
        else:
            lo = rotation_point
            hi = len(nums) - 1

        mid = -1

        while (lo <= hi):
            mid = (lo + hi) // 2

            if nums[mid] == target:
                break
            elif nums[mid] > target:
                hi = mid - 1
            else:
                lo = mid + 1

        if mid != -1 and nums[mid] == target:
            return mid
        else:
            return -1

# time O(log n)
# space O(1)

'''
69 Sqrt(x)
https://leetcode.com/problems/sqrtx/
'''
import math


class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0

        st = 1
        en = x

        while (True):
            m = (st + en) / 2
            sq_m = m * m

            if (sq_m == x) or (math.ceil(sq_m) >= x and math.floor(sq_m) < x):
                break

            elif sq_m < x:
                st = m

            else:
                en = m

        m = math.floor(m)

        if (m + 1) ** 2 == x:
            return m + 1
        else:
            return m


# time O(log n)
# space O(1)


'''
215. Kth Largest Element in an Array
https://leetcode.com/problems/kth-largest-element-in-an-array/
'''

# Quick Select algo
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums = nums
        k = len(nums) - k # k should be index of the sorted input array. If we need 2nd largest elem in array whose len is 7. Elem at index 6 of sorted array will be 1st greatest. Elem at index 5 will be 2nd
        # greatest. len(arr) - k = index of the kth largest elem => 7 - 2 = 5 th index holds the 2nd largest elem in array

        def rearrange(st_ind, end_ind):
            pivot = nums[end_ind]
            i = j = st_ind

            while (j < end_ind):
                if nums[j] < pivot:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
                    j += 1
                else:
                    j += 1

            nums[i], nums[end_ind] = nums[end_ind], nums[i]
            # print(nums)
            return i

        def randomized_rearrange(st_ind, end_ind):
            # ------------------------------------------ picking random pivot --------------------------------------------

            pivot_ind = random.randint(st_ind, end_ind)
            pivot = nums[pivot_ind]
            nums[pivot_ind], nums[end_ind] = nums[end_ind], nums[pivot_ind]

            # -------------------------------------------------------------------------------------------------------------------

            i = j = st_ind

            while (j < end_ind):
                if nums[j] < pivot:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
                    j += 1
                else:
                    j += 1

            nums[i], nums[end_ind] = nums[end_ind], nums[i]
            # print(nums)
            return i

        def quicksort(st_ind, end_ind):
            if st_ind > end_ind:
                return

            index_sorted_elem = rearrange(st_ind, end_ind)

            # If we remove the following if, elif, else statement, this is quick sort implmentation
            if index_sorted_elem == k:
                return

            elif k < index_sorted_elem:
                return quicksort(st_ind, index_sorted_elem - 1)

            else:
                return quicksort(index_sorted_elem + 1, end_ind)

        quicksort(0, len(nums) - 1)
        # print(nums)
        return nums[k]

# time: avg case O(n), worst case O(n ^ 2)
# space O(1)

# ------------------------------------------------------------- End of binary search block ---------------------------------------------------------------------------------------------






















# ------------------------------------------------------------- Dynamic prog block ---------------------------------------------------------------------------------------------

'''
70. Climbing stairs
https://leetcode.com/problems/climbing-stairs/

You are climbing a stair case. It takes n steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you 
climb to the top?

Note: Given n will be a positive integer.

Example 1:

Input: 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
Example 2:

Input: 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
'''

class Solution:
    def __init__(self):
        self.possible_steps = [1, 2]
        self.memo = {}

    def climbStairs(self, n: int) -> int:
        self.memo = {}

        def helper(n):
            if n == 0:
                return 1

            if n in self.memo:
                return self.memo[n]

            possible_options_for_n = 0

            for step_len in self.possible_steps:
                if n < step_len:
                    continue

                possible_options_for_n += helper(n - step_len)

            self.memo[n] = possible_options_for_n
            return self.memo[n]

        return helper(n)


# Follow up
'''
Amazon question
Recursive staircase problem
https://www.youtube.com/watch?v=5o-kdjv7FD0
'''


class Solution:
    def __init__(self):
        self.possible_steps = [1, 2]
        self.memo = {}

    def climbStairs(self, n: int) -> int:
        self.memo = {}

        def helper(n):
            if n == 0:
                return 1

            if n in self.memo:
                return self.memo[n]

            possible_options_for_n = 0

            for step_len in self.possible_steps:
                if n < step_len:
                    continue

                possible_options_for_n += helper(n - step_len)

            self.memo[n] = possible_options_for_n
            return self.memo[n]

        return helper(n)

#time O(n * k) where k is the num of possible steps
#space O(n)

#approach 2 DP This approach is not possible if possible_steps is not a constant
possible_steps = 0
memo_dp = {}
memo_dp[0] = 0
memo_dp[1] = memo_dp[2] = 1
memo_dp[3] = memo_dp[0] + memo_dp[1]
memo_dp[4] = memo_dp[3] + memo_dp[1]
memo_dp[5] = memo_dp[4] + memo_dp[2] + memo_dp[0]

for i in range(n+1):
    possible_steps += memo_dp[i]
    memo_dp[i] = possible_steps

#time O(n)
#space O(1)


'''
322 Coin change
https://leetcode.com/problems/coin-change/

You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you 
need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.
'''

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        self.memo = {}

        def helper(rem_amount):
            if rem_amount == 0:
                # self.min_coins_needed = min(self.min_coins_needed, num_coins)
                return 0

            elif rem_amount < 0:
                return float('+inf')


            if rem_amount in self.memo:
                return self.memo[rem_amount]

            num_coins_needed_for_curr_amount = float('+inf')

            for coin in coins:
                if rem_amount < coin:
                    continue

                num_coins_needed_for_this_possibility = helper(rem_amount - coin)
                num_coins_needed_for_curr_amount = min(num_coins_needed_for_curr_amount, \
                                                       num_coins_needed_for_this_possibility)

            self.memo[rem_amount] = 1 + num_coins_needed_for_curr_amount
            return self.memo[rem_amount]

        min_coins_needed = helper(amount)

        if min_coins_needed == float('+inf'):
            return -1
        else:
            return min_coins_needed

# time O(n * k) where n is the amount and k is the num of coins
# space O(n)

'''
377 Combinations Sum IV
https://leetcode.com/problems/combination-sum-iv/

Given an integer array with all positive numbers and no duplicates, find the number of possible combinations that add up to a positive integer target.
'''

# One key point to note is that, given a target 0, out func should return 1 because an empty array is a subarray of any given array
class Solution:
    def combinationSum4(self, nums, target):
        self.memo = {}

        def helper(rem_target):
            if rem_target == 0:
                return 1

            elif rem_target < 0:
                return float('-inf')

            if rem_target in self.memo:
                return self.memo[rem_target]

            num_poss_for_curr_target = 0

            for num in nums:
                if rem_target < num:
                    continue

                ret_value = helper(rem_target - num)

                if ret_value != float('-inf'):
                    num_poss_for_curr_target += ret_value

            self.memo[rem_target] = num_poss_for_curr_target
            return self.memo[rem_target]

        num_possibilities = helper(target)
        # print(num_possibilities)
        return num_possibilities

# time: O(n)
# space: O(n)

'''
300. Longest Increasing Subsequence
https://leetcode.com/problems/longest-increasing-subsequence/
'''

class Solution:
    def lengthOfLIS(self, nums):
        nums = [float('-inf')] + nums
        self.num_len = len(nums)
        self.memo = []
        self.memo = [[False] * self.num_len for _ in range(self.num_len)]

        def helper(i, j):
            if j == self.num_len:
                return 0

            if self.memo[i][j] != False:
                return self.memo[i][j]

            if nums[i] < nums[j]:
                take = 1 + helper(j, j + 1)
                skip = helper(i, j + 1)
            else:
                take = float('-inf')
                skip = helper(i, j + 1)

            self.memo[i][j] = max(skip, take)
            return self.memo[i][j]

        return helper(0, 1)


139. Word break
Leetcode
https://leetcode.com/problems/word-break/

Example
1:

Input: s = "leetcode", wordDict = ["leet", "code"]
Output: true

Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
Output: false

https://leetcode.com/problems/word-break/solution/
#solutions tab has good answers. See approach 2 and 3

# there are 2 available solutions. BFS and DP

# approach 1 - BFS
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """

        word_dict = {}
        visited_dict = {}
        for word in wordDict:
            word_dict[word] = True

        queue = [s]

        while (queue):
            # print queue
            curr_string = queue[0]
            queue = queue[1:]

            if not curr_string:
                return True

            for word in wordDict:
                # print 'word = ', word
                if curr_string.startswith(word):
                    substr = curr_string[len(word):]
                    # print 'substr = ', substr

                    if substr in visited_dict:
                        continue

                    visited_dict[substr] = True
                    queue.append(curr_string[len(word):])
                    # print 'queue at end = ', queue

        return False

# time O(n * k) n is the length of s and k is the no of words. We queue any i < len(s) once once. For each iteration of while, we loop through ip words
# space: O(k) + O(n) at most our queue can be as big as n. Like we said before each i < len(s) will get queued only once

# approach 2 - DP
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        self.word_dict = {}
        self.memo = {}

        for word in wordDict:
            self.word_dict[word] = True

        def helper(i):
            if i == len(s):
                return True

            if i in self.memo:
                return self.memo[i]

            for j in range(i + 1, len(s) + 1):
                if s[i:j] in self.word_dict and helper(j):
                    self.memo[i] = True
                    return self.memo[i]

            self.memo[i] = False
            return self.memo[i]

        return helper(0)

time: O(n ^ 2) recursive fn will execute n number of times and for loop inside recursive fn will do n iterations in the worst case

'''
198 House Robber
https://leetcode.com/problems/house-robber/submissions/

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping 
you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent 
houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.
'''

class Solution:
    def rob(self, nums: List[int]) -> int:
        self.len_nums = len(nums)
        self.memo = [['False'] * 2 for _ in range(self.len_nums)]

        def helper(ind, prev_house_robbed):
            if ind == self.len_nums:
                return 0

            if self.memo[ind][prev_house_robbed] != 'False':
                # print ('in memo')
                return self.memo[ind][prev_house_robbed]

            if prev_house_robbed == True:
                rob = float('-inf')
                skip = helper(ind + 1, False)
            else:
                rob = nums[ind] + helper(ind + 1, True)
                skip = helper(ind + 1, False)

            self.memo[ind][prev_house_robbed] = max(rob, skip)
            return self.memo[ind][prev_house_robbed]

        return helper(0, False)

time: O(n * 2) => O(n)

'''
213 House Robber II
https://leetcode.com/problems/house-robber-ii/

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. 
That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into
on the same night.
'''

class Solution:
    def rob(self, nums: List[int]) -> int:
        self.nums_copy = nums
        self.len_nums = len(nums) - 1

        if self.len_nums == 0:
            return nums[0]

        def helper(ind, prev_house_robbed):
            if ind >= self.len_nums:
                return 0

            if self.memo[ind][prev_house_robbed] != 'False':
                # print ('in memo')
                return self.memo[ind][prev_house_robbed]

            if prev_house_robbed == True:
                rob = float('-inf')
                skip = helper(ind + 1, False)
            else:
                rob = nums[ind] + helper(ind + 1, True)
                skip = helper(ind + 1, False)

            self.memo[ind][prev_house_robbed] = max(rob, skip)
            return self.memo[ind][prev_house_robbed]

        self.memo = [['False'] * 2 for _ in range(self.len_nums)]
        nums = self.nums_copy[0:len(self.nums_copy) - 1]
        with_house_one = helper(0, False)

        self.memo = [['False'] * 2 for _ in range(self.len_nums)]
        nums = self.nums_copy[1:len(self.nums_copy)]
        without_house_one = helper(0, False)

        return max(with_house_one, without_house_one)

# time O(n)
# space O(n)


'''
91 Decode ways
https://leetcode.com/problems/decode-ways/
'''

class Solution:
    def numDecodings(self, s: str) -> int:
        self.len_s = len(s)
        self.memo = {}

        def helper(ind):
            if ind == self.len_s:
                return 1

            if ind in self.memo:
                return self.memo[ind]

            if (ind != self.len_s - 1) and int(s[ind:ind + 2]) < 27 and int(s[ind:ind + 2]) > 9:
                take_two_digits = helper(ind + 2)
                take_one_digit = helper(ind + 1)
            else:
                take_two_digits = 0
                if s[ind] == '0':
                    take_one_digit = 0
                else:
                    take_one_digit = helper(ind + 1)

            self.memo[ind] = take_two_digits + take_one_digit
            return self.memo[ind]

        return helper(0)

# time O(n)
# space O(n)

'''
62 Unique Paths
https://leetcode.com/problems/unique-paths/

A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?
'''


class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        self.memo = [[False] * n for _ in range(m)]

        def helper(row, col):
            if row == (m - 1) and col == (n - 1):
                return 1

            if self.memo[row][col]:
                return self.memo[row][col]

            num_ways = 0

            if row == (m - 1):
                num_ways = helper(row, col + 1)

            elif col == (n - 1):
                num_ways = helper(row + 1, col)

            else:
                num_ways += helper(row + 1, col)
                num_ways += helper(row, col + 1)

            self.memo[row][col] = num_ways

            return self.memo[row][col]

        return helper(0, 0)

# time: O(m * n)
# space: O(m * n)

'''
55. Jump Game
Given an array of non-negative integers, you are initially positioned at the first index of the array. Each element in the array represents your maximum jump length at that position.
Determine if you are able to reach the last index
'''

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        #approach 1 time O(n) space O(1)
        reachable_ind = len(nums) - 1

        for j in range(reachable_ind, -1, -1):
            if j + nums[j] >= reachable_ind:
                reachable_ind = j

        return reachable_ind == 0

        self.len_nums = len(nums)
        self.memo = {}

        # approach 2 time O(n^2) space O(n)
        def helper(ind):
            if ind == (self.len_nums - 1):
                return True

            if ind in self.memo:
                return self.memo[ind]

            max_jump_poss = nums[ind]
            ind_copy = ind

            while (max_jump_poss > 0):
                ind_copy += 1

                if helper(ind_copy):
                    self.memo[ind] = True
                    return self.memo[ind]

                max_jump_poss -= 1

            self.memo[ind] = False
            return self.memo[ind]

        return helper(0)

'''
https://leetcode.com/discuss/interview-question/849799/Any-ideas-on-how-to-solve-this-Roblox-OA
num of ways a shopper can shop items given the costs of possibilities of each item and you are required to choose 1 possibility of each item. But you have only certain amount of money
'''

class Solution:
    def compute_possibilities(self):
        all_items = [[2, 3], [2, 3], [4], [1, 2]]
        k = 10
        #self.num_poss = 0
        memo = [[-1] * (k + 1) for _ in range(len(all_items))]
        print(memo)
        def helper(ind, rem_k):

            if ind == len(all_items) and rem_k >= 0:
                #self.num_poss += 1
                return 1

            if rem_k <= 0:
                return 0

            if memo[ind][rem_k] != -1:
                return memo[ind][rem_k]

            choices_for_curr_item = all_items[ind]
            num_poss_from_curr_index = 0

            for choice in choices_for_curr_item:
                num_poss_from_curr_index += helper(ind + 1, rem_k - choice)

            #print(ind, rem_k)
            memo[ind][rem_k] = num_poss_from_curr_index

            return memo[ind][rem_k]

        print(helper(0, k))
        #print(self.num_poss)

s = Solution()
s.compute_possibilities()

# time O(k * n * m) n is the num of items where m is the max possible options for any item
# space O(k * n)


'''
Construction Management - Roblox
https://leetcode.com/discuss/interview-question/849752/Roblox-SWE-Intern-OA-(Summer-2021)
Min cost to consturct houses. Cannot paint adjacent houses with the same material. 
'''

class Solution:
    def get_min_cost(self, costs):
        memo = [[-1] * (len(costs) + 1) for _ in range(len(costs[0]))]

        def helper(index, prev_house_material_index): # 2, 3
            if index == len(costs):
                return 0

            if prev_house_material_index and memo[index][prev_house_material_index] != -1:
                return memo[index][prev_house_material_index]

            available_options = costs[index]
            min_curr_cost = float('+inf')

            for material_index, cost in enumerate(available_options):
                if prev_house_material_index != None and material_index == prev_house_material_index:
                    continue

                else:
                    curr_cost = cost + helper(index + 1, material_index)

                min_curr_cost = min(min_curr_cost, curr_cost)

            if prev_house_material_index == None: prev_house_material_index = 0
            memo[index][prev_house_material_index] = min_curr_cost

            return memo[index][prev_house_material_index]

        print(helper(0, None))

costs = [[2,2,1], [1,2,3], [3,3,1]]
s = Solution()
s.get_min_cost(costs)

# time O(n * m * m) n is the num of houses and m is the num of materials # NEED TO VERIFY TOMORROW MORNING
# space O(n * m)


'''
Own Practice problem
num of ways you can add up to a given target using the given nums
'''

class Solution:
    def get_max_poss_vacc(self, vacs, amt):
        memo = [[-1] * len(vacs) for _ in range(amount)]

        def helper(index, rem_amt):
            if rem_amt == 0:
                return 1

            if rem_amt < 0 or index == len(vacs):
                return 0

            if memo[index][rem_amt] != -1:
                return memo[ind][rem_amt]

            skip = helper(index + 1, rem_amt) # 0 1
            take = helper(index + 1, rem_amt - vacs[index]) # 1 0
            memo[ind][rem_amt] = skip + take

            return memo[ind][rem_amt]

        print(helper(0, amt))

vacs = [0,1,2,3]
amt = 3
vacs = [5,1,1,2,2,5]
amt = 5
vacs = [7, 8, 9, 8, 7, 9, 8, 8, 10]
amt = 32
vacs = [9, 4, 2, 2, 6, 3, 2, 2, 1]
amt = 5
s = Solution()
s.get_max_poss_vacc(vacs, amt)


'''
University Career Fair - Roblox
https://leetcode.com/discuss/interview-question/854052/Roblox-Intern-OA-2020
How many meetings can be held
'''
# approach 1 n log n
end_times = [meetings[i] + durations[i] for i in range(len(meetings))]
sorted_end_time_start_time_tup_list = sorted(list(zip(end_times, meetings)))
prev_meeting = None
num_meetings = 0

for curr_meeting in sorted_end_time_start_time_tup_list:
    if prev_meeting and prev_meeting[0] > curr_meeting[1]:
        continue
    else:
        num_meetings += 1
        prev_meeting = curr_meeting

print(num_meetings)

def get_max_possible_meetings(meetings, durations):
    meeting_start_and_durations = list(zip(meetings, durations))
    meeting_start_and_durations.sort()
    max_possible_meetings = float('-inf')
    memo = [[-1] * (len(meetings) + 1) for _ in range(len(meetings) + 1)]

    def helper(index, prev_meeting_ind):
        nonlocal meeting_start_and_durations

        if index == len(meetings):
            return 0

        if prev_meeting_ind != -1 and memo[index][prev_meeting_ind] != -1:
            return memo[index][prev_meeting_ind]

        if prev_meeting_ind != -1:
            prev_meeting_end_time = meeting_start_and_durations[prev_meeting_ind][0] + meeting_start_and_durations[prev_meeting_ind][1]
        else:
            prev_meeting_end_time = -1

        curr_meeting = meeting_start_and_durations[index]

        if prev_meeting_end_time <= curr_meeting[0]:
            num_meetings_if_taken = 1 + helper(index + 1, index)
            num_meetings_if_skipped = helper(index + 1, prev_meeting_ind)
        else:
            num_meetings_if_taken = float('-inf')
            num_meetings_if_skipped = helper(index + 1, prev_meeting_ind)

        memo[index][prev_meeting_ind] = max(num_meetings_if_taken, num_meetings_if_skipped)
        return memo[index][prev_meeting_ind]

    return(helper(0, -1))

# time O(n ^ 2)
# space O(n ^ 2)

'''
Roblox OA
Max weight you can form by combining values in weights. At the same time, it should be less than max_capacity\
'''
def weightCapacity(weights, max_capacity):
    max_possible_weight = float('-inf')
    num_of_avail_weights = len(weights)

    def helper(index, weight_so_far):
        if index == num_of_avail_weights:
            #print(weight_so_far)
            return weight_so_far

        weight = weights[index]

        if weight + weight_so_far > max_capacity:
            take = float('-inf')
        else:
            take = helper(index + 1, weight+ weight_so_far)

        skip = helper(index + 1, weight_so_far)

        return max(skip, take)

    return helper(0,0)


print(weightCapacity([4,8,5,9], 20))
print(weightCapacity([1,3,5], 7))

'''
115 Distinct Subsequences - Marhworks
https://leetcode.com/problems/distinct-subsequences/
Given a string S and a string T, count the number of distinct subsequences of S which equals T.

S = "rabbbit", T = "rabbit", res = 3
S = "babgbag", T = "bag", res = 3
'''
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        len_s = len(s)
        len_t = len(t)
        # memo = [[False] * len_s for _ in range(len_t)]
        memo = {}

        def helper(s_ind, t_ind):
            if t_ind == len_t:
                return 1

            if s_ind == len_s:
                return 0

            if (t_ind, s_ind) in memo:
                return memo[t_ind, s_ind]

            '''
            if memo[t_ind][s_ind] != False:
                return memo[t_ind][s_ind]
            '''

            if s[s_ind] == t[t_ind]:
                take = helper(s_ind + 1, t_ind + 1)
                skip = helper(s_ind + 1, t_ind)
            else:
                take = 0
                skip = helper(s_ind + 1, t_ind)

            # memo[t_ind][s_ind] = skip + take
            # return memo[t_ind][s_ind]
            memo[t_ind, s_ind] = skip + take
            return memo[t_ind, s_ind]

        return (helper(0, 0))

# time O(m * n)
# space O(m * n)






# ------------------------------------------------------------- END of Dynamic prog block ---------------------------------------------------------------------------------------------























# ------------------------------------------------------------------------- Intervals block ---------------------------------------------------------------------------------------------

'''
56 Merge intervals
https://leetcode.com/problems/merge-intervals/submissions/
'''

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        new_intervals = []

        for interval in intervals:
            new_intervals.append((interval[0], interval[1]))

        intervals = sorted(new_intervals)
        merged_intervals = []

        for ind, interval in enumerate(intervals):
            # print (merged_intervals)
            if ind == 0:
                merged_intervals.append([interval[0], interval[1]])
            else:
                prev_interval = merged_intervals[-1]

                if prev_interval[1] >= interval[1]:
                    continue

                elif prev_interval[1] < interval[0]:
                    merged_intervals.append([interval[0], interval[1]])

                else:
                    prev_interval[1] = max(prev_interval[1], interval[1])

        return merged_intervals

# time O(n)
# space O(n)

'''
57. Insert intervals
https://leetcode.com/problems/insert-interval/
'''


class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        if not intervals:
            return [newInterval]

        new_interval_start = newInterval[0]
        new_interval_end = newInterval[1]
        merged_intervals = []
        new_interval_inserted = False

        if new_interval_start <= intervals[0][0]:
            merged_intervals.append(newInterval)
            new_interval_inserted = True

        def insert_interval(interval):
            prev_interval = merged_intervals[-1]

            if prev_interval[1] >= interval[1]:
                pass

            elif prev_interval[1] < interval[0]:
                merged_intervals.append([interval[0], interval[1]])

            else:
                prev_interval[1] = max(prev_interval[1], interval[1])

        for ind, interval in enumerate(intervals):
            if ind == 0 and merged_intervals:
                insert_interval(interval)

            elif ind == 0 and not merged_intervals:
                merged_intervals.append([interval[0], interval[1]])

            else:
                if new_interval_inserted == False and new_interval_start > merged_intervals[-1][1] and new_interval_start < interval[0]:
                    merged_intervals.append(newInterval)
                    new_interval_inserted = True

                insert_interval(interval)

            if new_interval_inserted == False and new_interval_start >= interval[0] and new_interval_start <= interval[1]:
                insert_interval(newInterval)
                new_interval_inserted = True

        if new_interval_inserted == False:
            merged_intervals.append(newInterval)

        return merged_intervals

# time O(n)
# space O(1)

'''
252 Meeting Rooms
https://leetcode.com/problems/meeting-rooms/

Can a person attend all the meetings
'''

class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        meetings_list = []

        for m in intervals:
            meetings_list.append((m[0], m[1]))

        intervals = sorted(meetings_list)

        for ind, meeting in enumerate(intervals):
            if ind == 0:
                prev_meeting_end_time = meeting[1]
                continue
            else:
                if meeting[0] < prev_meeting_end_time:
                    return False

                prev_meeting_end_time = meeting[1]

        return True

# time O(n)
# space O(n)


'''
253. Meeting Rooms II (See the followup questions as well)
https://leetcode.com/problems/meeting-rooms-ii/submissions/

Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of conference rooms required.
'''


class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        meetings_list = []
        max_rooms_needed = 0
        rooms_needed = 0
        ordered_meet_en_time = []

        for m in intervals:
            meetings_list.append((m[0], m[1]))

        processed_meeting_times = sorted(meetings_list)

        for meet_time in processed_meeting_times:  # O(n)
            st_time = meet_time[0]
            en_time = meet_time[1]

            while (ordered_meet_en_time and ordered_meet_en_time[0] <= st_time):  # O(1)
                heapq.heappop(ordered_meet_en_time)  # O(log n)
                rooms_needed -= 1

            heapq.heappush(ordered_meet_en_time, en_time)  # log n
            rooms_needed += 1
            max_rooms_needed = max(rooms_needed, max_rooms_needed)

        return max_rooms_needed


'''
Mock interview with chen. Follow up 
'''

Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of conference rooms required.
Output: 2

max_rooms_needed = 0
rooms_needed = 0
meeting_times = [[10:00, 11:00],[10:30,12:30],[13:00, 18:00]] #[[10,11][11,12]]
meeting_times = [[10, 11], [11,11:30],
processed_meeting_times = []

for meet_time in meeting_times: # O(n)
    st_time = meet_time[0]
    en_time = meet_time[1]
    hr_st_time = float(st_time.split(':')[0])
    min_st_time = float(st_time.split(':')[1])
    min_st_time = min_st_time / 60
    st_time = hr_st_time + min_st_time
    en_time = #
    tup = (st_time, en_time)
    processed_meeting_times.append(tup)

processed_meeting_times.sort() # O(N log N)

'''
Sample i/p test cases
[[10, 11], [10,10:30]] n = 2
[[10, 11], [11,11:30]] n = 1
[[10, 5], [10:30,11], [1:30,3:30], [4:30,6:30]] n = 2
[[10, 9], []]
[]
[['a','b']]
'''

# time O(n log n)
# space O(n)


'''
435 Non-overlapping Intervals
https://leetcode.com/problems/non-overlapping-intervals/
'''


class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        min_removes = 0
        i = 0

        while (i < len(intervals)):
            interval = tuple(intervals[i])
            intervals[i] = interval
            i += 1

        # intervals = sorted(intervals, key=itemgetter(1))
        intervals.sort(key=lambda x: x[1])
        prev_interval = None

        for interval in intervals:
            if prev_interval and interval[0] < prev_interval[1]:
                min_removes += 1
                continue

            prev_interval = interval

        return min_removes


# time O(n log n)
# O(1)

# ------------------------------------------------------------- END of intervals block ---------------------------------------------------------------------------------------------



























# ------------------------------------------------------------- Binary block or Bit manipulation block ---------------------------------------------------------------------------------------------
'''
371. Sum of Two Integers
https://leetcode.com/problems/sum-of-two-integers/

Calculate the sum of two integers a and b, but you are not allowed to use the operator + and -
'''

class Solution:
    def getSum(self, a: int, b: int) -> int:
        x, y = abs(a), abs(b)

        if x < y:
            a, b = b, a
            x, y = abs(a), abs(b)

        sign = 1 if a > 0 else -1

        if a * b > 0:

            while (y > 0):
                sum_without_carry = x ^ y
                y = (x & y) << 1
                x = sum_without_carry

        else:

            while (y > 0):
                sub_without_borrow = x ^ y
                y = ((~x) & y) << 1
                x = sub_without_borrow

        return x * sign

# time O(1) because each integer contains 32 bits.
# space O(1)

'''
191 Number of 1 bits
https://leetcode.com/problems/number-of-1-bits/

Write a function that takes an unsigned integer and return the number of '1' bits it has (also known as the Hamming weight).
'''

# approach 1
class Solution:
    def hammingWeight(self, n: int) -> int:
        num_one_bits = 0
        i = 0

        while (i < 32 and n > 0):
            if n & 1 == 1:
                num_one_bits += 1
            n >>= 1
            i += 1

        return num_one_bits

# approach 2
class Solution:
    def hammingWeight(self, n: int) -> int:
        num_one_bits = 0
        i = 0
        mask = 1

        while (i < 32 and n > 0):
            if n & mask != 0:
                num_one_bits += 1
            mask <<= 1
            i += 1

        return num_one_bits

'''
383 Counting Bits
Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate the number of 1's in their binary representation and return them as an array.

'''

class Solution:
    def countBits(self, num: int) -> List[int]:
        if num == 0:
            return [0]

        num_bits_dict = {0: 0, 1: 1}
        curr_two_pow = 0
        curr_two_pow_val = 1
        res = [0, 1]

        for n in range(2, num + 1):
            if n == 2 * curr_two_pow_val:
                num_one_bits = 1
                num_bits_dict[n] = num_one_bits
                curr_two_pow += 1
                curr_two_pow_val = n
            else:
                remaining_val = n - curr_two_pow_val
                num_one_bits = 1 + num_bits_dict[remaining_val]
                num_bits_dict[n] = num_one_bits

            # print(num_bits_dict)
            # print(num_one_bits)
            res.append(num_one_bits)

        return res

# time O(n)
# space O(n)


'''
268 Missing Number
https://leetcode.com/problems/missing-number/solution/
'''

class Solution:
    def missingNumber(self, nums):
        missing = len(nums)
        for i, num in enumerate(nums):
            missing ^= i ^ num
        return missing

# time O(n)
# space O(1)


'''
190 Reverse bits
https://leetcode.com/problems/reverse-bits/
'''

# approach 1
class Solution:
    def reverseBits(self, n: int) -> int:
        output = 0
        mask_pow = 31
        ind = 0

        while ind < 32:
            least_sig_bit = n & 1

            if least_sig_bit == 1:
                output ^= (2 ** mask_pow)

            # print(bin(output))
            n >>= 1
            mask_pow -= 1
            ind += 1

        return output

# approach 2
def reverseBits(self, n):
    ret, power = 0, 31
    while n:
        ret += (n & 1) << power
        n = n >> 1
        power -= 1
    return ret

# time O(n)
# space O(1)







# --------------------------------------------------------------------------- Graph block ---------------------------------------------------------------------------------------------

'''
207 Course Schedule
https://leetcode.com/problems/course-schedule/

There are a total of numCourses courses you have to take, labeled from 0 to numCourses-1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?
'''


class Node:
    def __init__(self, val):
        self.val = val
        self.children = []
        self.active_recursion_state = False


# print(s.canFinish(4, [[1,0],[2,1],[3,1],[2,3]]))
class Solution:
    # approach 1 (approach 3 is the best approach)
    def check_cycle_course_2(self, node):
        if node in self.overall_discovered_nodes:
            return False

        if node.active_recursion_state:
            return True

        # visited_nodes = list(visited_nodes)
        # visited_nodes.append(node)
        node.active_recursion_state = True

        for child in node.children:
            if self.check_cycle_course(child):
                return True

        node.active_recursion_state = False
        self.overall_discovered_nodes.add(node)
        return False

    # approach 2
    def check_cycle_course(self, node, visited_nodes):
        if node in self.overall_discovered_nodes:
            return False

        if node in visited_nodes:
            return True

        visited_nodes.add(node)

        for child in node.children:
            if self.check_cycle_course(child, visited_nodes):
                return True

        visited_nodes.remove(node)
        self.overall_discovered_nodes.add(node)

        return False

    def canFinish(self, numCourses, prerequisites):
        if not prerequisites: return True

        self.overall_discovered_nodes = set()

        zero_indegree_set = set()
        node_val_map = {}

        for course_prereq_assoc in prerequisites:
            course = course_prereq_assoc[0]
            prereq = course_prereq_assoc[1]

            if prereq in node_val_map:
                prereq_node = node_val_map[prereq]

            else:
                prereq_node = Node(prereq)
                zero_indegree_set.add(prereq)
                node_val_map[prereq] = prereq_node

            if course in node_val_map:
                course_node = node_val_map[course]

                if course in zero_indegree_set:
                    zero_indegree_set.remove(course)

            else:
                course_node = Node(course)
                node_val_map[course] = course_node

            prereq_node.children.append(course_node)

        # print(zero_indegree_set)

        if not zero_indegree_set:
            return False

        for course in zero_indegree_set:
            course_node = node_val_map[course]

            # if self.check_cycle_course(course_node):
            if self.check_cycle_course(course_node, set()):
                return False

        if len(self.overall_discovered_nodes) < len(node_val_map.keys()):
            return False

        return True

# time O(V + E)
# space O(V + E)

# approach 3 Topological sorting
'''
 - In order to find a global order, we can start from those nodes which do not have any prerequisites (i.e. indegree of node is zero), we then incrementally add new nodes to the global order, 
 following the dependencies (edges).
 - Once we follow an edge, we then remove it from the graph.
 - With the removal of edges, there would more nodes appearing without any prerequisite dependency, in addition to the initial list in the first step.
 - The algorithm would terminate when we can no longer remove edges from the graph. There are two possible outcomes:
 -- 1). If there are still some edges left in the graph, then these edges must have formed certain cycles, which is similar to the deadlock situation. It is due to these cyclic dependencies that we cannot 
 remove them during the above processes.
 -- 2). Otherwise, i.e. we have removed all the edges from the graph, and we got ourselves a topological order of the graph.
'''

class GNode(object):
    """  data structure represent a vertex in the graph."""
    def __init__(self):
        self.inDegrees = 0
        self.outNodes = []

class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        from collections import defaultdict, deque
        # key: index of node; value: GNode
        graph = defaultdict(GNode)

        totalDeps = 0
        for relation in prerequisites:
            nextCourse, prevCourse = relation[0], relation[1]
            graph[prevCourse].outNodes.append(nextCourse)
            graph[nextCourse].inDegrees += 1
            totalDeps += 1

        # we start from courses that have no prerequisites.
        # we could use either set, stack or queue to keep track of courses with no dependence.
        nodepCourses = deque()
        for index, node in graph.items():
            if node.inDegrees == 0:
                nodepCourses.append(index)

        removedEdges = 0
        while nodepCourses:
            # pop out course without dependency
            course = nodepCourses.pop()

            # remove its outgoing edges one by one
            for nextCourse in graph[course].outNodes:
                graph[nextCourse].inDegrees -= 1
                removedEdges += 1
                # while removing edges, we might discover new courses with prerequisites removed, i.e. new courses without prerequisites.
                if graph[nextCourse].inDegrees == 0:
                    nodepCourses.append(nextCourse)

        if removedEdges == totalDeps:
            return True
        else:
            # if there are still some edges left, then there exist some cycles
            # Due to the dead-lock (dependencies), we cannot remove the cyclic edges
            return False

# time O(V + E)
# space O(V + E)


'''
417 Pacific Atlantic Water Flow
https://leetcode.com/problems/pacific-atlantic-water-flow/
'''

class Solution:
    def pacificAtlantic(self, matrix):
        if not matrix:
            return []
        res = []
        num_rows = len(matrix)
        num_cols = len(matrix[0])
        pacific_reach_matrix = [[False] * num_cols for _ in range(num_rows)]
        atlantic_reach_matrix = [[False] * num_cols for _ in range(num_rows)]
        visited = [[False] * num_cols for _ in range(num_rows)]

        def do_bfs(pacific=False, atlantic=False):
            nonlocal visited, pacific_reach_matrix, atlantic_reach_matrix
            queue = []
            all_dirs = [(0, -1), (-1, 0), (1, 0), (0, 1)]

            for row in range(num_rows):
                for col in range(num_cols):

                    if pacific and pacific_reach_matrix[row][col] == 1:
                        queue.append([row, col])

                    elif atlantic and atlantic_reach_matrix[row][col] == 1:
                        queue.append([row, col])

            while (queue):
                row, col = queue.pop()

                visited[row][col] = True

                for direction in all_dirs:
                    new_row = row + direction[0]
                    new_col = col + direction[1]

                    if new_row > -1 and new_row < num_rows and new_col > -1 and new_col < num_cols and matrix[new_row][new_col] >= matrix[row][col] \
                            and visited[new_row][new_col] == False:

                        if pacific:
                            pacific_reach_matrix[new_row][new_col] = 1
                        else:
                            atlantic_reach_matrix[new_row][new_col] = 1

                        queue.append([new_row, new_col])

        row = 0

        for col in range(num_cols):
            pacific_reach_matrix[row][col] = 1

        row = num_rows - 1

        for col in range(num_cols):
            atlantic_reach_matrix[row][col] = 1

        col = 0

        for row in range(num_rows):
            pacific_reach_matrix[row][col] = 1

        col = num_cols - 1

        for row in range(num_rows):
            atlantic_reach_matrix[row][col] = 1

        do_bfs(pacific=True)
        visited = [[False] * num_cols for _ in range(num_rows)]
        do_bfs(atlantic=True)

        for row in range(num_rows):
            for col in range(num_cols):
                if pacific_reach_matrix[row][col] == 1 and atlantic_reach_matrix[row][col] == 1:
                    res.append([row, col])

        return res

# time O(n)
# space O(n)


'''
200. Number of Islands
https://leetcode.com/problems/number-of-islands/
'''

from collections import deque


class Solution:
    def numIslands(self, grid):
        if not grid:
            return 0

        self.col_len = len(grid[0])
        self.row_len = len(grid)
        # discovery_grid = [[False] * self.col_len for _ in range(self.row_len)]
        num_islands = 0
        '''
          [ ["1","1","1"],
            ["0","1","0"],
            ["1","1","1"]]
        '''

        def do_bfs(row, col):
            queue = deque()
            queue.append((row, col))
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            in_queue = set()
            in_queue.add((row, col))

            while queue:
                row, col = queue.popleft()
                grid[row][col] = '0'

                for d in directions:
                    new_row = row + d[0]
                    new_col = col + d[1]

                    if new_row < 0 or new_row >= self.row_len or new_col < 0 or new_col >= self.col_len or grid[new_row][new_col] == "0" or (
                    new_row, new_col) in in_queue:
                        continue

                    in_queue.add((new_row, new_col))
                    queue.append((new_row, new_col))

            # print(discovery_grid)

        for row in range(self.row_len):
            for col in range(self.col_len):
                if grid[row][col] == "0":
                    continue
                else:
                    print('in else')
                    do_bfs(row, col)
                    num_islands += 1

        return num_islands

# time O(m * n)
# apce O(m * n) # you queue may have m * n elements present in it if all the elems in the matrix are 1


'''
269. Alien Dictionary
https://leetcode.com/problems/alien-dictionary/

The following is a variation of alien dictionary. Not the actual problem
'''
from collections import deque, OrderedDict

class CharNode:
    def __init__(self, val: str):
        self.val = val
        self.parents = set()
        self.next = set()


class Solution:
    def __init__(self):
        self.ordered_chars = []
        self.chars_dict = {}

    def alienOrder(self, words):
        super_parents = list(self.construct_nodes_and_get_super_parents(words))
        print(super_parents)
        for char in self.chars_dict:
            print('\n', char)
            char_node = self.chars_dict[char]
            print(char_node.next)
            print(char_node.parents)

        self.order_chars(super_parents)
        return ''.join(self.ordered_chars)

    def order_chars(self, super_parents):
        queue = deque(super_parents)
        visited_dict = {}

        while (queue):
            parent_char = queue.popleft()
            print('\nparent_char = ', parent_char)
            char_node = self.chars_dict[parent_char]

            if not char_node.parents and not parent_char in visited_dict:
                self.ordered_chars.append(parent_char)
                visited_dict[parent_char] = True

            for child in char_node.next:
                child_node = self.chars_dict[child]
                if parent_char in child_node.parents: child_node.parents.remove(parent_char)

                queue.append(child)

    def construct_nodes_and_get_super_parents(self, words):
        super_parents = OrderedDict()

        for word in words:

            for ind, char in enumerate(word):

                if char in self.chars_dict:
                    char_node = self.chars_dict[char]
                else:
                    char_node = CharNode(char)
                    self.chars_dict[char] = char_node
                    super_parents[char] = True

                if ind != 0:
                    parent = word[ind - 1]
                    if parent == char: continue
                    char_node.parents.add(parent)
                    parent_node = self.chars_dict[parent]
                    parent_node.next.add(char)
                    if char in super_parents: super_parents.pop(char)

        return super_parents


s = Solution()
order = s.alienOrder([
  "wrt",
  "wrf",
  "er",
  "ett",
  "rftt"
])
print(order)


# The following is the solution to the actual problem

from collections import defaultdict, deque


class CharNode(object):
    def __init__(self, val):
        self.val = val
        self.next = []
        self.indegree = 0


class Solution(object):
    def get_char_node(self, char):
        if char in self.char_node_map:
            char_node = self.char_node_map[char]
        else:
            char_node = CharNode(char)
            self.char_node_map[char] = char_node

        return char_node

    def get_zero_indeg_nodes(self):
        for _, node in self.char_node_map.items():
            if node.indegree == 0:
                self.char_node_map.pop(node.val)
                return node

        return None

    def alienOrder(self, words):
        """
        :type words: List[str]
        :rtype: str
        """
        self.char_node_map = {}
        same_starting_chars_dict = defaultdict(list)

        for word in words:
            same_starting_chars_dict[''].append(word)

        queue = deque()
        queue.append(same_starting_chars_dict.popitem()[1])

        while (queue):
            same_starting_chars_dict = defaultdict(list)
            curr_words = queue.popleft()
            # visited_chars = set()

            if len(curr_words) == 1:
                continue

            for ind, word in enumerate(curr_words):
                if not word: continue

                char = word[0]
                rem_word = word[1:]

                char_node = self.get_char_node(char)

                if ind == 0:
                    parent_c_node = char_node
                    same_starting_chars_dict[char].append(word[1:])
                    continue

                if char == parent_c_node.val:
                    same_starting_chars_dict[char].append(word[1:])
                    continue

                parent_c_node.next.append(char_node)
                parent_c_node = char_node
                char_node.indegree += 1
                same_starting_chars_dict[char].append(word[1:])

            print('\n', same_starting_chars_dict)

            for key, value in same_starting_chars_dict.items():
                if len(value) == 1:
                    continue
                queue.append(value)

            # print(queue)
            '''
            for _, node in self.char_node_map.items():
                print(node.val)
                print(node.next)
                print(node.indegree)
            '''

        order = []
        visited_chars = set()

        for char in self.char_node_map:
            visited_chars.add(char)

        while (self.char_node_map):
            zero_indeg_node = self.get_zero_indeg_nodes()

            if not zero_indeg_node:
                return ''

            order.append(zero_indeg_node.val)

            for neighbor in zero_indeg_node.next:
                if neighbor.indegree == 1:
                    neighbor.indegree = 0
                else:
                    neighbor.indegree -= 1

        for word in words:
            for char in word:
                if char not in visited_chars:
                    order.append(char)

        return ''.join(order)

# CHeckout leetcode solutions tab for algorithm. We are using topological sort

'''
261 Graph Valid Tree
https://leetcode.com/problems/graph-valid-tree/

Given n nodes labeled from 0 to n-1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.
'''
# approach 1 - DFS where we skip the parent node of a node since Parent.neighbors will have Child and Child.neighbors will contain Parent O(V + E)
def validTree(self, n: int, edges: List[List[int]]) -> bool:
    if len(edges) != n - 1: return False

    adj_list = [[] for _ in range(n)]
    for A, B in edges:
        adj_list[A].append(B)
        adj_list[B].append(A)

    seen = set()

    def dfs(node, parent):
        if node in seen: return;
        seen.add(node)
        for neighbour in adj_list[node]:
            if neighbour == parent:
                continue
            if neighbour in seen:
                return False
            result = dfs(neighbour, node)
            if not result: return False
        return True

    # We return true iff no cycles were detected,
    # AND the entire graph has been reached.
    return dfs(0, -1) and len(seen) == n


# approach 2 - union find - O (V + (E * V))
class Solution:
    def get_parent(self, node):
        # print('get_parent of = ', node)
        if self.elem_parent_dict[node] == -1:
            return node

        return self.get_parent(self.elem_parent_dict[node])

    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if len(edges) < n - 1:
            return False

        self.elem_parent_dict = {}

        for i in range(n):
            self.elem_parent_dict[i] = -1

        for edge in edges:
            node_1 = edge[0]
            node_2 = edge[1]

            parent_node_1 = self.get_parent(node_1)
            parent_node_2 = self.get_parent(node_2)

            if parent_node_1 == parent_node_2:
                return False

            else:
                self.elem_parent_dict[parent_node_1] = parent_node_2

        return True

# time O (V + (E * V)) For each edge in the graph we call union find to find out if there exisits a cylce in the graph
# space O(V)


# Extension or Application fo the prev problem - We use union find in Kruskal algo to find min spanning tree of a graph as below
'''
Amazon OA
https://leetcode.com/discuss/interview-question/796241/Amazon-OA2-SDE-1(New-Grad)-2021-(Coding-2Questions-70mins)
https://leetcode.com/problems/connecting-cities-with-minimum-cost/discuss/344867/Java-Kruskal's-Minimum-Spanning-Tree-Algorithm-with-Union-Find
https://leetcode.com/problems/connecting-cities-with-minimum-cost/discuss/344835/Python-simple-Union-Find-Solution
'''

from collections import defaultdict


class GetShortestRoutes():
    def __init__(self, connections, n):
        self.parents_dict = defaultdict(str)
        self.curr_connections = connections
        self.min_required_connections = []
        self.num_sets = len(connections)

    def find_parent(self, server_node):
        if self.parents_dict[server_node] == server_node: return server_node

        return self.find_parent(self.parents_dict[server_node])

    def combine_sets(self, server_node_1, server_node_2, parent_1, parent_2):

        if parent_1 < parent_2:
            self.parents_dict[parent_2] = parent_1
        else:
            self.parents_dict[parent_1] = parent_2
        self.num_sets -= 1

    def form_min_server_connections(self):
        curr_connections_tups = []

        for con in self.curr_connections:
            server_1 = con[0]
            server_2 = con[1]
            tup = (con[2], server_1, server_2)
            curr_connections_tups.append(tup)
            self.parents_dict[server_1] = server_1
            self.parents_dict[server_2] = server_2

        curr_connections_tups.sort()

        for con_tup in curr_connections_tups:
            server_1 = con_tup[1]
            server_2 = con_tup[2]
            val = con_tup[0]
            parent_server_1 = self.find_parent(server_1)
            parent_server_2 = self.find_parent(server_2)

            if parent_server_1 == parent_server_2:
                continue
            else:
                self.combine_sets(server_1, server_2, parent_server_1, parent_server_2)
                self.min_required_connections.append([server_1, server_2, val])

        print('req conn = ', self.min_required_connections)

        if self.num_sets == 1:
            return self.min_required_connections
        else:
            return []


connections = [
    ['a', 'b', 1],
    ['b', 'c', 4],
    ['b', 'd', 6],
    ['d', 'e', 5],
    ['c', 'e', 1]]
connections = [[1, 2, 5], [1, 3, 6], [2, 3, 1]]
connections = [[1, 2, 3], [3, 4, 4]]
gsr = GetShortestRoutes(connections, len(connections))
print(gsr.form_min_server_connections())

# time O (V + (E * V)) For each edge in the graph we call union find to find out if there exisits a cylce in the graph

'''
323. Number of Connected Components in an Undirected Graph
https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/submissions/

Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write a function to find the number of connected components in an undirected graph.
'''

class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        self.num_disjoint_sets = n
        parent_dict = {}
        node_rank_dict = {}

        def find_parent(node):
            if parent_dict[node] == -1:
                return node

            return find_parent(parent_dict[node])

        def union(node_1, node_2):
            if node_rank_dict[node_1] > node_rank_dict[node_2]:
                parent_dict[node_2] = node_1

            elif node_rank_dict[node_2] > node_rank_dict[node_1]:
                parent_dict[node_1] = node_2

            else:
                parent_dict[node_1] = node_2
                node_rank_dict[node_2] += 1  # We increase rank only when 2 nodes are of the same rank. Try to visualize in you mind to understand better. Think of rank as height

            self.num_disjoint_sets -= 1  # For some reason this variable was not accessible without self

        for i in range(n):
            parent_dict[i] = -1
            node_rank_dict[i] = 0

        for edge in edges:
            node_1 = edge[0]
            node_2 = edge[1]
            parent_1 = find_parent(node_1)
            parent_2 = find_parent(node_2)

            if parent_1 != parent_2:
                union(parent_1, parent_2)

        return self.num_disjoint_sets

# time: O(E log V)
# space: O(V)


'''
133. Clone Graph
https://leetcode.com/problems/clone-graph/
'''

from collections import deque

"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""


class Solution:
    # approach 1
    def cloneGraph(self, node):
        if not node:
            return None

        queue = deque()
        visited = {}
        visited[node] = Node(node.val)
        queue.append(node)

        while (queue):
            n = queue.popleft()

            for neighbor in n.neighbors:
                if neighbor not in visited:
                    visited[neighbor] = Node(neighbor.val)
                    queue.append(neighbor)

                visited[n].neighbors.append(visited[neighbor])

        return visited[node]

    # approach 2
    def __init__(self):
        # Dictionary to save the visited node and it's respective clone
        # as key and value respectively. This helps to avoid going in an infinite loop.
        self.visited = {}

    def cloneGraph(self, node):
        if not node:
            return node

        if node in self.visited:
            return self.visited[node]

        clone_node = Node(node.val, [])

        self.visited[node] = clone_node

        if node.neighbors:
            clone_node.neighbors = [self.cloneGraph(n) for n in node.neighbors]

        return clone_node


# time O(n)
# space O(n)


# --------------------------------------------------------------------------- End of Graph block ---------------------------------------------------------------------------------------------















# --------------------------------------------------------------------------- Tree block ---------------------------------------------------------------------------------------------
'''
100 Same Tree
https://leetcode.com/problems/same-tree/
'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        self.same_tree = True

        def helper(p, q):
            if (p and q == None) or (q and p == None):
                self.same_tree = False
                return

            if not p and not q:
                return

            if p.val != q.val or self.same_tree == False:
                self.same_tree = False
                return

            helper(p.left, q.left)
            helper(p.right, q.right)

        helper(p, q)
        return self.same_tree

# time O(min(m,n))
# space O(min(m,n))

'''
226. Invert Binary Tree
https://leetcode.com/problems/invert-binary-tree/

testcase: ip:[1,2] op:[1, null, 2]
'''

class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None

        def helper(node):
            if not node:
                return None

            temp_left = node.left
            node.left = helper(node.right)
            node.right = helper(temp_left)

            return node

        helper(root)
        return root

# eg testcase: ip:[1,2] op:[1, null, 2]
# time O(n)
# space O(n)


'''
124. Binary Tree Maximum Path Sum
https://leetcode.com/problems/binary-tree-maximum-path-sum/
'''

class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        self.max_sum = float('-inf')

        def helper(node):
            if not node:
                return 0

            left_sum = helper(node.left)
            right_sum = helper(node.right)
            curr_sum = left_sum + right_sum + node.val
            curr_sum_left = left_sum + node.val
            curr_sum_right = right_sum + node.val
            curr_max = max(node.val, curr_sum, curr_sum_left, curr_sum_right)
            self.max_sum = max(curr_max, self.max_sum)
            max_return_value = curr_max = max(node.val, curr_sum_left, curr_sum_right)

            return curr_max

        helper(root)
        return self.max_sum

# time O(n)
# space O(n)


'''
102 Binary Tree Level Order Traversal
https://leetcode.com/problems/binary-tree-level-order-traversal/submissions/
'''
from collections import defaultdict, deque


class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        levels_dict = defaultdict(list)
        queue = deque()
        levels_dict[0].append(root.val)
        queue.append((root, 0))
        max_ht = 0
        levels_list = []

        while queue:
            node, ht = queue.popleft()
            child_ht = ht + 1
            max_ht = max(max_ht, child_ht)

            if node.left:
                levels_dict[child_ht].append(node.left.val)
                queue.append((node.left, child_ht))

            if node.right:
                levels_dict[child_ht].append(node.right.val)
                queue.append((node.right, child_ht))

        for i in range(max_ht):
            levels_list.append(levels_dict[i])

        return levels_list


'''
297. Serialize and Deserialize Binary Tree
https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
'''


class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """

        def helper(node):
            if not node:
                return 'None'

            left_string = helper(node.left)
            right_string = helper(node.right)

            return [node.val, left_string, right_string]

        serialized_tree = json.dumps(helper(root))

        return serialized_tree

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """

        def helper(node_list):
            if node_list == 'None':
                return None

            node_val = node_list[0]
            node_obj = TreeNode(node_val)
            node_obj.left = helper(node_list[1])
            node_obj.right = helper(node_list[2])

            return node_obj

        data = json.loads(data)
        return helper(data)


# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))

# time: O(n)
# space: O(m) m is equal to the string len of concatenation of all the values in the tree


572. Subtree of Another tree (Amazon - Kevin youtube)
https://leetcode.com/problems/subtree-of-another-tree/

#We can also do this problem with inorder traversal. Doing inorder traversal of both the
#trees and storing the values in 2 different lists and checking if the list of subtree
#is a sublist of parent inorder list. In that case we need to store the None value of the
#leaf node's left and right child seperately. Otherwise we wont get correct result

# appraoch 1
is_sub_tree = False

def check_if_subtree(root, c_root):
    if not root and not c_root:
        return True

    if type(root) != type(c_root) or root.val != c_root.val:
        return False

    return check_if_subtree(root.left, c_root.left) and \
    check_if_subtree(root.right, c_root.right)

def do_dfs_parent(root):
    global is_sub_tree

    if not root:
        return None

    if root.val == c_root.val:
        if check_if_subtree(root, c_root):
            is_sub_tree = True
            return True

    do_dfs_parent(root.left)
    do_dfs_parent(root.right)

do_dfs_parent(p_root)
print (is_sub_tree)

# time: O(nm) - worst case we might end up checking if its a subtree of parent tree for every node of the parent tree. If n is the number of nodes in the parent tree and m is
# the number of nodes in the child tree, we will check if m nodes in the sub tree occur in the parent tree for each node in parent tree whose size is n

# space = O(n) - At any point during the recursion, your stack will not be larger than n (assuming n is larger than m)

# approach 2

class Solution(object):

    def isSubtree(self, s, t):
        main_tree = []
        sub_tree = []
        my_l = []

        def preorder(node):
            if not node:
                my_l.append("Null")
                return
            my_l.append(node.val)
            preorder(node.left)
            preorder(node.right)

        preorder(s)
        main_tree = my_l
        my_l = []
        preorder(t)
        sub_tree = my_l
        # have to check if subtree is a sublist of main tree
        main_tree = "_".join(str(i) for i in main_tree)
        main_tree = '_' + main_tree
        sub_tree = "_".join(str(i) for i in sub_tree)
        sub_tree = '_' + sub_tree
        return sub_tree in main_tree

# time O(nm)
# space O(max(n,m))

'''
105 Construct Binary Tree from Preorder and Inorder Traversal
https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
'''


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        # In each recursive call, we always pop out the first univisited preorder elem and use it as root
        # https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/discuss/34579/Python-short-recursive-solution

        # approach 0 - more intutive approach time: O(n ^ 2) check the first comment in the prev link
        preorder.reverse()

        def helper(inorder):
            if not inorder:
                return

            curr_root_ele = preorder.pop()
            curr_root = TreeNode(curr_root_ele)
            root_inorder_index = inorder.index(curr_root_ele)
            curr_root.left = helper(inorder[:root_inorder_index])
            curr_root.right = helper(inorder[root_inorder_index + 1:])

            return curr_root

        return helper(inorder)

        # approach 1 time: O(n) space: O(n) - Improvement to the prev approach. We use indices instead of sliced array in recursive calls
        if not preorder:
            return None

        preorder.reverse()
        inorder_dict = {val: ind for ind, val in enumerate(inorder)}

        def helper(inorder_start, inorder_end):

            if inorder_start > inorder_end:
                return None

            preord_ele = preorder.pop()

            preorder_node = TreeNode(preord_ele)
            preorder_node.left = helper(inorder_start, inorder_dict[preord_ele] - 1)
            preorder_node.right = helper(inorder_dict[preord_ele] + 1, inorder_end)

            return preorder_node

        return helper(0, len(inorder) - 1)

        # approach 2 time: O(n ^ 3) space: O(n) very ineffective approach. We were unclear if taking the first unvisited elem in preorder list will be our current root. So, it became very ineffective

        preorder_dict = {}

        for ind, num in enumerate(preorder):
            preorder_dict[num] = ind

        def helper(curr_inorder):
            if not curr_inorder:
                return None

            earliest_index = float('+inf')

            for ind, ele in enumerate(curr_inorder):
                if preorder_dict[ele] < earliest_index:
                    earliest_index = preorder_dict[ele]
                    earlies_occuring_ele = ele
                    root_inorder_list_ind = ind

            node = TreeNode(earlies_occuring_ele)
            node.left = helper(curr_inorder[:root_inorder_list_ind])
            node.right = helper(curr_inorder[root_inorder_list_ind + 1:])

            return node

        return helper(inorder)


'''
106 Construct Binary Tree from Inorder and Postorder Traversal
https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
'''


class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        # approach 1 time: O(n ^ 2) space O(n)

        def helper(inorder):
            if not inorder:
                return

            curr_root_ele = postorder.pop()
            curr_root = TreeNode(curr_root_ele)
            root_inorder_index = inorder.index(curr_root_ele)
            curr_root.right = helper(inorder[root_inorder_index + 1:])
            curr_root.left = helper(inorder[:root_inorder_index])

            return curr_root

        return helper(inorder)

        # approach 2

        inorder_dict = {val: ind for ind, val in enumerate(inorder)}

        def helper_2(inorder_st_ind, inorder_en_ind):
            if inorder_st_ind > inorder_en_ind:
                return None

            curr_root_ele = postorder.pop()
            curr_root = TreeNode(curr_root_ele)
            curr_root.right = helper_2(inorder_dict[curr_root_ele] + 1, inorder_en_ind)
            curr_root.left = helper_2(inorder_st_ind, inorder_dict[curr_root_ele] - 1)

            return curr_root

        return helper_2(0, len(inorder) - 1)

# approach 2
# time O(n)
# space O(n) -> inorder dict as well as recursion stack


'''
98 Validate Binary Search Tree
https://leetcode.com/problems/validate-binary-search-tree/submissions/
'''

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:

        def helper(node, min_val, max_val):
            if not node:
                return True

            if node.val >= max_val or node.val <= min_val:
                return False

            return helper(node.left, min_val, node.val) and helper(node.right, node.val, max_val)

        return helper(root, float('-inf'), float('+inf'))

# time O(n)
# space O(n)


'''
230 Kth Smallest Element in a BST
https://leetcode.com/problems/kth-smallest-element-in-a-bst/
'''

class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        num_visited_elems = 0
        kth_elem = None

        def helper(node):
            nonlocal num_visited_elems
            nonlocal kth_elem

            if kth_elem != None:
                return

            if not node:
                return

            helper(node.left)
            num_visited_elems += 1

            if num_visited_elems == k:
                kth_elem = node.val
                return

            helper(node.right)

        helper(root)
        return kth_elem


# time: O(n)
# space: O(n)

# approach 2 - stack based
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        stack = []
        node = root

        while (node): # It's just node NOT NODE.LEFT
            stack.append(node)
            node = node.left

        while (True):
            curr_node = stack.pop()
            k -= 1

            if k == 0:
                return curr_node.val

            elif curr_node.right:
                node = curr_node.right

                while (node): # It's just node NOT NODE.LEFT
                    stack.append(node)
                    node = node.left

# time O(n) # this is worst case. In avg case, it performs better because there are no un wanted recursive calls. Avg case: O(H + K) refer solution tab leetcode
# space O(n) # size of stack atmost height of tree. worst case left sided tree

'''
94 Binary Tree Inorder Traversal
https://leetcode.com/problems/binary-tree-inorder-traversal/
'''


class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        # approach 1
        op = []

        def helper(node):
            nonlocal op

            if not node:
                return

            helper(node.left)
            op.append(node.val)
            helper(node.right)

        # helper(root)

        # approach 2
        stack = []

        if root: stack.append(root)
        visited = set()

        while (stack):
            node = stack.pop()

            while (node and id(node) not in visited):  # pay close attention to the use of visited here
                stack.append(node)
                node = node.left

            node = stack.pop()
            op.append(node.val)
            visited.add(id(node))
            node = node.right

            if node: stack.append(node)

        return op


# time: O(n)
# space: O(n)

'''
235. Lowest Common Ancestor of a Binary Search Tree
https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
'''

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        lca = None

        def helper(node):
            nonlocal lca

            if lca:
                return None, None

            if not node:
                return None, None

            node_1_found_left, node_2_found_left = helper(node.left)
            node_1_found_right, node_2_found_right = helper(node.right)

            if node.val == p.val:
                node_1_found = True
            else:
                node_1_found = False

            if node.val == q.val:
                node_2_found = True
            else:
                node_2_found = False

            node_1_found = node_1_found or node_1_found_left or node_1_found_right
            node_2_found = node_2_found or node_2_found_left or node_2_found_right

            if node_1_found and node_2_found and not lca:
                lca = node

            return node_1_found, node_2_found

        helper(root)
        return lca

# time O(n)
# space O(n)

'''
208 Implement Trie (Prefix Tree)
https://leetcode.com/problems/implement-trie-prefix-tree/

Trie trie = new Trie();

trie.insert("apple");
trie.search("apple");   // returns true
trie.search("app");     // returns false ******PAY CLOSE ATTENTION TO THIS EXAMPLE******
trie.startsWith("app"); // returns true
trie.insert("app");   
trie.search("app");     // returns true
'''


class TrieNode:
    def __init__(self, val, end_of_word=False):
        self.val = val
        self.next = {}
        self.end_of_word = end_of_word


class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie_root = TrieNode('')

    def insert(self, word: str) -> None:  # time O(m) space O(m)
        """
        Inserts a word into the trie.
        """
        i = 0
        parent = self.trie_root

        while (i < len(word)):
            char = word[i]

            if char in parent.next:
                trie_node = parent.next[char]
            else:
                trie_node = TrieNode(char)
                parent.next[char] = trie_node

            parent = trie_node
            i += 1

        trie_node.end_of_word = True

    def search(self, word: str) -> bool:  # time O(m) space O(1)
        """
        Returns if the word is in the trie.
        """
        # print('search word = ', word)
        parent = self.trie_root

        for char in word:
            if char in parent.next:
                parent = parent.next[char]
            else:
                # print('ret false due to char = ', char)
                return False

        if parent.end_of_word == True:
            return True
        else:
            return False

    def startsWith(self, prefix: str) -> bool:  # time O(m) space O(1)
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        parent = self.trie_root

        for char in prefix:
            if char in parent.next:
                parent = parent.next[char]
            else:
                return False

        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)


'''
211. Design Add and Search Words Data Structure
https://leetcode.com/problems/design-add-and-search-words-data-structure/
'''


class TrieNode:
    def __init__(self, val):
        self.val = val
        self.next = {}
        self.word_ending = False


class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode(1)

    def addWord(self, word: str) -> None:  # time O(M), M is len of word
        """
        Adds a word into the data structure.
        """
        i = 0
        trie_node = self.root

        while (i < len(word)):
            ch = word[i]

            if ch in trie_node.next:
                trie_node = trie_node.next[ch]
            else:
                new_trie_node = TrieNode(ch)
                trie_node.next[ch] = new_trie_node
                trie_node = new_trie_node

            i += 1

        trie_node.word_ending = True # ************************************ IMPTNT ***********************************************************

    def search(self, word: str) -> bool:  # time O(N), N is the num of words added till now. Space: O(M) M is len of search word
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        """
        trie_node = self.root
        word_found = False

        def dfs(trie_node, ind):
            nonlocal word_found

            if word_found:
                return True

            if ind == len(word):
                if trie_node.word_ending:
                    word_found = True
                    return
                else:
                    return

            ch = word[ind]

            if ch == '.':
                for next_key, next_node in trie_node.next.items():
                    if word_found:  # This is just a practical performance enhancement
                        return
                    else:
                        dfs(next_node, ind + 1)


            else:
                if trie_node.next.get(ch):
                    return dfs(trie_node.next[ch], ind + 1)
                else:
                    return

        dfs(trie_node, 0)
        return word_found


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)


'''
101 Symmetric Tree
https://leetcode.com/problems/symmetric-tree/
'''

from collections import deque, defaultdict

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        inorder = []  # [n,3,n,2,n,4,n], [1], [n,4,n,2,n,3,n]

        queue = deque()
        level_order_dict = defaultdict(list)
        queue.append((root, 0))

        while (queue):
            node, level = queue.popleft()

            if not node:
                level_order_dict[level].append('null')
                continue

            level_order_dict[level].append(node.val)
            queue.append((node.left, level + 1))
            queue.append((node.right, level + 1))

        for _, level_elems in level_order_dict.items():

            num_eles_in_level = len(level_elems)

            for i, ele in enumerate(level_elems):
                if i == num_eles_in_level // 2: break

                apt_ele_from_last = level_elems[num_eles_in_level - i - 1]

                if ele != apt_ele_from_last:
                    return False

        return True

        # The below approach is using dfs. It failed for a few cases. See submissions tab for more details
        '''def helper(node):
            nonlocal inorder

            if not node:
                inorder.append('null')
                return

            if not node.left and not node.right:
                inorder.append(node.val)
                return

            helper(node.left)
            inorder.append(node.val)
            helper(node.right)

        helper(root)
        len_inorder = len(inorder)

        for i, ele in enumerate(inorder):
            if i == len_inorder // 2: break

            apt_ele_from_last = inorder[len_inorder - i - 1]

            if ele != apt_ele_from_last:
                return False

        return True
        '''


# time O(n)
# space O(n)

'''
103 Binary Tree Zigzag Level Order Traversal
https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
'''
from collections import defaultdict, deque


class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        queue = deque()
        level_order_dict = defaultdict(list)
        queue.append((root, 0))
        max_level = 0
        op = []

        while (queue):
            node, level = queue.popleft()

            if not node:
                continue

            level_order_dict[level].append(node.val)
            queue.append((node.left, level + 1))
            queue.append((node.right, level + 1))
            max_level = max(max_level, level)

        for level in range(max_level + 1):
            if level & 1 == 0:
                op.append(level_order_dict[level])
            else:
                level_order_dict[level].reverse()
                op.append(level_order_dict[level])

        return op


# time O(n)
# space O(n)


'''
108 Convert Sorted Array to Binary Search Tree
https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/submissions/
'''


class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        def helper(st, en):
            if st > en:
                return None

            mid = (st + en) // 2
            root = TreeNode(nums[mid])
            root.left = helper(st, mid - 1)
            root.right = helper(mid + 1, en)

            return root

        return helper(0, len(nums) - 1)


# time O(n)
# space O(n)


'''
116 Populating Next Right Pointers in Each Node
https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
'''
from collections import deque


class Solution:
    def connect(self, root: 'Node') -> 'Node':
        queue = deque()
        queue.append((root, 0))
        prev_node = None
        prev_node_ht = -1

        while (queue):
            node, curr_node_ht = queue.popleft()

            if not node:
                continue

            if prev_node_ht != curr_node_ht:
                prev_node = node
                prev_node_ht = curr_node_ht

            else:
                prev_node.next = node
                prev_node = node

            queue.append((node.left, curr_node_ht + 1))
            queue.append((node.right, curr_node_ht + 1))

        return root


# time O(n)
# space O(n)


'''
116 Populating Next Right Pointers in Each Node
https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
'''
from collections import deque


class Solution:
    def connect(self, root: 'Node') -> 'Node':
        # approach 1 time: O(n) space O(1)
        if not root or not root.left:
            return root

        root.left.next = root.right
        node = root.left
        same_level_prev_node = None
        leftmost_node_next_level = None

        while (node):
            if not node.left:  # No need do anything in the last level
                break

            node.left.next = node.right

            if same_level_prev_node:
                same_level_prev_node.next = node.left
                same_level_prev_node = node.right
            else:
                same_level_prev_node = node.right

            if not leftmost_node_next_level:
                leftmost_node_next_level = node.left

            if node.next:
                node = node.next

            elif leftmost_node_next_level:  # moving to the next level of the tree
                node = leftmost_node_next_level
                leftmost_node_next_level = None
                same_level_prev_node = None

            else:
                node = None

        return root

        # approach 1 written in a better way
        leftmost = root

        # Once we reach the final level, we are done
        while leftmost.left:

            # Iterate the "linked list" starting from the head
            # node and using the next pointers, establish the 
            # corresponding links for the next level
            head = leftmost
            while head:

                # CONNECTION 1
                head.left.next = head.right

                # CONNECTION 2
                if head.next:
                    head.right.next = head.next.left

                # Progress along the list (nodes on the current level)
                head = head.next

            # Move onto the next level
            leftmost = leftmost.left

        # approach 2 time O(n) space O(n)
        '''queue = deque()
        queue.append((root,0))
        prev_node = None
        prev_node_ht = -1

        while(queue):
            node, curr_node_ht = queue.popleft()

            if not node:
                continue

            if prev_node_ht != curr_node_ht:
                prev_node = node
                prev_node_ht = curr_node_ht

            else:
                prev_node.next = node
                prev_node = node

            queue.append((node.left, curr_node_ht + 1))
            queue.append((node.right, curr_node_ht + 1))

        return root'''


# time O(n)
# space O(n) If we don't take recursion stack or queue stack into account, then the space is O(1)

# --------------------------------------------------------------------------- End of Tree block ---------------------------------------------------------------------------------------------






















# ------------------------------------------------------------- LinkedList block ---------------------------------------------------------------------------------------------

'''
206. Reverse Linked List (recursively)
https://leetcode.com/problems/reverse-linked-list/
'''


class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        self.new_root = None

        def helper(node, prev_node):
            if not node:
                return None

            helper(node.next, node)
            node.next = prev_node

            if not self.new_root:
                self.new_root = node

            return

        helper(head, None)
        return self.new_root

    # Iterative sol
    prev_node = None
    node = head

    while (node):
        temp = node.next
        node.next = prev_node
        prev_node = node
        node = temp

    return prev_node

# time O(n)
# space O(n)

'''
141. Linked List Cycle
https://leetcode.com/problems/linked-list-cycle/
'''

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        slow_ptr = fast_ptr = head

        if slow_ptr and slow_ptr.next == None:  # when there is only one elem or when the list is empty, there is no cycle
            return False

        while (fast_ptr != None):
            slow_ptr = slow_ptr.next

            if fast_ptr.next != None:
                fast_ptr = fast_ptr.next.next
            else:
                fast_ptr = fast_ptr.next

            if slow_ptr == fast_ptr:
                return True

        return False

# time O(n)
# space O(1)


'''
23 Merge k Sorted Lists
https://leetcode.com/problems/merge-k-sorted-lists/
'''
import heapq

class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """

        head = point = ListNode(0)
        q = []

        for l in lists:
            if l:
                heapq.heappush(q, (l.val, id(l), l))

        while q:
            val, _id, node = heapq.heappop(q)
            point.next = ListNode(val)
            point = point.next
            node = node.next

            if node:
                heapq.heappush(q, (node.val, id(node), node))

        return head.next

# time worst case: (N log N) N is the total number of elems in all the sub lists. This scenario occurs [[1,1,1], [1,1,1], [1,1,1]]
# time avg case: (N log k) where k is the number of linked lists in the input [[-1,0,1], [2,3,4], [5,6,7]]
# space O(N)

# ------------------------------------------------------------- END of LinkedList block ---------------------------------------------------------------------------------------------








# ------------------------------------------------------------- Matrix block ---------------------------------------------------------------------------------------------
'''
1582. Special Positions in a Binary Matrix
https://leetcode.com/problems/special-positions-in-a-binary-matrix/

Given a rows x cols matrix mat, where mat[i][j] is either 0 or 1, return the number of special positions in mat.

A position (i,j) is called special if mat[i][j] == 1 and all other elements in row i and column j are 0 (rows and columns are 0-indexed).
'''
from collections import defaultdict


class Solution:
    def numSpecial(self, mat: List[List[int]]) -> int:
        rows_dict = defaultdict(int)
        cols_dict = defaultdict(int)
        num_rows = len(mat)
        num_cols = len(mat[0])
        res = 0

        for i in range(num_rows):
            for j in range(num_cols):
                if mat[i][j] == 1:
                    rows_dict[i] += 1
                    cols_dict[j] += 1

        for i in range(num_rows):
            for j in range(num_cols):
                if mat[i][j] == 1 and rows_dict[i] == 1 and cols_dict[j] == 1:
                    res += 1

        return res

# time O(n * m)
# space O(n + m)

'''
48. Rotate Image
https://leetcode.com/problems/rotate-image/
'''


class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # start state
        """ 
        00 01 02
        10 11 12
        20 21 22
        """

        # end state
        """
        20 10 00
        21 11 01
        22 12 02
        """

        # reverse the rows
        '''
        02 01 00
        12 11 10
        22 21 20
        '''
        # interchange rows and cols
        """
        20 10 00
        21 11 01
        22 12 02
        """
        len_row = len_col = len(matrix)

        for i in range(len_row):
            for j in range(len_col):
                if i < j:
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

            matrix[i].reverse()


# time O(n ^ 2) where n is the num of rows OR the num of cols in matrix
# space O(1)


'''
54 Spiral Matrix
https://leetcode.com/problems/spiral-matrix/
'''


class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        '''
        00 01 02
        10 11 12
        20 21 22
        30 31 32
        '''

        # matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        if not matrix:
            return []

        m = len(matrix)
        n = len(matrix[0])
        total_unvisited_eles_in_mat = m * n
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        i = 0
        j = -1  # ********************* NOTICE -1 ************
        directions_ind = 0
        op = []

        while (total_unvisited_eles_in_mat > 0):
            curr_dir = directions[directions_ind]
            i += curr_dir[0]
            j += curr_dir[1]

            while (True):
                op.append(matrix[i][j])
                total_unvisited_eles_in_mat -= 1
                matrix[i][j] = None

                new_i = i + curr_dir[0]
                new_j = j + curr_dir[1]

                if (new_i < 0 or new_i >= m) or (new_j < 0 or new_j >= n) or matrix[new_i][new_j] == None:
                    break  # *********************** condition to break out of loop is in if ***********************

                i = new_i
                j = new_j

            directions_ind += 1

            if directions_ind >= 4:
                directions_ind = 0

        return op

# time O(n)
# space O(1)


'''
73. Set Matrix Zeroes
https://leetcode.com/problems/set-matrix-zeroes/
'''

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        if not matrix:
            return []

        num_rows = len(matrix)
        num_cols = len(matrix[0])

        # appraoch 1 # time O(n ^ 2 * m ^ 2)


        def set_row_zero(row_num):
            for c in range(num_cols):
                if matrix[row_num][c] != 0:
                    matrix[row_num][c] = None

        def set_col_zero(col_num):
            for r in range(num_rows):
                if matrix[r][col_num] != 0:
                    matrix[r][col_num] = None

        for i in range(num_rows):
            for j in range(num_cols):
                if matrix[i][j] == 0:
                    set_row_zero(i)
                    set_col_zero(j)

        for i in range(num_rows):
            for j in range(num_cols):
                if matrix[i][j] == None:
                    matrix[i][j] = 0


        # approach 2 O(nm)
        first_row_zero = False
        first_col_zero = False

        if 0 in matrix[0]:
            first_row_zero = True

        for i in range(num_rows):
            if matrix[i][0] == 0:
                first_col_zero = True

        #print(first_col_zero)

        for i in range(1, num_rows):
            for j in range(1, num_cols):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0

        #print(matrix)

        for i in range(1, num_rows):
            for j in range(1, num_cols):

                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0

        if first_row_zero:
            for j in range(num_cols):
                matrix[0][j] = 0

        if first_col_zero:
            for i in range(num_rows):
                matrix[i][0] = 0

        #print(matrix)

        #print(matrix)

        return matrix

# time O(nm)


'''
79 Word Search
https://leetcode.com/problems/word-search/
'''


class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        self.word_found = False
        len_rows = len(board)
        len_cols = len(board[0])
        len_word = len(word)
        all_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        visited = set()

        def dfs(cell, word_ind):
            if word_ind == len_word:
                self.word_found = True
                return True

            next_char = word[word_ind]

            for d in all_dirs:
                try:
                    new_row = cell[0] + d[0]
                    new_col = cell[1] + d[1]
                    tup = (new_row, new_col)

                    if board[new_row][new_col] == next_char and self.word_found == False and tup not in visited and new_row > -1 and new_col > -1:
                        visited.add(tup)
                        dfs(tup, word_ind + 1)
                        visited.remove(tup)

                except IndexError:
                    pass

        for i in range(len_rows):
            for j in range(len_cols):
                if self.word_found == False and board[i][j] == word[0]:
                    visited = set()
                    visited.add((i, j))
                    dfs((i, j), 1)

        return self.word_found


# time O(nm * 4 ** l) For each cell in the matrix of size (n * m), you might do 4 ** l operations where l is the len of word
# space O(nm)
# ------------------------------------------------------------- END OF Matrix block -------------------------------------------------------------------------------------------








# ------------------------------------------------------------- Stacks block -------------------------------------------------------------------------------------------

'''
1130. Minimum Cost Tree From Leaf Values
https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/
'''
class Solution:
    def mctFromLeafValues(self, A):
        # appraoch 2 time O(n) (Not mine - don't copy this approach - plaguarism issue)
        res = 0
        stack = [float('inf')]

        for a in A:

            while stack[-1] <= a:
                mid = stack.pop()
                res += mid * min(stack[-1], a)

            stack.append(a)

        while len(stack) > 2:
            res += stack.pop() * stack[-1]
        return res

    # approach 1 O(n ^ 2)
    def mctFromLeafValues(self, arr: List[int]) -> int:
        res = 0

        while (len(arr) > 1):
            curr_min = min(arr)
            index = arr.index(curr_min)

            if index == 0:
                other_leaf = arr[index + 1]

            elif index == len(arr) - 1:
                other_leaf = arr[index - 1]

            elif arr[index + 1] < arr[index - 1]:
                other_leaf = arr[index + 1]

            else:
                other_leaf = arr[index - 1]

            arr.pop(index)
            prod_of_leaves = curr_min * other_leaf
            res += prod_of_leaves

        return res


'''
42. Trapping Rain Water
https://leetcode.com/problems/trapping-rain-water/solution/
'''
# See the solutions tab to understand stack approach. Start reading from Brute Force, Dynamic Prog and then read stack based approach to understand easily
class Solution:
    def trap(self, height: List[int]) -> int:
        # approach 1 time O(n) space O(1)
        stack = []
        total_water = 0

        for i, h in enumerate(height):
            while (stack and stack[-1][1] < h):
                top_i, top_h = stack.pop()

                if not stack:
                    break

                min_boundary_h = min(h, stack[-1][1])# ********************** imptnt, our right boundary is 'i', left boundary is stack[-1] *************
                rect_h = min_boundary_h - top_h
                w = i - stack[-1][0] - 1 # ********************** imptnt, our right boundary is 'i', left boundary is stack[-1] *************
                total_water += rect_h * w

            stack.append((i, h))

        return total_water


        # approach 2 # time O(n) space O(n)
        total_water = 0
        forward_array = []
        backward_array = []

        for h in height:
            if forward_array == [] or forward_array[-1] < h:
                forward_array.append(h)
            else:
                forward_array.append(forward_array[-1])

        height.reverse()

        for h in height:
            if backward_array == [] or backward_array[-1] < h:
                backward_array.append(h)
            else:
                backward_array.append(backward_array[-1])

        height.reverse()
        backward_array.reverse()

        for i, h in enumerate(height):
            height_of_water_at_index = min(forward_array[i], backward_array[i]) - h

            if height_of_water_at_index > 0: total_water += height_of_water_at_index

        return total_water



# --------------------------------------------------------------- Backtracking Block ------------------------------------------------------------------------------------------------------------------------

'''
Efficient Janitor - Roblox
https://leetcode.com/discuss/interview-question/490066/Efficient-Janitor-Efficient-Vineet-(Hackerrank-OA)
https://leetcode.com/discuss/interview-question/452959/factset-oa-efficient-janitor-problem
'''
def getMinTrips(weights, max):
    global res
    visited = [False for i in range(len(weights))]
    dfs(weights, visited, 0.0, 1, max)
    return res


def dfs(weights, visited, w, tmp, max):
    global res
    if tmp > res:
        return

    if isAllVisited(visited):
        res = min(res, tmp)
        return

    i = 0

    while(i < len(weights)):
        if visited[i] == False:
            visited[i] = True

            if w + weights[i] <= max:
                dfs(weights, visited, w + weights[i], tmp, max)
            else:
                dfs(weights, visited, weights[i], tmp + 1, max)

            visited[i] = False

        i += 1


def isAllVisited(visited):
    if False in visited:
        return False
    else:
        return True


#weights = [1.99, 1.01, 2.5, 1.5, 1.01]
weights = [2.8,2.7,0.1,0.05,0.15,0.2]
max = 3.0
res = float('+inf')
print(getMinTrips(weights, max))

'''
526 Beautiful Arrangement
https://leetcode.com/problems/beautiful-arrangement/
'''


class Solution:
    def countArrangement(self, N: int) -> int:
        self.num_poss = 0
        n = N

        def callback(curr_ind, curr_list): # approach 1
            # nonlocal num_poss
            if curr_ind > n:
                # op_list.append(curr_list)
                self.num_poss += 1
                return

            curr_set = set(curr_list)

            for i in range(1, n + 1):
                if i in curr_set:
                    continue

                if i % curr_ind == 0 or curr_ind % i == 0:
                    callback(curr_ind + 1, curr_list + [i])

        callback(1, [])


        def callback(curr_ind, visited): # approach 2 (Better approach)
            # nonlocal num_poss
            if curr_ind > n:
                # op_list.append(curr_list)
                self.num_poss += 1
                return

            # curr_set = set(curr_list)

            for i in range(1, n + 1):
                if visited[i - 1]:
                    continue

                if i % curr_ind == 0 or curr_ind % i == 0:
                    visited[i - 1] = True
                    callback(curr_ind + 1, visited)
                    visited[i - 1] = False

        callback(1, [False for i in range(n)])
        return self.num_poss


# time: O(k). k refers to the number of valid permutations.
# space: O(n). visited array of size n is used. The depth of recursion tree will also go upto n. Here, n refers to the given integer n.


'''
47 Permutations II
https://leetcode.com/problems/permutations-ii/
'''
from collections import defaultdict


# approach 1
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        op = []
        len_nums = len(nums)
        visited_indices = set()

        def helper(formed_list, list_len):
            nonlocal len_nums, visited_indices, op

            if list_len == len_nums:
                op.append(formed_list)
                return

            curr_ind_visited_nums_set = set()

            for i, n in enumerate(nums):
                if n in curr_ind_visited_nums_set or i in visited_indices:
                    continue

                else:
                    curr_ind_visited_nums_set.add(n)
                    visited_indices.add(i)
                    helper(formed_list + [n], list_len + 1)
                    visited_indices.remove(i)

        helper([], 0)
        return op


# time: O(k) k is the num of valid permutations
# space: O(n ^ 2) at any point the max depth of recusriosn will be O(n) and at each level we will have a curr_ind_visited_nums_set (whose size is n). So, we can upper bound it by n ^ 2

# approach 2 (Look at what we are iterating inside the recursive fn. We want a given number at a given index of the permuted array, to appear only once. So we form a dictionay whose keys are unique
# numbers in the input and iterate through that to populate our permuted array)

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        results = []

        def backtrack(comb, counter):
            if len(comb) == len(nums):
                # make a deep copy of the resulting permutation,
                # since the permutation would be backtracked later.
                results.append(list(comb))
                return

            for num in counter:
                if counter[num] > 0:
                    # add this number into the current combination
                    comb.append(num)
                    counter[num] -= 1
                    # continue the exploration
                    backtrack(comb, counter)
                    # revert the choice for the next exploration
                    comb.pop()
                    counter[num] += 1

        backtrack([], Counter(nums))

        return results


# time O(k)
# space O(n)

'''
Split the given string into Primes : Digit DP
https://www.geeksforgeeks.org/split-the-given-string-into-primes-digit-dp/
'''
import math

input_str = '13499315'
len_inp = len(input_str)
all_subsets = []
min_len_subset = None
len_of_min_subset = float('+inf')

def check_prime(num):
    if num == 1:
        return False

    for n in range(2, math.ceil(math.sqrt(num)) + 1):
        if num % n == 0:
            return False

    return True


def get_all_combinations(i, curr_subset, curr_subset_len):
    global len_of_min_subset
    global min_len_subset

    if i == len_inp:
        all_subsets.append(curr_subset)

        if curr_subset_len < len_of_min_subset:
            min_len_subset = curr_subset
            len_of_min_subset = curr_subset_len

        return

    for j in range(i + 1, len_inp + 1):
        #print(input_str[i:j])
        #print(check_prime(int(input_str[i:j])))

        if check_prime(int(input_str[i:j])):
            curr_num = int(input_str[i:j])
            get_all_combinations(j, curr_subset + [input_str[i:j]], curr_subset_len + 1)

    return

get_all_combinations(0, [], 0)
print(all_subsets)
print(min_len_subset)

# time O(2 ^ n)
# space O(n)
# --------------------------------------------------------------- End of Backtracking Block ---------------------------------------------------------------------------------------------------------------




























# --------------------------------------------------------------- Sliding Window Block ---------------------------------------------------------------------------------------------------------------

'''
3 Longest Substring Without Repeating Characters
https://leetcode.com/problems/longest-substring-without-repeating-characters/
'''


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        end = 0  # 4
        start = 0  # 1
        curr_window = set()  # bca
        max_len = 0  # 3

        while (start < len(s)):

            while (end < len(s) and s[end] not in curr_window):
                curr_window.add(s[end])  #
                end += 1  #
                max_len = max(max_len, end - start)  #

            curr_window.remove(s[start])  #
            start += 1

        return max_len


# time O(n)
# space O(n)


'''
159. Longest Substring with At Most Two Distinct Characters
https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/
'''

from collections import defaultdict


class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        uniq_char_set = set()  # e b
        st = 0  # 2
        en = 0  # 4
        len_s = len(s)
        curr_char_occurances = defaultdict(list)  # {e:[0,2], b:[3]}
        max_len = 0

        while (st < len_s):  # 0 1 2 3

            while (en < len_s and len(uniq_char_set) < 3):
                uniq_char_set.add(s[en])
                curr_char_occurances[s[en]].append(en)
                en += 1

                if len(uniq_char_set) > 2:
                    break
                else:
                    max_len = max(max_len, en - st)

            char_at_st = s[st]  # c

            if curr_char_occurances[char_at_st][-1] == st:
                curr_char_occurances.pop(char_at_st)
                uniq_char_set.remove(char_at_st)

            st += 1

        return max_len

# time O(n)
# space O(n)


'''
424 Longest Repeating Character Replacement
https://leetcode.com/problems/longest-repeating-character-replacement/
'''
from collections import defaultdict


class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        start = 0
        end = 0
        curr_window = [0 for i in range(26)]
        max_len = 0
        s_len = len(s)

        while (start < s_len):

            window_len = end - start
            min_replacable_chars = window_len - max(curr_window)

            while (end < s_len):
                char_at_end = s[end]
                curr_window[ord(char_at_end) - ord('A')] += 1
                end += 1
                window_len = end - start
                min_replacable_chars = window_len - max(curr_window)

                if min_replacable_chars > k:
                    break
                else:
                    max_len = max(max_len, window_len)

            char_at_start = s[start]
            curr_window[ord(char_at_start) - ord('A')] -= 1
            start += 1

        return max_len

# time O(n)
# space O(1)

# --------------------------------------------------------------- End of Slinding window block ---------------------------------------------------------------------------------------------------------------



























# ------------------------------------------------------------------------ heaps block ---------------------------------------------------------------------------------------------------------------

'''
295 Find Median from Data Stream
https://leetcode.com/problems/find-median-from-data-stream/
'''
import heapq


class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.min_heap = []  # 10,11,12,13,14
        self.min_heap_size = 0
        self.max_heap = []  # -5,-4,-3,-2
        self.max_heap_size = 0

        heapq.heapify(self.max_heap)
        heapq.heapify(self.min_heap)

    def addNum(self, num: int) -> None:  # O(log n)
        heapq.heappush(self.max_heap, -num)
        top = heapq.heappop(self.max_heap)
        heapq.heappush(self.min_heap, -top)
        self.min_heap_size += 1

        if self.max_heap_size == self.min_heap_size or self.max_heap_size + 1 == self.min_heap_size:
            return
        else:
            top = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -1 * top)
            self.min_heap_size -= 1
            self.max_heap_size += 1

        '''if self.min_heap == [] or num >= self.min_heap[0]:

            if self.min_heap_size > self.max_heap_size:
                top_ele = heapq.heappop(self.min_heap)
                heapq.heappush(self.max_heap, -1 * top_ele)
                self.max_heap_size += 1
                self.min_heap_size -= 1

            heapq.heappush(self.min_heap, num)
            self.min_heap_size += 1

        else:

            if self.max_heap_size == self.min_heap_size:
                if self.max_heap and num > -1 * self.max_heap[0]:
                    ele_to_push_min_heap = num
                else:
                    ele_to_push_min_heap = -1 * heapq.heappop(self.max_heap)
                    heapq.heappush(self.max_heap, -1 * num)

                heapq.heappush(self.min_heap, ele_to_push_min_heap)
                self.min_heap_size += 1

            else:
                heapq.heappush(self.max_heap, -1 * num)
                self.max_heap_size += 1
        '''

    def findMedian(self) -> float:  # O(1)
        if self.min_heap_size > self.max_heap_size:
            return self.min_heap[0]
        else:
            mid_ele_1 = -1 * self.max_heap[0]
            mid_ele_2 = self.min_heap[0]
            return (mid_ele_1 + mid_ele_2) / 2

# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()

'''
DIP - Google

Given a list of points and a number k, find the k closest points to the origin.

def findClosestPointsOrigin(points, k):
  # Fill this in.

print (findClosestPointsOrigin([[1, 1], [3, 3], [2, 2], [4, 4], [-1, -1]], 3))
# [[-1, -1], [1, 1], [2, 2]]
'''

'''
Questions to ask the interviewer
1 - What should I do if k is larger than the len of list
2 - What should I do if k = 1 and there are 2 points on the graph with the same shortest distance from origin?
'''
import heapq
import math

points = [[-1, -1], [1, 1], [2, 2]]
origin = (0,0)
distance_list = []
heapq.heapify(distance_list)
k = 3

for point in points:
    x_dist = point[0] - origin[0]
    y_dist = point[1] - origin[1]
    dist_sq = x_dist ** 2 + y_dist ** 2
    d = math.sqrt(dist_sq)
    heapq.heappush(distance_list, (d,point[0], point[1]))

while(k > 0 and distance_list):
    print(heapq.heappop(distance_list))
    k -= 1

# time O(n) + k log n
# space O(n)

'''
https://oss.1point3acres.cn/forum/202002/16/105235vyhn4ghwyfz9czzt.jpg!c
'''
import heapq

s = 'ababyz'
s_list = [char for char in s]
heapq.heapify(s_list)
prev_char = float('-inf')
ss = [prev_char] # a b y z
secondary_list = [] # a a b

while(s_list):
    prev_char = heapq.heappop(s_list)
    ss.append(prev_char)

    while(s_list and s_list[0] == prev_char):
        secondary_list.append(heapq.heappop(s_list))

heapq._heapify_max(secondary_list)

while(secondary_list):
    ss.append(heapq.heappop(secondary_list))

ss = ''.join(ss[i] for i in range(1, len(ss)))
print(ss)

# time: n log n


'''
Amazon OA
Max profit for Amazon basics
'''


def highest_profit(numSuppliers, inventory, order):
    import heapq
    heap = []
    for item in inventory:
        if item > 0:
            heapq.heappush(heap, item * (-1))

    profit = 0

    while order > 0 and heap:

        a = heapq.heappop(heap)
        profit += a * (-1)
        a += 1
        if a:
            heapq.heappush(heap, a)

        order -= 1

    return profit


print
'Highest Profit: ', highest_profit(2, [3, 5], 6)

'''
Amazon OA
max_num_boxes
'''

import heapq


def maxUnits(num, boxes, unitSize, unitsPerBox, truckSize):
    heap = []

    for i in range(len(boxes)):
        units_per_box = unitsPerBox[i]
        heapq.heappush(heap, (-units_per_box, boxes[i]))

    ret = 0

    while truckSize > 0 and heap:
        curr_max = heapq.heappop(heap)
        max_boxes = min(truckSize, curr_max[1])
        truckSize -= max_boxes
        ret += max_boxes * (curr_max[0] * -1)

    return ret


# test cases
print(maxUnits(3, [1, 2, 3], 3, [3, 2, 1], 3))
print(maxUnits(3, [2, 5, 3], 3, [3, 2, 1], 50))

# --------------------------------------------------------------- End of Slinding window block ---------------------------------------------------------------------------------------------------------------



























# --------------------------------------------------------------- Linked list block ---------------------------------------------------------------------------------------------------------------

'''
25. Reverse Nodes in k-Group
https://leetcode.com/problems/reverse-nodes-in-k-group/
'''
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        prev_group_last_node = None
        new_head = None

        def helper_reverse(node, k_copy):
            nonlocal prev_group_last_node, new_head

            node_count = 0
            st_node = node
            prev_node = None

            while(node and node_count < k):
                node = node.next
                node_count += 1

            if node_count < k:
                if prev_group_last_node: prev_group_last_node.next = st_node
                return None

            node = st_node

            while (k_copy > 0 and node):
                temp_node = node.next
                node.next = prev_node
                prev_node = node
                node = temp_node
                k_copy -= 1

            if new_head == None:
                new_head = prev_node

            if prev_group_last_node:
                prev_group_last_node.next = prev_node

            prev_group_last_node = st_node
            return node

        node = head
        new_head = None
        node_num = 1
        st_node_reverse_group = head

        while (st_node_reverse_group):
            st_node_reverse_group = helper_reverse(st_node_reverse_group, k)

        return new_head

# time O(n)
# space O(1)

# --------------------------------------------------------------- End of Linked list block ---------------------------------------------------------------------------------------------------------------
























# ------------------------------------------------------------------ OA questions block ------------------------------------------------------------------------------------------------
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

"""
Input
-----
voters -
Voter arrival data (array of strings) sorted by arrival timestamp
Format of each entry - "<arrivalTimestamp>,<votingTime>,<numChildren>,<toleranceTime>"

numMachines -
Number of voting machines

queueSize -
Size of the polling place's queue

Output
------
Return an array of integers of size `numMachines+1` where the 0-th index represents the total number of voters who successfully cast votes and indices 1 to `numMachines` 
represent the number of votes cast at each voting machine.
"""

'''import heapq
from collections import deque

def get_max_times_to_iterate(voters):
    max_time_to_iterate = 0

    for voter in voters:
        voter_info = voter.split(',')
        in_time = int(voter_info[0])
        max_wait_time = in_time + int(voter_info[3])
        max_time_to_iterate = max(max_time_to_iterate, max_wait_time)

    return  max_time_to_iterate

def solution(voters, numMachines, queueSize):
    max_time_to_iterate = get_max_times_to_iterate(voters)
    max_queue_size = queueSize
    num_machines = numMachines
    voters_data = deque(voters)
    curr_queue_size = 0 # 0 2
    vote_end_times = [] # (end_time, machine_id) (25, 0), (25, 1)
    voters_queue = deque([]) # (voter_id, num_people, time_needed_to_vote) #[(18,2)]
    max_waittimes_heap = []  # (max_wait_time, num_people, voter_id)
    voters_left_due_to_time_expiry = set()
    free_machines = [i for i in range(num_machines)]
    heapq.heapify(vote_end_times)
    heapq.heapify(max_waittimes_heap)
    heapq.heapify(free_machines)
    machine_vote_counts = deque([0 for i in range(num_machines)])
    voter_id = 0


    for curr_time in range(0, max_time_to_iterate + 2):
        if curr_time == 60:
            print()
            pass
        if voters_data == [] and voters_queue == []:
            break
        # num people who voted and are leaving
        while(vote_end_times and vote_end_times[0][0] <= curr_time):
            _, free_machine_id = heapq.heappop(vote_end_times)
            heapq.heappush(free_machines, free_machine_id)

            # num of people who are going to vote at curr time
            while (voters_queue and free_machines):
                voter_going_to_vote, space_to_be_freed, time_needed_to_vote = voters_queue.popleft()

                if voter_going_to_vote not in voters_left_due_to_time_expiry:
                    free_machine_id = heapq.heappop(free_machines)
                    machine_vote_counts[free_machine_id] += 1
                    heapq.heappush(vote_end_times, (curr_time + time_needed_to_vote, free_machine_id))

        # num of people who are leaving the queue at curr time due to time expiry
        while(max_waittimes_heap and max_waittimes_heap[0][0] <= curr_time):
            _, num_people_left_queue, voter_id_to_remove = heapq.heappop(max_waittimes_heap)
            curr_queue_size -= num_people_left_queue
            voters_left_due_to_time_expiry.add(voter_id_to_remove)


        # people who come in at curr time
        while(voters_data and int(voters_data[0].split(',')[0]) == curr_time):
            voter_info = voters_data.popleft().split(',')
            in_time, time_needed_to_vote, num_children = int(voter_info[0]), int(voter_info[1]), int(voter_info[2])
            total_space_needed = 1 + num_children
            max_wait_time = in_time + int(voter_info[3])

            if free_machines:
                free_machine_id = heapq.heappop(free_machines)
                machine_vote_counts[free_machine_id] += 1
                heapq.heappush(vote_end_times, (curr_time + time_needed_to_vote, free_machine_id))

            else:
                if curr_queue_size + total_space_needed <= max_queue_size:
                    heapq.heappush(max_waittimes_heap, (max_wait_time, total_space_needed, voter_id))
                    voters_queue.append((voter_id, total_space_needed, time_needed_to_vote))
                    curr_queue_size += total_space_needed

            voter_id += 1

    machine_vote_counts.appendleft(sum(machine_vote_counts))
    return list(machine_vote_counts)
'''
