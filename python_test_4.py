# *************************** BACKTRACKING TIME COMPL SEARCH ***********************

Time compl. of backtracking prob can be calculated using the following 2 factors

1) Fan out (meaning how many edges can come out of a node, in the worst case). Lets call this m

2) How deep in the worst case can the recursion stack (or recursion tree) grow '?'. Lets call this n

So, time comp is m ** n in the worst case


# *************************** BINARY SEARCH ***********************

Binary search tips:

Use st < en as condition in your while loop as much as possible. Try not to use st <= en as it will lead to unwanted complications

1)
mid = (st + en) // 2 => mid = math.floor((st + en) / 2)
You can have "en = mid" in any of your if or elif or else conditions and you code WILL NOT GO INTO INFINITE LOOP
Reason is math.floor =>
eg:

while st < en
st = 4
en = 5
mid = (4 + 5) // 2 => 4
if(...): en = mid => en = 4

2)
For the same eg above. if you have
st = mid
in any of your if or elif or else. There is a possibility of infinite loop when

st = 4
en = 5
mid = (4 + 5) // 2 => 4
if(...): st = mid => st = 4 (**DANGER** Infinite loop)

Solution => mid = math.ceil((st + en) / 2) => math.ceil((4 + 5) / 2) => 5
st = 4
en = 5
mid = math.ceil((4 + 5) / 2) => 5
if(...): st = mid => st = 5 (**SOLUTION** loop terminates as st = 5 and en = 5)

3)
If you want to get the last occurance of an element in a sorted array

mid = (st + en) // 2

if mid_ele == target:
    en = mid # You don't want to loose track of the index mid because the index mid could be the right most index holding value target

conversely
If you want to get the first occurance of an element in a sorted array

mid = math.ceil((st + en) / 2)

if mid_ele == target:
    st = mid # You don't want to loose track of the index mid because the index mid could be the left most index holding value target

4)

If you want to get the index of the first smallest ele less than the target

if mid_ele < target:
    st = mid # Now since you are using st = mid, to avoid infinite loop scenario, you have to use mid = math.ceil((st + en) /2)


Conversely, if you want to get the index of the first biggest ele greater than the target

if mid_ele > target:
    en = mid # Now since you are using en = mid, you DO NOT have to think of using math.ceil

'''
1. Two Sum
https://leetcode.com/problems/two-sum/
'''

from collections import defaultdict


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        my_dictionary = defaultdict(int)

        for i, num in enumerate(nums):
            if target - num in my_dictionary:
                return (i, my_dictionary[target - num])

            my_dictionary[num] = i


'''
time O(n)
space: O(n)
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

'''
2. Add Two Numbers
https://leetcode.com/problems/add-two-numbers/
'''


class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        num1 = 0
        num2 = 0
        carry = 0
        res_link_list_last_inserted_node = None

        while (l1 or l2):  # 2 or 5
            if l1:
                num1 = l1.val
                l1 = l1.next
            else:
                num1 = 0

            if l2:
                num2 = l2.val
                l2 = l2.next
            else:
                num2 = 0

            sum_of_nums = num1 + num2 + carry
            dig = sum_of_nums % 10

            new_node = ListNode(dig)

            if res_link_list_last_inserted_node:
                res_link_list_last_inserted_node.next = new_node  # N(0, N(8)) || N(7, N(0))

                res_link_list_last_inserted_node = new_node  # N(0, N(8)) || N(7, N(0))

            else:
                head = res_link_list_last_inserted_node = new_node

            carry = sum_of_nums // 10

        if carry:
            res_link_list_last_inserted_node.next = ListNode(carry)

        return head


'''
time: O(m + n)
sapce: O(max(m, n))
'''
'''
num1 = 4
num2 = 6
carry = 0

sum_of_nums = 10
dig = 0
nn = N(7, None) || N(0, None)
nn.next = N(7)

res_link_list_last_inserted_node = None || N(7, None) || N(0, N(7))
'''

'''
3. Longest Substring Without Repeating Characters
https://leetcode.com/problems/longest-substring-without-repeating-characters/
'''


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        i = 0
        j = 0
        current_window_dict = {}
        unique_chars_max_window_len = 0

        while (j < len(s)):
            str_i = s[i]
            str_j = s[j]

            while str_j in current_window_dict and current_window_dict:
                current_window_dict.pop(str_i)
                i += 1
                str_i = s[i]

            unique_chars_max_window_len = max(unique_chars_max_window_len, j - i + 1)

            current_window_dict[str_j] = j
            j += 1

        return unique_chars_max_window_len


"""
Insert at tail and delete from front
Watch out for the condition in the inner while loop. It's "str_j"

time: O(n)
space: O(n)
"""

'''
5. Longest Palindromic Substring
https://leetcode.com/problems/longest-palindromic-substring/
'''

from collections import deque


class Solution:
    def longestPalindrome(self, s: str) -> str:
        max_pal = s[0]
        max_len_pal = 1

        '''
        def check_pal_2(i, j, pal):
            pal_len = 0

            while(i > -1 and j < len(s) and s[i] == s[j]): # O(n)
                pal = s[i] + pal + s[j] # Theoritically O(n) but python implementation COULD makes this efficient at times but should NOT be assumed 
                to happen all the time. https://stackoverflow.com/questions/34008010/is-the-time-complexity-of-iterative-string-append-actually-on2-or-on . 
                It should however be perceived as O(n)
                pal_len += 2

                i -= 1
                j += 1

            return pal_len, pal
        '''

        def check_pal(i, j, pal):
            pal_len = 0
            pal = deque(pal)

            while (i > -1 and j < len(s) and s[i] == s[j]):  # O(n)
                pal.appendleft(s[i])  # O(1)
                pal.append(s[j])  # O(1)
                pal_len += 2

                i -= 1
                j += 1

            return pal_len, ''.join(pal)  # O(n)

        for ind in range(0, len(s)):  # O(n)
            curr_len_1, pal_1 = check_pal(ind, ind + 1, [])  # O(n) BUT check_pal_2 will give O(n ^ 2)

            curr_len_2, pal_2 = check_pal(ind - 1, ind + 1, [s[ind]])  # O(n) BUT check_pal_2 will give O(n ^ 2)

            curr_len_2 += 1

            if curr_len_1 > max_len_pal:
                max_len_pal = curr_len_1
                max_pal = pal_1

            if curr_len_2 > max_len_pal:
                max_len_pal = curr_len_2
                max_pal = pal_2

        return max_pal


"""
Note: the question asks you for the actual longest pal substring (NOT THE LEN OF the longest SUBS)

check_pal
time O(n ^ 2)
space O(n)

check_pal_2
time O(n ^ 3)
space O(n)

Better practical runtime appraoch https://leetcode.com/submissions/detail/276814925/ However, theoritically it's same as deque approach
"""

'''
6. ZigZag Conversion
https://leetcode.com/problems/zigzag-conversion/
'''


class Solution:
    def convert(self, s: str, numRows: int) -> str:

        if numRows == 1:  # Note: without this check the foll ex will fail -> eg: ABC || numRows = 1
            return s

        res_mat = []

        for i in range(numRows):
            res_mat.append([])

        row_ind = 0
        cycle = 'down'

        for char in s:
            res_mat[row_ind].append(char)

            if cycle == 'down':
                if row_ind == numRows - 1:
                    row_ind -= 1
                    cycle = 'up'
                else:
                    row_ind += 1

            else:
                if row_ind == 0:
                    row_ind += 1
                    cycle = 'down'
                else:
                    row_ind -= 1

        res = []

        for i in range(numRows):
            res.extend(res_mat[i])

        return ''.join(res)


'''
time: O(n)
space: O(n)
'''


'''
7. Reverse Integer
https://leetcode.com/problems/reverse-integer/
'''


class Solution:
    def reverse(self, x: int) -> int:
        rev_num = 0
        mul_factor = 1

        if x < 0:
            mul_factor = -1
            x = abs(x)

        while (x):
            last_dig = x % 10
            x = x // 10

            rev_num = rev_num * 10 + last_dig

            rev_num_copy = rev_num * mul_factor if mul_factor == -1 else rev_num

            if not (-2 ** 31 <= rev_num_copy <= 2 ** 31 - 1):
                return 0

        return rev_num * mul_factor


'''
Don't try to get remainder for diving neg number by positive number to get the last digit. You won't get the desired result
eg: divmod(-17, 4) => (-5, 3)

Notice how you formed the 'if' in 'while' loop. Just copied the portion of question to suit my needs
time O(n) where n is the num of digits in n
space O(n)

'''

'''
8. String to Integer (atoi)
https://leetcode.com/problems/string-to-integer-atoi/
'''


class Solution:
    def myAtoi(self, s: str) -> int:
        s = s.strip()

        mult_factor = 1

        if s and s[0] == '+':
            s = s[1:]  # Do Not do s.lstrip('+'). eg: '++1' should give 0 as result

        elif s and s[0] == '-':
            s = s[1:]
            mult_factor = -1

        if not s:  # eg: "+"
            return 0

        res = 0

        for char in s:

            if char.isnumeric() == False:
                return res * mult_factor

            res = res * 10 + int(char)

            res_copy = res * mult_factor

            if res_copy < -2 ** 31:
                return -2 ** 31

            if res_copy >= 2 ** 31:
                return 2 ** 31 - 1

        return res * mult_factor


"""
Test cases that failed
" ++1" || expected: 0
"3.14159" || expected: 3
"21474836460" || expected: 2147483647 (2**31 - 1) || not 2147483648 (2**31)
"  -0012a42" || expected: -12 || forgot to add: res '* mult_factor' in the for loop return
"  +  413" || expected: 0 || python > "   413".split() -> ["413"]

time: O(n) where n is the len of string
space: O(n)
"""

'''
9. Palindrome Number
https://leetcode.com/problems/palindrome-number/
'''


class Solution:
    def isPalindrome(self, x: int) -> bool:
        orig_x = x

        if x < 0:
            return False

        rev_num = 0

        while x:
            last_dig = x % 10
            x = x // 10
            rev_num = rev_num * 10 + last_dig

        return orig_x == rev_num


'''
time: O(n)
space O(n)
'''

'''
11. Container With Most Water
https://leetcode.com/problems/container-with-most-water/
'''


class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        max_water = 0

        while (left < right):
            left_ht = height[left]
            right_ht = height[right]

            width = right - left

            collected_water = min(left_ht, right_ht) * width

            max_water = max(max_water, collected_water)

            left, right = (left + 1, right) if right_ht > left_ht else (left, right - 1)

        return max_water


'''
Renamed the pointers to 'left' and 'right' to add make the code more understandable

time: O(n)
space: O(1)
'''

'''
12. Integer to Roman
https://leetcode.com/problems/integer-to-roman/
'''


class Solution:
    def intToRoman(self, num: int) -> str:
        num_to_roman_dict = {
            1: 'I',
            5: 'V',
            10: 'X',
            50: 'L',
            100: 'C',
            500: 'D',
            1000: 'M'
        }

        thousands_dig, num = (num // 1000, num % 1000)
        hundreds_dig, num = (num // 100, num % 100)
        tens_dig, num = (num // 10, num % 10)
        units_dig = num

        roman_str = ''

        if thousands_dig:
            roman_str += num_to_roman_dict[1000] * thousands_dig

        if hundreds_dig:
            if hundreds_dig == 9:
                roman_str += 'CM'

            elif hundreds_dig == 4:
                roman_str += 'CD'

            elif hundreds_dig == 5:
                roman_str += num_to_roman_dict[500]

            elif hundreds_dig > 5:
                roman_str += num_to_roman_dict[500] + num_to_roman_dict[100] * (hundreds_dig - 5)

            else:
                roman_str += num_to_roman_dict[100] * hundreds_dig

        if tens_dig:
            if tens_dig == 9:
                roman_str += 'XC'

            elif tens_dig == 4:
                roman_str += 'XL'

            elif tens_dig == 5:
                roman_str += num_to_roman_dict[50]

            elif tens_dig > 5:
                roman_str += num_to_roman_dict[50] + num_to_roman_dict[10] * (tens_dig - 5)

            else:
                roman_str += num_to_roman_dict[10] * tens_dig

        if units_dig:
            if units_dig == 9:
                roman_str += 'IX'

            elif units_dig == 4:
                roman_str += 'IV'

            elif units_dig == 5:
                roman_str += num_to_roman_dict[5]

            elif units_dig > 5:
                roman_str += num_to_roman_dict[5] + num_to_roman_dict[1] * (units_dig - 5)

            else:
                roman_str += num_to_roman_dict[1] * units_dig

        return roman_str


'''
There are 3 special cases for each of hundereds, tens and units digit. They are 9, 5 and 4
For thousands digit there are no special cases because the question states that the max input value can be 3999

checkout this submission for an even concise approach: https://leetcode.com/submissions/detail/422213816/

time O(1)
space O(1) assuming the max value constaint of 3999 holds always. num_to_roman_dict will always stay the same regardless of the input value
'''

'''
13. Roman to Integer
https://leetcode.com/problems/roman-to-integer/
'''


class Solution:
    def romanToInt(self, s: str) -> int:
        rom_to_num_dict = {
            'I': 1,
            'IV': 4,
            'V': 5,
            'IX': 9,
            'X': 10,
            'XL': 40,
            'L': 50,
            'XC': 90,
            'C': 100,
            'CD': 400,
            'D': 500,
            'CM': 900,
            'M': 1000
        }

        i = 0
        res = 0

        while (i < len(s)):
            char_i = s[i]

            if i < len(s) - 1:
                char_i_plus_one = s[i + 1]
            else:
                char_i_plus_one = ' '

            # Computation part begins

            if char_i + char_i_plus_one in rom_to_num_dict:
                res = res + rom_to_num_dict[char_i + char_i_plus_one]
                i += 2

            else:
                res = res + rom_to_num_dict[char_i]
                i += 1

        return res


'''
One place I got confused is, I was performing 
res = res * 10 + rom_to_num_dict[i]
which is incorrect because, win our dictionary we have actual values representing characters. eg: M = 1000 and not 1
if 
input = MI || op = 1001 => 1000 + 1
if you do 
res = res * 10 + rom_to_num_dict[i] => you will end up with res = 1000 * 10 + 1 (in the second iteration of while loop)

time: O(n)
space: O(1) provided that the max input value is 3999 || rom_to_num_dict will always be the same regardless of the input
'''

'''
14. Longest Common Prefix
https://leetcode.com/problems/longest-common-prefix/
'''


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        smallest_word = strs[0]

        for word in strs:
            if len(word) < len(smallest_word):
                smallest_word = word

        while (smallest_word):  # O(n)

            is_longest_pref = True

            for word in strs:  # O(m)

                if word[:len(smallest_word)] == smallest_word:  # O(n)
                    continue

                else:
                    is_longest_pref = False
                    break

            if is_longest_pref:
                return smallest_word

            smallest_word = smallest_word[:-1]

        return ''


'''
time: O(n ^ 2 * m)
space: O(n)

where n is the len of the longest word and m is the number of words
'''

'''
15. 3Sum
https://leetcode.com/problems/3sum/
'''


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        # [-4, -1, -1, 0, 1, 2]
        nums_dict = {val: ind for ind, val in enumerate(nums)}
        res = set()

        for i, num_i in enumerate(nums):

            # The following if is an arbitrary run time improvement strategy
            if num_i > 0:
                break

            j = len(nums) - 1
            prev_j = None

            while (j > i):
                num_j = nums[j]

                # The following if condition is an arbitrary run time improvement strategy
                if num_j < 0:
                    break

                '''
                The following if condition won't work for test case [1,1,-2]
                if prev_j and prev_j == num_j:
                    break
                '''

                num_req_to_get_a_triplet = -1 * (num_i + num_j)

                if num_req_to_get_a_triplet in nums_dict and \
                        nums_dict[num_req_to_get_a_triplet] != i and \
                        nums_dict[num_req_to_get_a_triplet] != j:
                    triplet = [num_i, num_req_to_get_a_triplet, num_j]
                    triplet.sort()
                    res.add(tuple(triplet))

                prev_j = num_j
                j -= 1

        return res


'''
time: O(n ^ 2)
space:O(n)

On paper (or in theory) this approach is O(n ^ 2) but is 2x slower than https://leetcode.com/submissions/detail/422635227/ which is also O(n ^ 2)
'''

'''
16. 3Sum Closest
https://leetcode.com/problems/3sum-closest/
'''


class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        closest_diff = float(+inf)
        res = None
        i = 0

        while (i < len(nums) - 2):
            j = i + 1
            k = len(nums) - 1

            while j < k:
                curr_sum = nums[i] + nums[j] + nums[k]
                current_diff = curr_sum - target

                if abs(current_diff) < closest_diff:
                    closest_diff = abs(current_diff)
                    res = curr_sum

                if current_diff < 0:
                    j += 1
                else:
                    k -= 1

            i += 1

        return res


'''
time: O(n ^ 2)
space: O(1)
'''


'''
17. Letter Combinations of Phone number
https://leetcode.com/problems/letter-combinations-of-a-phone-number/
'''


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        res = []
        digits_dict = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }

        # approach 1
        def recurse(num_str, substr):

            if not num_str:
                if substr:
                    res.append(substr)
                return

            dig = num_str[0]
            possible_chars = digits_dict[dig]

            for char in possible_chars:
                recurse(num_str[1:], substr + char)  # O(n) + O(n) in the worst case

        # approach 2
        def recurse_2(i, substr):

            if i == len(digits):
                if substr:
                    res.append(substr)
                return

            dig = digits[i]
            possible_chars = digits_dict[dig]

            for char in possible_chars:
                recurse_2(i + 1, substr + char)  # O(n) in the worst case

        # approach 3
        def recurse_3(i, substr_list):

            if i == len(digits):
                if substr_list:
                    res.append(''.join(substr_list))
                return

            dig = digits[i]
            possible_chars = digits_dict[dig]

            for char in possible_chars:
                recurse_3(i + 1, substr_list + [char])  # O(n) in the worst case

        # approach 4
        def recurse_4(i, substr_list):

            if i == len(digits):
                if substr_list:
                    res.append(''.join(substr_list))
                return

            dig = digits[i]
            possible_chars = digits_dict[dig]

            for char in possible_chars:
                substr_list.append(char)

                recurse_4(i + 1, substr_list)  # O(1)

                substr_list.pop()

        # recurse(digits, '')

        # recurse_2(0, '')

        # recurse_3(0, [])

        recurse_4(0, [])

        return res


'''
Notice all the 4 approaches above and look at the submissions tab to see how the best approach (approach_4) runs faster than the other approaches 
and try to understand why approach 4 runs faster by looking at the comments in the code

The following is the time compl of the best case
time O((4 ^ n) * n) where n is the length of the input. In backtracking problems, MOST CASES, the number of elements in the result will be the time complexity
space O(n) without counting the space occupied by the result varibable. The max size of recursion stack at any given point will be O(n)
'''


'''
18. 4Sum
https://leetcode.com/problems/4sum/
'''


class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        res = []
        nums.sort()  # DO NOT ASSUME THAT THE INPUT IS SORTED

        for i in range(0, len(nums) - 3):  # O(n)
            for j in range(i + 1, len(nums) - 2):  # O(n)

                k = j + 1
                l = len(nums) - 1

                while (k < l):  # O(n)
                    curr_sum = nums[i] + nums[j] + nums[k] + nums[l]

                    if curr_sum - target == 0:
                        quad = [nums[i], nums[j], nums[k], nums[l]]
                        quad.sort()
                        res.append(tuple(quad))

                        k += 1
                        l -= 1


                    elif curr_sum - target < 0:
                        k += 1
                    else:
                        l -= 1

        return set(res)


'''
time: O(n ^ 3)
space: O(1) without counting the space required for storing the result. Size of result can be bounded by nC4 which is "how many combinations of 
4 can you get from an input set whose length is n. nCr = n!/ (r! *(n-r)!)"
'''


'''
19. Remove Nth Node From End of List
https://leetcode.com/problems/remove-nth-node-from-end-of-list/
'''


class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # 1 2 3 4 5 6 7
        # n = 2
        # 1 2 3 4 5 7
        # node to del = 6
        # stop at node 5

        node = head
        node_number = 1
        node_to_delete = head
        node_to_modify = None

        '''
        eg: 1
        # 1 2 3 4 5 6 7
        # n = 2
        # 1 2 3 4 5 7
        # node to del = 6
        # stop at node 5

        n = 2
        node = 1 2 3 4 5 6 7 N
        ntm_ = N N N 1 2 3 4 5
        ntd_ = 1 1 1 2 3 4 5 6
        nn__ = 1 2 3 4 5 6 7 8
        '''

        while node:
            if node_number >= n + 1:
                node_to_modify = node_to_delete
                node_to_delete = node_to_delete.next

            node = node.next
            node_number += 1

        if node_to_modify:
            node_to_modify.next = node_to_delete.next

        else:
            return head.next

        return head

    '''
        eg: 2
        # 1 2 3
        # n = 1
        # 1 2 3
        # node to del = 3
        # stop at node 2

        n = 1
        node = 1 2 3 N
        ntm_ = N N 1 2
        ntd_ = 1 1 2 3
        nn__ = 1 2 3 4
    '''

    '''
        eg: 3
        # 1 
        # n = 1
        # 1 
        # node to del = 1
        # stop at node 0

        n = 1
        node = 1 N
        ntm_ = N N
        ntd_ = 1 1
        nn__ = 1 2
    '''

    '''
    Failed test cases in the first submit try
    [1,2]
    2   
    '''


'''
20. Valid Parentheses
https://leetcode.com/problems/valid-parentheses/
'''


class Solution:
    def isValid(self, s: str) -> bool:
        stack = []

        open_brackets_set = set(['(', '[', '{'])

        brackets_map_dict = {
            ')': '(',
            '}': '{',
            ']': '['
        }

        for char in s:
            if char in open_brackets_set:
                stack.append(char)

            elif stack and stack[-1] == brackets_map_dict[char]:
                stack.pop()

            else:
                return False

        if stack:
            return False

        return True


'''
time: O(n)
space: O(n) #size of stack in the worst case will be equal to size of input

Failed testcase in first submission
'['
'''


'''
21. Merge Two Sorted Lists
https://leetcode.com/problems/merge-two-sorted-lists/
'''


class Solution:
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        head = None

        while l1 or l2:
            if l1:
                l1_val = l1.val
            else:
                l1_val = float('+inf')

            if l2:
                l2_val = l2.val
            else:
                l2_val = float('+inf')

            if l1_val < l2_val:
                node = ListNode(l1_val)
                l1 = l1.next

            else:
                node = ListNode(l2_val)
                l2 = l2.next

            if not head:
                head = node
                prev_node = node

            else:
                prev_node.next = node
                prev_node = node

        return head


'''
l1 and l2 pointers should be incremented in the 3rd if else. Made a mistake by incrementing them in the 1st and 2nd if else

time: O(max(m, n))
space: O(m + n)
'''

'''
22. Generate Parentheses
https://leetcode.com/problems/generate-parentheses/
'''


class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        '''
        n = 3
        _ _ _ _ _ _

        each blank has 2 options either '(' or ')'

        num_close cannot be greater thatn num_open

        3 parameters to recursive fun
        '''
        res = []

        def recurse(formed_string: str, num_open: int, num_close: int):
            # base case
            if len(formed_string) == n * 2:
                res.append(formed_string)
                return

            if num_open == num_close:
                recurse(formed_string + '(', num_open + 1, num_close)

            elif num_open == n:
                recurse(formed_string + ')', num_open, num_close + 1)

            elif num_open > num_close:
                recurse(formed_string + '(', num_open + 1,
                        num_close)  # NOTE string concatenation is O(m + n). Although python implementation handles this case more efficiently.
                # But should NOT be assumed that this concatenation will be done in O(1) time.
                # https://stackoverflow.com/questions/34008010/is-the-time-complexity-of-iterative-string-append-actually-on2-or-on
                recurse(formed_string + ')', num_open, num_close + 1)

        def recurse_2(formed_string: [], num_open: int, num_close: int):
            # base case
            if len(formed_string) == n * 2:
                res.append(''.join(formed_string))
                return

            if num_open == num_close:
                formed_string.append('(')

                recurse_2(formed_string, num_open + 1, num_close)

                formed_string.pop()

            elif num_open == n:
                formed_string.append(')')

                recurse_2(formed_string, num_open, num_close + 1)

                formed_string.pop()

            elif num_open > num_close:
                formed_string.append('(')
                recurse_2(formed_string, num_open + 1, num_close)
                formed_string.pop()

                formed_string.append(')')
                recurse_2(formed_string, num_open, num_close + 1)
                formed_string.pop()

        # recurse('', 0, 0)

        recurse_2([], 0, 0)

        return res


'''
Note the difference in both the approaches (recurse and recurse_2). Same as 5. Longest Palindromic Substring

time: 2 ^ (2n - 2) 
eg when n = 3 we have 2 * n = 6 spots _ _ _ _ _ _ to fill. In the first spot only '(' can come and in the last spot only ')' is valid. So now we 
have '(' _ _ _ _ ')' only 4 spots left and each spot has 2 possibilities either '(' or ')'

space: O(n) without considering the space occupied by the output. n is the max recursion depth.  We can upper bound Size of op by 2 ^ (2n - 2) 
'''

'''
23. Merge k Sorted Lists
https://leetcode.com/problems/merge-k-sorted-lists/
'''

import heapq


class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        min_heap = []

        for l in lists:
            if l:
                heapq.heappush(min_heap, (l.val, id(l), l))

        head = None

        while min_heap:
            smallest_ele_in_heap, _, node = heapq.heappop(min_heap)
            new_node = ListNode(smallest_ele_in_heap)

            if head:
                prev_node.next = new_node

            else:
                head = new_node

            prev_node = new_node

            if node.next:
                node = node.next
                heapq.heappush(min_heap, (node.val, id(node), node))

        return head

    '''
    def mergeKLists_2(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        head = None

        while lists:

            curr_iteration_min_val = float('+inf')

            for i, l in enumerate(lists):

                if l and l.val < curr_iteration_min_val:
                    curr_iteration_min_val = l.val
                    curr_iteration_min_val_list_index = i

            if curr_iteration_min_val == float('+inf'):
                break

            new_node = ListNode(curr_iteration_min_val)
            min_val_list = lists[curr_iteration_min_val_list_index]

            if min_val_list.next:
                lists[curr_iteration_min_val_list_index] = min_val_list.next

            else:
                lists.pop(curr_iteration_min_val_list_index)

            if head:
                prev_node.next = new_node
                prev_node = new_node

            else:
                prev_node = head = new_node


        return head
    '''


'''

The following are time and space compl of the best solution => tool 112 ms
time: O(N log k) 
N is the total number of nodes in all the input lists and k is the number of linked lists. Note: the size of heap will always be equal to k because 
you are storing only current head node of each linked list in your heap

space: O(N)

For the less optimal solution the space is still O(N) but time is O(N * m) => Took 4564 ms
'''

'''
24. Swap Nodes in Pairs
https://leetcode.com/problems/swap-nodes-in-pairs/

Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes 
(i.e., only nodes themselves may be changed.)

Input: head = [1,2,3,4]
Output: [2,1,4,3]
'''


class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        eg 1: 1, 2, 3, 4
        2 -> 1      |  1 -> 4 -> 3
        node: 1, 3  |  3, None
        head: 1, 2  |  2
        prev: n, 1  |  1, 3
        nnod: 2     |  4

        Initally did not take into consideration the odd number of nodes scenario (eg: 2). So, the following testcase failed. Later added the line
        prev_node.next = next_node if next_node else node
        to handle this case

        eg 2: 1, 2, 3
        2 -> 1      |
        node: 1, 3  |
        head: 1, 2  |
        prev: n, 1  |
        nnod: 2     |
        '''

        '''
        approach 1
        ----------

        node = head
        prev_node = None

        while node: 
            next_node = node.next 

            if prev_node:
                prev_node.next = next_node if next_node else node
            else:
                head = next_node if next_node else node

            prev_node = node

            if next_node:
                node = next_node.next
                next_node.next = prev_node
            else:
                node = None

        if prev_node: prev_node.next = node

        return head
        '''

        # approach 2
        '''
        eg 1: 1, 2, 3, 4
        0 -> 2 -> 1  || 1 -> 4 -> 3
        node: 1, 3   || N
        head: 0      ||
        prev: 0, 1   || 3
        nnod: 2      || 4
        nnit: 3      || N
        '''
        dummy_head = prev_node = ListNode(0)
        node = head

        while node:
            next_node = node.next

            prev_node.next = next_node if next_node else node

            node_for_next_iteration = next_node.next if next_node else None

            if next_node:
                next_node.next = node

            prev_node = node
            node = node_for_next_iteration

        # Missed out the following line. Added it only after the test case 1,2,3,4 failed.
        if prev_node and prev_node.next != None: prev_node.next = None

        return dummy_head.next


'''
Both the approaches have the same runtime, time and space compl. but approach 2 is a bit easier to understand because of DUMMY_HEAD
time: O(n)
space: O(1)
'''

'''
25. Reverse Nodes in k-Group
https://leetcode.com/problems/reverse-nodes-in-k-group/
'''


class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:

        '''
        eg: [1,2,3,4,5] k =2

        node 1, 2 || 3
        nn = 0, 1 || 2
        nl = [1]  || [1,2]

        k_plus_one_node = 3
        nl = [1, 2] => [1] => []
        prev_node = 0 || 2
        k_gr_du_h = 0

        curr_node = 2 1
        0 -> 2 -> 1

        2 -> 1 -> 2
        k_group_head = 2
        new_head_to_explore = 3

        '''

        return_head = None
        new_head_to_explore = head
        prev_k_group_tail = None

        def reverse_k_nodes(head):
            node = head
            num_nodes = 0
            nodes_list = []

            while (node and num_nodes < k):
                num_nodes += 1
                nodes_list.append(node)
                node = node.next

            # If we do not have atleast k nodes, we should not reverse the nodes
            if num_nodes < k:
                return head, None, None

            k_plus_one_node = node
            k_group_dummy_head = prev_node = ListNode(0)  # Notice the use of dummy_head

            while (nodes_list):
                curr_node = prev_node.next = nodes_list.pop()
                prev_node = curr_node

            return k_group_dummy_head.next, prev_node, k_plus_one_node

        while (new_head_to_explore):
            curr_k_group_head, curr_k_group_tail, new_head_to_explore = reverse_k_nodes(new_head_to_explore)

            if prev_k_group_tail:
                prev_k_group_tail.next = curr_k_group_head

            if not return_head:
                return_head = curr_k_group_head

            if curr_k_group_tail and not new_head_to_explore:
                curr_k_group_tail.next = None

            prev_k_group_tail = curr_k_group_tail

        return return_head


'''
time: O(n)
space: O(n) nodes_list variable can hold n nodes in the worst case
'''

'''
26. Remove Duplicates from Sorted Array
https://leetcode.com/problems/remove-duplicates-from-sorted-array/
'''


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        '''
        1,2,3,3,3,4,4,4,5
        1,2,3,4,5

        i = 1, 2, 3, 4, 5
        j = 1, 2, 3, 5, 8

        curr_j = 3 || 4


        '''

        # approach 1
        '''
        i = 1
        j = 1

        if len(nums) <= 1:
            return len(nums)

        while i < len(nums) and j < len(nums):
            if nums[i] <= nums[i - 1]:
                curr_j = nums[j]

                while j < len(nums) and nums[j] == curr_j:
                    j += 1

                if j == len(nums):
                    break

                nums[i] = nums[j]


            i += 1

            if j < i: j += 1

        return i
        '''

        # approach 2
        index = 0
        prev_num = 'a'

        for num in nums:
            if num == prev_num:
                continue
            else:
                nums[index] = num
                prev_num = num
                index += 1

        return index


'''
Both approaches are O(n) but the approach 2 is simpler and easier to understand
time: O(n)
space: O(1)
'''

'''
27. Remove Element
https://leetcode.com/problems/remove-element/
'''


class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i = 0
        j = 1

        while i < len(nums):

            if nums[i] == val:

                while j < len(nums) and nums[j] == val:
                    j += 1

                if j == len(nums):
                    break

                nums[i], nums[j] = nums[j], nums[i]

                j += 1

            i += 1

            if j == i:
                j += 1

        return i


'''
Failed test case: 
nums = [2] and k = 3
changed the while condition 
while i < len(nums) and j < len(nums) 
to 
while i < len(nums)

time: O(n)
space: O(1)
'''


'''
28. Implement strStr()
https://leetcode.com/problems/implement-strstr/
'''


class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        needle_len = len(needle)
        haystack_len = len(haystack)

        if haystack_len < needle_len:
            return -1

        if not needle:
            return 0

        for ind, char in enumerate(haystack):
            if ind + needle_len > haystack_len:
                break

            j = ind
            k = 0

            while j < (ind + needle_len) and j < len(haystack) and k < needle_len and haystack[j] == needle[k]:
                j += 1
                k += 1

            if k == needle_len:
                return ind

            # The following implementaion is more pythonic. The above approach times out. The below approach works fine
            # Both the approaches are O(n ^ 2) time. The below approach has space O(n) but the above approach has O(1) space complexity

            # if haystack[ind: ind + needle_len] == needle:
            #    return ind

        return -1


'''
time: O(n ^ 2)
space: O(n)
'''

'''
29. Divide Two Integers
https://leetcode.com/problems/divide-two-integers/
'''


class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        res = 0
        mult_factor = 1

        if dividend < 0 and divisor < 0:
            dividend = -dividend
            divisor = -divisor

        elif dividend < 0:
            dividend = -dividend
            mult_factor = -1

        elif divisor < 0:
            divisor = -divisor
            mult_factor = -1

        orig_dividend = dividend
        orig_divisor = divisor
        divisor_mult_factor = 1

        while orig_dividend >= orig_divisor:

            while dividend >= divisor:
                res += divisor_mult_factor

                dividend -= divisor

                # Following are 2 ways for multiplying divisor by 2
                # divisor += divisor
                divisor = divisor << 1

                # divisor_mult_factor += divisor_mult_factor
                divisor_mult_factor = divisor_mult_factor << 1

            orig_dividend = dividend

            # --------------------------

            '''
            approach 1
            divisor = orig_divisor
            '''
            # Following 2 lines are approach 2 (efficient than approach 1) check solution tab to if unable to understand
            divisor = divisor >> 1
            divisor_mult_factor = divisor_mult_factor >> 1

            # --------------------------

            if mult_factor < 0 and -res < (-2 ** 31):  # ((mult_factor * res) > (2**31 - 1)) or ((mult_factor * res) < (-2 ** 31)): Removed '*' symbol
                return 2 ** 31 - 1

            elif mult_factor > 0 and res > (2 ** 31 - 1):
                return 2 ** 31 - 1

        return -res if mult_factor < 0 else res


'''
Failed test cases
1) -2147483648
1

2) -2147483648
-1

Notice 
1) elif condition - did not have mult_factor > 0 inititally which caused the testcase -2147483648, 1 to fail
elif mult_factor > 0 and res > (2**31 - 1):

2) Notice that we are not using '*' anywhere. Rather we use -variable for negation. It is equal to -1 * variable

time: O(log n)
space: O(1)
'''

'''
30. Substring with Concatenation of All Words
https://leetcode.com/problems/substring-with-concatenation-of-all-words/

You are given a string s and an array of strings words of the same length. Return all starting indices of substring(s) in s that is a 
concatenation of each word in words exactly once, in any order, and without any intervening characters.

You can return the answer in any order.



Example 1:

Input: s = "barfoothefoobarman", words = ["foo","bar"]
Output: [0,9]
Explanation: Substrings starting at index 0 and 9 are "barfoo" and "foobar" respectively.
The output order does not matter, returning [9,0] is fine too.
Example 2:

Input: s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]
Output: []
Example 3:

Input: s = "barfoofoobarthefoobarman", words = ["bar","foo","the"]
Output: [6,9,12]
'''


class Solution:
    def findSubstring(self, s, words):
        # approach 1
        op = []
        word_len = len(words[0])
        words_dict = defaultdict(int)
        num_words = len(words)

        for word in words:
            words_dict[word] += 1

        def check_if_all_subs_at_ind(ind, start_ind):
            nonlocal word_len, words_dict

            if not words_dict:
                op.append(start_ind)
                return True

            new_word = s[ind: ind + word_len]  # O(m) m is the len of each word (will be same for all words)

            if new_word in words_dict:
                if words_dict[new_word] == 1:
                    words_dict.pop(new_word)
                else:
                    words_dict[new_word] -= 1

                check_if_all_subs_at_ind(ind + word_len, start_ind)
                words_dict[new_word] += 1

        i = 0

        while (i < len(s)):  # O(n) n is the num of chars in s
            if len(s) - i < word_len * num_words:
                break

            new_word = s[i: i + word_len]

            if new_word in words_dict:

                if words_dict[new_word] == 1:
                    words_dict.pop(new_word)
                else:
                    words_dict[new_word] -= 1

                check_if_all_subs_at_ind(i + word_len,
                                         i)  # O(m) * O(m) * O(m) * ... => worst case there will be 'n' such occurances for each func call => O(m ** n)

                words_dict[new_word] += 1

            i += 1

        return op

    # approach 2
    def findSubstring_2(self, s: str, words: List[str]) -> List[int]:
        word_len = len(words[0])
        visited_indices = set()
        num_words = len(words)
        res = []

        def check_if_all_words_exist_from_index(i: int, num_visited_words: int):

            if num_visited_words == num_words:
                return True

            # missed 'i + word_len' in "[i: i+word_len]" and instead had just "[i:word_len]"
            substring_of_full_string = s[i:i + word_len]  # O(m) m is the len of each word (will be same for all words)

            for word_ind, word in enumerate(words):  # O(w)  w is the num of words in input
                if word_ind in visited_indices:
                    continue

                if substring_of_full_string == word:
                    visited_indices.add(word_ind)

                    if check_if_all_words_exist_from_index(i + word_len, num_visited_words + 1):
                        visited_indices.remove(word_ind)
                        return True

                    visited_indices.remove(word_ind)

            return False

        for i in range(0, len(s)):  # O(n) n is the num of chars in s
            if len(s) - i < word_len * num_words:
                break

            if check_if_all_words_exist_from_index(i, 0):  # O(m + w) * O(m + w) * O(m + w) * ... => worst case there will be
                # 'w' such occurances for EACH func call => O(m + w) ** w
                res.append(i)

        return res


'''

approach 2 timed out. 

approach 1 (best approach)
time: O(n) * (O(m) ** n)
space: O(m ** n) -> max recursion depth (recursion stack size)


approach 2
time: O(n) * (O(m + w) ** w)
space: O(m + w) ** w -> Your recursion depth

Verify once if the time and space compl are correct
'''

'''
31. Next Permutation
https://leetcode.com/problems/next-permutation/
'''


class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # eg
        # ips  [3,9,8,5,6,7, 4]
        # .     [3,9,8,5,7,6,4] => [3,9,8,6,7,5,4] => [3,9,8,6,4,5,7]
        #      [3,9,8,5,2,3, 4]

        '''
        The following code gives run time improvisation in real time but has same theoritical time complexity

        rearrangement_possible = False
        curr_min = nums[0]

        for i in range(1, len(nums)):
            if curr_min < nums[i]:
                rearrangement_possible = True
                break
            else:
                curr_min = nums[i]

        #print(rearrangement_possible)
        if not rearrangement_possible:
            nums.sort()
            return
        '''

        def index_of_closest_small_num_on_left(i):
            '''
            Function returns the index of the smallest number to the left of it
            '''
            for j in range(i - 1, -1, -1):
                if nums[j] < nums[i]:
                    return j

            return -1

        def rev_nums_between_left_and_right(m, n):
            '''
            Function reverses the digits between m and n index (both m and n included)
            '''
            while m < n:
                nums[m], nums[n] = nums[n], nums[m]
                m += 1
                n -= 1

        smallest_left_ind_to_change = float('-inf')
        left_right_index_pair = None

        for right_ind in range(len(nums) - 1, -1, -1):
            left_ind = index_of_closest_small_num_on_left(right_ind)

            if left_ind == -1:
                continue

            # We want to increase the rightmost possible index of a number to get to next largest permutation

            # eg in a number wxyz => increasing the number at digit y will result in a smaller number than increasing the number at digit x

            # eg number: 8 6 7 9 => swapping 8 and 9 => 9 6 7 8 whereas swapping 7 and 6 gives 8 7 6 9

            if left_ind > smallest_left_ind_to_change:
                left_right_index_pair = (left_ind, right_ind)
                smallest_left_ind_to_change = left_ind

        if not left_right_index_pair:
            nums.sort()
            return

        left_ind, right_ind = left_right_index_pair

        nums[left_ind], nums[right_ind] = nums[right_ind], nums[left_ind]

        rev_nums_between_left_and_right(left_ind + 1, len(nums) - 1)

        return


'''
time O(n)
space: O(1)
'''

'''
32. Longest Valid Parentheses
https://leetcode.com/problems/longest-valid-parentheses/

Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

Example 2:

Input: s = ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()".
Example 3:
'''


class Solution:
    def longestValidParentheses(self, s: str) -> int:
        '''
        eg testcases
        () ( ()
        ((())
        () )( ()
        '''
        # approach 1 O(n ** 2)
        '''
        len_of_longest_valid_paren = 0

        def get_longest_paren_from_i(i):
            st = []
            max_valid_paren_len_from_i = 0

            for j in range(i, len(s)):
                if st and s[j] == ')':
                    st.pop()

                    if not st:
                        max_valid_paren_len_from_i = j - i + 1

                elif s[j] == '(':
                    st.append('(')


                else: # stack empty and char = )
                    break

            return max_valid_paren_len_from_i


        for i in range(0, len(s)):
            len_longest_valid_paren_from_i = get_longest_paren_from_i(i)
            len_of_longest_valid_paren = max(len_of_longest_valid_paren, len_longest_valid_paren_from_i)

        return len_of_longest_valid_paren
        '''

        # approach 2 O(n)
        '''
        eg testcases
        () ( ()  
        ((())
        () )( ()  
        '''

        # Notice the use of -1 when initializing the stack.

        # See the video / visual of using stack approach to understand better

        # We keep track of the "starting index - 1" (the index where the current valid parenthesis string starts - 1) in variable st

        st = [-1]
        longest_valid_paren_len = 0

        for i, char in enumerate(s):

            if char == ')':
                st.pop()

                if st:
                    longest_valid_paren_len = max(longest_valid_paren_len, i - st[-1])
                else:
                    st.append(i)

            else:
                st.append(i)

        return longest_valid_paren_len


'''
Best approach (approach 2)
time: O(n)
space: O(n)

Other approach (approach 1)
time: O(n ** 2)
space: O(n)
'''

'''
33. Search in Rotated Sorted Array
https://leetcode.com/problems/search-in-rotated-sorted-array/

eg:
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
'''


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        st = 0
        en = len(nums) - 1

        while st < en:
            mid = (st + en) // 2
            mid_ele = nums[mid]

            # First find the rotation point
            # Notice the use of '>='. Initally had only '>' and the code failed for input [3,1], target = 1
            if mid_ele >= nums[st] and mid_ele > nums[en]:
                st = mid + 1  # Rotation point is on the right of mid

            else:
                en = mid  # Rotation point COULD be mid or somewhereon the right of mid

            '''
            The following are the cases handled by else block

            elif mid_ele > nums[st] and mid_ele < nums[en]:
                en = en - 1

            elif mid_ele < nums[st] and mid_ele < nums[en]:
                en = en - 1
            '''
        rotation_point = st
        rotation_point_ele = nums[rotation_point]

        if target >= rotation_point_ele and target <= nums[len(nums) - 1]:
            st = rotation_point
            en = len(nums) - 1

        else:
            st = 0
            en = rotation_point - 1

        while st <= en:
            mid = (st + en) // 2
            mid_ele = nums[mid]

            if mid_ele == target:
                return mid

            elif target < mid_ele:
                en = mid - 1

            else:
                st = mid + 1

        return -1


'''
time: O(log n)
space: O(1)
'''

'''
34. Find First and Last Position of Element in Sorted Array
https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
'''

import math


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        '''
        Notice from the old approach, how handling of "if not nums:" case simplified binary search while loop

        Try to keep "st < en" to keep your binary search simple.
        '''
        if not nums:
            return [-1, -1]

        if target > nums[-1] or target < nums[0]:
            return [-1, -1]

        '''if len(nums) == 1:
            return (0,0) if target == nums[0] else (-1,-1)
        '''

        left_most_occurance = right_most_occurance = -1

        st = 0
        en = len(nums) - 1

        # first find the left position
        while st < en:
            mid = (st + en) // 2
            mid_ele = nums[mid]

            if mid_ele == target:
                en = mid


            elif mid_ele < target:
                st = mid + 1

            else:
                en = mid - 1

        if nums[st] == target:
            left_most_occurance = st

        else:
            return (-1, -1)

        # Notice the value we are assigning to st. We know the left_most_occurance of a value. So, the right most occurance WILL be towards the right
        # So, we initialize st to left_most_occurance
        st = left_most_occurance
        en = len(nums) - 1

        # first find the right position
        while st < en:
            mid = math.ceil((st + en) / 2)
            mid_ele = nums[mid]

            if mid_ele == target:
                st = mid

                if st == en: break

            elif mid_ele < target:
                st = mid + 1

            else:
                en = mid - 1

        right_most_occurance = en

        return (left_most_occurance, right_most_occurance)


'''
time: O(log n)
space: O(1)
'''

'''
35. Search Insert Position
https://leetcode.com/problems/search-insert-position/

Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be 
if it were inserted in order.

You must write an algorithm with O(log n) runtime complexity.

Input: nums = [1,3,5,6], target = 5
Output: 2

'''


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:

        # Handling special cases like the followinf if and elif can be extremely helpful to overcome corner cases.
        # eg: [1,3] target 0 will fail without the following if and elif case

        if target < nums[0]:
            return 0

        elif target > nums[len(nums) - 1]:
            return len(nums)

        st = 0
        en = len(nums) - 1

        while st < en:
            mid = (st + en) // 2
            mid_ele = nums[mid]

            if target == mid_ele:
                return mid

            elif target > mid_ele:
                st = mid + 1

            else:
                en = mid

        return en

        '''
        Notice how changing 
        en = mid - 1
        to
        en = mid

        allowed us to comment out the following lines of code. Ideas like this may come up in the interview. If it comes up in mind, use it otherwise
        The following is still good and works

        if target > nums[en]:
            return en + 1

        elif target <= nums[en]: # "=" added to handle case [1], target = 1
            return en
        '''


'''
time: O(log n)
space: O(1)
'''

'''
36. Valid Sudoku
https://leetcode.com/problems/valid-sudoku/
'''

from collections import defaultdict


class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row_dict = defaultdict(set)
        col_dict = defaultdict(set)
        grid_dict = defaultdict(set)

        for i in range(len(board)):
            for j in range(len(board[0])):
                cell_ele = board[i][j]

                # Missed the following if
                if cell_ele == '.':
                    continue

                grid = f'{i // 3}-{j // 3}'

                if (cell_ele in row_dict[i]) or (cell_ele in col_dict[j]) or (cell_ele in grid_dict[grid]):
                    return False

                row_dict[i].add(cell_ele)
                col_dict[j].add(cell_ele)
                grid_dict[grid].add(cell_ele)

        return True


'''
time: O(n) n represents the number of cells in the sudoku board
space: O(n)
'''

'''
38. Count and Say
https://leetcode.com/problems/count-and-say/

Input: n = 4
Output: "1211"
Explanation:
countAndSay(1) = "1"
countAndSay(2) = say "1" = one 1 = "11"
countAndSay(3) = say "11" = two 1's = "21"
countAndSay(4) = say "21" = one 2 + one 1 = "12" + "11" = "1211"
'''


class Solution:
    def countAndSay(self, n: int) -> str:
        current_str = start_str = '1'

        while n > 1:

            prev_val = None
            new_str_list = []

            for char in current_str:

                if not prev_val:
                    counter = 1

                elif char != prev_val:
                    new_str_list.append(str(counter) + prev_val)

                    counter = 1

                else:
                    counter += 1

                prev_val = char

            new_str_list.append(str(counter) + prev_val)

            n -= 1
            current_str = ''.join(new_str_list)

        return current_str


'''
time: O(n * x) where n is the input and x is the number of characters in the output when n = 29. Since 30 is the max input, we will loop through the 
result of n = 29 to form the last output string in our count and say sequence

space: O(x) new_str_list will hold atmost x elements
'''

'''
39. Combination Sum
https://leetcode.com/problems/combination-sum/

Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen 
numbers sum to target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

 

Example 1:

Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.

'''

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        '''
        This is a perfect example of a confusing problem. Confusion here is how can we use memoization here.

        Easiest way is to treat this as a backtracking problem (approach 3). However we can use memoization as stated in recurse_2 (approach2)
        which is faster than backtracking in practice

        Haven't calculated the time complexity of this problem. Have to do it later. Look at Leetcode solutions tab for backtracking approach time compl

        If the question had asked for "the number of ways we can add upto the target, then using memoization is the best approach. Refer recurse_4 to
        find out how the solution would have looked like for that kind of a problem"
        '''

        # ----------------****---------------------------------****---------------------------------****-----------------

        memo = {}

        def recurse_1(i, curr_sum):
            if (i, curr_sum) in memo:
                return memo[(i, curr_sum)]

            if curr_sum == target:
                return [[]]


            local_result = []

            for ind in range(i, len(candidates)):
                if curr_sum + candidates[ind] > target:
                    continue

                result = recurse(ind, curr_sum + candidates[ind])

                if type(result) == list:

                    for item in result:
                        '''
                        let's assume: memo[(2,5)] = [[1,2], [3,4]]
                        
                        Let's say at some point later in the recursion, in the 1 st line of for loop we have
                        
                        ind = 2
                        candidates[2] = 5
                        current_sum = 0
                        Therefore => result = recurse_1(2,5)
                        Since, (2, 5) is in memo, in the appropriate recursive call, we return the list [[1,2], [3,4]]
                        
                        result = [[1,2], [3,4]] || Remember this is a shallow copy and not deep copy. The pointers pointing to [1,2] and [3,4] are 
                        pointing to the location of memo[(2,5)]
                        
                        item.append(candidats[ind]) => result = [[1,2,5], [3,4,5]] => We have changed the list pointed to by memo[(2,5)] inadvertantly 
                        and is wrong. You have modified the correct result of a pre-computed combination by a wrong value.
                        
                        # local_result = [result[0], result[1]]
                        
                        eg scenario which leads to incorrect result is 
                        candidates = [2,7,6,3,5,1]
                        target = 9
                        
                        However we have fixed this issue in recurse_2 by taking a DEEP COPY. For the same eg stated above, recurse_2 gives correct results
                        
                        '''

                        item.append(candidates[ind])
                        local_result.append(item)


                memo[(i, curr_sum)] = local_result

                return local_result

        # res = recurse_1(0, 0)

        # return res



        # ----------------****---------------------------------****---------------------------------****-----------------


        memo = {}

        def recurse_2(i, curr_sum):
            if (i, curr_sum) in memo:
                # print('memo hit 1')
                return memo[(i, curr_sum)]

            if curr_sum == target:
                return [[]]

            local_result = []

            for ind in range(i, len(candidates)):
                if curr_sum + candidates[ind] > target:
                    continue

                result = recurse_2(ind, curr_sum + candidates[ind])

                if result:

                    for item in result:
                        temp_item = list(item) # DEEP COPY TO FIX ISSUE STATED IN RECURSE_1
                        temp_item.append(candidates[ind])
                        local_result.append(temp_item)


            memo[(i, curr_sum)] = local_result

            return local_result

        res = recurse_2(0, 0)

        return res


        # ----------------****---------------------------------****---------------------------------****-----------------

        '''
        Leetcode solution - backtacking
        '''

        results = []

        def backtrack(remain, comb, start):
            if remain == 0:
                # make a deep copy of the current combination
                results.append(list(comb))
                return
            elif remain < 0:
                # exceed the scope, stop exploration.
                return

            for i in range(start, len(candidates)):
                # add the number into the combination
                comb.append(candidates[i])
                # give the current number another chance, rather than moving on
                backtrack(remain - candidates[i], comb, i)
                # backtrack, remove the number from the combination
                comb.pop()

        # backtrack(target, [], 0)

        # return results

        # ----------------****---------------------------------****---------------------------------****-----------------

        '''
        The following is just a different way of implementing the problem using backtracking approach along with memoization
        '''

        res = set()
        memo = {}

        def recurse_3(ind, tup_list):
            tup = tuple(tup_list)
            tup_sum = sum(tup)

            if (ind, tup) in memo:
                return

            if tup_sum == target:
                res.add(tup)
                return

            elif tup_sum > target:
                return

            for i in range(ind, len(candidates)):
                tup_list.append(candidates[i])

                recurse_3(i, tup_list)

                tup_list.pop()

            memo[(ind, tup)] = True

        # recurse_3(0, [])

        # return res

        # ----------------****---------------------------------****---------------------------------****-----------------

        # recurse_4 is not the solution for this question. It is a solution for the number of ways we can add up to the target

        def recurse_4(ind, curr_sum):
            if curr_sum == target:
                return 1

            if (ind, curr_sum) in memo:
                return memo[(ind, curr_sum)]

            num_ways_to_add_to_target_from_ind_from_curr_sum = 0

            for i in range(ind, len(candidates)):
                if curr_sum + candidates[i] > target:
                    continue

                else:
                    # notice the variables passed to the recursive function in the following line. It's "i" and NOT "ind".
                    # Initially passed "ind" in the following line (careless mistake)
                    num_ways_to_add_to_target_from_ind_from_curr_sum += recurse_4(i, curr_sum + candidates[i])

            memo[(ind, curr_sum)] = num_ways_to_add_to_target_from_ind_from_curr_sum

            return memo[(ind, curr_sum)]

        return recurse_4(0, 0)

        '''
        recurse_4 will consider all elements in "input list" more than once. 
        
        eg: [2,3,6,7] and target = 7 => output 2 => [[7], [2,2,3]] 2 is considered twice even though only one 2 is present in input and 3 considered once)
        
        This assumes that we have infinite 2's, inifite 3's,... at our disposal
        
        But what if the question says we have only one 2 and one 3 and one 6, .. Meaining we can count an element only once to add up to the target ?
        
        Sol for such a scenario would be to change the recursive call to 
        
        num_ways_to_add_to_target_from_ind_from_curr_sum += recurse_3(i + 1, curr_sum + candidates[i])
        
        This ensures that in subsequent recursive calls, we do not consider the first i elements of input because we have already visited them 
        
        if we make the above change for the same 
        
        eg: [2,3,6,7] and target = 7 => output 1
        
        because only 7 will adds up to the target. There is no other combination WITHOUT REPETITION OF ELEMENYS that adds to the target
        
        Another eg: 
        - candidates = [2,7,6,3,5,1] target = 9 => output = 21 (Repetition allowed. Same ele can be counted infinite times)
        
        - candidates = [2,7,6,3,5,1] target = 9 => output = 4 (Repetition NOT allowed. Same ele canNOT be counted multipe times)
        
        
        '''

'''
Time compl. of backtracking prob can be calculated using the following 2 factors

1) Fan out (meaning how many edges can come out of a node, in the worst case). For this question, fan out factor is 'n'
2) How deep in the worst case can the recursion stack (or recursion tree) grow?. For this question, max depth of recursion tree is 'n'

So, time comp is n ** n in the worst case

Space compl is O(n)

Slightly more accurate representation of time and space compl. can be observed in the solution tab on leetcode

'''

'''
40. Combination Sum II
https://leetcode.com/problems/combination-sum-ii/

Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

Each number in candidates may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.



Example 1:

Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
'''


class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        '''
        Like the previous question, this question asks for the actual combination of numbers (NOT the number of possible combinations)
        So, backtracking is the best possible approach
        '''
        # approach 1

        memo = {}
        candidates.sort()

        # [1, 1, 2, 5, 6, 7, 10]

        def recurse_2(i, curr_sum):
            if (i, curr_sum) in memo:
                # print('memo hit 1')
                return memo[(i, curr_sum)]

            if curr_sum == target:
                return [[]]

            local_result = []

            for ind in range(i, len(candidates)):
                if ind > i and candidates[ind] == candidates[ind - 1]:
                    continue

                if curr_sum + candidates[ind] > target:
                    continue

                result = recurse_2(ind + 1, curr_sum + candidates[ind])

                if result:

                    for item in result:
                        temp_item = list(item)  # DEEP COPY TO FIX ISSUE STATED IN RECURSE_1
                        temp_item.append(candidates[ind])
                        local_result.append(temp_item)

            memo[(i, curr_sum)] = local_result

            return local_result

        # res = recurse_2(0, 0)
        # return res

        # approach 2

        res = []

        def backtrack(i, curr_sum, curr_comb):
            if curr_sum == target:
                # Note the usage of "list(curr_comb)" -> This is a deep copy => Shallow copy won't work because curr_comb is a list object
                # Code will fail if you try using shallow copy like this "res.append(curr_comb)"
                res.append(list(curr_comb))
                return

            for ind in range(i, len(candidates)):
                if ind > i and candidates[ind] == candidates[ind - 1]:
                    continue

                if curr_sum + candidates[ind] > target:
                    continue

                curr_comb.append(candidates[ind])
                backtrack(ind + 1, curr_sum + candidates[ind], curr_comb)
                curr_comb.pop()

        backtrack(0, 0, [])

        return res


'''
Time compl. of backtracking prob can be calculated using the following 2 factors

1) Fan out (meaning how many edges can come out of a node, in the worst case). For this question, fan out factor is 2. If you think in a more abstract sense, 
there are only 2 possibilities. Either there is an edge between n and n + 1 or there is no edge between n and n + 1

2) How deep in the worst case can the recursion stack (or recursion tree) grow?. For this question, max depth of recursion tree is 'n'

So, time comp is 2 ** n in the worst case

Explanation from solution tab is as follows (The following also makes sense)
In the worst case, our algorithm will exhaust all possible combinations from the input array. Again, in the worst case, let us assume that each number is unique. 
The number of combination for an array of size N would be 2 ** N
 i.e. each number is either included or excluded in a combination.


Since there is a sort operation at the top, total time compl is

time: (n log n) + (2 ** n )

Space compl is O(n)

'''

'''
41. First Missing Positive
https://leetcode.com/problems/first-missing-positive/

Given an unsorted integer array nums, return the smallest missing positive integer.

You must implement an algorithm that runs in O(n) time and uses constant extra space.
'''


class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:

        max_val = (2 ** 31) + 1

        for i in range(len(nums)):

            if nums[i] <= 0:
                nums[i] = max_val

        for i in range(len(nums)):
            num_i = abs(nums[i])

            if num_i > len(nums):
                continue

            # handle duplicates in input. eg: [1, 1] should output
            if nums[num_i - 1] > 0:
                nums[num_i - 1] = -1 * nums[num_i - 1]

        for i in range(0, len(nums)):
            if nums[i] > 0:
                return i + 1

        return len(nums) + 1


'''
Notice the submissions tab. The above approach has 40x more running time because we are trying to replace the -ve and 0 values with 2 ** 31.
Since 2 ** 31 need a lot of bits in binary to represent, this takes more time is what I think is making this solution to take a lot of time

If the interviewer says, we cannot store any value greater than 2 ** 31 in the environemnt, instead of using 2 ** 31 + 1, 
use value "1" to replace all -ve and 0's. But before making this replacement, check if 1 is present in nums

if 1 not in nums:
    return 1

for i in range(n):
    if nums[i] <= 0 or nums[i] > n:
        nums[i] = 1

time: O(n)
space: O(1)

Failed test cases
1) [2,1]
2) [1,1]
3) [1]
'''

'''
42. Trapping Rain Water
https://leetcode.com/problems/trapping-rain-water/
'''


class Solution:
    def trap(self, height: List[int]) -> int:
        # approach 1 (stack)
        '''
        i = 0  ||
        tw= 0  ||
        st= [] => st[1]

        i = 1
        tw = 0 => 0
        st = [0] => []
        wid = 1
        h = 0

        i = 2
        tw = 0 =>
        st[1] => [1,2]

        i = 3
        tw = 0 => 0 => 1
        st = [1,2] => [1] =>[] => [3]
        ht = 2

        sti = 2
        wid = 1
        h = 0

        sti = 1
        wid = 2
        h = 1

        i = 4
        tw = 1
        st = [3] => [3,4]

        i = 5
        tw = 1
        st = [3,4]  => [3,4,5]

        i = 6
        tw = 1 => 2
        st => [3,4,5] => [3,4] => [3] => [3,6]
        ht = 1

        sti = 5
        wid = 0
        h =. 0

        sti = 4
        wid = 1
        h =. 1

        i = 7
        tw = 2
        st = [3,6]

        sti = 6
        wid = 0
        ht = 1
        '''
        total_water = 0
        st = []

        for i, ht in enumerate(height):

            # Try to understand with the following eg why the following condition should not have "<=" like this "height[st[-1]] <= ht"
            '''
            height = [7,6,6]
            st = [0,1] 
            ht = 6
            i = 2
            now the height of your right wall is ht = 6
            If you have <=, you will pop out the top most stack => stack_top_index = 1
            width = 2 - 1 = 1
            h = height[st[-1]] - height[stack_top_ind] = height[0] - height[1] = 7 - 6 = 1
            total_water = 1 * 1 = 1 AND THIS IS WRONG. YOU CANNOT HOLD ANY WATER BETWEEN [7, 6, 6] 
            '''
            while st and height[st[-1]] < ht:
                stack_top_index = st.pop()

                if not st:
                    break

                '''
                Try and understand why the following line is (i - st[-1]) - 1 instead of "i - stack_top_index" 
                eg: [4,2,0,3,2,5]
                The abpve ex will fail if you use "i - stack_top_index"
                Notice the case when the stack becomes 
                st = [0,3]
                i = 5
                ht = 5
                '''

                width = (i - st[-1]) - 1

                '''
                Try and understand why the min(a,b) in the following line is
                You should pick the smallest wall (on the left or right) as your boundary
                eg: [4,2,3] 
                '''
                h = min(height[st[-1]], ht) - height[stack_top_index]

                total_water += width * (h)
                prev_ht = h

            st.append(i)

        return total_water

        # approach 2 (dynamic prog)

        '''total_water = 0

        left_max = [0]
        right_max = [0]

        for i in range(0, len(height)):

            if height[i] > left_max[-1]:
                left_max.append(height[i])

            else:
                left_max.append(left_max[-1])

        height.reverse()

        for i in range(0, len(height)):

            if height[i] > right_max[-1]:
                right_max.append(height[i])

            else:
                right_max.append(right_max[-1])


        right_max.reverse()
        height.reverse()

        for i, ht in enumerate(height):
            water_that_can_be_held = min(left_max[i], right_max[i]) - ht

            if water_that_can_be_held > 0:
                total_water += water_that_can_be_held


        return total_water
     '''


'''
time: O(n)
space O(1) for approach 1
space O(n) for approach 2
'''

'''
45. Jump Game II
https://leetcode.com/problems/jump-game-ii/

Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.
'''


class Solution:
    def jump(self, nums: List[int]) -> int:
        # approach 1 - Greedy
        jump_from_ind = 0
        jumps = 0
        last_ind = len(nums) - 1

        # Note: The while loop end condition is "< last_ind" and not "<= len(nums)". The latter will lead to infinite loop

        while jump_from_ind < last_ind:

            max_jump_from_jump_from_ind = nums[jump_from_ind]
            optimal_jump_to_ind = jump_from_ind
            jumps += 1

            '''
            test case that failed: (without the following line) # Should not increment jump one more time if you can reach last ind from your current position
            1) [1,2]
            '''
            if max_jump_from_jump_from_ind >= last_ind:
                break

            for jump_to_ind in range(jump_from_ind + 1, jump_from_ind + max_jump_from_jump_from_ind + 1):

                if jump_to_ind + nums[jump_to_ind] >= last_ind:
                    optimal_jump_to_ind = last_ind
                    jumps += 1
                    break

                if (jump_to_ind + nums[jump_to_ind]) > (optimal_jump_to_ind + nums[optimal_jump_to_ind]):
                    optimal_jump_to_ind = jump_to_ind

            jump_from_ind = optimal_jump_to_ind

        return jumps

        # approach 2 - DP

        '''
        # Second approach
        last_ind = len(nums) - 1
        min_jumps = len(nums)
        memo = [[0] * (len(nums) + 1) for i in range(len(nums))]

        def recurse(ind, num_jumps_so_far):
            nonlocal min_jumps

            if memo[ind][num_jumps_so_far]:
                return

            if ind == last_ind:
                min_jumps = num_jumps_so_far if num_jumps_so_far < min_jumps else min_jumps
                return

            jump_len = nums[ind]

            for i in range(ind + 1, ind + jump_len + 1):
                if i > last_ind:
                    break

                recurse(i, num_jumps_so_far + 1)

            memo[ind][num_jumps_so_far] = True

        recurse(0, 0)

        return min_jumps
        '''


'''
test case that failed:
1) [1,2]


Best approach (Greedy)
time: O(n)
space: O(1)

second approach (Dynamic Prog using memo)
time: O(n ^ 2)
space: O(n)
'''

'''
46. Permutations
https://leetcode.com/problems/permutations/

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
'''

# Perfect example of backtracking problem
# No optimizations can be applied here. Tried to apply optimization in the for loop but later realized that nothing can be optimized
# eg input =  [1,2,3,4,5]
'''
Let's assume the case when [3, _, _, _, _, _] 3 is at index 0. For the index 1, there are 4 possibilities and there is no way to realize that 3 is the only number that is present in our comb. 

You can't do 2 for loops like the below as well
for i in range(0, ind):
    # blah blah

for i in range(ind + 1, len_nums):
    # blah blah

because say, now you have [3,2, _, _, _] in your comb, what will your ind variable be???

So, the following is the only approach 

Also notice that the question states that we only have distinct numbers in our input. If we have duplicates in the input, follow the same approach but instead of adding 
the number to the comb list, add the indices to the comb list

At the end, replace each element in all_permutations variable by nums[element] where element is the index 
'''


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        all_permutations = []
        len_nums = len(nums)

        def recurse(comb):
            if len(comb) == len_nums:
                all_permutations.append(list(comb))
                return

            comb_set = set(comb)

            for i in range(0, len_nums):
                if nums[i] in comb_set:
                    continue

                comb.append(nums[i])
                recurse(comb)
                comb.pop()

        recurse([])

        return all_permutations


'''
time: O(n ^ n) => We will have n branches originating from each elem of our recursion tree and there will be n levels
space: O(n)
'''

'''
47. Permutations II
https://leetcode.com/problems/permutations-ii/

Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.



Example 1:

Input: nums = [1,1,2]
Output:
[[1,1,2],
 [1,2,1],
 [2,1,1]]

'''
from collections import Counter


class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # approach 1
        all_permutations = []
        len_nums = len(nums)
        nums_counter = Counter(nums)

        def recurse(comb):

            nonlocal nums_counter

            if len(comb) == len_nums:
                all_permutations.append(list(comb))
                return

            for key in nums_counter:
                if nums_counter[key] == 0:
                    continue

                nums_counter[key] -= 1
                comb.append(key)
                recurse(comb)
                comb.pop()
                nums_counter[key] += 1

        recurse([])
        return all_permutations

        # approach 2
        all_permutations = []
        len_nums = len(nums)

        def recurse(comb):
            if len(comb) == len_nums:
                all_permutations.append(list(comb))
                return

            comb_set = set(comb)

            for i in range(0, len_nums):
                if i in comb_set:
                    continue

                comb.append(i)
                recurse(comb)
                comb.pop()

        recurse([])

        res = set()

        for permutation in all_permutations:

            sub_res = []

            for ele in permutation:
                sub_res.append(nums[ele])

            res.add(tuple(sub_res))

        return res


'''
Notice the use of Counter to handle the duplicates scenario. ** IMPORTANT **

approach 1
time: O(n ^ n) In the worst case we will have n edges emerging from a node and the recursion tree will be of depth n
space: O(n)

approach 2
time: O((n ^ n) + (n ^ n)). The second n ^ n represents the number of times the for loop will occur. In the worst case we will have n ^ n total elements in our all_permutations list
space: O(n)
'''

'''
48. Rotate Image
https://leetcode.com/problems/rotate-image/

eg:
Input: matrix = [
[1,2,3],
[4,5,6],
[7,8,9]]

Output: [
[7,4,1],
[8,5,2],
[9,6,3]]
'''


class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        '''
        transpose (odd num of rows)
        1 4 7
        2 5 8
        3 6 9
        '''
        '''
        reverse rows
        7 4 1
        8 5 2
        9 6 3
        '''
        '''
        input (even rows and even cols)
        1 2 3 4
        5 6 7 8
        9 1 1 2
        3 4 5 6

        Transpose
        1 5 9 3
        2 6 1 4
        3 7 1 5
        4 8 2 6
        '''

        num_rows = len(matrix)
        num_cols = len(matrix[0])
        visited = set()

        for i in range(num_rows):

            for j in range(num_cols):

                # Notice that you only have to check (j,i) and do not need to check (i,j) because row 1 and col 2 will be visited only once.
                # The configuration where we will face issue is when row is 2 and col is 1. Because we have already swapped [1][2] with [2][1]. Swapping it again will make undo the changes of swap
                if (j, i) in visited:
                    continue

                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

                visited.add((i, j))

        for row in matrix:
            row.reverse()

        return matrix


'''
Note: When transposing a matrix, you need to go over the entire matrix. Trying to stop at the center cell of matrix did not work for 
matrix = 
[
[5,1,9,11],
[2,4,8,10],
[13,3,6,7],
[15,14,12,16]
]

Notice if you stop transposing at the end of row = 1. 6 and 16 swap and 7 and 12 swap will not happen. So, you have to visit all the cells. 

Time: O(n)
space: O(n)
'''

'''
49. Group Anagrams
https://leetcode.com/problems/group-anagrams/

Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
'''

from collections import defaultdict, Counter


class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # approach 1

        d = defaultdict(list)

        for word in strs:
            chars_freq_items = list(Counter(word).items())
            chars_freq_items.sort()
            chars_freq_items = tuple(chars_freq_items)
            d[chars_freq_items].append(word)

        return d.values()

        # approach 2
        ans = collections.defaultdict(list)
        for s in strs:
            count = [0] * 26

            for c in s:
                count[ord(c) - ord('a')] += 1

            ans[tuple(count)].append(s)

        return ans.values()


'''
Note: for problems including english alphabets, remember you only have 26 chars and hence you can use something like approach 2 and assume the operation to be in constant time.

(1, 2) is not equal to (2,1) 
((1,2), (3,4)) is not equal to ((3,4), (1,2))

time: O(w * c) where w is the num of words and c is the max chars per word.
approach 1's sort inside for loop can be treated as constant because chars_freq_items can utmost have length 27 because there are only 27 alphabets i english

both approaches theoritically take the same time complexity 

space: O(w * c)
'''

'''
51. N-Queens
https://leetcode.com/problems/n-queens/
'''


class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:

        rows_containing_queen = set()
        cols_containing_queen = set()
        diag_containing_queen = set()
        anti_diag_containing_queen = set()

        mat = [['.'] * n for i in range(n)]
        num_rows = num_cols = n
        res = []
        explored_possibilities = set()

        def can_place_queen_at_cell(row, col):

            if row in rows_containing_queen or col in cols_containing_queen:
                return False

            # Uncomment the following 2 lines for more efficiency. This line is used for checking if there are any queens in the diagonal. Refer to the solutions tab to understand this concept.
            # Note, this is just a hack. You don't have to know this. The code will run fine even when the following lines are commented.

            # if (row - col) in diag_containing_queen or (row + col) in anti_diag_containing_queen:
            #    return False

            # Conventinal check for checking if any queen is there is any of the diagonals
            all_dirs = [(1, 1), (-1, -1), (1, -1), (-1, 1)]

            for direction in all_dirs:

                temp_row = row
                temp_col = col

                while temp_row > -1 and temp_row < num_rows and temp_col > -1 and temp_col < num_cols:

                    if mat[temp_row][temp_col] == 'Q':
                        return False

                    temp_row += direction[0]
                    temp_col += direction[1]

            return True

        def backtrack(curr_row, num_queens_left):
            if num_queens_left == 0:
                local_res = [list(row) for row in mat]
                res.append(local_res)
                return

            r = curr_row

            # Note, you only have to loop on the cols. No need to have an outer for loop "for r in range(num_rows)"
            # Because, once you keep a queen on a row, there is no way you can have another queen in the same row and similarly you should
            # have atleast one queen in one row.
            # So, the only thing we care about is, what are the possible cols in "curr_row + 1" that can hold my next queen

            # If you have an outer for loop for rows, you are wasting a lot of time. For ex. let's say there are 9 rows. You have placed 5 queens on the
            # first 5 rows and now you realize that there is no col on row 6 that can accomodate a queen, with an outer loop, you will still go to the row 7 and
            # try all possibilities for the 6th queen. This is not needed. Because no matter how you try, you cannot place a queen on row 6.

            # The diag and anti diag check are just random optimizations that I noticed from solutions tab. The code will run just fine even without the diag and anti diag hack.
            # I mean you can replace that by conventional diagonal checks using the for loop in can_place_queen_at_cell

            for c in range(num_cols):

                if can_place_queen_at_cell(r, c):
                    # if (r,c) in explored_possibilities:
                    #    continue

                    mat[r][c] = 'Q'
                    rows_containing_queen.add(r)
                    cols_containing_queen.add(c)
                    diag_containing_queen.add(r - c)
                    anti_diag_containing_queen.add(r + c)

                    backtrack(curr_row + 1, num_queens_left - 1)

                    mat[r][c] = '.'
                    rows_containing_queen.remove(r)
                    cols_containing_queen.remove(c)
                    diag_containing_queen.remove(r - c)
                    anti_diag_containing_queen.remove(r + c)

        backtrack(0, n)

        new_res = []

        # The following for loop is for formatting the output to the required format
        for r in res:
            new_res.append(''.join(row) for row in r)

        return new_res


'''
N is the number of queens

Time: O(N!)
Unlike the brute force approach, we will only place queens on squares that aren't under attack. For the first queen, we have NN options. For the next queen, we won't attempt to place it in the 
same column as the first queen, and there must be at least one square attacked diagonally by the first queen as well. Thus, the maximum number of squares we can consider for the second queen is 
N - 2. For the third queen, we won't attempt to place it in 2 columns already occupied by the first 2 queens, and there must be at least two squares attacked diagonally from the first 2 queens. 
Thus, the maximum number of squares we can consider for the third queen is N - 4N−4. This pattern continues, resulting in an approximate time complexity of N!.

space: O(N ** 2)

Refer solutions tab if you find this explanation difficult to understand
'''

