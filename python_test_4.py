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
