Hi, here is your problem today. This problem was recently asked by Facebook:

Given a list of words, for each word find the shortest unique prefix. You can 
assume a word will not be a substring of another word (ie play and playing wont be
in the same words list)

Example
Input: ['joma', 'john', 'jack', 'techlead']
Output: ['jom', 'joh', 'ja', 't']

class Node(object):
    def __init__(self, val, next_nodes=[]):
        self.val = val
        self.next = next_nodes

#the floowing function is not yet tested. But thought this is simpler than the one below
def inser_trie_node(root, word):
    while(word):
        char = word[0]
        new_root = None

        for neighbor in root.next:
            if neighbor.val == char:
                new_root = neighbor
                break

        if not new_root:
            new_root = Node(char)
            root.next.append(new_root)

        root = new_root
        word = word[1:]


def build_trie(word, node):
    print 'build trie called ', word
    if not word:#According to the question, this case is not possible
        return unique_prefix
    
    prev_node = node
    while(True):
        char_in_trie = False
        for child in node.next:
            if word[0] == child.val:
                node = child
                word = word[1:]
                char_in_trie = True
        if not char_in_trie:
            prev_node = node
            break
    
    for char in word:
        print 'for loop char = ', char
        print 'prev_node = ', prev_node.val
        new_trie_node = Node(char, [])
        prev_node.next = prev_node.next + [new_trie_node]
        prev_node = new_trie_node


def build_prefix(word, node):
    if not node:
        return
    
    index = 0
    while(True):
        print 'node = ', node.val
        print 'next = ', node.next
        if len(node.next) == 1:
            return word[:index]

        for next_node in node.next:
            if word[index] == next_node.val:
                index += 1
                node = next_node
                break


root = Node(1)
unique_prefix_list = []
index = 0

while(index < len(inputs)):
    word = inputs[0]
    unique_prefix = build_trie(word, root)
    index += 1

while(index < len(inputs)):
    word = inputs[index]
    unique_prefix = build_prefix(word,root)
    unique_prefix_list.append(unique_prefix)
    index += 1



DIP
Hi, here is your problem today. This problem was recently asked by Facebook:

Given an expression (as_ a list) in reverse polish notation, evaluate the expression.
Reverse polish notation is where the numbers come before the operand.
Note that there can be the 4 operands '+', '-', '*', '/'. You can also assume the 
expression will be well formed.

Example
Input: [1, 2, 3, '+', 4, '*', '-']
Output: -19
The equivalent expression of the above reverse polish notation would be 
(1 - ((2 + 3) * 4)).

input_list = [1, 2, 3, '+', 4, '*', '-']
stack = []
for item in input_list:
    if type(item) == int:
        stack.append(item)
    else:
        num_1 = stack.pop()
        num_2 = stack.pop()
        
        if item == '+':
            stack.append(num_2 + num_1)
        if item == '-':
            stack.append(num_2 - num_1)
        if item == '*':
            stack.append(num_2 * num_1)
        if item == '/':
            stack.append(num_2 / num_1)


DIP
https://leetcode.com/problems/evaluate-reverse-polish-notation/
Hi, here is your problem today. This problem was recently asked by Google:

Given a nested dictionary, flatten the dictionary, where nested dictionary keys 
can be represented through dot notation.

Example:
my_d = {
  'a': 1,
  'b': {
    'c': 2,
    'd': {
      'e': 3,
      'f': 4
    },
    'g': {
      'h': 3,
      'i': 4
    },
  }
}

Output: {
  'a': 1,
  'b.c': 2,
  'b.d.e': 3,
  'b.d.f': 4,
  'b.g.h': 3, 
  'b.g.i': 4
}


all_flat_dictionaries = []
def flatten_dictionary(sub_dictionay, current_val):#
    global all_flat_dictionaries
    if type(sub_dictionay) != dict:
        val = str(sub_dictionay)
        all_flat_dictionaries.append(current_val + '.' + val)
        return

    else:
        for key_2 in sub_dictionay.keys():
            sub_d = sub_dictionay[key_2]
            flatten_dictionary(sub_d, current_val +'.'+ key_2)#eg:(sub_d['d'],
            #'b.d')

for key in my_d.keys():
    if type(my_d[key]) == dict:
        flatten_dictionary(my_d[key], key)#eg:(my_d['b''], 'b')
    else:
        output_str = (key+'.'+str(my_d[key]))
        all_flat_dictionaries.append(output_str)

print all_flat_dictionaries
#['a.1', 'b.c.2', 'b.d.e.3', 'b.d.f.4', 'b.g.i.4', 'b.g.h.3']

output_d = {}
for item in all_flat_dictionaries:
    key = '.'.join(item.split('.')[:-1])
    val = item.split('.')[-1]
    output_d[key] = val


Leetcode - Find first and last pos of an elem in a sorted array
https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-
array/submissions/
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        returned_pos = self.find_element(nums, target, 0)
        if returned_pos == -1:
            return [-1, -1]
        i = returned_pos
        j = returned_pos
        while(i >= 0):
            if nums[i] == target:
                i -= 1
            else:
                break
        
        while(j < len(nums)):
            if nums[j] == target:
                j += 1
            else:
                break
        return [i+1, j-1]
        
    def find_element(self, nums, target, pos):
        if not nums:
            return -1
        
        mid =  len(nums) / 2
        if nums[mid] == target:
            return pos+mid
        elif nums[mid] > target:
            return self.find_element(nums[0:mid], target, pos)
        else:
            return self.find_element(nums[mid+1:], target, pos+mid+1)
            

Leetcode 
Validate Sudoku
https://leetcode.com/problems/valid-sudoku/solution/

class Solution:
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        # init data
        rows = [{} for i in range(9)]
        columns = [{} for i in range(9)]
        boxes = [{} for i in range(9)]

        # validate a board
for i in range(9):
    for j in range(9):
                num = board[i][j]
                if num != '.':
                    num = int(num)
                    #the foll line could be a bit confusing. Don't worry abt it
                    #You can always follow a any strategy to find out eles in 
                    #every 3 X 3 grid
                    box_index = ((i / 3 ) * 3) + (j / 3)
                    
                    # keep the current cell value
                    rows[i][num] = rows[i].get(num, 0) + 1
                    columns[j][num] = columns[j].get(num, 0) + 1
                    boxes[box_index][num] = boxes[box_index].get(num, 0) + 1
                    
                    # check if this value has been already seen before
                    if rows[i][num] > 1 or columns[j][num] > 1 or \
                    boxes[box_index][num] > 1:
                        return False         
        return True


DIP Ask prof Kasthu how to go about problems like these
Build BST from Post order traversal
https://www.geeksforgeeks.org/construct-a-binary-search-tree-from-given-postorder/

post_order_list = [1,3,2,4,8,10,9,7,5]

root = post_order_list[-1]

#This is my solution O(n^2) eg for worst case would a bst which has only right
#childs untested code
def build_bst(my_l, should_gt_than, should_lt_than):
    if not my_l:
        return None

    ele = my_l.pop()

    if not(ele > should_gt_than and ele < should_lt_than):
        return None

    node = Node(ele)
    #this can be optimized further. Traverese backwards through the my_l till the
    #point where the ele in my_l is greater than root. Take these ele in a list
    #called right_subtree_list
    #Put The remaining ele are in the other list left_subtree_list.
    #node.r = build_bst(right_subtree_list)#we don't need other params
    #node.l = build_bst(left_subtree_list)
    child = my_l[-1]
    if chile > ele:
        node.r = build_bst(my_l, ele, should_lt_than)#all the ele on the right 
        #sub tree should be greater than the current ele
    
    for index, child in enumerate(my_l):
        if child < ele:
            node.l = build_bst(my_l[ind:], should_gt_than, ele)#all the ele on the
            #left sub tree should be greater than the current ele
            break

#Alternate O(n) solution:
https://www.geeksforgeeks.org/construct-a-binary-search-tree-from-given-postorder/

Build Binary tree from Inorder and_ post_order - next problem



Rectangle Intersection
DIP
Given two rectangles, find the area of intersection.

bottom_left should be maximized and top right should be minimized
2 rectangles:
bl -> bottom left
tr -> top right

rect_1 = (bl_1, bl_2),                        (tr_1, tr_2)
rect_1 = (bl_3, bl_4),                         (tr_3, tr_4)
inters = (max(bl_1,bl_3), max(bl_2, bl_4)), (min(tr_1, tr_3), min(tr_2, tr_4))

eg:
r_1 =            (2,3) (4,5)
r_2 =            (1,2) (3,4)
intersect_rect = (2,3) (3,4)

intersect_area = (x2-x1) * (y2-y1) 
= (3-2) * (4-3) = 1


Merge Sort:

my_l = [10,5,3,7,9,4,1,2]

def merge_sort(arr):
    if len(arr) == 1:
        return arr

    if len(arr_1) == 2:
        if arr_1[0] > arr_1[1]:
            return [arr_1[1], arr_1[0]]
        else:
            return arr_1

    mid = len(arr)/2
    arr_1 = merge_sort([:mid]) #T(n/2)
    arr_2 = merge_sort([mid+1:]) #T(n/2)
    merge_sorted_arrays(arr_1, arr_2) #cn
#You know how to merge 2 sorted arrays


Quick Sort:
my_l = [10,5,3,7,9,4,1,2]

def quick_sort(arr):
    if not arr or len(arr) == 1:
        return

    pivot = arr[-1]
    pos = place_pivot_in_correct_pos(pivot) #cn
    quick_sort(arr[:pos]) #T(r-1)
    quick_sort(arr[pos+1:]) #T(n-r)


Subset Sum

my_l = [11,6,5,1,7,13,12]
target = 15
no_of_subsets = 0
memo = {}

def subset_sum(pos, remaining_target):
    if remaining_target == 0:
        no_of_subsets += 1
        return True

    if pos < 0:
        return False

    if memo[pos - 1][target - my_l[pos]] == True:
        return True
    elif memo[pos - 1][target] == True:
        return True
    elif memo[pos - 1][target] == 'unvisited':
        pass

    with_ele = subset_sum(pos - 1, remaining_target - my_l[pos])
    without_ele = subset_sum(pos - 1, remaining_target)
    memo[pos][remaining_target] = with_ele or without_ele

    return memo[pos][remaining_target]


Algorithms class_
words = ['bot', 'hear', 'a', 'heart', 'hand', 'and', 'saturn', 'spin']
string = 'BOTHEARTHANDSATURNSPIN'.lower()
memo = ['unvisited' for i in range(len(string))]

def is_splittable(st, end):
    print 'start = ', st
    print 'end = ', end
    print 'word = ', string[st:end]
    if st >= len(string):
        return True

    if end >= len(string):
        if string[st:end] in words:
            return True
        return False
    
    if memo[st] == 'unvisited':
        pass
    elif memo[st] == True:
        print 'ret True because of memo'
        return True
    elif memo[st] == False:
        print 'ret false from memo'
        return False
    else:
        pass
    #elif memo[st][end] == 'unvisited':
    #    pass
    with_word = False
    if string[st:end] in words:
        with_word = is_splittable(end, end+1)
    
    #if not with_word:
    without_word = is_splittable(st, end+1)
    
    memo[st]= with_word or without_word
    return memo[st]


LIS
lis_array = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15] 

The above input should return 6 since the longest increasing subsequence is 
0, 2, 6, 9 , 11, 15.

lis_array = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
lis_array = [8,9,10,11, 7, 15]
memo = [False for i in range(len(lis_array))]

def lis(st_index, prev_element):
    if st_index < 0:
        return 0
    
    
    if memo[st_index]:
        if lis_array[st_index] < prev_element:
            return 1 + memo[st_index]
        else:
            return memo[st_index]
    

    with_element = float('-inf')
    if lis_array[st_index] < prev_element:
        with_element = 1 + lis(st_index - 1, lis_array[st_index])

    without_element = lis(st_index - 1, prev_element)
    print st_index, with_element, without_element
    memo[st_index] = max(with_element, without_element)
    return max(with_element, without_element)

lis(len(lis_array) - 1, 25)



def lis(index, prev_element):
    if index > len(lis_array):
        return 0
    
    if memo[index] != False:
        return memo[index]

    if lis[index] < prev_element:
        without_element = lis(index + 1, lis[index])
    
    with_element = lis(index + 1, lis[index])
    without_element = lis(index + 1, prev_element)

    return max(with_element, without_element)


array = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 6, float('+inf')]
array = [1,4,2,3,5, float('+inf')]
#array = nums_list
mat = [['unvisited' for j in range(len(array)+1)] for i in range(len(array) - 1)]

for i in range(len(array) - 1):
    mat[i][len(array)] = 0


def lis_bigger(i,j, curr_seq):
    if j >= len(array):
        #print '\ni = ',i
        #print 'j = ',j
        #print 'curr_seq = ', curr_seq
        return curr_seq

    if mat[i][j] != 'unvisited':
        return curr_seq + mat[i][j]
    
    if array[j] <= array[i]:
        mat[i][j] = lis_bigger(i, j+1, curr_seq)
        return mat[i][j]
    
    take = lis_bigger(j, j+1, curr_seq + [array[i]])
    skip = lis_bigger(i, j+1, curr_seq)
    #mat[i][j] = max(take, skip)
    if len(take) > len(skip):
        mat[i][j] = take
    else:
        mat[i][j] = skip

    return mat[i][j]

lis_bigger(0,1,[])


array = [1,4,2,3,5, float('+inf')]
array = [0, 8, 7, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15, float('+inf')]

nums_string = file('input.txt', 'r').read()
nums_list = []
for num in nums_string.split('\n'):
    print num
    nums_list.append(int(num))

array = nums_list
diff_array = []

for index, item in enumerate(array[1:]):
    diff_array.append(item - array[index - 1])

mat = [[[] for j in range(len(array)+1)] for i in range(len(array))]
for i in range(len(array) - 1):
    mat[i][len(array)] = []

j = len(array) - 2

while(j >= 0):
    for i in range(len(array)):

        if array[i] > array[j]:
            mat[i][j] = mat[i][j+1]
        else:
            take = mat[j][j + 1] #[1][2]
            skip = mat[i][j + 1] #[0][2]
            if len(skip) > 1 + len(take):
                mat[i][j] = mat[i][j+1]
            else:
                mat[i][j] = [array[j]] + mat[j][j+1]
            #if (1 + len(take)) > len(skip):
            #    mat[i][j] = [array[j]] + mat[i][j + 1]
            #else:
            #    mat[i][j] = mat[j][j+1]
    j -= 1


Leetcode
https://leetcode.com/problems/count-and-say/submissions/

The count-and-say sequence is the sequence of integers with the first five terms as 
following:

1.     1
2.     11
3.     21
4.     1211
5.     111221
1 is read off as_ "one 1" or 11.
11 is read off as_ "two 1s" or 21.
21 is read off as_ "one 2, then one 1" or 1211.

import itertools

class Solution(object):
    def countAndSay(self, n):
        new_num = '1'
        index = 1
        
        while(index < n):
            curr_num = new_num
            new_num = ''
            
            for ele, occur in itertools.groupby(curr_num):
                new_num += (str(len(list(occur))) + ele)
            #print new_num
            
            index += 1
            
        return new_num


leetcode
https://leetcode.com/problems/combination-sum/submissions/
Input: candidates = [2,3,5], target = 8,
A solution set is:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]

class Solution(object):
    all_combs = []
    my_list = []
    
    def combinationSum(self, candidates, target):
        res = []
        candidates.sort()
        self.dfs(candidates, target, 0, [], res)
        return res

    def dfs(self, nums, target, index, path, res):
        if target < 0:
            return  # backtracking
        if target == 0:
            res.append(path)
            return 
        for i in xrange(index, len(nums)):
            self.dfs(nums, target-nums[i], i, path+[nums[i]], res)


DIP
Hi, here is your problem today. This problem was recently asked by Google:

Given a sorted list of numbers, and two integers low and high representing the lower
and upper bound of a range, return a list of (inclusive) ranges where the numbers 
are missing. A range should be represented by a tuple in the format of 
(lower, upper).

Here is an example and some starting code:

def missing_ranges(nums, low, high):
  # Fill this in.
  
print(missing_ranges([1, 3, 5, 10], 1, 10))
# [(2, 2), (4, 4), (6, 9)]
nums = [1, 5, 10]
low = 2
high = 10
missing_tup_list = []
st_ind = None

for ind, num in enumerate(nums):
    print 's = ', st_ind
    if num >= low and st_ind == None:
        st_ind = ind
    print st_ind

    if num >= high:
        en_ind = ind
        break

if nums[st_ind] > low: #if we don't have the low in input
    missing_tup_list.append((low, nums[st_ind] - 1))

if nums[en_ind] < high: #if we don't have the high in input
    missing_tup_list.append((nums[en_ind - 1] + 1, high))
    en_ind = en_ind - 1

nums = nums[st_ind:en_ind + 1]
for ind, ele in enumerate(nums):
    if ind == 0:
        continue
    if ele - 1 != nums[ind - 1]:
        prev_array_ele = nums[ind - 1]
        missing_tup_list.append((prev_array_ele + 1, ele - 1))


DIP

Hi, here is your problem today. This problem was recently asked by Google:

Given a positive integer, find the square root of the integer without using any 
built in square root or power functions (math.sqrt or the ** operator). Give 
accuracy up to 3 decimal points.

sol: https://www.geeksforgeeks.org/square-root-of-a-number-without-using-sqrt-function/

num = 20
# We first figure out the integers between which 20's sqrt falls which is 4 and 5
# Then we do binary search between 4 and 5 untill the len of mid is 1 greater than
# the num of decimal points the question asks for (or)
# if mid * mid = input
def binary_search(st, en, num):
    if st > en:
        return st, en

    mid = (st + en) / 2
    
    if (len(str(mid).split('.')[1]) >= 5) or (mid * mid == num):
        return mid

    print 'mid = ', mid
    
    if mid * mid > num:
        return binary_search(st, mid, num)
    else:
        return binary_search(mid, en, num)

i = 0
while(i < num):
    if i * i == num:
        print 'sqrt of num is = ', i
        break
    if i * i > num:
        break
    i += 1

if i * i > num:
    j = i - 1 #The sqrt of num is between i and j

binary_search(float(i),float(j),float(num))


DIP

Hi, here is your problem today. This problem was recently asked by AirBNB:

The power function calculates x raised to the nth power. If implemented in O(n) it
would simply be a for loop over n and_ multiply x n times. Instead implement this
power function in O(log n) time. You can assume that n will be a non-negative int.

#incomplete program - need to complete it
num = 2
power = 10
curr_power = 1
power_val_dict = {1:num}
sum_of_powers_calculated_so_far = 1

while((curr_power * 2) < power):
    if sum_of_powers_calculated_so_far > power:
        break
    print 'curr_power = ', curr_power
    num = num * num
    curr_power = curr_power * 2
    sum_of_powers_calculated_so_far += curr_power
    power_val_dict[curr_power] = num

req_power = power - curr_power
power_val_dict[curr_power] * power_val_dict[req_power]



https://leetcode.com/problems/first-missing-positive/
Find first missing positive number
Given an unsorted integer array, find the smallest missing positive integer.

Example 1:

Input: [1,2,0]
Output: 3
Example 2:

Input: [3,4,-1,1]
Output: 2
Example 3:

Input: [7,8,9,11,12]
Output: 1

class Solution(object):
    def firstMissingPositive(self, nums):
        index = 0
        while(index < len(nums)): #removing all -ve nums from the list
            #print nums
            if nums[index] < 1:
                nums = nums[:index] + nums[index + 1:]
                continue
            index += 1
            
        #print 'nums after 1st while = ', nums
        
        index = 0
        while(index < len(nums)): #negating the indices of input (corresponding to 
        #ele in array)
            ele_at_index = abs(nums[index])
            if ele_at_index <= len(nums):
                if nums[ele_at_index - 1] > 0: nums[ele_at_index - 1] = \
                -1 * nums[ele_at_index - 1]#eg case: [1,1]
            index += 1
        
        #print 'nums after 2nd while = ', nums
        
        index = 0
        while(index < len(nums)): #the first non negative index is the ans
            if nums[index] > 0:
                return index + 1
            index += 1
            
        #print 'nums after 3rd while = ', nums
        return len(nums) + 1 #if no non negative ints are found (eg: [1,2,3] your 
        #ans should be 4)



DIP
Hi, here is your problem today. This problem was recently asked by Uber:

Given a square 2D matrix (n x n), rotate the matrix by 90 degrees clockwise.

Here is an example and some starting code:

def rotate(mat):
  # Fill this in.

mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# Looks like
# 1 2 3
# 4 5 6
# 7 8 9

# should be transformed like
# 7 4 1
# 8 5 2
# 9 6 3
print(rotate(mat))
# [[7, 4, 1], [8, 5, 2], [9, 6, 3]]

mat = [
['00', '01', '02'],
['10', '11', '12'],
['20', '21', '22'],
]

[20, 10, 00]
[21, 11, 01]
[22, 12, 02]

num_rows = num_cols = len(mat)
new_mat = [[False] * num_rows] * num_rows

row = 0
col = 0

print 'num_rows = ', num_rows
while(row < num_rows):#0, 1, 2
    print 'r while = ', row
    col = 0
    while(col < num_rows): #0, 1, 2
        print 'r = ', row
        print 'c = ', col
        new_mat[row][col] = mat[col][row]
        col += 1
    
    print 'new_r = ', new_mat[row]
    new_mat[row] = new_mat[row][::-1]
    row += 1
    

DIP
Hi, here is your problem today. This problem was recently asked by Twitter:

Given 2 strings s and t, find and return all indexes in string s where t is an anagram.

Here is an example and some starter code:

print(find_anagrams('acdbacdacb', 'abc'))
# [3, 7]

#Need to test this code
anagram_list = []
anag_dict = {}
#            0123 45
input_string = 'zacdbacdacb'
string_2 = 'abc'
len_string_2 = len(string_2)

for char in string_2:
    if char in anag_dict:
        anag_dict[char] += 1
    else:
        anag_dict[char] = 1

index = 0

while(index < len(input_string)):
    print 'index = ', index
    char = input_string[index]
    print 'char = ', char

    if char in anag_dict:
        print 'in if anag_dict = ', anag_dict
        new_dict = anag_dict.copy()

        new_end_ind = index + len_string_2
        new_start_index = index

        if new_end_ind > len(input_string):
            break

        #the following is the code to check if substr is anagram of str_2
        while(new_start_index < new_end_ind):
            char = input_string[new_start_index]
            if char in new_dict:
                new_dict[char] = new_dict[char] - 1
                if new_dict[char] == 0:
                    new_dict.pop(char)
                new_start_index += 1
            else:
                index = new_start_index + 1
                break

        if len(new_dict.keys()) == 0:
            anagram_list.append(index)
            index = new_end_ind
        else:
            index = new_start_index
    else:
        index += 1


Leetcode
https://leetcode.com/problems/jump-game-ii/

class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        index = 0
        jumps = 0
        if len(nums) == 1:
            return 0
        
        while(index < len(nums)):# 0
            #print 'while index = ', index
            max_allowed_jump_from_index = index + nums[index] #2
            if max_allowed_jump_from_index >= len(nums) - 1:
                jumps += 1
                break
            ideal_jump_index = max_allowed_jump_from_index #2
            max_possible_jump = float('-inf')
            index += 1
            
            while(index <= max_allowed_jump_from_index and index < len(nums)):#0<=2
                if index + nums[index] > max_possible_jump:# 0+2 > -inf; 1+3 > 2
                    max_possible_jump = index + nums[index] # 2
                    ideal_jump_index = index # 0 2
                index += 1
            
            index = ideal_jump_index
            jumps += 1
        
        return jumps
            

Leetcode
3 sum closest
https://leetcode.com/problems/3sum-closest/

class Solution:
    # @return an integer
    def threeSumClosest(self, num, target):
        num.sort()
        result = num[0] + num[1] + num[2]
        for i in range(len(num) - 2):
            j, k = i+1, len(num) - 1
            while j < k:
                sum = num[i] + num[j] + num[k]
                if sum == target:
                    return sum
                
                if abs(sum - target) < abs(result - target):
                    result = sum
                
                if sum < target:
                    j += 1
                elif sum > target:
                    k -= 1
            
        return result


DIP
Hi, here is your problem today. This problem was recently asked by Apple:

Given 2 binary trees t and s, find if s has an equal subtree in t, where the 
structure and the values are the same. Return True if it exists, otherwise ret False.


#Do inorder traversal of both the trees and see if the inord trav list of subtree
#is a sublist of the main tree


DIP
Hi, here is your problem today. This problem was recently asked by Microsoft:

Given a binary tree, find the level in the tree where the sum of all nodes on that
level is the greatest.

#DO a bfs have a dict[level] = [values in that level]
#Finally find the level which has highest val


DIP
Hi, here is your problem today. This problem was recently asked by Microsoft:

Given a list of sorted numbers (can be both negative or positive), return the list
of numbers squared in sorted order.

print(square_numbers([-5, -3, -1, 0, 1, 4, 5]))
# [0, 1, 1, 9, 16, 25, 25]
You have to do it in O(n) time

sorted_list = [-5, -3, -1, 0, 1, 4, 5]
squares_list = []

i = 0
j = len(sorted_list) - 1

while(i <= j):
    if i == j:
        squares_list.append(abs(sorted_list[j]) ** 2)
        break
    elif abs(sorted_list[i]) == abs(sorted_list[j]):
        squares_list.append(abs(sorted_list[i]) ** 2)
        squares_list.append(abs(sorted_list[j]) ** 2)
        i += 1
        j -= 1
    elif abs(sorted_list[i]) > abs(sorted_list[j]):
        squares_list.append(abs(sorted_list[i]) ** 2)
        i += 1
    else:
        squares_list.append(abs(sorted_list[j]) ** 2)
        j -= 1

squares_list.reverse()
print squares_list


Leetcode
https://leetcode.com/problems/4sum/
4 sum

class Solution(object):
    def fourSum(self, nums, target):
        nums.sort()
        four_pairs = {}
        #print 'nums = ', nums
        i = 0
        while(i < len(nums) - 3): # we are stopping at len(nums) - 4 because we
        #need only sets of 4 eles (not  3 ele set or 2 ele set)
            #if nums[i] > target:
            #    break
            j = len(nums) - 1 
            while(j > i+2):#We keep i and j as fixed, we move the pointers l and k
            #depending on the sum 
                #print 'i = ', i
                #print 'j = ', j
                
                sum_i_j = nums[i] + nums[j]
                #print 'sum_i_j = ', sum_i_j
                k = i+ 1
                l = j - 1
                #print 'k = ', k
                #print 'l = ', l
                while(k < l):
                    #print 'loop k = ', k
                    #print 'loop l = ', l
                    if sum_i_j + nums[k] + nums[l] == target:
                        four_pairs[(nums[i], nums[j], nums[k], nums[l])] = 1 #we 
                        #have dict because we only need unique quad sets
                        #print 'four_pairs = ', four_pairs
                        k += 1
                        l -= 1
                    elif sum_i_j + nums[k] + nums[l] < target:
                        k += 1
                    elif sum_i_j + nums[k] + nums[l] > target:
                        l -= 1
                j -= 1
            i += 1
        
        return four_pairs.keys()
        

DIP
Hi, here is your problem today. This problem was recently asked by Amazon:

Given a 2d n x m matrix where each cell has a certain amount of change on the floor
your goal is to start from the top left corner mat[0][0] and end in the bottom 
right corner mat[n - 1][m - 1] with the most amount of change. You can only move 
either left or down.

Here is some starter code:

def max_change(mat):
  # Fill this in.

mat = [
    [0, 3, 0, 2],
    [1, 2, 3, 3],
    [6, 0, 3, 2]
]

print(max_change(mat))
# 13
mat = [
    [0, 3, 0, 2],
    [1, 2, 3, 3],
    [6, 0, 3, 2]
]

i = 1
while(i < len(mat)):
    j = 0
    while(j < len(mat[0])):
        if j == 0:
            mat[i][j] = mat[i][j] + mat[i-1][j]
        else:
            mat[i][j] = mat[i][j] + max(mat[i-1][j], mat[i][j-1])
        j += 1
    i += 1

print 'ans = ', mat[i-1][j-1]


Leetcode
Merge K sorted lists

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
from Queue import PriorityQueue

class Solution(object):
    def mergeKLists(self, lists):

        #Recommended solution
        
        head = point = ListNode(0)
        q = PriorityQueue()
        for l in lists:
            if l:
                q.put((l.val, l))
        while not q.empty():
            val, node = q.get()
            point.next = ListNode(val)
            point = point.next
            node = node.next
            if node:
                q.put((node.val, node))
        return head.next
    
        #my solution
        op_list= None
        head = None
        i = 0
        while(lists):#iterate utill there is atleast 1 linked list left in 'lists'
        # array
            min_val = float('+inf')
            i = 0
            #print 'lists = ', lists
            while(i < len(lists)): #iterate through all the linked lists to find
            # whose 1st node has min val
                list = lists[i]
                if not list: #special case: eg inp - [[0,1], []]
                    lists = lists[:i] + lists[i+1:]
                    continue
                    
                elif list.val < min_val:
                    min_val = list.val
                    min_val_list_index = i
                i += 1
            #print 'min_val = ', min_val
            
            if min_val == float('+inf'):#special case when ip is [[]]
                continue
                
            lists[min_val_list_index] = lists[min_val_list_index].next #point the 
            #node of the list from which we have taken the min elem to point to 
            #next node
            
            if lists[min_val_list_index] == None: #if the list mentioned in the 
            #prev line has no elements left, remove the node the input "lists"
                lists = lists[:min_val_list_index] + lists[min_val_list_index+1:]
                
            if not op_list:
                op_list = ListNode(min_val)
                head = op_list
            else:
                #new_node = ListNode(min_val)
                op_list.next = ListNode(min_val)#new_node
                op_list = op_list.next
                
        return head


Merge Sorted array variant
https://leetcode.com/problems/merge-sorted-array/submissions/
Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as_ one
sorted array.

Note:

The number of elements initialized in nums1 and nums2 are m and n respectively.
You may assume that nums1 has enough space (size that is greater or equal to m + n)
to hold additional elements from nums2.
Example:

Input:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

Output: [1,2,2,3,5,6]

#This is a problem in which you have to startpopulating your merged array from the
#end. Or else you will face complications

class Solution(object):
    def merge(self, nums1, m, nums2, n):
        m -= 1
        n -= 1
        index = len(nums1) - 1
        
        while(n > -1 and m > -1):
            #print '\nnums1 = ', nums1[:m+1]
            #print 'nums2 = ', nums2[:n+1]
            #print nums1
            #print m
            #print n
            if nums1[m] > nums2[n]:
                nums1[index] = nums1[m]
                m -= 1
            else:
                nums1[index] = nums2[n]
                n -= 1
            index -= 1
        
        #print 'm = ', m
        #print 'n = ', n
        if len(nums2[:n+1]) > 0:
            #print 'in if'
            i = index
            while(n > -1):
                #print 'in whi i, n = ', i, n
                nums1[i] = nums2[n]
                i -= 1
                n -= 1
                #print 'end = ', nums1


Leetcode:
https://leetcode.com/problems/clone-graph/
Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a val (int) and a list (List[Node]) of its neighbors.

"""
# Definition for a Node.
class Node(object):
    def __init__(self, val = 0, neighbors = []):
        self.val = val
        self.neighbors = neighbors
"""
class Solution(object):
    def cloneGraph(self, node):
        if not node:
            return node
        nodes_dict = {}
        queue = [(node, None)]
        start_node = None
        graph_list = []
        
        while(queue):
            node_and_parent = queue[0]
            parent = node_and_parent[1]
            node = node_and_parent[0]
            queue = queue[1:]
            val = node.val
            
            if node in nodes_dict:#this node has already been visited. SO, just add
            # this node to it's parent's neig list and DO NOT REVISIT THIS NODE 
            #BECAUSE you will create duplicate neighbors for this node. eg: see node
            # 3. It can be visited from 2 and 4. So, your queue will have 2 entries for
            #3 (one when you visited 2 and the other when you visited 4). 
                new_node = nodes_dict[node]
                parent.neighbors.append(new_node)
                continue
            else:
                new_node = Node(val)
                nodes_dict[node] = new_node
            
            if not start_node: start_node = new_node
            else:
                parent.neighbors.append(new_node)
            
            for neighbor in node.neighbors:
                if neighbor in nodes_dict: #the neighbors has already been visited
                    new_node.neighbors.append(nodes_dict[neighbor])
                    continue
                else:    
                    queue.append((neighbor, new_node))
                    
        return start_node


Leetcode
Implement an iterator over a binary search tree (BST). Your iterator will be 
initialized with the root node of a BST.

Calling next() will return the next smallest number in the BST.
Note:

next() and hasNext() should run in average O(1) time and uses O(h) memory, where h
is the height of the tree.
You may assume that next() call will always be valid, that is, there will be at
least a next smallest number in the BST when next() is called.
class BSTIterator(object):
    last_popped_node = None
    stack = []
    def __init__(self, root):
        self.inorder_left(root)

    def next(self):
        """
        @return the next smallest number
        :rtype: int
        """
        
        if self.last_popped_node and self.last_popped_node.right:
            self.inorder_left(self.last_popped_node.right)
            
        if self.stack:
            self.last_popped_node = self.stack[0]
            self.stack = self.stack[1:]
        #else:
        #    print 'end of tree'
        
        #print 'end = ', self.stack
        #print 'lpn = ', self.last_popped_node.val
        return self.last_popped_node.val
        

    def hasNext(self):
        """
        @return whether we have a next smallest number
        :rtype: bool
        """
        #print 'has_next ', self.stack
        if self.stack or (self.last_popped_node and self.last_popped_node.right):
            return True
    
    def inorder_left(self, root):
        while(root):
            #print 'while = ', root.val
            self.stack = [root] + self.stack
            root = root.left
        
        return True

# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()


DIP

Hi, here is your problem today. This problem was recently asked by Facebook:

Given a number n, find the least number of squares needed to sum up to the number.

Here is an example and some starting code:

print(square_sum(13))
# Min sum is 32 + 22
# 2

#We can think of this problem as a variation of sum of nums in an array that make
#up a target. Of all the poss results, take the result which is composes of 
#least number of elements.

#Have a list consisting of squares of nums like this
#squares_list = [1,4,9,16,25,36,..sqrt(n)]

squares_list = []
i = 0
while(i**i <= num):
    squares_list.append(i*i)

def target_sum(i, remaining_target, curr_list):
    if remaining_target == 0
    if i > len(squares_list):
        return False


Leetcode
https://leetcode.com/problems/add-and-search-word-data-structure-design/
(usually word search related problems are best when done using tries)
have not used trie here but many sols have used tries

#Not my solution but the best solution.
class WordDictionary(object):
    def __init__(self):
        self.word_dict = collections.defaultdict(list)
        

    def addWord(self, word):
        if word:
            self.word_dict[len(word)].append(word)

    def search(self, word):
        if not word:
            return False
        if '.' not in word:
            return word in self.word_dict[len(word)]
        for v in self.word_dict[len(word)]:
            # match xx.xx.x with yyyyyyy
            for i, ch in enumerate(word):
                if ch != v[i] and ch != '.':
                    break
            else:
                return True
        return False

#MY solution using bfs
class WordDictionary(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.words_dict = {}
        

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: None
        """
        self.words_dict[word] = True
        

    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot 
        character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        if word in self.words_dict:
            return True
        
        poss_words = self.words_dict.keys()
        queue = []
        queue.append((word, poss_words))
        
        while(queue):
            rem_word, poss_words = queue[0]
            #print '\nrem_word = ', rem_word
            #print 'poss_words = ', poss_words
            queue = queue[1:]
            
            if not rem_word:
                return False
            
            first_char = rem_word[0]
            matching_word_list = []
            
            for word in poss_words:
                if word  == rem_word:
                    return True
                if len(word) != len(rem_word):
                    continue
                if word and (first_char == '.' or first_char == word[0]):
                    if len(rem_word) == 1 and len(word) == 1:
                        return True
                    else:
                        matching_word_list.append(word[1:])
                    
                    
            if matching_word_list:
                rem_word = rem_word[1:]
                queue.append((rem_word, matching_word_list))
                
        return False


Leetcode
https://leetcode.com/problems/alien-dictionary/

There is a new alien language which uses the latin alphabet. However, the order 
among letters are unknown to you. You receive a list of non-empty words from the 
dictionary, where words are sorted lexicographically by the rules of this new 
language. Derive the order of letters in this language.

Example 1:

Input:
[
  "wrt",
  "wrf",
  "er",
  "ett",
  "rftt"
]

Output: "wertf"
Example 2:

Input:
[
  "z",
  "x"
]

Output: "zx"

#think of this prob as as directed graph where characters are nodes and the direct
#ion of the arrows specify the lexiographic order

#UNTESTED CODE
class Node(object):
    def __init__(self,val):
        self.val = val
        selt.next = {}
        
class Solution(object):
    def alienOrder(self, words):
        """
        :type words: List[str]
        :rtype: str
        """
        visited_dict = {}
        no_parent_nodes = {}
        for word in words:
            first_char = word[0]
            if not first_char in visited_dict:
                no_parent_nodes[first_char] = True
                
            for char in word:
                if prev_node and prev_node.val == char:
                    continue
                
                if char in visited_dict:
                    char_node = visited_dict[char]
                else:
                    char_node = Node(char)
                    visited_dict[char] = char_node
                    if char in no_parent_nodes:
                        no_parent_nodes[char] = char_node
                
                if not prev_node:
                    continue
                    
                if char_node in prev_node.next:
                    prev_node = char_node
                    continue
                else:
                    prev_node.next[char_node] = True
                    if char in no_parent_nodes:
                        no_parent_nodes.pop(char)
        
        print 'no_parent_nodes = ', no_parent_nodes
        print 'visited_dict = ', visited_dict


DIP
Hi, heres your problem today. This problem was recently asked by Twitter:

Given a matrix, transpose it. Transposing a matrix means the rows are now the 
column and vice-versa.

Heres an example:

def transpose(mat):
  # Fill this in.

o/p:
# [[1, 4],
#  [2, 5], 
#  [3, 6]]

mat = [
    [1, 2, 3],
    [4, 5, 6],
]
num_cols = len(mat[0]) #3
num_rows = len(mat) #2
tran_mat = []

for i in range(num_cols):#0, 1, 2
    row = []
    for j in range(num_rows):#0, 1
        row = row + [mat[j][i]]
        print 'i j= ',i, j
        #print 'tran mat i,j = ', tran_mat[i][j]
        print 'mat j,i = ', mat[j][i]
        #tran_mat[i][j] = mat[j][i]
        #print tran_mat

    print 'row = ', row
    tran_mat.append(row)

print tran_mat


DIP
Hi, here s your problem today. This problem was recently asked by Apple:

Given a list of strings, find the list of characters that appear in all strings.

Here s an example and some starter code:

def common_characters(strs):
  # Fill this in.

print(common_characters(['google', 'facebook', 'youtube']))
# ['e', 'o']

words_list = ['google', 'facebook', 'youtube']
words_list = [set(word) for word in words_list]
op_set = words_list[0]
for word in words_list[1:]:
    op_set = op_set.intersection(word)

print op_set
          
11. Leetcode
container-with-most-water
https://leetcode.com/problems/container-with-most-water/solution/

class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        #[1,8,6,2,5,4,8,3,7]
        
        l = 0
        r = len(height) - 1
        max_collected_water = 0
        
        while(l < r):
            width = r - l
            if height[l] < height[r]:
                h = height[l]
                l += 1
            else:
                h = height[r]
                r -= 1
            max_water_current_config = width * h
            max_collected_water = max(max_collected_water, max_water_current_config )
            
        return max_collected_water

17. Letter Combinations of a Phone Number
https://leetcode.com/problems/letter-combinations-of-a-phone-number/

comb_list = ['abc', 'def', 'ghi']
all_combs = []

def get_all_combs(comb_ind, formed_comb):
    global all_combs
    if len(formed_comb) == len(comb_list):
        all_combs.append(formed_comb)
        return

    for letter in comb_list[comb_ind]:
        get_all_combs(comb_ind + 1, formed_comb + letter)

Time: O(4 ^ n) because at most we will have 4 choices for every position (where position is a num
ranging from 1 - n)

19. Remove Nth Node From End of List
https://leetcode.com/problems/remove-nth-node-from-end-of-list/
#Clarification to ask the interviewer. What if n is larger than the length of list

22. Generate Paranthesis
https://leetcode.com/problems/generate-parentheses
n = 3
all_possible_combs = []

def gen_par(formed_string, no_of_open_braces, no_of_close_braces):
    if len(formed_string) == n * 2:
        all_possible_combs.append(formed_string)
        return

    if no_of_open_braces == n:
        gen_par(formed_string + ')', no_of_open_braces, no_of_close_braces + 1)
    elif no_of_open_braces == no_of_close_braces:
        gen_par(formed_string + '(', no_of_open_braces + 1, no_of_close_braces)
    else:
        gen_par(formed_string + ')', no_of_open_braces, no_of_close_braces + 1)
        gen_par(formed_string + '(', no_of_open_braces + 1, no_of_close_braces)

Time: O(2 ^ (n * 2)) for every position, we have 2 poss - one is '(' and other is ')' (where position
is a num ranging from 1 - (n * 2) )
We can upper bound the above way

23. Merge K sorted list
https://leetcode.com/problems/merge-k-sorted-lists/

If you do it without heap, time will be O(nk), but with heap its O(n log k). See the solutions tab of the
above link

25. Reverse Nodes in k-Group
https://leetcode.com/problems/reverse-nodes-in-k-group/
# 1->2->3->4->5->6 en_node = 5
# 1<-2<-3<-4
# |
# |->5 -> 6
node = head

def reverse_part_linked_list(st_node, en_node):
    prev_node = st_node
    node = node.next

    while(node and node != en_node):
        tmp = node.next
        node.next = prev_node
        prev_node = node
        node = tmp

    st_node.next = en_node
    return

while(node):
    k_copy = k
    st_node = node

    while(k_copy != 0):

        if not node:
            break

        node = node.next
        k -= 1

    reverse_part_linked_list(st_node, node)

Time: O(n)
space: O(1)

29. Divide two integers
https://leetcode.com/problems/divide-two-integers/
https://leetcode.com/problems/divide-two-integers/solution/ # See approach 2. See also video demo

dividend = 99
divisor = 3
res = 1
positive = (dividend < 0) is (divisor < 0)

while (dividend > divisor):
    temp = divisor
    increase_factor = 1
    while (dividend > temp):
        dividend -= temp
        res += increase_factor
        temp <<= 1  # left shifting by 1 to multiple the current divisor in while loop (temp) by 2
        increase_factor <<= 1  # since we multiplied the divisor, we should also multiple the increase_factor by 2

if is_negative:
    res *= -1

print(res)

30. Substring with Concatenation of All Words
#Need to correct the solution to this problem
https://leetcode.com/problems/substring-with-concatenation-of-all-words/

from collections import defaultdict

s = "barfoothefoobarman",
words = ["foo", "bar"]
word_dict = {}
const_word_length = len(words[0])
index = 0
for word in words:
    if word in word_dict:
        word_dict[word] += 1
    else:
        word_dict[word] = 1

# One hint given in the question is all words are of the same length
# If the interviwer says you are given a list of words like in the above question, you can ask if all the words are
# of same length. Sometimes, these minute details will help you like in the below while loop
# (curr_substr = s[0: const_word_length]). Now you know exactly
# you need to take const_word_length characters (3 in this example) to check if the substr is in your dictionary

while (s and len(s) >= (len(words) * const_word_length) + 1): #The part after and is a slight improvisation. But it
    # does not affect theoritical runtime
    curr_substr = s[0: const_word_length]

    if curr_substr in word_dict:
        # all the words in the word dictionary should be in the following range
        required_len = len(words) * const_word_length

        if check_subs(s[0: required_len], word_dict):
            output_list.append(index)

    s = s[1:]
    index += 1


def check_subs(main_substring, word_dict):
    new_dict = defaultdict(int)
    while (main_substring):
        substr = main_substring[0:const_word_length]

        if not substr in word_dict: return False

        new_dict[substr] += 1

        if new_dict[substr] > word_dict[substr]: return False

        main_substring = main_substring[const_word_length:]

    return not (main_substring) #retruns True if main_substring is empty

#Time: O(n * m * k) - n is the length of s and m is the number of words and k is the len of a word in word_list
#Space = O(m + n) #dictionaries we created to hold the words (default_dict and word_dict). n is the len of s.
#In the variable curr_substr we hold the value of s. Worst case curr_substr size is equal to s

32. Longest Valid Parentheses
https://leetcode.com/problems/longest-valid-parentheses/

stack = []
inp_str = ')()((())'
max_len = 0
index = 0
st_ind = None

while(index < len(inp_str)):
    if inp_str[index] == '(':
        stack.append(index)

    else:
        if stack:
            open_brace_index = stack.pop()

            if not stack:
                curr_len = index - st_ind if st_ind and st_ind < open_brace_index else index - open_brace_index
            else:
                curr_len = index - open_brace_index

            max_len = max(max_len, curr_len)

            if not st_ind:
                st_ind = open_brace_index

        else:
            st_ind = None

    index += 1

print (max_len)
#Time: O(n)


34. Find First and Last Position of Element in Sorted Array
https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/

nums = [5,7,7,8,8,10]
inds = [0,1,2,3,4,5]
target = 8

class Solution:
    # returns leftmost (or rightmost) index at which `target` should be inserted in sorted
    # array `nums` via binary search.
    def extreme_insertion_index(self, nums, target, find_left_boundary):
        lo = 0
        hi = len(nums)

        while lo < hi:
            mid = (lo + hi) // 2

            if nums[mid] > target or (find_left_boundary and target == nums[mid]):
                hi = mid
            else:
                lo = mid+1

        return lo

    def searchRange(self, nums, target):
        #Find the left boundary of the element
        left_idx = self.extreme_insertion_index(nums, target, True)

        # assert that `left_idx` is within the array bounds and that `target`
        # is actually in `nums`.
        if left_idx == len(nums) or nums[left_idx] != target:
            return [-1, -1]

        #Find the right boundary of the element
        return [left_idx, self.extreme_insertion_index(nums, target, False) - 1]


42. Trapping rain water
https://leetcode.com/problems/trapping-rain-water/
Refer image 42_lc.png


walls = [2,1,0,1,3]
walls = [0,1,0,2,1,0,1,3,2,1,2,1]
indices_stack = []
curr_ind = 0
total_water = 0

while curr_ind < len(walls):
    
    if indices_stack and walls[indices_stack[-1]] < walls[curr_ind]:
        print '\n\nin if ', curr_ind

        while(walls[indices_stack[-1]] < walls[curr_ind]):
            print 'in while stack = ', indices_stack
            poped_index = indices_stack.pop()
            
            if not indices_stack:
                break
            
            min_height_boundary = min(walls[curr_ind], walls[indices_stack[-1]])
            height_of_popped_wall = walls[poped_index]
            available_height_to_store_water = min_height_boundary - height_of_popped_wall
            breadth = curr_ind - indices_stack[-1] - 1
            print 'h = ', available_height_to_store_water
            print 'b = ', breadth
            total_water += available_height_to_store_water * breadth

    indices_stack.append(curr_ind)
    curr_ind += 1

print total_water



43. Multiply strings
leetcode
https://leetcode.com/problems/multiply-strings/
# 456 * num1
# 123   num2
#------------
#   1 3 6 8
#   9 1 2
# 4 5 6
#------------
# 5 6 0 8 8
#------------

#prod_dict =  prod_dict =  {0: [8], 1: [6, 2], 2: [3, 1, 6], 3: [1, 9, 5], 4: [4]}
class Solution(object):
    def multiply(self, num1, num2):
        if num1 == '0' or num2 == '0':
            return '0'
        
        num1 = num1[::-1]
        num2 = num2[::-1]
        product_dict = {}
        #num1 456
        #num2 123
        
        for ind_1, n1 in enumerate(num1): #6 5 4
            dig_pos = ind_1 #0
            carry = 0
            for n2 in num2: #3 2 1
                prod_val = int(n1) * int(n2) #
                if dig_pos in product_dict:
                    product_dict[dig_pos].append(carry + prod_val % 10)
                else:
                    product_dict[dig_pos] = [carry + prod_val % 10]
                    
                carry = prod_val / 10
                dig_pos += 1
            
            if carry:
                dig_pos = dig_pos
                product_dict[dig_pos] = [carry]
            
        #print '\n prod_dict = ', product_dict
        
        my_l = []
        carry = 0
        for i in range(len(product_dict.keys())):
            #print 'c = ', carry
            #print product_dict[i]
            total_val = carry + sum(product_dict[i])
            #print 'tot = ', total_val
            sum_val = total_val % 10
            carry = total_val / 10
            my_l.append(str(sum_val))
        
        if carry:
            my_l.append(str(carry))
        #print my_l
        final_val = ''.join(my_l)[::-1]
        return final_val


Leetcode
49. Group anagrams

#leetcode sol - req more space less time
from collections import Counter
import json

class Solution:
    def groupAnagrams(self, strs):
        ans = collections.defaultdict(list)
        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord('a')] += 1
            ans[tuple(count)].append(s)
        return ans.values()
#my sol - req more time less space
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        
        anagram_dict = {}
        
        for word in strs:
            c = Counter(word)
            json_val = json.dumps(sorted(c.items()))
            if json_val in anagram_dict:
                anagram_dict[json_val].append(word)
            else:
                anagram_dict[json_val] = [word]
                
        #print anagram_dict
        return anagram_dict.values()
  

Leetcode
Best Time to Buy and Sell Stock II
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        stack = []
        max_profit = 0
        
        if not prices:
            return 0
        
        index = 0
        len_prices = len(prices)
        #your stack size should always be 1. It holds the value of the stock you 
        #are holding at the moment
        while(index < len_prices - 1):#[7,1,3,5,4,6,4]
            if not stack: #Stack will be empty if you sold the stock in your 
            #previous transaction or if this is your first iteration
                stack.append(prices[index])
                index += 1
            
            elif prices[index] > stack[-1]:#you can only consider selling the 
            #stock if the current price is greater than the one you already 
            #purchased and have in your stack
                if prices[index] > prices[index + 1]:#In the previous step, we 
                #only considered selling, We make a final decision in this step. 
                #If the price of the stock the next day is higher than the price 
                #today, you should sell it the next day because you cannot sell and
                #purchase on the same day. So, if you sell it today, you will make
                #less profit than if you had sold it the next day.
                    profit = prices[index] - stack.pop()
                    max_profit += profit
                index += 1
            else:
                stack = [prices[index]]#Notice carefully here, We are INITIALIZING
                #the stack because the calue of the stock is less today, than it
                #was yesterday. So, you are purchasing it today. Remember what we
                #said initially, we can ONLY HOLD ONE STOCK at a particular point
                #in time
                index += 1
                
        
        if stack and prices[-1] > stack[-1]:#If we had purchased a stock on the
        #day before the last day, we check if the price on the last day is greater
        #than the price yesterday and if so, we sell it and make profit
            max_profit += prices[-1] - stack.pop()
            
        #print max_profit
        return max_profit
            

Leetcode

Counting Elements
Solution
Given an integer array arr, count element x such that x + 1 is also in arr.

If there are duplicates in arr, count them seperately.

 

Example 1:

Input: arr = [1,2,3]
Output: 2
Explanation: 1 and 2 are counted cause 2 and 3 are in arr.
Example 2:

Input: arr = [1,1,3,3,5,5,7,7]
Output: 0
Explanation: No numbers are counted, cause there is no 2, 4, 6, or 8 in arr.
Example 3:

Input: arr = [1,3,2,3,5,0]
Output: 3
Explanation: 0, 1 and 2 are counted cause 1, 2 and 3 are in arr.
Example 4:

Input: arr = [1,1,2,2] or [1,1,2]
Output: 2
Explanation: Two 1s are counted cause 2 is in arr.
 

Constraints:

1 <= arr.length <= 1000
0 <= arr[i] <= 1000

class Solution(object):
    def countElements(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        nums_dict = {}
        for ele in arr:
            if ele in nums_dict:
                nums_dict[ele] += 1
            else:
                nums_dict[ele] = 1
                
        arr = list(set(arr))
        arr.sort()
        count = 0
        
        for index, ele in enumerate(arr[:-1]):
            if ele + 1 == arr[index + 1] :
                count += nums_dict[ele]
        
        return count


LC 51. n-queens problem
https://leetcode.com/problems/n-queens
all_combs = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1)]

def shrink_possible_rows(row, col, mat):
    directions_remaining = 4
    
    for comb in all_combs:
        row_copy = row
        col_copy = col
        while(True):
            new_row = row_copy + comb[0]
            new_col = col_cpy + comb[1]
            if is_valid(new_row) and is_valid(new_col):
                mat[new_row][new_col] = 'l'#locked
            else:
                break
            row_copy = new_row
            col_copy = new_col
    return mat

def place_queen(mat, row, col):
    mat[row][col] = 'q'
    mat = shrink_possible_rows(row,col)
    return mat

for col in range(n): #keep the first queen in all cells of first row and recursiely
    #find the position of second queen using callback
    row = 0
    mat = initialize_matrix()
    mat = place_queen(mat, row, col)
    callback(mat, row + 1)

def callback(mat, row):
    if row == n:# we have placed all the queens on the board
        result_list.append(mat)
        return

    queen_placed = False
    for col in range(n):
        if mat[row][col] != 'l':
            mat = place_queen(mat, row, col)
            queen_placed = True
            callback(mat, row + 1)

    if not queen_placed:
        return

LC 54. Spiral matrix
https://leetcode.com/problems/spiral-matrix/

Given a matrix of m x n elements (m rows, n columns), return all elements of the
matrix in spiral order.

Example 2:

Input:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]

[
[00,01,02,03],
[10,11,12,13],
[20,21,22,23],
]
op: [00,01,02,03,13,23,22,21,20,10,11,12]
all_combs = [(0,1),(1,0),(0,-1),(-1,0)]

num_of_visited_cells = 0
num_cells = len(mat) * len(mat[0])
start_cell = (0,0)

while(num_of_visited_cells < num_cells):
    for comb in all_combs:
        new_row = (cell + comb)[0]
        new_col = (cell + comb)[1]
        
        while(is_valid_row(new_row) and is_valid_col(new_col) and \
            mat[new_row][new_col] != 'v'):
            output_list.append(mat[new_row][new_col])
            mat[new_row][new_col] = 'v'
            cell = cell + comb
            num_of_visited_cells += 1



58. Length of Last Word
Return the length of the last word in the given sentence

59. Spiral Matrix II
https://leetcode.com/problems/spiral-matrix-ii/

#approach 1
n = 3
A = [[0] * n for _ in range(n)]
i, j, di, dj = 0, 0, 0, 1

for k in xrange(n*n):
    print '\nk = ', k
    print 'A = ', A
    A[i][j] = k + 1
    
    if A[(i + di) % n][(j + dj) % n]:
        di, dj = dj, -di
    
    i += di
    j += dj

#My approach
num_rows = 3
end_num = num_rows ** 2
curr_num = 1
directions = [(0,1), (1,0), (0,-1), (-1,0)]
direction_index = 0
row = 0
col = 0

def valid_cell(cell):
    if cell >= 0 and cell < num_rows:
        return True
    else:
        return False

def get_direction_index(direction_index):
    if direction_index == len(directions) - 1:
        direction_index = 0
    else:
        direction_index += 1

while(curr_num < num_rows):
    mat[row][col] = curr_num
    curr_direction = directions[direction_index]
    
    if valid_cell(row + curr_direction[0]) and valid_cell(col + curr_direction[1]):
        if mat[row + curr_direction[0]][col + curr_direction[1]] == False:
            row, col = row + curr_direction[0], col + curr_direction[1]
        else:
            direction_index = get_direction_index(direction_index)
    else:
        direction_index = get_direction_index(direction_index)


Time: O(n ** 2)
space: O(1)
60. permutations sequence
https://leetcode.com/problems/permutation-sequence/
The set [1,2,3,...,n] contains a total of n! unique permutations.

By listing and labeling all of the permutations in order, we get the following
sequence for n = 3

"123"
"132"
"213"
"231"
"312"
"321"
Given n and k, return the kth permutation sequence.

Note:

Given n will be between 1 and 9 inclusive.
Given k will be between 1 and n! inclusive.
Example 1:

Input: n = 3, k = 3
Output: "213"
Example 2:

Input: n = 4, k = 9
Output: "2314"

solution - refer the link below to understand fully: 
'''
https://leetcode.com/problems/permutation-sequence/discuss/22507/%22Explain-like-\
I'm-five%22-Java-Solution-in-O(n)
'''

Rotate List
https://leetcode.com/problems/rotate-list/

Given a linked list, rotate the list to the right by k places, where k is 
non-negative.

Example 1:

Input: 1->2->3->4->5->NULL, k = 2
Output: 4->5->1->2->3->NULL
Explanation:
rotate 1 steps to the right: 5->1->2->3->4->NULL
rotate 2 steps to the right: 4->5->1->2->3->NULL
Example 2:

Input: 0->1->2->NULL, k = 4
Output: 2->0->1->NULL
Explanation:
rotate 1 steps to the right: 2->0->1->NULL
rotate 2 steps to the right: 1->2->0->NULL
rotate 3 steps to the right: 0->1->2->NULL
rotate 4 steps to the right: 2->0->1->NULL
.next = None

#One thing to be careful about is if k is greater than 2*len(list). We have to get
#proper k, we have to k = k % list_len

class Solution(object):
    def rotateRight(self, head, k):
        len_list = 0
        node = head
        while(node):#We use this loop only to find len of list
            node = node.next
            len_list += 1

        if k > len_list:
            k = k % len_list

        curr_pos = 0
        queue = []
        node = head
        while(node):
            queue = [node.val] + queue
            if curr_pos >= k:
                val = queue.pop()
                node.val = val
            node = node.next
            curr_pos += 1
        
        #we would have moved the first k elements to their appt pos. We still have
        #to move the last few elem in the queue to the first of the queue
        #For example 1, at this point, the queue will be [4,5]
        node = head
        while(queue):
            val = queue.pop()
            node.val = val
            node = node.next

        return head


Unique path
https://leetcode.com/problems/unique-paths/

Example 1:

Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Right -> Down
2. Right -> Down -> Right
3. Down -> Right -> Right

Example 2:

Input: m = 7, n = 3
Output: 28

#BFS and each time we reach the end count path += 1
#but the above is not an efficient approach as you will visit the same cells more
#than once.

#Do recursion and convert it into dynamic programming ans like below
https://leetcode.com/problems/unique-paths/solution/ #O(n)


Unique Path II 
#Unique path + obstacles (obstacles marked as 1)
https://leetcode.com/problems/unique-paths-ii/

#same approach as previous problem. Just convert 1's in the matrix to 'B'


minimum-path-sum
https://leetcode.com/problems/minimum-path-sum/
prev_val = 0
for row in mat:
    #initializing first col += prev row's first col. Because there is only way to
    #reach this cell (i.e. from the top)
    row[0] += prev_val
    prev_val = row[0]

prev_val = 0
for col in row[0]:
    #similarly, initializing first row += prev col's first row.
    mat[0][col] = mat[0][col] + prev_val
    prev_val = mat[0][col]

for row_ind, row in enumerate(mat):
    for col_ind, col in row:
        if row_ind == 0 or col_ind == 0:
            continue
        else:
            mat[row_ind][col_ind] = mat[row_ind][col_ind] + \
            min(mat[row_ind - 1][col_ind], mat[row_ind][col_ind - 1])

return mat[n-1][n-1]



Valid number
https://leetcode.com/problems/valid-number/discuss/23728/A-simple-solution-in-
Python-based-on-DFA
#This is a bit vague question. We have used more like a pattern matching method
#to validate our input. If the "current index" of the input has a "value" belonging
#to a particular type (say digit, blank or e), what are the possibilities for the
#next digit.
#All possible types are "blank, sign, digit, ., e"
#In the end, we check if the end state is valid in our list "state". If the input
# ends in a state referenced by indices (3,5,8,9), then it's a valid number else not
class Solution(object):
  def isNumber(self, s):
      """
      :type s: str
      :rtype: bool
      """
      #define a DFA
      state = [{}, #index 0
              {'blank': 1, 'sign': 2, 'digit':3, '.':4}, #index 1
              {'digit':3, '.':4},#index 2
              {'digit':3, '.':5, 'e':6, 'blank':9},#index 3
              {'digit':5},#index 4
              {'digit':5, 'e':6, 'blank':9},#index 5
              {'sign':7, 'digit':8},#index 6
              {'digit':8},#index 7
              {'digit':8, 'blank':9},#index 8
              {'blank':9}]#index 9
      currentState = 1#Since the "currentState" is 1, a number can start with any
      #of the keys inside state[1]. 
      #Let's assume, the first number is a digit, then the currentState becomes 3(
      #according to the last line of the below for loop). so, the digit should be
      #one of the keys specified inside state[3] otherwise we return False
      for c in s:
          if c >= '0' and c <= '9':
              c = 'digit'
          if c == ' ':
              c = 'blank'
          if c in ['+', '-']:
              c = 'sign'
          if c not in state[currentState].keys():
              return False
          currentState = state[currentState][c]
      if currentState not in [3,5,8,9]:
          return False
      return True

Plus one
https://leetcode.com/problems/plus-one/
Question is add 1 to the given number
#One corned case is when the last digit is a 9, you will get a carry that you have 
#to progate further

add_binary
https://leetcode.com/problems/add-binary/
#simple binary addition

Test Justification
https://leetcode.com/problems/text-justification/
Given an array of words and a width maxWidth, format the text such that each line 
has exactly maxWidth characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack _as many words as_ 
you can in each line. Pad extra spaces ' ' when necessary so that each line has 
exactly maxWidth characters.

Extra spaces between words should be distributed as_ evenly as_ possible. If the 
number
of spaces on a line do not divide evenly between words, the empty slots on the left
will be assigned more spaces than the slots on the right.

For the last line of text, it should be left justified and no extra space is 
inserted between words.

Note:

A word is defined as_ a character sequence consisting of non-space characters only.
Each word s length is guaranteed to be greater than 0 and not exceed maxWidth.
The input array words contains at least one word.

words = ["This", "is", "an", "example", "of", "text", "justification."]
maxWidth = 16
Output:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]

output_dict = {}
line_num = 0
curr_line_words = []
curr_line_words_len = 0
max_width = 16
words = ["What","must","be","acknowledgment","shall","be"]

#the below function formats the given words with appt spaces
def format_line(curr_line_words, curr_line_words_len, max_width):
    op_line = ''
    req_spaces = max_width - curr_line_words_len
    no_of_words = len(curr_line_words)
    if no_of_words == 1:
        spaces = ' '.join([' ' for _ in range(req_spaces)])
        return curr_line_words[0] + spaces
    min_space_bet_words = req_spaces / (no_of_words - 1)
    extra_spaces = req_spaces % (no_of_words - 1)
    print 'extra_spaces = ', extra_spaces

    for word in curr_line_words:
        spaces = ''.join([' ' for _ in range(min_space_bet_words)])
        op_line += (word + spaces)
        if extra_spaces > 0:
            op_line += ' '
            extra_spaces -= 1

    return op_line

while(words):
    #print 'curr_line_words = ', curr_line_words
    curr_word = words[0]
    words = words[1:]
    if curr_line_words_len + len(curr_line_words) + len(curr_word) <= max_width:
        #len_of_words_in_curr_line + num of spaces + current_word len
        curr_line_words.append(curr_word)
        curr_line_words_len += len(curr_word)
    else:
        print 'line before format = ', curr_line_words
        curr_line = format_line(curr_line_words, curr_line_words_len, max_width)
        print 'after format = ', curr_line
        output_dict[line_num] = curr_line
        line_num += 1
        curr_line_words = [curr_word]
        curr_line_words_len = len(curr_word)

if curr_line_words:
    print 'line before format = ', curr_line_words
    curr_line = format_line(curr_line_words, curr_line_words_len, max_width)
    print 'after format = ', curr_line


sqrtx
https://leetcode.com/problems/sqrtx/

#I've worked out a modified version of the problem to get sqrt of 'x' upto 3 
#decimal places 

def callback(st, en, x):
    mid = float((st + en) / 2)#ALT: typecast to float
    print mid
    if mid * mid == x:
        return mid

    if len(str(mid).split('.')) > 1 and len(str(mid).split('.')[1]) >= 4:
        return math.ceil(mid)

    if mid * mid < x:
        return callback(mid, en, x)
    else:
        return callback(st, mid, x)

x = 16
callback(0, x, x)
x = 20
callback(0, x, x)



Simplify path
https://leetcode.com/problems/simplify-path/

Given an absolute path for a file (Unix-style), simplify it. Or in other words, 
convert it to the canonical path.

In a UNIX-style file system, a period . refers to the current directory. 
Furthermore, a double period .. moves the directory up a level.

Note that the returned canonical path must always begin with a slash /, and there 
must be only a single slash / between two directory names. The last directory name
(if it exists) must not end with a trailing /. Also, the canonical path must be the
shortest string representing the absolute path.


Input: "/home/"
Output: "/home"
Explanation: Note that there is no trailing slash after the last directory name.
Example 2:

Input: "/../"
Output: "/"
Explanation: Going one level up from the root directory is_ a no-op, as_ the root
level is the highest level you can go.
Example 3:

Input: "/home//foo/"
Output: "/home/foo"
Explanation: In the canonical path, multiple consecutive slashes are replaced by a
single one.
Example 4:

Input: "/a/./b/../../c/"
Output: "/c"
Example 5:

Input: "/a/../../b/../c//.//"
Output: "/c"
Example 6:

Input: "/a//b////c/d//././/.." 
Output: "/a/b/c"

#input_string = "/a//b////c/d//././/.." 
input_string = "/a/../../b/../c//.//"
#input_string = "/a/./b/../../c/"
cleaned_input = ''
index = 0
#In the following loop we are cleaning the input. eg: we remove '.' and '//' which
#does nothing
while(index < len(input_string)):
    char = input_string[index]
    if not cleaned_input:
        cleaned_input = char
        index += 1
        continue
    
    if char == '.' and input_string[index + 1] == '.':
        cleaned_input += '..'
        index += 2
        continue
    
    if char == '.' and input_string[index + 1] != '.':
        index += 1
        continue
    
    if cleaned_input[-1] == '/' and char == '/':
        index += 1
        continue
    
    cleaned_input += char
    index += 1 

print cleaned_input
#/a/b/../../c/
stack = []
index = 0
while(index < len(cleaned_input)):
    char = cleaned_input[index]
    if char == '/':
        index += 1
        continue
    if char == '.' and cleaned_input[index + 1] == '.':
        if stack:
            stack.pop()
        index += 2
        continue
    stack.append(char)
    index += 1


Edit Distance
https://leetcode.com/problems/edit-distance/discuss/159295/Python-solutions-and
-intuition

Given two words word1 and word2, find the minimum number of operations required to
 convert word1 to word2.

You have the following 3 operations permitted on a word:

Insert a character
Delete a character
Replace a character
Example 1:

Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with_ 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
Example 2:

Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with_ 'e')
enention -> exention (replace 'n' with_ 'x')
exention -> exection (replace 'n' with_ 'c')
exection -> execution (insert 'u')

https://leetcode.com/problems/edit-distance/discuss/159295/Python-solutions-and
-intuition

Look at the solution stated in the above link. This is a basic recursion problem
where there are 3 possibilities at each step
1 - Insert a char
2 - Replace a char
3 - delete a char

Once you decode the recursion part, the rest is just memoizing the results

def minDistance(self, word1, word2, i, j, memo):
    """Memoized solution"""
    if i == len(word1) and j == len(word2):
        return 0

    #We have reached end of word1. Now we need to remove extra chars from word2 to
    #transform it to word1
    if i == len(word1): 
        return len(word2) - j
    
    #We have reached end of word2. Now we need to insert extra chars to word2 to 
    #transform it to word1
    if j == len(word2):
        return len(word1) - i

    if (i, j) not in memo:
        if word1[i] == word2[j]:
            ans = self.minDistance2(word1, word2, i + 1, j + 1, memo)
        else: 
            insert = 1 + self.minDistance2(word1, word2, i, j + 1, memo)
            delete = 1 + self.minDistance2(word1, word2, i + 1, j, memo)
            replace = 1 + self.minDistance2(word1, word2, i + 1, j + 1, memo)
            ans = min(insert, delete, replace)
        memo[(i, j)] = ans
    return memo[(i, j)]


73 - set matrix zeroes
https://leetcode.com/problems/set-matrix-zeroes/
Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it
in-place.

Example 1:

Input: 
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]
Output: 
[
  [1,0,1],
  [0,0,0],
  [1,0,1]
]

Example 2:

Input: 
[
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]
Output: 
[
  [0,0,0,0],
  [0,4,5,0],
  [0,3,1,0]
]

mat = [
  [1,1,1],
  [1,0,1],
  [1,1,1]
]

def set_falase_row_col(row,col):
    global mat
    directions = [(0,1),(1,0),(0,-1),(-1,0)]
    for direction in directions:
        curr_row = row
        curr_col = col
        while(curr_row > -1 and curr_row < len(mat) and curr_col > -1 and \
            curr_col < len(mat[0])):
            if mat[curr_row][curr_col] != 0:
                mat[curr_row][curr_col] = 'False1'
            curr_row += direction[0]
            curr_col += direction[1]

row_len = len(mat)
col_len = len(mat[0])
for row in range(row_len):
    for col in range(col_len):
        if mat[row][col] == 0:
            set_falase_row_col(row,col)

for row in range(row_len):
    for col in range(col_len):
        if mat[row][col] == 'False1':
            mat[row][col] = 0
    print mat[row]


74- Search a 2D matrix
https://leetcode.com/problems/search-a-2d-matrix/solution/

Write an efficient algorithm that searches for a value in an m x n matrix. This 
matrix has the following properties:

Integers in each row are sorted from left to right.
The first integer of each row is greater than the last integer of the previous row.
Example 1:

Input:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 3
Output: true
Example 2:

Input:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 13
Output: false

mat = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
num_rows = len(mat)
num_cols = len(mat[0])

def binary_search(st, en, search_num):
    global mat
    global num_cols
    global num_rows
    if st > en:
        return False

    mid = (st + en) / 2
    print mid
    row_num = mid / num_cols
    col_num = mid % num_cols

    if mat[row_num][col_num] == search_num:
        return True

    if mat[row_num][col_num] < search_num:
        #NOTE it's mid + 1 and not just "mid"
        return binary_search(mid + 1, en, search_num)
    else:
        #NOTE it's mid - 1 and not just "mid"
        return binary_search(st, mid - 1, search_num)


st = 0
en = (num_rows * num_cols) - 1
search_num = 16

binary_search(st, en, search_num)


75 - Sort colors
https://leetcode.com/problems/sort-colors/
Given an array with n objects colored red, white or blue, sort them in-place so that
objects of the same color are adjacent, with the colors in the order red, white and
blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and
blue respectively.


Follow up:

A rather straight forward solution is a two-pass algorithm using counting sort.
First, iterate the array counting number of 0s, 1s, and 2s, then overwrite array 
with total number of 0s, then 1s and followed by 2s.

Algorithm

Initialise the rightmost boundary of zeros : p0 = 0. During the algorithm execution
nums[idx < p0] = 0.

Initialise the leftmost boundary of twos : p2 = n - 1. During the algorithm
execution nums[idx > p2] = 2.

Initialise the index of current element to consider : curr = 0.

While curr <= p2 :

If nums[curr] = 0 : swap currth and p0th elements and move both pointers to the
right.

If nums[curr] = 2 : swap currth and p2th elements. Move pointer p2 to the left.

If nums[curr] = 1 : move pointer curr to the right.
Could you come up with a one-_pass algorithm using only constant space?


76- Minimum window substring
https://leetcode.com/problems/minimum-window-substring/solution/

#See also version 2 which is same as version 1 but version 2 is written by me
#version 1
def minWindow(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: str
    """

    if not t or not s:
        return ""

    # Dictionary which keeps a count of all the unique characters in t.
    dict_t = Counter(t)

    # Number of unique characters in t, which need to be present in the desired window.
    required = len(dict_t)

    # left and right pointer
    l, r = 0, 0

    # formed is used to keep track of how many unique characters in t are present
    # in the current window in its desired frequency.
    # e.g. if t is "AABC" then the window must have two A's, one B and one C. 
    #Thus formed would be = 3 when all these conditions are met.
    formed = 0

    # Dictionary which keeps a count of all the unique characters in the current window.
    window_counts = {}

    # ans tuple of the form (window length, left, right)
    ans = float("inf"), None, None

    while r < len(s):

        # Add one character from the right to the window
        character = s[r]
        window_counts[character] = window_counts.get(character, 0) + 1

        # If the frequency of the current character added equals to the desired 
        #count in t then increment the formed count by 1.
        if character in dict_t and window_counts[character] == dict_t[character]:
            formed += 1

        # Try and contract the window till the point where it ceases to be 
        #'desirable'.
        while l <= r and formed == required:
            character = s[l]

            # Save the smallest window until now.
            if r - l + 1 < ans[0]:
                ans = (r - l + 1, l, r)

            # The character at the position pointed by the `left` pointer is no 
            #longer a part of the window.
            window_counts[character] -= 1
            if character in dict_t and window_counts[character] < dict_t[character]:
                formed -= 1

            # Move the left pointer ahead, this would help to look for a new window.
            l += 1    

        # Keep expanding the window once we are done contracting.
        r += 1    
    return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]


#version 2 - written by me
s = "ADOBECODEBANC"
t = "ABC"
num_unique_chars = 0
min_window = s
t_dict = {}
i = 0
j = 0

for char in t:
    if char in t_dict:
        t_dict[char] += 1
    else:
        num_unique_chars += 1
        t_dict[char] = 1


while j < len(s):
    char = s[j]

    if char in t_dict:
        t_dict[char] -= 1

        if t_dict[char] == 0:
            num_unique_chars -= 1

    if num_unique_chars == 0:
        
        while(num_unique_chars == 0):

            print '\nnum_unique_chars is 0', s[i:j+1]
            print 't_dict = ', t_dict
            if num_unique_chars == 0: 
                curr_window = s[i : j+1]
                
                if len(curr_window) < len(min_window):
                    min_window = curr_window
            
            char_going_out_window = s[i]
            print 'char_going_out_window = ', char_going_out_window
            
            if char_going_out_window in t_dict:
                t_dict[char_going_out_window] += 1

                if t_dict[char_going_out_window] > 0:
                    num_unique_chars += 1

            i += 1
    j += 1

Complexity Analysis

Time Complexity: 
#O(S) - we orst case we visit each elem twice
O(|S| + |T|) where |S| and |T| represent the lengths of
strings S and T. In the worst case we might end up visiting every element of 
string S twice, once by left pointer and once by right pointer. |T| represents the 
length of string T.

#O(T) - The dictionary which stores values in T
Space Complexity: O(|S| + |T|). |S|when the window size is equal to the entire 
string S. |T| when TT has all unique characters.


77. Combinations
https://leetcode.com/problems/combinations
Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.

Example:

Input: n = 4, k = 2
Output:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]

all_combs = []
def callback(nums, comb):
    global k
    print 'comb = ', comb
    if len(comb) == k:
        all_combs.append(comb)

    for ind, num in enumerate(nums):
        callback(nums[ind + 1: ], comb + [num])

n = 4
k = 2
nums = [str(i) for i in range(1, n + 1)]
callback(nums,[])
print all_combs

Time and Space = nCr = n! / ((n-r)! * r!)


78. Subsets
#see time complexity calculation
https://leetcode.com/problems/subsets/
https://www.youtube.com/watch?v=LdtQAYdYLcE
#tested the code below. It works fine.
Input: nums = [1,2,3]
Output:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]

#solving using bitmaps
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        output = []
        
        for i in range(2**n, 2**(n + 1)):
            # generate bitmask, from 0..00 to 1..11
            bitmask = bin(i)[3:]
            
            # append subset corresponding to that bitmask
            output.append([nums[j] for j in range(n) if bitmask[j] == '1'])
        
        return output

Time = O(n * 2^n) -> (2 ^ n) instead of 'n' because inner for loop is_ dependent on the
number of variables in the output at a given time. 2 ^ n because because for each of the 
input numbers, we have 2 choices, either pick or not pick. Imagine recursion tree
space = time

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        output = [[]]
        
        for num in nums:
            output += [curr + [num] for curr in output]
        
        return output

Time = O(n * 2^n) -> (2 ^ n)

class Solution(object):
    res = []
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        self.res = [[]]
        
        for set_length in range(1, len(nums) + 1):
            self.generate_subset(set_length, nums, [])
            
        return self.res
        
        
    def generate_subset(self, set_length, nums, curr_set):
        if set_length == 0:
            #print curr_set
            self.res.append(curr_set)
            return
        
        for i in range(len(nums)):
            self.generate_subset(set_length-1, nums[i+1:], curr_set + [nums[i]])


Time = O(n * 2^n)
space = time


79 - word search - This is a classic example of DFS. BFS is not the best approach here
#See time complexity calculation 
https://leetcode.com/problems/word-search/
Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent"
cells are those horizontally or vertically neighboring. The same letter cell may not be
used more than once.

board = [
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.

Do a DFS from each cell in_ all 4 directions.

Complexity Analysis

Time Complexity: O(n * ( 4 ^ l))
where n is the number of cells in the board and l is the length of the word to be 
matched.

For the backtracking function, its execution trace would be visualized _as a 4-ary tree,
each of the branches represent a potential exploration in the corresponding direction.
Therefore, in the worst case, the total number of invocation would be the number of nodes
in a full 4-nary tree, which is about 4 ^ l (where l can be thought of as_ height of the
recursion tree)

We iterate through the board for backtracking, i.e. there could be n times invocation 
for the backtracking function in the worst case.

Space Complexity: O(l) where l is the length of the word to be matched.

The main consumption of the memory lies in the recursion call of the backtracking 
function.
The maximum length of the call stack would be the length of the word.


80 Remove Duplicates from Sorted Array II
#Note the time complexity is O(n^2) and not O(n)
# Refer EPI 5.5 for o(n) solution
https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/

Given a sorted array nums, remove the duplicates in-place such that duplicates appeared 
at most twice and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input
array in-place with O(1) extra memory.

Example 1:

Given nums = [1,1,1,2,2,3],

Your function should return length = 5, with the first five elements of nums being 1, 1, 
2, 2 and 3 respectively.

It doesnt matter what you leave beyond the returned length.
Example 2:

Given nums = [0,0,1,1,1,1,2,3,3],

Your function should return length = 7, with the first seven elements of nums being
modified to 0, 0, 1, 1, 2, 3 and 3 respectively.

It doesnt matter what values are set beyond the returned length.

class Solution(object):
    def removeDuplicates(self, nums):
        # Initialize the counter and the array index.
        i, count = 1, 1
        # Start from the second element of the array and process
        # elements one by one.
        while i < len(nums):
            # If the current element is a duplicate, 
            # increment the count.
            if nums[i] == nums[i - 1]:
                count += 1
                # If the count is more than 2, this is an
                # unwanted duplicate element and hence we 
                # remove it from the array.
                if count > 2:
                    nums.pop(i)
                    # Note that we have to decrement the
                    # array index value to keep it consistent
                    # with the size of the array.
                    i-= 1
            else:
                # Reset the count since we encountered a different element
                # than the previous one
                count = 1
            # Move on to the next element in the array
            i += 1    
                
        return len(nums)

Time- O(n ^ 2) coz python list - pop() is O(1), but pop(index) is O(n) (since the whole 
rest of the list has to be shifted). #O(n) solution is available for this prob. refer solutions tab in leetcode
sapce - O(1)


81. Search in Rotated Sorted Array II
https://leetcode.com/problems/search-in-rotated-sorted-array-ii/
approach 1, minie below uses O(log n) for time and_ O(n) for_ space
approach 2 give in below link uses O(log n) time and O(1) space
https://leetcode.com/problems/search-in-rotated-sorted-array-ii/discuss/28218/
My-8ms-C%2B%2B-solution-(o(logn)-on-average-o(n)-worst-case)

Suppose an array sorted in ascending order is rotated at some pivot unknown to you 
beforehand.

(i.e., [0,0,1,2,2,5,6] might become [2,5,6,0,0,1,2]).

You are given a target value to search. If found in the array return true, otherwise 
return false.

Example 1:

Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true
Example 2:

Input: nums = [2,5,6,0,0,1,2], target = 3
Output: false

arr = [2,3,4,5,6,0,1,2]
arr = [7,8,100,0,1,2,2,3,3,3,3,4,4,5,6]

def find_pivot(st, en):
    print '\n\nst = ', st
    print 'en = ', en
    global arr
    if st >= en:
        return en #arr_len = odd, pivot will point to the greater num. eg:pos 2 in
        #arr = [7,8,100,0,1,2,2,3,3,3,3,4,4,5,6], if len is even we return pos 3

    mid = (st + en) / 2
    print 'mid = ', mid
    print 'st elem = ', arr[st]
    print 'mid elem = ', arr[mid]
    
    if arr[mid] == arr[st]:
        return st
    if arr[mid] < arr[st]:
        return find_pivot(st, mid - 1)
    else:
        return find_pivot(mid + 1, en)

pivot = find_pivot(0, len(arr) - 1)
if pivot > 0 and arr[pivot] < arr[pivot - 1]:
    pivot -= 1


if arr[pivot] > target:
    arr = arr[pivot + 1:] + [0 : pivot]
else:
    arr = arr[pivot:] + arr[0 : pivot - 1]

Now do binary search on arr to get your target

time = O(log n)
space = O(n)


82. Remove Duplicates from Sorted List II
https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/

#Input: 1->2->3->3->4->4->5
#Output: 1->2->5

#Input: 1->1->2
#Output: 1->2

#The below was my first approach. It was getting complicated when I tried to solve an
#edge case where head is duplicated and all the following nodes are also duplicates.
#Solution to handle this case is to create a dummy head like below
dummy = prev_node = ListNode(0)
dummy.next = head

#Input: 1->2->3->3->4->4->5
#Output: 1->2->5

#prev_node = head
#node = prev_node.next
node = dummy

while(node):
    print '\nprev node = ', prev_node.val
    print 'node = ', node.val
    if not node.next:
        break
    print 'next node = ', node.next.val
    if node.val == node.next.val:
        curr_node = node
        while(node):
            if node.val == curr_node.val:
                node = node.next
            else:
                break

        prev_node.next = node
        if not node:
            break
    else:
        prev_node = node
        node = node.next
        
return head

time: O(n)
space: O(1)


#approach 2 but this uses n space because we create a dictionary and keep track of dups
d = {}
node = head
while(node):
    if node.val in d:
        d[node.val] += 1
    else:
        d[node.val] = 1
    node = node.next

node = head
new_head = None
while(node):
    if d[node.val] > 1 and not new_head:
        new_head = node
        break
    node = node.next

head = new_head
if not head:
    return None
#Insert the while loop in approach 1 here

time: O(n)
space: O(n)

83: Remove Duplicates from Sorted List 
https://leetcode.com/problems/remove-duplicates-from-sorted-list/
#Easy version of above problem (82)

#Input: 1->2->3->3->4->4->5
#Output: 1->2->5

#Input: 1->1->1->2->3
#Output: 2->3


84. Largest Rectangle in Histogram
https://leetcode.com/problems/largest-rectangle-in-histogram/

Input: [2,1,5,6,2,3]
Output: 10

inp = [2,1,5,6,2,3]
#Brute force approach
max_water = float('-inf')
for ind, item in enumerate(inp):
    min_histo = item
    ind_2 = ind
    while(ind_2 < len(inp)):
        min_histo = min(inp[ind_2], min_histo)#2 1 1
        height = min_histo
        width = (ind_2 - ind + 1)#1 2 3
        contained_water =  height * width
        max_water = max(contained_water, max_water)
        ind_2 += 1

print 'max_water = ', max_water

time: O(n^2)
space: O(1)

#Stack approach time:O(n)
https://leetcode.com/problems/largest-rectangle-in-histogram/solution/
See the animation in the link to understand concept.

#Problem is very similar to trapping rain water. Only modification is we initialize stack
#with -1. And we also have a while loop at the end.
histo_bars = [3, 2, 1, 5, 6, 2, 3]
hb = histo_bars
#histo_ind = [0, 1, 2, 3, 4, 5, 6]
indices_stack = [-1] #We need this -1. If the min height bar appears somewhere in the mid
#of array.Here the min elem 1 occurs at index 2.So, in the last while loop, when we tryTo
#calc the max area formed when histo_bar at index 2 (i.e the one with height 1 is takenIn
#-to account, we will calc area like this h_b[2] * len(histo_bars) - stack[-1] = 1*7 = 7)
index = 0
max_area = float('-inf')

while(index < len(hb)):
    if len(indices_stack) > 1 and hb[indices_stack[-1]] > hb[index]:
        while(len(indices_stack) > 1 and hb[indices_stack[-1]] > hb[index]):
            top_ind = indices_stack.pop()
            bar_ht = hb[top_ind]
            width = index - indices_stack[-1] - 1
            rect_area = bar_ht * width
            max_area = max(max_area, rect_area)

    indices_stack.append(index)
    index += 1

while(len(indices_stack) > 1):
    top_ind = indices_stack.pop()
    bar_ht = hb[top_ind]
    #We know all the bars on the right hand side of the current bar are higher than its 
    #ht
    width = len(hb) - indices_stack[-1] - 1 
    rect_area = bar_ht * width
    max_area = max(max_area, rect_area)


1 intersting different thing about the stack we have used is. We store not only the elems
in the stack but also their position in the array


85. Maximal Rectangle
https://leetcode.com/problems/maximal-rectangle/
see approach 2 and 3 in the below link
time of approach 2 is O(n^2 m) try to understand why this running time
time of approach 2 is O(nm) try to understand why this running time
https://leetcode.com/problems/maximal-rectangle/solution/
So, basically 
i- we compute the height of histogram in each cell.
ii - At the end we use the max area covered by histogram using the algorithm we found in
     problem no. 84


86. Partition List
https://leetcode.com/problems/partition-list/

Example:

Input: head = 1->4->3->2->5->2, x = 3
Output: 1->2->2->4->3->5

#approach 1
Have 2 pointers i and j. i will point at a point in the list. This point marks the first 
elem that is greater than x. Actually i will point at 1 pos before that elem stated in 
the prev line.
so, i will point at index 0 in our example
j will move 1 step fwd in each iteration of the list.

if the elem pointed by j is smaller than x, we create a newNode after i with_ the elem
pointed by j and and move i to the newly created node.


if head.val > x:
    dummy_head = NewNode('dummyhead')
    dummyhead.next = head
    i = dummy_head
    j = dummy_head
else:
    dummy_head = None
    i = head
    j = head

while(j): #time: O(n), space: O(n)-because we are creating duplicate nodes
    if j.val < x:
        new_node = NewNode(j.val)
        new_node.next = i.next
        i.next = new_node
        i = new_node

    j = j.next

#approach 2
#just replace the while loop above with_ the following while_
while(j.next):#time: O(n) and space O(1)
    if j.next.val >= x:
        j = j.next
    else:
        new_node = NewNode(j.next.val)
        new_node.next = i.next
        i.next = new_node
        i = new_node
        j.next = j.next.next

#approach 3
create 2 new list less_than_list and greater_than_equal_list.
iterate through inp list by a pointer i and add a new elem to less_than_list if the val
is less than x. If val >= x, add a new node to the greater_than_equal_list.
Finally delete the node in the inp list we just visited.
Point the  next of last elem in less_than_list to head of greater_than_equal_list
time: O(n)
space: O(1)


87 Scramble String
https://leetcode.com/problems/scramble-string/

#my algorithm - similar to merge sort
def split_string(s):#this function will all possible scrambles for the given s.
    if len(s) == 2:
        permuted_str = [[s[0], s[1]], [s[1], s[0]]]
        return permuted_str
    if len(s) == 1:
        return s
    
    mid = len(s) / 2
    portion_1 = split_string(s[0:mid])
    portion_2 = split_string(s[mid:])
    return permutation(portion_1, portion_2)

eg:
#                              [g,r,e,a,t]
#                              /         \
#                             [g,r]     [e,a,t]
#                              |         /  \
#                           [gr,rg]    [e]   [a,t]
#                              |        |     |
#                              |        |     [at, ta]
#                              |        |     /
#                              |        \    /
#                              |         \  /
#                              |        [eat,eta,ate,tae]
#                               \      /
#                                [great, greta, grate, grtae,
#                                 rgeat, rgeta, rg,ate,rgtae]   


#Not able to clearly understand the solution discussed in forums. This is a better sol
class Solution:
# @return a boolean
def isScramble(self, s1, s2):
    n, m = len(s1), len(s2)
    if n != m or sorted(s1) != sorted(s2):
        return False
    if n < 4 or s1 == s2:
        return True
    f = self.isScramble
    for i in range(1, n):
        if f(s1[:i], s2[:i]) and f(s1[i:], s2[i:]) or \
           f(s1[:i], s2[-i:]) and f(s1[i:], s2[:-i]):
            return True
    return False


88. Merge sorted array Easy problem
https://leetcode.com/problems/merge-sorted-array/

89. Grey code
https://leetcode.com/problems/gray-code/

Input: 2
Output: [0,1,3,2]
Explanation:
00 - 0
01 - 1
11 - 3
10 - 2

n = 3
output_list = []
memo = {}
start_number = '0' * n

def callback(formed_string):
    global n, memo, output_list

    for ind, val in enumerate(formed_string):
        new_val = '1' if val == '0' else '0'
        new_string = formed_string[0:ind] + new_val + formed_string[ind+1:]

        if new_string in memo:
            continue

        memo[new_string] = True
        output_list.append(new_string)
        callback(new_string)

    return

memo[start_number] = True
output_list.append(start_number)
callback(start_number)
print output_list


time: (2 ^ n) * n, which is equal to 2^n
#Recursion tree will look like this
000 -> 100 -> 110 -> 010 -> 011 -> 111 -> 101 -> 001. We are making 2 ^ n number of rec
calls and for loop iterates n number of times within it(If we have to upper bound algo)
space: O(2 ^ n)

#WRONG- RECURSION TREE WONT LOOK LIKE THIS. IT WILL FIRST GO DEEP
#                           000   
#                         /  |  \  
#                      100  010  001
#                    /  |    |
#                 110  101   011
#                  |
#                 111
#

#the following is iterative java solution. this is also a good sol
https://leetcode.com/problems/gray-code/discuss/29891/Share-my-solution
#the following approach is not relevent to the question. Just did it for practice
n = 3
inp_variants = ['0', '1']
output_list = []
#We need to form string of len n
def callback(formed_string):
    global output_list, inp_variants, n
    if len(formed_string) == n:
        output_list.append(formed_string)
        return

    for i in range(2):
        callback(formed_string + inp_variants[i])

callback('')

90. subsets-ii
https://leetcode.com/problems/subsets-ii/

Input: [1,2,2]
Output:
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]
inp = ['1','2','2']
inp = ['1','2','3']
op_dict = {}

def callback(formed_string, choices, desired_len):
    print '\nformed_string ', formed_string
    print 'desired_len = ', desired_len
    print 'choices = ', choices
    if len(formed_string) == desired_len:
        op_dict[formed_string] = True
        return

    for ind, ele in enumerate(choices):
        new_choices = choices[ind + 1:]
        if len(formed_string + ele) + len(new_choices) < desired_len:
            break
        callback(formed_string + ele, new_choices, desired_len)

for i in range(1, len(inp)+1):
    callback('', inp, i)

Time = 2 ^ n
space = size of output list which is O(2 ^ n)
#approach 2 - Iterative approach 
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        output = {'':True}
        
        for num in nums:
            for curr in output.keys():
                output[curr + [num]] = True
        
        return output.keys()

Time = O(n * 2^n) -> (2 ^ n)
space = size of output list which is O(2 ^ n)


91. Decode Ways - Google Question - Daily coding problem
Refer 91_lc.jpg
https://leetcode.com/problems/decode-ways/

A message containing letters from A-Z is being encoded to numbers using the following 
mapping:

'A' -> 1
'B' -> 2
...
'Z' -> 26

Example 2:

Input: "226"
Output: 3
Explanation: It could be decoded as_ "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).


#approach 2:
# There are only 2 possibilities at each index.
# 1 - Either take that number at that index alone (and count the num of poss for elements from index + 1) or
# 2 - Take that number and the number at index + 1 (and count the num of poss for elements from index + 2)
# And the number of poss available from that index is the sum of 1 and 2

string = '226'
string = '12'
string = '12211'
memo = {}

def callback(index):
    global memo
    if index >= len(string) - 1:
        return 1

    if index in memo:
        return memo[index]

    count_only_number = callback(index + 1)
    count_number_and_next = callback(index + 2) if index < len(string) - 1 and \
    int(string[index: index + 2]) < 27 else 0

    total_possibilities = count_only_number + count_number_and_next
    memo[index] = total_possibilities

    return total_possibilities

callback(0)

Recursion tree:
#                     ind_0 (1)
#                   /         \
#                 ind_1(2)     ind_2(m)
#                  /      \
#               ind_2(3)   ind_3(m)
#                 /     \
#              ind_3(4)  ind_4(m)
#               /
#             ind_4(5)

#approach 3 - this is exactly O(n). approach 2 cannot be called O(n) because of string slicing operation done in the
#line count_number_and_next
#https://www.youtube.com/watch?v=qli-JCrSwuk
#Questions to ask interviewer
#1) Will there be 0's in the input? What should I do if I encounter 0's in the input
#2) Will there be values other than numbers in the input? What should I do if I encounter non numerals in input
main_data = '1111111'
memo = {}

def num_ways(data):
    global memo

    if not data:
        return 1

    elif len(data) == 1:
        return 1

    if data in memo:
        print ('return from memo ', data)
        return memo[data]

    else:
        if int(data[0:2]) <= 26:
            num_poss_ways = num_ways(data[1:]) + num_ways(data[2:])
        else:
            num_poss_ways = num_ways(data[1:])

        memo[data] = num_poss_ways

    return memo[data]

print (num_ways(main_data))
print (memo)

#Time O(n)
#space O(n)

92. Reverse linked list
Reverse a linked list from position m to n. Do it in_ one-pass.

Note: 1 ≤ m ≤ n ≤ length of list.

Example:

Input: 1->2->3->4->5->NULL, m = 2, n = 4
Output: 1->4->3->2->5->NULL

node = head
counter = 1

if m == 1: #edge case
    dummy_head = Node('dummy')
    dummy_head.next = head

while(node):
    next_node = node.next

    if counter + 1 == m:
        node_before_m = node

    if counter == m:
        m_th_node = node
        prev_node = node

    if counter > m and counter < n:
        node.next = prev_node
        prev_node = node

    if counter == n:
        node_before_m.next = node
        m_th_node.next = next_node
        node.next = prev_node

    node = next_node
    counter += 1

time: O(n)
space: O(1)


93. Restore IP addresses
https://leetcode.com/problems/restore-ip-addresses/
#Refer dia 93_lc to see recursion tree.
#memoizing is not of use here (because, as far as I can see you will not reevaluate the
#same sub problem more than once according to the recursion tree dia)

Given a string containing only digits, restore it by returning all possible valid IP 
address combinations.

A valid IP address consists of exactly four integers (each integer is between 0 and 255) 
separated by single points.

Example:

Input: "25525511135"
Output: ["255.255.11.135", "255.255.111.35"]

#Need to see how to memoize the results for better runtime
inp = '25525511135'
output_list = []

def callback(index, formed_string):
    global output_list
    global memo
    
    if formed_string.split('.') > 5:#this is an optimization step
        return False

    if index >= len(inp):
        #do something
        if len(formed_string.split('.')) == 5:
            print 'formed_string = ', formed_string
            output_list.append(formed_string)

        return 
    
    for ind in range(1,4): #loop will iterate at most 3 because inp[ind
    #:new_index] should be less than 255. So we can consider this as a constant when calc
    #time comple
        new_index = index + ind
        if int(inp[index:new_index]) > 255:
            break
        else:
            callback(new_index, formed_string + inp[index:new_index] + '.')

callback(0, '')

#calculating time and space is a bit complex but following are the worst case according
#one of the comments under solution tab. The solution tab says this is a constant time
#algo. check out that as well.
Time O(3^n) worst
Space O(3^n) worst


94. Binary tree inorder traversal
https://leetcode.com/problems/binary-tree-inorder-traversal/

def callback(node):
    if not node:
        return

    callback(node.left)
    op_list.append(node.val)
    callback(node.right)

time: O(n), n is the size of the binary tree. In the worst case (left sided binary tree)
we will have to make n recursive calls
space: O(n), we need to store all the values of nodes of the tree in op_list


95. Unique Binary Search Trees II
https://leetcode.com/problems/unique-binary-search-trees-ii/



#Almost got it correct. But Not perfectly correct. See the approach 2 for correct ans
#Approach 1 
roots_all_poss_trees = []

def constuct_tree_with_root_i(node_i, left_poss, right_poss):
    if not left_poss and not right_poss:
        return

    for ind, left_ele in enumerate(left_poss):
        new_lnode = Node(left_ele)
        node_i.left = new_lnode
        constuct_tree_with_root_i(new_lnode, left_poss[0:ind], left_poss[ind + 2:])

        for ind_2, right_ele in enumerate(right_poss):
            new_rnode = Node(right_ele)
            node_i.right = new_rnode
            constuct_tree_with_root_i(new_rnode, right_poss[0:ind], right_poss[ind + 2:])

    roots_all_poss_trees.append(node_i)


all_roots = range(n)
for i, ele in enumerate(all_roots):
    root = node(i)
    constuct_tree_with_root_i(root, all_roots[:left_poss], all_roots[i+2:])


#Approach 2: solutions tab of the problem
class Solution:
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        def generate_trees(start, end):
            if start > end:
                return [None,]
            
            all_trees = []
            for i in range(start, end + 1):  # pick up a root
                # all possible left subtrees if i is choosen to be a root
                left_trees = generate_trees(start, i - 1)
                
                # all possible right subtrees if i is choosen to be a root
                right_trees = generate_trees(i + 1, end)
                
                # connect left and right subtrees to the root i
                for l in left_trees:
                    for r in right_trees:
                        current_tree = TreeNode(i)
                        current_tree.left = l
                        current_tree.right = r
                        all_trees.append(current_tree)
                    print all_trees
                    print '\n'
            print '-------------------\n'
            
            return all_trees #NOTE THE RETURN HERE **** VERY IMPORTANT *****
        
        return generate_trees(1, n) if n else []

See the execution order to clearly understand the recursion:

[TreeNode{val: 3, left: None, right: None}]


-------------------

[TreeNode{val: 2, left: None, right: TreeNode{val: 3, left: None, right: None}}]


[TreeNode{val: 2, left: None, right: None}]


-------------------

[TreeNode{val: 2, left: None, right: TreeNode{val: 3, left: None, right: None}}, 
TreeNode{val: 3, left: TreeNode{val: 2, left: None, right: None}, right: None}]
.
.
.
.


class Solution(object):
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        if n == 0:
            return [[]]
        return self.dfs(1, n+1)
        
    def dfs(self, start, end):
        if start == end:
            return None
        result = []
        for i in xrange(start, end):
            for l in self.dfs(start, i) or [None]:
                for r in self.dfs(i+1, end) or [None]:
                    node = TreeNode(i)
                    node.left, node.right  = l, r
                    result.append(node)
        return result

 
96. Unique Binary Search Trees
https://leetcode.com/problems/unique-binary-search-trees/

#the following is not the correct algorithm. See solutions tab of the prob to
#understand
def cb(left, right):
    if not left and not right:
        return 0
    
    for index, item in enumerate(left):
        left = 1 + cb(left[0:index], left[index + 1: ])

    for index, item in enumerate(right):
        right = 1 + cb(right[0:index], right[index + 1: ])

    return left * right
for i in range(len(n)):
    cb(n[0:i], n[i+1:n])


97. Interleaving String
https://leetcode.com/problems/interleaving-string/

Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.

Example 1:

Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
Output: true
Example 2:

Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
Output: false

s1 = "aabcc"
s2 = "dbbca"
s3 = "aadbbcbcac"
memo = {}

def callback(ind_1, ind_2):
    if ind_1 + ind_2 > (len(s3) - 2):
        return True
    
    global memo
    dict_key = str(ind_1) + str(ind_2)
    is_mix = False
    s3_ind = ind_1 + ind_2

    if dict_key in memo:
        return memo[dict_key]
    
    if ind_1 < len(s1) and s1[ind_1] == s3[s3_ind]:
        is_mix = callback(ind_1 + 1, ind_2)

    if ind_1 < len(s2) and not is_mix and s2[ind_2] == s3[s3_ind]:
        is_mix = callback(ind_1, ind_2 + 1)

    memo[dict_key] = is_mix

    return is_mix

callback(0,0)


98. Validate Binary Search Tree
https://leetcode.com/problems/validate-binary-search-tree/

min_val = float('-inf')
max_val = float('+inf')

def callback(min_val, max_val, node):
    if not node:
        return True

    if node.val < min_val and node.val > max_val:
        return callback(min_val, node.val, node.l) and \
        callback(node.val, max_val, node.r)
    else:
        return False

callback(min_val, max_val, root)

99. Recover Binary Search Tree
https://leetcode.com/problems/recover-binary-search-tree/

min_val = float('-inf')
max_val = float('+inf')
misplaced_node = None

#The following approach is wrong. If the root val and the right most val in the right 
#subtree are swapped, the following algo won't work. There can be many other cases
#this might not work as well.
def callback(min_val, max_val, node):
    if not node:
        return True

    if node.val < min_val and node.val > max_val:
        l_ret_node = callback(min_val, node.val, node.l)
        
        if type(l_ret_node) == TreeNode():
            if node.val > l_ret_node.val:
                node.val, l_ret_node.val = l_ret_node.val, node.val
        
        r_ret_node = callback(node.val, max_val, node.r)
        
        if type(r_ret_node) == TreeNode():
            if node.val < r_ret_node.val:
                node.val, r_ret_node.val = r_ret_node.val, node.val
    
    else:
        return node

callback(min_val, max_val, root)


#algorithm
- To identify swapped nodes, track the last node pred in the inorder traversal 
(i.e. the predecessor of the current node) and compare it with current node value. 
If the current node value is smaller than its predecessor pred value, the swapped node is
here.

- There are only two swapped nodes here, and hence one could break after having the 
second node identified.


See the solutions tab for answer. space O(h) solution is_not diff. But O(1) using morris
algo is not easy. 
see this link to understand morris algo
- https://www.cnblogs.com/AnnieKim/archive/2013/06/15/morristraversal.html
- https://leetcode.com/problems/recover-binary-search-tree/discuss/32559/
Detail-Explain-about-How-Morris-Traversal-Finds-two-Incorrect-Pointer


100. Same tree
https://leetcode.com/problems/same-tree/

def check_same_tree(node_1, node_2):
    if not node_1 and not node_2:
        return True

    if node_1 and not node_2:
        return False

    if node_2 and not node_1:
        return False

    if node_1.val != node_2.val:
        return False

    check_same_tree(node_1.left, node_2.left)
    check_same_tree(node_1.right, node_2.right)

127. Word Ladder
https://leetcode.com/problems/word-ladder/

from collections import defaultdict, deque


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        words_dict = defaultdict(list)
        end_word_found = False
        total_transformations = 0
        visited_set = {beginWord}

        for word in wordList:
            for ind, char in enumerate(word):
                formed_word = word[:ind] + '*' + word[ind + 1:]
                words_dict[formed_word].append(word)

        queue = deque()
        queue.append((beginWord, 1))
        # print(words_dict)

        while (queue and end_word_found == False):  # O(W)
            word_in_queue, num_transformations = queue.popleft()
            # print(word_in_queue, num_transformations)

            for ind, char in enumerate(word_in_queue):  # O(n)
                formed_word = word_in_queue[:ind] + '*' + word_in_queue[ind + 1:]  # O(n)

                if formed_word in words_dict:
                    for close_enough_word in words_dict[formed_word]:  # O(W)
                        if close_enough_word == endWord:
                            end_word_found = True
                            total_transformations = num_transformations + 1
                        else:
                            if close_enough_word in visited_set:
                                continue
                            visited_set.add(close_enough_word)
                            queue.append((close_enough_word, num_transformations + 1))

        return total_transformations

refer https://leetcode.com/discuss/interview-question/736717/Google-or-Phone-or-Start-to-End-with-Safe-states

# time: O(W^2 * N^2) # W is the num of words and N is the length of each word
# space: O(W * N) # For each word we will have N different combinations and each combination is a key in the words_dict

152. Max Product subarray
https://leetcode.com/problems/maximum-product-subarray/
from _collections import deque

inp = [2, 3, 2, -2, -4, -5, 103]
max_prod = float('-inf')
"""
#O(n^2) solution
for i, elem_1 in enumerate(inp):
    current_prod = elem_1
    max_prod = max(max_prod, current_prod)

    for j, elem_2 in enumerate(inp[i+1:]):
        current_prod = current_prod * elem_2
        max_prod = max(max_prod, current_prod)
"""
no_of_minus_to_right = 0
i = len(inp) - 1
no_of_minus_to_right_list = deque()
prod = 1

while(i >= 0):
    no_of_minus_to_right_list.appendleft(no_of_minus_to_right)

    if inp[i] < 0:
        no_of_minus_to_right += 1

    i -= 1

print(no_of_minus_to_right_list)
i = 0

while(i < len(inp)):

    if inp[i] < 0:
        if no_of_minus_to_right_list[i] > 0:
            prod *= inp[i]
        else:
            prod = 1
            i += 1
            continue
    else:
        prod *= inp[i]

    max_prod = max(prod, max_prod)
    i += 1

print(max_prod)


224. Basic Calculator
https://leetcode.com/problems/basic-calculator/
Advaced version of the above problem in geeksforgeeks
https://www.geeksforgeeks.org/expression-evaluation/

238. Product of the array except self
https://leetcode.com/problems/product-of-array-except-self/

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # if len(nums) == 1:
        #    return nums

        # [1,2,3,4]
        aux_arr = [1] * (len(nums))  # [1,1,1,1]
        # [1, 1, 2, 6]

        for i in range(1, len(nums)):  # 1 2 3
            aux_arr[i] = aux_arr[i - 1] * nums[i - 1]

        aux_arr_rev = [1] * (len(nums))  # [1,1,1,1]
        # [24,12,4,1]
        for i in range(len(nums) - 2, -1, -1):  # 3
            # print (i)
            aux_arr_rev[i] = aux_arr_rev[i + 1] * nums[i + 1]

        # print (aux_arr)
        # print(aux_arr_rev)

        for i in range(len(aux_arr)):
            aux_arr[i] *= aux_arr_rev[i]

        return aux_arr


239. Sliding window max
https://leetcode.com/problems/sliding-window-maximum
https://www.geeksforgeeks.org/sliding-window-maximum-maximum-of-all-subarrays-of-
size-k/

stack based problem: 
You cannot place a larger elem on top of a smaller element.
Looks like its O(nk) but its not because of the above point. We REMOVE the smaller elems
in the left of the current elem in each iteration.

Follow the sol given in geeksforgeeks
A double ended queue is just a data structure where you can push and pop from both
ends in O(1). You can implement the below approach using python lists as_ well
because removing ele at an index and appending elem to a list are O(1)

#if unable to follow the login, execute the program and see. print statements are more
#intutive
#expected_op = [3,3,5,5,6,7] 
from collections import deque

nums = [1,3,-1,-3,5,3,6,7] 
k = 3
deque_index_stack = deque()
max_sliding_window = []
curr_ind = 0


while(curr_ind < len(nums)):
    print '\n\ncurr_ind = ', curr_ind
    print 'ele curr_ind = ', nums[curr_ind]
    print 'deque_index_stack = ', deque_index_stack
    if deque_index_stack and curr_ind - deque_index_stack[0] == k:
        print 'popping left'
        deque_index_stack.popleft()

    while(deque_index_stack and nums[curr_ind] > nums[deque_index_stack[-1]]):
        print 'popping right'
        deque_index_stack.pop()
    
    deque_index_stack.append(curr_ind)
    index_of_max_elem_in_interval = deque_index_stack[0]

    if curr_ind >= k - 1: # We need to start appending eles to output only when curr_ind
        #reached the interval size (k)
        max_sliding_window.append(nums[index_of_max_elem_in_interval])

    curr_ind += 1

298. Binary Tree Longest Consequtive sequence (google - Kevin youtube)
https://leetcode.com/problems/binary-tree-longest-consecutive-sequence/

#Our solution finds the longest consecutive length and the node from which that len can
#attained
class Node(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

root = Node(1)
three_node = Node(3)
four_node = Node(4)
five_node = Node(5)
six_node = Node(6)
two_node = Node(2)
root.right = three_node
three_node.left = two_node
three_node.right = four_node
four_node.right = five_node
five_node.left = six_node

len_of_longest_seq = float('-inf')
root_of_longest_seq = None

def dfs(node, prev_node_val):
    global len_of_longest_seq
    global  root_of_longest_seq

    if not node:
        return 0

    len_left_seq = dfs(node.left, node.val)
    len_right_seq = dfs(node.right, node.val)
    curr_max_len = max(len_left_seq, len_right_seq) + 1 # + 1 because we need to include 
    #curr node in the sequence

    if prev_node_val and prev_node_val + 1 == node.val:
        return curr_max_len

    else:
        if curr_max_len > len_of_longest_seq:
            len_of_longest_seq = curr_max_len
            root_of_longest_seq = node

        return 0

dfs(root, None)
print (len_of_longest_seq)
print (root_of_longest_seq.val)

Time: O(n)
space: O(n) worst case because the it could be left  or right sided tree



735. Asteroid collision
https://leetcode.com/problems/asteroid-collision/

my_l = [-2, -1, 1, 2]
output_list = []
index = 0
stack = []

while(index < len(my_l)):
    item = my_l[index]

    if item < 0:
        output_list.append(item)
    else:
        break

    index += 1

while(index < len(my_l)):
    print ('\n')
    print (index)
    print (stack)
    item = my_l[index]

    if item > 0:
        stack.append(item)
    else:
        while(stack and stack[-1] < abs(item)):
            stack.pop()

        if stack and stack[-1] == abs(item):
            stack.pop()
        elif not stack:
            output_list.append(item)

    index += 1

output_list.extend(stack)
print (output_list)

Time: O(n) - Since we are popping elem from stack in_ second while_ loop, its O(n)
Space: O(n) - Again its possible that our stack can grow upto size of input


904. Fruits in basket
https://leetcode.com/problems/fruit-into-baskets/solution/


[3,3,1,3,1,2,1,1,1,2,3,3,4]

#approach 1 
#sliding window solution (We are finding the longest window with the help of 2ptrs)
#the sliding window is an irritating sol. Have this as the last resort
def totalFruit(self, tree):
    count, i = {}, 0
    for j, v in enumerate(tree):
        count[v] = count.get(v, 0) + 1
        if len(count) > 2:
            count[tree[i]] -= 1
            if count[tree[i]] == 0: del count[tree[i]] #making sure that the above
            # if is executed only when index of 'i' is less than the point (where
            # only 2 keys can be present)
            i += 1 #incrementing i ONLY when there are more than 2 keys in count{}
    
    return j - i + 1 #Now fast pointer j value - slow pointer i (i gets increment
    # -ed only during times when you have more than 2 keys in dict). 'i' can never
    # equal or be greater than j at any cost because 
    # 1) j increments during each iteration
    # 2) i increments only when a particular cond (more than 2 keys in count)
    # is satisfied in each iteration
    # ans = Total no of iterations - total iterations when a particular condition-
    # - (more than 2 keys in count)is satisfied 

#approach 2
trees = [3,3,3,1,2,1,1,2,3,3,4]
k = 2
i = 0
j = 0
basket_dict = {}
num_fruits = 0
unique_fruits = 0
max_fruits = 0

while(j < len(trees)):
    print '\n j = ', j
    fruit = trees[j]
    num_fruits += 1

    if fruit in basket_dict:
        basket_dict[fruit] += 1
    else:
        basket_dict[fruit] = 1
        unique_fruits += 1

        print 'unique_fruits = ', unique_fruits

        if unique_fruits > k:
            while(unique_fruits > k):
                fruit_going_out_of_window = trees[i]
                num_fruits -= 1
                basket_dict[fruit_going_out_of_window] -= 1

                if basket_dict[fruit_going_out_of_window] == 0:
                    basket_dict.pop(fruit_going_out_of_window)
                    unique_fruits -= 1

                i += 1

    max_fruits = max(max_fruits, num_fruits)
    j += 1

#approach 3
Iterator solution:
What are iterators?
eg:
import itertools
tree = [1,0,1,1,4,1,4,1,2,3]
for ele, v in itertools.groupby(tree)
...     print 'ele = ', ele
...     print 'v = ', list(v)
... 
ele =  1
v =  [1]
ele =  0
v =  [0]
ele =  1
v =  [1, 1]
ele =  4
v =  [4]
ele =  1
v =  [1]
ele =  4
v =  [4]
ele =  1
v =  [1]
ele =  2
v =  [2]
ele =  3
v =  [3]
>>> blocks = [(ele, len(list(v)))
            for ele, v in itertools.groupby(tree)]
>>> blocks
[(1, 1), (0, 1), (1, 2), (4, 1), (1, 1), (4, 1), (1, 1), (2, 1), (3, 1)]

class Solution(object):
    def totalFruit(self, tree):
        blocks = [(k, len(list(v)))
                  for k, v in itertools.groupby(tree)]

        ans = i = 0
        while i < len(blocks):
            # We'll start our scan at block[i].
            # types : the different values of tree[i] seen
            # weight : the total number of trees represented
            #          by blocks under consideration
            types, weight = set(), 0

            # For each block from i and going forward,
            for j in xrange(i, len(blocks)):
                # Add each block to consideration
                types.add(blocks[j][0])
                weight += blocks[j][1]

                # If we have 3 types, this is not a legal subarray
                if len(types) >= 3:
                    i = j-1
                    break

                ans = max(ans, weight)

            # If we go to the last block, then stop
            else:
                break

        return ans


1055. Shortest Way to Form String
https://leetcode.com/problems/shortest-way-to-form-string/

#one mistake I made when I tried to solve the problem second time was I took a more compl
#-ex approach to solve it by using BFS. We should know 1 thing. Just because its google
#doesn't mean, we will always be asked bfs or dfs or other complicated questions.
class Solution(object):
    def shortestWay(self, source, target):
        """
        :type source: str
        :type target: str
        :rtype: int
        """
        source_dict = {}
        dup_string = source
        no_of_subsequences = 0
        
        for index, char in enumerate(source):
            if char in source_dict:
                source_dict[char].append(index)
            else:
                source_dict[char] = [index]
                
        for index, char in enumerate(target):
            if not char in source_dict:
                return -1
        
        while(target):
            char = target[0]
            if not char in source_dict:
                return -1
            
            if not char in dup_string:
                dup_string = source
                no_of_subsequences += 1
            
            char_index = dup_string.index(char)
            dup_string = dup_string[char_index+1:]
            target = target[1:]
        
        no_of_subsequences += 1
        #print 'no_of_subsequences = ', no_of_subsequences
        return no_of_subsequences

1239. Maximum Length of a Concatenated String with Unique Characters (Microsoft- Kevin)
https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-
characters/

#Not the solution to the problem given. The following is a solution if we need to find 
#the longest continuous subsequence of words without repeating characters
from collections import deque
arr = ["cha","rvb","tca","ers"]
deq_arr = deque()
curr_substr = ''

for word in arr:
    d = {}
    for char in word:
        d[char] = True

    deq_arr.append(d)

char_dict = {}
j = 0

while(j < len(arr)):
    word = arr[j]

    for char in word:
        if char in char_dict:
            while(True):
                word_at_zero = deq_arr.popleft()
                if char in word_at_zero:
                    for dup_char in word_at_zero:
                        char_dict.pop(dup_char)
                    break
                else:
                    for dup_char in word_at_zero:
                        char_dict.pop(dup_char)

            char_dict[char] = True
        else:
            char_dict[char] = True

    if len(char_dict.keys()) > len(curr_substr):
        curr_substr = ''.join(char_dict.keys())

    j += 1

print ('curr_substr = ', curr_substr)

time: O(nk)
space: O(nk) because the char_dict dict will have n * k number of entities in it.

arr = ["cha","rebv","act","ers"]

def callback(i, formed_string):
    if i == len(arr):
        print ('formed_str = ', formed_string)
        return formed_string

    can_be_included = True

    for char in arr[i]:
        if char in formed_string: 
            can_be_included = False
            break

    if can_be_included:
        skip = callback(i + 1, formed_string)
        take = callback(i + 1, formed_string + arr[i])
    else:
        skip = callback(i + 1, formed_string)
        take = skip
        
    if len(skip) > len(take):
        return skip
    else:
        return take

ret_val = callback(0, '')
print ('ret = ', ret_val)

time: O(2 ^ n). Theoritically we cannot achieve a better runtime than this. But we can do
some hacks to improve the runtime like using dictionary or set for formed_string. Your 
lookup will be faster in the for loop. we can also use set union and_ intersections for_
better runtime but Theoritically this is the best.

Refer the below link for how to use sets to solve this problem
https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-
characters/discuss/419204/3-Solutions%3A-Backtracking-Recursive-and-DP-solutions-
(With-Video-explanations)
space: O(n)

1466. Reorder Routes to Make All Paths Lead to the City Zero
https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/

from collections import deque
n = 6
connections = [[1,0],[2,0]]
indegree_dict = {}
outdegree_dict = {}
no_of_directions_to_change = 0
visited_dict = {}

def do_dfs(city):
    global no_of_directions_to_change
    global visited_dict
    queue = deque()
    queue.append(city)

    while(queue):
        city = queue.popleft()
        visited_dict[city] = True

        if city in outdegree_dict:
            for out_city in outdegree_dict[city]:
                if out_city == 0 or out_city in visited_dict:
                    continue
                else:
                    no_of_directions_to_change += 1
                    queue.append(out_city)

        if city in indegree_dict:
            for in_city in indegree_dict[city]:
                if in_city == 0 or in_city in visited_dict:
                    continue
                else:
                    queue.append(in_city)

for c in connections:
    if c[1] in indegree_dict:
        indegree_dict[c[1]].append(c[0])
    else:
        indegree_dict[c[1]] = [c[0]]

    if c[0] in outdegree_dict:
        outdegree_dict[c[0]].append(c[1])
    else:
        outdegree_dict[c[0]] = [c[1]]

print (indegree_dict)
print (outdegree_dict)

if 0 in indegree_dict:
    for in_city in indegree_dict[0]:
        do_dfs(in_city)

if 0 in outdegree_dict:
    for out_city in outdegree_dict[0]:
        no_of_directions_to_change += 1
        do_dfs(out_city)

print ("no_of_directions_to_change = ", no_of_directions_to_change)


Google Question
https://www.youtube.com/watch?v=V0xjK_6ZoEY
Form a tree according to the heirarchy provided

class Node():
    def __init__(self, val, next=None):
        self.val = val
        self.next = next if next else []

hierarchy_list = [
     ('animal', 'mammal'),
     ('animal', 'bird'),
     ('lifeform', 'animal'),
     ('cat', 'lion'),
     ('mammal', 'cat'),
     ('animal', 'fish')
 ]

#We need to form a tree with appt hierarchy.
parent_dict = {}
child_dict = {}

for tup in hierarchy_list:
    parent = tup[0]
    child = tup[1]

    if parent in parent_dict:
        parent_dict[parent].append(child)
    else:
        parent_dict[parent] = [child]

    if child in child_dict:
        child_dict[child].append(parent)
    else:
        child_dict[child] = [parent]

root = set(parent_dict) - set(child_dict)
root = root.pop()

def construct_hierarchy(value):
    node = Node(value)

    if not value in parent_dict:
        return node

    for child in parent_dict[value]:
        node.next.append(construct_hierarchy(child))

    return node

tree = construct_hierarchy(root)
print ('hello')


Google question:
find and remove cycle in tree
related question: https://www.geeksforgeeks.org/detect-cycle-in-a-graph/?ref=lbp
https://www.geeksforgeeks.org/detect-cycle-undirected-graph/

class Node():
    def __init__(self, val, next=None):
        self.val = val
        self.next = next if next else []

root = Node(1)
two_node = Node(2)
three_node = Node(3)
four_node = Node(4)
five_node = Node(5) #This is the only node that forms the cycle. It has 2 parents 
#(two_node and 3_node)
five_node_v2 = Node(5)
six_node = Node(6)
root.next = [two_node, three_node]
two_node.next = [four_node, five_node]
three_node.next = [five_node, six_node, five_node_v2]
visited_dict = {}

def find_cycle_tree(node, caller=None):
    global visited_dict
    if not node:
        return

    cycle = False

    if id(node) in visited_dict:
        cycle =  True

    visited_dict[id(node)] = True
    i = 0

    while(i < len(node.next)):
        neighbor = node.next[i]
        is_cycle = find_cycle_tree(neighbor, node)

        if is_cycle:
            node.next.remove(neighbor)
            continue

        i += 1

    return cycle

find_cycle_tree(root)


#https://aonecode.com/facebook-coding-interview-questions/Minimum-Time-to-Complete-Tasks
#tried doing it with dict. The code is getting too big
#Ask clarifications regarding the question if needed
#Here you can ask will the prereq list given will always be valid?

#Prefect example of why you should not try doing top sorting problems without graphs
#The following program is complicated than using the graph version
"""
project_list = [
    'A|B C|4',
    'B|C D|3',
    'C|D|3',
    'D|2',
]
prereq_satisfied_projects = set() #will there be more than 1 project with zero prereq?
projects_prereq_dict = {}
is_prereq_of_dict = {}
project_work_dict ={}

for project in project_list:
    parts_of_string = project.split('|')
    key = parts_of_string[0]
    prereqs = parts_of_string[1].split() if len(parts_of_string) > 2 else []
    work_units_needed =parts_of_string[-1]
    projects_prereq_dict[key] = prereqs
    project_work_dict[key] = work_units_needed

    for prereq in prereqs:
        if prereq in is_prereq_of_dict:
            is_prereq_of_dict[prereq].append(key)
        else:
            is_prereq_of_dict[prereq] = [key]

    if not prereqs: prereq_satisfied_projects.add(key)

print(projects_prereq_dict)
print(is_prereq_of_dict)
print(prereq_satisfied_projects)

while(projects_prereq_dict):
    project = prereq_satisfied_projects.pop()
    projects_prereq_dict.pop(p)

    if not project in is_prereq_of_dict:
        continue

    for p in is_prereq_of_dict[project]:
        projects_prereq_dict[p].remove(project)

        if projects_prereq_dict[p] == []:
            prereq_satisfied_projects.add(p)

"""

#Try doing it with graph
from collections import deque

class Node(object):
    def __init__(self, val, no_of_prereq=None, work_needed=None):
        self.val = val
        self.no_of_prereq = no_of_prereq
        self.is_prereq_of = []
        self.work_needed = work_needed

project_list = [
    'A|B C|4',
    'B|C D|3',
    'C|D|3',
    'D|2',
]

prereq_satisfied_projects = deque() #will there be more than 1 project with zero prereq? 
#Thre is no reason behind using deque here. Just for practice
projects_node_relationship_dict = {}
project_work_dict ={}
total_work_needed = 0

for project_specifics in project_list:
    parts_of_string = project_specifics.split('|')
    project = parts_of_string[0]
    prereqs = parts_of_string[1].split() if len(parts_of_string) > 2 else []
    work_units_needed =parts_of_string[-1]

    if project in projects_node_relationship_dict:
        project_node = projects_node_relationship_dict[project]
    else:
        project_node = Node(project)
        projects_node_relationship_dict[project] = project_node

    project_node.work_needed = work_units_needed
    project_node.no_of_prereq = len(prereqs)

    for prereq in prereqs:

        if prereq in projects_node_relationship_dict:
            prereq_node = projects_node_relationship_dict[prereq]
        else:
            prereq_node = Node(prereq)
            projects_node_relationship_dict[prereq] = prereq_node

        prereq_node.is_prereq_of.append(project)

for project in projects_node_relationship_dict:
    project_node = projects_node_relationship_dict[project]

    if project_node.no_of_prereq == 0:
        prereq_satisfied_projects.append(project)

print('Project execution order:')
while(projects_node_relationship_dict):
    main_proj = prereq_satisfied_projects.popleft()
    print(main_proj)
    main_proj_node = projects_node_relationship_dict[main_proj]
    total_work_needed += int(main_proj_node.work_needed)
    projects_node_relationship_dict.pop(main_proj)

    for depenedent_project in main_proj_node.is_prereq_of:
        depenedent_project_node = projects_node_relationship_dict[depenedent_project]
        depenedent_project_node.no_of_prereq -= 1

        if depenedent_project_node.no_of_prereq == 0:
            prereq_satisfied_projects.append(depenedent_project)

print(total_work_needed)

#time: O(n^2). Worst case each project can have n-1 prereq. Since we have 2 for loops its
#O(n^2)
#space: O(n^2). Since each vertex in worst case will be connected to every other vertex,
#We need space for vertices and edges. we will have n vertices and each of those n 
#vertices will have n-1 edges going out of it. So, it's n * n-1 = n^2

#https://aonecode.com/facebook-coding-interview-questions/Longest-Monotonic-Path-in-
Binary-Tree-III

longest_monotonic_path = -1

def callback(node, parent_val=None, monotonous_seq_len=0):
    global longest_monotonic_path
    
    if not node:
        return None

    if node.val > parent_val:
        monotonous_seq_len += 1
        longest_monotonic_path = max(longest_monotonic_path, monotonous_seq_len)
        callback(node.left, node.val, monotonous_seq_len)
        callback(node.right, node.val, monotonous_seq_len)
    else:
        callback(node.left, node.val, 1)
        callback(node.right, node.val, 1)

#time:O(n). we visit each node once
#space: O(n). height of the tree. Worst case, we might get a left sided tree


#https://aonecode.com/facebook-coding-interview-questions/regex-dictionary
#This regex supports only '?'

class regex(object):
    def __init__(self):
        self.word_dict = {}
        
    def add_word(self, word):
        word_len = len(word)
        
        if word_len in self.word_dict:
            self.word_dict[word_len].append(word)
        else:
            self.word_dict[word_len] = [word]
    
    def search_word(self, search_term):
        len_search_term = len(search_term)
        
        for word in self.word_dict[len_search_term]:
            self.do_bfs(search_term, word)
        
    def do_bfs(self, word, search_term):
        pass
        #Do bfs and see if this word matches search term
    
#TIME: def add() is O(1), def search() is O(nk) where n is the num of words to add in  
#dictionary and k is the length of the longest word in the dictionary. k is the time 
#representing the do_bfs()
#space: O(n) we have a dictionary of size n 

https://aonecode.com/google-coding-interview-questions/Word-Break-Combinations
https://leetcode.com/problems/concatenated-words/
https://leetcode.com/problems/concatenated-words/discuss/95652/Java-DP-Solution

#Given a set of words, find all words that are concatenations of other words in the set.

# Given a set of words, find all words that are concatenations of other words in the set.
# Questions to ask before coding
# What should I return when there is no such combinations possible?
# Will there be duplicates in the input?
# The following solution is an inefficient solution. Follow the solution given in the above lin
from collections import deque
words = ['foo', 'foobar', 'foobartoo', 'too', 'bar']
words_dict ={}
possible_concatenations = []

for word in words:
    words_dict[word] = True


def do_bfs(word, remamining_words):
    global possible_concatenations
    queue = deque()
    queue.append(([word], remamining_words))

    while(queue):
        curr_seq, remamining_words = queue.popleft()
        curr_seq_string = ''.join(curr_seq)

        if curr_seq_string in words_dict and len(remamining_words) < len(words) - 1: # second condition makes sure that we have more than 1 word in
            # the curr_seq_string
            possible_concatenations.append(curr_seq)

        for index, word in enumerate(remamining_words):
            new_remaining_words = remamining_words[:index] + remamining_words[index + 1:]
            queue.append((curr_seq + [word], new_remaining_words))


for index, word in enumerate(words):
    do_bfs(word, words[:index] + words[index + 1:])

print(possible_concatenations)

#THE FOLLOWING TIME COMPLEXITY CALC IS WRONG
#Refer word-break-combination.jpg
#time: O(n! * n). the while loop in do_bfs will execute O(n!) times
#space: O(n!) In each iteration of the while loop, we build a fresh "curr_seq+
#new_remaining_words" variable which is equal to size n

#correct time and space - In efficient sol
#time: O(n ^ n)
#space: O(n)


Efficient sol time
# time: n * (n ^ 2) -> (n ^ 2) comes from prev problem. For each word in input we call the fun we wrote in
# https://leetcode.com/problems/word-break/solution/

----------------------------------------------------------------------------------------
3 problem types we need more familiarity with_
1 - Window problems - Stack based problems - Rectangle histogram
2 - backtracking - n queens problem
3 - Greedy problem - matlab interview question
--------------------------------------------------------------

https://leetcode.com/explore/interview/card/google/59/array-and-strings/3048/

https://www.geeksforgeeks.org/must-do-coding-questions-for-companies-like-amazon-
microsoft-adobe/?ref=leftbar-rightbar
https://www.geeksforgeeks.org/google-interview-preparation/

Remove active 
check date when new post is created
Password validation
passw reset - Do if you have time

Leetcode Unsolved:
3
https://leetcode.com/problems/median-of-two-sorted-arrays/discuss/2481/Share-my-O(log(min
(mn)))-solution-with-explanation
12
https://leetcode.com/problems/integer-to-roman/solution/
https://leetcode.com/problems/implement-strstr/solution/ (last sliding wind sol)

stack problems 
(You cannot place a larger elem on top of a smaller elem)
42
https://leetcode.com/problems/trapping-rain-water/solution/ (sol 3 in sol tab)

'''
DRW
'''
# Q1
from collections import defaultdict


def fragments(A, mean):
    seen = defaultdict(int)
    curr = 0
    seen[0] = 1
    ret = 0
    for item in A:
        print('\nseen = ', seen)
        print("ret = ", ret)
        curr += item - mean
        print('curr = ', curr)
        ret += seen[curr]
        seen[curr] += 1
    return ret


print(fragments([0, 4, 3, -1], 2))


# Q2
def uniqueLetterString(self, S):
    index = {c: [-1, -1] for c in string.ascii_uppercase}
    res = 0
    for i, c in enumerate(S):
        k, j = index[c]
        res += (i - j) * (j - k)
        index[c] = [j, i]
    for c in index:
        k, j = index[c]
        res += (len(S) - j) * (j - k)
    return res % (10 ** 9 + 7)


# Q3
pairs_rem_pyramids = get_all_pyramid_pairs_with_same_comb()  # [[(pair_1), [rem_dominos_1]],[(pair_2), [rem_dominos_2]], ...]

for pair in pairs_rem_pyramids:
    pair_foward = pair[0]
    remaining_dominoes = pair[1]
    row_2_pair, row_3_choices = get_possible_choices_for_row2(pair_foward, remaining_dominoes)

    pair_backward = (pair[0][1], pair[0][0])
    get_possible_choices_for_row2(pair_foward, remaining_dominoes)
    row_2_pair, row_3_choices = get_possible_choices_for_row2(pair_foward, remaining_dominoes)

'''
DRW submitted ans
'''


def get_num_of_unichar_possibilities_in_range(occurance_3, occurances_list):
    occurance_1 = occurances_list[0]
    occurance_2 = occurances_list[1]
    num_possibilities_for_single_appearance = (occurance_3 - occurance_2) * (occurance_2 - occurance_1)

    return num_possibilities_for_single_appearance

def solution(S):
    char_occurances_dict = {}
    num_unichars_in_substrings = 0
    len_s = len(S)

    for char in string.ascii_uppercase:
        char_occurances_dict[char] = [-1, -1]

    for ind, char in enumerate(S):
        num_unichars_in_substrings += get_num_of_unichar_possibilities_in_range(ind, char_occurances_dict[char])
        char_occurances_dict[char] = [char_occurances_dict[char][1], ind]

    for key in char_occurances_dict:
        if char_occurances_dict[key][0] == -1 and char_occurances_dict[key][1] == -1:
            continue

        num_unichars_in_substrings += get_num_of_unichar_possibilities_in_range(len_s, char_occurances_dict[key])

    return num_unichars_in_substrings % 1000000007


def solution(A):
    max_so_far = min_so_far = max_prod = A[0]

    for ind, num in enumerate(A):
        if ind == 0:
            continue

        max_so_far, min_so_far = max(num, max_so_far * num, min_so_far * num), min(num, max_so_far * num, min_so_far * num)
        max_prod = max(max_prod, max_so_far, min_so_far)

        if max_prod > 1000000000:
            return 1000000000.0

    return max_prod


'''
DRW OA
Decompose a given positive integer as a sum of odd nums (K). Find the max len of k
'''

n = 8
nums = [i for i in range(1, n + 1) if i & 1 == 1]
memo = [[False] * (n + 1) for _ in range(n + 1)]

def callback(i, n, formed_arr):
    if n == 0:
        return 0, formed_arr

    if i == len(nums) or n < 0:
        return float('-inf'), []

    curr_num = nums[i]

    if memo[i][n] != False:
        return memo[i][n]

    take, ret_take_arr = callback(i + 1, n - nums[i], formed_arr + [curr_num])
    skip, ret_skip_arr = callback(i + 1, n, formed_arr)
    #print(ret_take_arr)
    memo[i][n] = (take + 1, ret_take_arr) if take + 1 > skip else (skip, ret_skip_arr)
    print(memo[i][n])

    return memo[i][n]

print(callback(0, n, []))

# Akuna capital web intern Prog 1
line = '00000000'

if len(line) != 8:
    print('Invalid')

ind = 7
res = 0
power = 0
hex_converter_dict = {'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15}
dec_converter_dict = {10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F'}

while (ind > 1):
    char = line[ind]

    if char.isnumeric():
        res += int(char) * (16 ** power)
    elif char.upper() in hex_converter_dict:
        res += (hex_converter_dict[char.upper()] * (16 ** power))
    else:
        print('INVALID')
    power += 1
    ind -= 1

sum_res = 0

while (res > 0):
    sum_res = sum_res + res % 10
    res = res // 10

quotient = sum_res
expected_check_sum = ''
curr_check_sum = line[0:2]

while (quotient > 0):
    rem = quotient % 16

    if rem < 10:
        expected_check_sum = str(rem) + expected_check_sum
    else:
        expected_check_sum = dec_converter_dict[rem] + expected_check_sum

    quotient = quotient // 16

if not expected_check_sum:
    expected_check_sum = '00'

print('expected_check_sum = ', expected_check_sum)
curr_check_sum = line[0:2]
if curr_check_sum == expected_check_sum:
    print('VALID')
else:
    print('INVALID')

# Akuna capital web intern  Prog 2

import fileinput
from collections import defaultdict


class StatisticsCalculator:
    def __init__(self):
        self.busy_city = ''
        self.busy_city_visits = 0
        self.city_visits_dict = defaultdict(int)
        self.customer_distance_dict = defaultdict(int)
        self.max_travel_customer_id = ''
        self.max_travel_dist = 0
        self.total_dist = 0

    def process(self, line: str) -> str:
        splited_line = line.split(':')
        cust_id = splited_line[0]
        city_1 = splited_line[1]
        city_2 = splited_line[2]
        dist = int(splited_line[3])
        self.total_dist += dist

        self.city_visits_dict[city_1] += 1
        self.city_visits_dict[city_2] += 1
        self.customer_distance_dict[cust_id] += dist

        if self.customer_distance_dict[cust_id] > self.max_travel_dist:
            self.max_travel_dist = self.customer_distance_dict[cust_id]
            self.max_travel_customer_id = cust_id

        elif self.customer_distance_dict[cust_id] == self.max_travel_dist:
            self.max_travel_dist = self.customer_distance_dict[cust_id]
            self.max_travel_customer_id = min(self.max_travel_customer_id, cust_id)

        if self.city_visits_dict[city_1] > self.busy_city_visits:
            self.busy_city_visits = self.city_visits_dict[city_1]
            self.busy_city = city_1

        elif self.city_visits_dict[city_1] == self.busy_city_visits:
            self.busy_city = min(self.busy_city, city_1)

        if self.city_visits_dict[city_2] > self.busy_city_visits:
            self.busy_city_visits = self.city_visits_dict[city_2]
            self.busy_city = city_2

        elif self.city_visits_dict[city_2] == self.busy_city_visits:
            self.busy_city = min(self.busy_city, city_2)

        res_str = str(self.total_dist) + ':' + self.max_travel_customer_id + ':' + self.busy_city

        pass


sc = StatisticsCalculator()
inp = '''0FF1CE18:NYC:SEATTLE:2414
C0FFEE1C:SEATTLE:HAWAII:4924
0FF1CE18:SEATTLE:NYC:2414
C0FFEE1C:HAWAII:SEATTLE:4924'''

for i in inp.split('\n'):
    print(i)
    sc.process(i)




#----------------------------------------------------------------- python test 3 redundant code ---------------------------------------------------------



#https://www.youtube.com/watch?v=pQfagNu3p54

#Question to ask interviewer. If we have 2 flights like this (4,9) (9,15), where end time of flight 1 is same as
#the end time of flight 2, should we consider at time 9, there are 2 flights or 1 flight?
#Are the start times in the input sorted
import heapq
flight_times = [
    (4,8),
    (2,5),
    (17,20),
    (10,21),
    (9, 18),
]
flight_times = sorted(flight_times)
end_time_heap = []
heapq.heapify(end_time_heap)
max_flights = 0
curr_flights_in_air = 0

for flight_time in flight_times:
    start_time = flight_time[0]
    end_time = flight_time[1]
    heapq.heappush(end_time_heap, end_time)

    while(end_time_heap and end_time_heap[0] <= start_time): #beware of "=" here. We need it
        heapq.heappop(end_time_heap)
        curr_flights_in_air -= 1

    curr_flights_in_air += 1
    max_flights = max(max_flights, curr_flights_in_air)

print(max_flights)


#https://aonecode.com/amazon-online-assessment-partition-string

s = 'bbeadcxede'
#exp_out = ['bb', 'eadcxede']
s = 'baddacx'
#exp_out = ['b', 'adda', 'c', 'x']
char_last_occurence_dict = {}
substr_list = []

for i, char in enumerate(s):
    if char in char_last_occurence_dict:
        char_last_occurence_dict[char] = i
    else:
        char_last_occurence_dict[char] = i

i = 0
curr_breakpoint = char_last_occurence_dict[s[i]] #last occurrence of char in s
curr_partition_start = i

while(i < len(s)):

    if i == curr_breakpoint:
        substr_list.append(s[curr_partition_start:i+1])
        curr_partition_start = i + 1

        if i < len(s) - 1:
            curr_breakpoint = char_last_occurence_dict[s[i + 1]]
    else:
        if char_last_occurence_dict[s[i]] > curr_breakpoint:
            curr_breakpoint = char_last_occurence_dict[s[i]]

    i += 1

if curr_partition_start < len(s):
    substr_list.append(s[curr_partition_start:])

print(substr_list)


#https://aonecode.com/amazon-online-assessment-find-substrings
import heapq
s = 'bbeadcxede'
#exp_out = ['bb', 'eadcxede']
s = 'baddacx'
possible_substrings = []

def get_all_combs(i, formed_string):
    global possible_substrings

    if i > len(s):
        possible_substrings.append(formed_string)
        return

    skip = get_all_combs(i + 1, formed_string)
    take = get_all_combs(i + 1, formed_string + s[i])

    return

#We make a call to the fun we made in #https://aonecode.com/amazon-online-assessment-
#partition-string
#This functions returns us the partitioned version of the substrings we have in
#possible_substrings
#We create a list of tuples of the form (no_of_partitions, sum_of_len_of_words_in_part,
#partition_str)
#We heapify this list heapq.heapify_max(l) until we get all tuples whose no_of_partitions
#is greatest.
my_l = [('no_of_partitions', 'sum_of_len_of_words_in_part', 'partition_str'), '.....']
heapq.heapify(my_l)
max_partitions, sum_of_len_of_words_in_part, partition_str = my_l.heappop()
min_len_of_words_in_part = sum_of_len_of_words_in_part
min_len_partitioned_str = partition_str

while(max_partitions == my_l[0][0]):
    max_partitions, sum_of_len_of_words_in_part, partition_str = my_l.heappop()

    if sum_of_len_of_words_in_part < min_len_of_words_in_part:
        min_len_of_words_in_part = sum_of_len_of_words_in_part
        min_len_partitioned_str = partition_str


print(min_len_partitioned_str)

#time: O(2^n)
#space: O(2^n)#We build formed string variable in each recursive call

#Random practice problem. Might need this for next problem
#Binary search find left and right most index of number in sorted list
def binary_search(st, en):
    mid = (st + en) // 2

    if mid < target:
        return binary_search(mid + 1, en)
    elif mid > target:
        return binary_search(st, mid + 1)
    elif st != target:#To get the right nost index replace thi line as en != target and retrun b_s(mid+1, en)
        return binary_search(st + 1, mid)
    else:
        return st
#Time: O(log n)

#Google question
#https://techdevguide.withgoogle.com/paths/foundational/find-longest-word-in-dictionary-that-subsequence-of-given-string#code-challenge
s = "abppplee"
[2, 2, 3, 4, -1, -1, -1, -1]
d = {"able", "ale", "apple", "bale", "kangaroo"}
#d = {"ale", "apple"}
char_dict = {}
max_substring = ''

def check_subs_ptr(word):
    ptr_1 = 0
    ptr_2 = 0

    while(ptr_1 < len(s) and ptr_2 < len(word)):
        if s[ptr_1] == word[ptr_2]:
            ptr_1 += 1
            ptr_2 += 1
        else:
            ptr_1 += 1

    if ptr_2 == len(word):
        return True
    else:
        return False

for word in d:
    #approach 1
    if check_subs_ptr(word): #O(mn) m is the no of words and n is the len of the string
        print (word)
        if len(word) > len(max_substring): max_substring = word

    # approach 2
    if check_subs(char_dict.copy(), word): #O(m * log n)
        print(word)
        if len(word) > len(max_substring): max_substring = word

for ind, char in enumerate(s): #O(n)
    if char in char_dict:
        char_dict[char].append(ind)
    else:
        char_dict[char] = [ind]

def check_subs(char_dictionary, word):
    prev_char_ind_in_s = -1

    for char in word:
        if not char in char_dictionary:
            return False
        elif char_dictionary[char][-1] < prev_char_ind_in_s:
            return False
        else:
            occurances = char_dictionary[char]
            updated_occurances = [occurance for occurance in occurances if occurance > prev_char_ind_in_s]
            #You can also find the least occurance greater than the prev_char_ind_in_s using binary_search
            #This reduced the runtime from O(n) to (log n) for this particular line
            #updated_occurances = binary_search(occurances, prev_char_ind_in_s)
            if not updated_occurances: return False
            char_dictionary[char] = updated_occurances[1:]
            if not char_dictionary[char]: char_dictionary.pop(char)
            prev_char_ind_in_s = updated_occurances[0]

    return True

#approach 1 Time O(mn) or O(wn) where w is the total no of chars in all words
#approach 2 TIme O(n + w * log n)

Google practice
https://techdevguide.withgoogle.com/paths/foundational/stringsplosion-problem-ccocodcode/#!
'''
stringSplosion("Code") → "CCoCodCode"
stringSplosion("abc") → "aababc"
stringSplosion("ab") → "aab"
'''

inp = 'code'
op_str = ''
prev_str = ''
i = 0

while(i < len(inp)):
    char = inp[i]
    op_str += prev_str + char
    prev_str += char
    i += 1

print (op_str)
#Time: O(n)

https://techdevguide.withgoogle.com/paths/foundational/maxspan-problem-return-largest-span-array/#!
Consider the leftmost and righmost appearances of some value in an array. We will say that the "span" is the number of
elements between the two inclusive. A single value has a span of 1. Returns the largest span found in the given array.
(Efficiency is not a priority.)

maxSpan([1, 2, 1, 1, 3]) → 4
maxSpan([1, 4, 2, 1, 4, 1, 4]) → 6
maxSpan([1, 4, 2, 1, 4, 4, 4]) → 6

from collections import defaultdict

elem_ind_dict = defaultdict(list)
inp = [1, 4, 2, 1, 4, 4, 4]
inp = [1, 2, 1, 1, 3]
max_span = 0

for ind, elem in enumerate(inp):
    if elem in elem_ind_dict:
        curr_span = ind - elem_ind_dict[elem][0] + 1
        max_span = max(max_span, curr_span)
    else:
        elem_ind_dict[elem].append(ind)

print (max_span)

#Time: O(n)

Google question
https://www.youtube.com/watch?v=VX2oZkDJeGA
Finding loop in array
#We have solved this problem by constructing graph and vertices
#questions to ask interviewer


arr = [3, 4, 1, 2, 9]
arr = [4, 2, 1, 4, 7]
arr_len = len(arr)
# Questions for interviewer
# Will there be out of bounds index in the array. eg. in this array will there be an element like 10 where the array #size is just 5 (indices range from 0 to 4 - inclusive)
# Let’s consider the case when it only has valid indices in the array for now

# Algo - approach 1 (inefficient - time O(n) space O(n) )
# 1) Go through the array elem by elem
# 2) Start a dfs from each element. if you reach the same element within the dfs recursive calls, there is a cycle.
# Another question for the interviewer. Will there be duplicates in the array?
# Let’s assume initially we don’t have duplicates

do_dfs(index[0], {}, 0) #Assuming we only start from the start of the array and check if we encounter loops

def do_dfs(index, indices_dict):
    if index >= arr_len:
        return True

    if indices_dict.has_key(index):
        return False

    indices_dict[index] = True
    return do_dfs(arr[index], indices_dict)

#Algo - appraoch 2 Rabbit and tortoise approach time O(n) space O(1)
# Have 2 pointers - fast_ptr and slow_ptr
# fast_ptr moves 2 places in a single move and slow pointer moves by 1 pos in a single move
# if the fast ptr catches the clow pointer, there is a loop
# If the fast ptr reaches a point where the array element points to an out of bounds index, there is no loop

'''
Count num of unival trees in the given tree
https://www.youtube.com/watch?v=7HgsS8bRvjo
Algo
1) If the returned left and right values are none, unival_trees += 1
2) If the returned left and right values are same as root, unival_trees += 1
  - Handle cases when only left or right child is present as well
3) One thing I missed due to carelessness in the above algo is what if
 - a node has a value of 3
 - its parent has value of 3
 - its left child has value of 2
the above algorithm will fail. We need to make additional checks
'''
print('tree creation start')

class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

root = Node(0)
l1 = Node(1)
r1 = Node(0)
root.left = l1
root.right = r1
l2 = Node(1)
r2 = Node(0)
r1.left = l2
r1.right = r2
l3 = Node(1)
r3 = Node(1)
l2.left = l3
l2.right = r3
print('tree creation done')
unival_trees = 0

def count_unival_trees(node):
    global unival_trees

    if not node:
        return None

    left_ret_val = count_unival_trees(node.left)
    right_ret_val = count_unival_trees(node.right)

    if node.left and node.right:
        if left_ret_val == right_ret_val == node.val:
            unival_trees += 1
        else:
            return False
    elif node.left:
        if node.val != left_ret_val:
            return False
        else:
            unival_trees += 1
    elif node.right:
        if node.val != node.right:
            return False
        else:
            return True
    else:
        unival_trees += 1

    return node.val

count_unival_trees(root)
#Time O(n)
#space O(n)


#google question
#https://leetcode.com/discuss/interview-question/723230/Google-or-Phone-or-Kth-missing-number
#questions to ask interviewer
#1 - Will there be duplicates in the array?
#2 - will there be negative intergers in the array?
#3 - Should I handle case where there may be 0 in the input?
sorted_arr = [4, 5, 14, 16, 17]
k = 4

#missing_elements = [1,2,3,6,7,8,9,...]
#no_of_missing_elements_before_i = [3,3,11,12,12]

#This problem can be thought of as "where will you place k in the no_of_missing_elements_before_i list"?
#That's mostly the solution to your problem

#output = 5

#testcase 1
#sorted_arr = [2,4,6]
#k = 8

#testcase 2
#sorted_arr = [2,4,10,15]
#k = 8
#missing = [1,3,5,6,7,8,11,'12',13,14]
st = 0
en = len(sorted_arr) - 1

if k > sorted_arr[en]: #We want to find the 8th missing elem. If we are given an empty array. all missing elems are
    # 1,2,3,4,5,6,7,8,9,10,11,... Here if we want to find the 8th missing element, we will do len(arr) + 8 = 8. Lets
    # assume our arr = [3]. Now our 8th missing elem will be len(arr) + 8 = 9.
    print ('missing elem is = ', len(sorted_arr) + k)
else:
    while(st < en):
        mid_ind = (st + en) // 2
        mid_ele = sorted_arr[mid_ind]
        imaginory_binary_search_elem = mid_ele - (mid_ind + 1)

        if imaginory_binary_search_elem < k:
            st = mid_ind + 1
            en = en
        else: #mid_ele - (mid_ind + 1) >= k:
            st = st
            en = mid_ind - 1

    print ('st = ', st)
    print ('en = ', en)
    no_of_elems_missing_before_k = sorted_arr[en] - (en + 1)
    print ('missing elem = ',sorted_arr[en] + (k - no_of_elems_missing_before_k) )

#time O(log n)
#space O(1)

'''
https://leetcode.com/problems/missing-element-in-sorted-array/
Given a sorted array A of unique numbers, find the K-th missing number starting from the leftmost number of the array.
'''
#Questions to ask interviewer
#1) Should I handle cases where kth missing element is greater than the largest elem in the array?
#2) Will k always be a positive number? Or should I handle -ve numbers and 0's as well?

sorted_arr = [4, 7, 9, 12, 14]
no_of_missing_eles =   [0, 2, 3, 5, 6] #This is the array on which you have to binary search on
missing_eles_ind =     [0, 1, 2, 3,  4]
missing_ele=           [5, 6, 8, 10, 11, 13]
#Question is to find k in this imaginary array
k = 4
st_ele = sorted_arr[0]
lo = 0
hi = len(sorted_arr) - 1

while(lo < hi): # loop breaks when it finds out lo and hi (indices correspond to the elements between which the missing
    #element is
    mid_ind = (lo + hi) // 2 #2 3
    mid_ele = sorted_arr[mid_ind] #9 12
    mid_ele_proper_ind = mid_ele - st_ele #5 8
    no_of_elems_missing_bef_mid = mid_ele_proper_ind - mid_ind  #3 5

    if no_of_elems_missing_bef_mid == k:
        lo = mid_ind
        break

    if no_of_elems_missing_bef_mid > k: #i2
        lo = lo #3
        hi = mid_ind - 1 #3
    else:#i1
        lo = mid_ind + 1 #3
        hi = hi #4

#while breaks at lo=3 and hi=2
prev_ind = lo - 1
proper_index = sorted_arr[prev_ind] - st_ele
no_of_elems_missing_bef_prev_ind = proper_index - prev_ind
remaining_k = k - no_of_elems_missing_bef_prev_ind
print ('ans = ', sorted_arr[prev_ind] + remaining_k)

#time: log n
#space 1

#Google question
#https://leetcode.com/discuss/interview-question/707835/Google-or-Phone-or-Find-origin-of-malware-transmission
'''
In a network of nodes, a node can transmit a malware to other nodes and infect them. Find out the nodes who have
started the malware transmission from the network traces.

Assumptions:
Network trace is like A->B, i.e. A transmitted malware to B
There are no cycles

Ex:
Network traces: [[1 -> 2 ], [1 -> 3], [2 -> 4], [2 -> 5], [6 -> 4]]
Ans: 1 and 6
'''

# Questions to ask interviewer
# Can I assume each unique number uniquely represents a tower?
# Should I expect cycles like this in input? [[1 -> 2], [2 -> 1]]

class Node(object):
    def __init__(self, val, out_deg, in_deg):
        self.val = val
        self.out_deg = out_deg
        self.in_deg = in_deg

transmitter_dict = {}
network = [['1', '2' ], ['1', '3'], ['2', '4'], ['2', '5'], ['6', '4']]
zero_indegree_set = set()

for item in network:
    sender = item[0]
    reciever = item[1]

    if sender in transmitter_dict: sender_node = transmitter_dict[sender]
    else:
        sender_node = Node(sender, 1, 0)
        transmitter_dict[sender] = sender_node
        zero_indegree_set.add(sender_node)

    if reciever in transmitter_dict:
        reciever_node = transmitter_dict[reciever]
        if reciever_node in zero_indegree_set:
            zero_indegree_set.pop(reciever_node)
    else:
        reciever_node = Node(reciever, 0, 1)
        transmitter_dict[reciever] = reciever_node

print ('Nodes that transmitted virus are ', zero_indegree_set)

for item in zero_indegree_set:
    print (item.val)

#time O(n) n representing the number of connections or the length of the list given as input
#space O(n)


#https://leetcode.com/discuss/interview-question/707842/Google-or-Phone-or-Time-to-reach-message-to-all-the-employees
#Questions to ask the interviewer
# Can I assume that one employee will report only to 1 manager? Or will an employee report to multiple managers?
# Can I assume that I wont encounter cycles in the input?

employee_id_node_dict = {} #maps id number to empliyee node

class Node(object):
    def __init__(self, emp_id, msg_pass_time, reports=[]):
        self.emp_id = emp_id
        self.reports = reports #list of people who report to this guy
        self.msg_pass_time = msg_pass_time

node_0 = Node(0, 4)
node_1 = Node(1, 3)
node_2 = Node(2, 2)
node_3 = Node(3, 4)
node_4 = Node(4, 3)
node_5 = Node(5, 3)
node_6 = Node(6, 3)
node_7 = Node(7, 6)
node_8 = Node(8, 3)
node_9 = Node(9, 3)
node_10 = Node(10, 3)

employee_id_node_dict = {0:node_0, 1:node_1, 2:node_2, 3:node_3, 4:node_4, 5:node_5, 6:node_6, 7:node_7, 8:node_8,\
                         9:node_9, 10:node_10}

node_0.reports = [1,2,3]
node_1.reports = [4,5]
node_2.reports = [6]
node_3.reports = [7,8,9]
node_7.reports = [10]

def dfs(node, time):
    max_time_from_node = time
    node_val = node.emp_id
    reports = node.reports

    for emp_id in node.reports:
        time_to_pass_message = node.msg_pass_time
        emp_node = employee_id_node_dict[emp_id]
        max_time_from_node = max(max_time_from_node, dfs(emp_node, time + time_to_pass_message))

    return max_time_from_node

print ('max_time_for_message_to_reach_employees = ' , dfs(node_0, 0))

#time: O(n) assuming one employee has only one manager, we visit a node only once
#space: O(n ^ 2) we create n nodes and for each employee node we might have reports list of size n (in the worst case)
# How I got confused here a bit
# We might think of creating a dictionary with each employee as key and their reports as list of values might bring
# down space complexity to O(n + n) 'n' representing nodes and second 'n' representing the dictionary.
# But this is in correct because, each of your dictionary key can hold a list of size n. which will again lead to n^2

'''
Google Interview question
https://leetcode.com/discuss/interview-question/704810/Google-or-Phone-or-Longest-Subarray-Midpoint

Coding. Given an array of 1s and 0s. Your method should return midpoint of the longest subarray of zeros.
Given 0100010 => the longest subarray is 0100010 => you should return the index of 0100010 => return 3.
'''
#here the question is clear. if they said you should find the longest subarray of zeros, you should ask them what
# should I return after finding that longest subarray of zeros? The length of the subarray or the starting ind?
# Can i assume that the elements in the input will contain only 0's and 1's
# What should I return if there are no zeros in the input?
# If my longest subarray of zeros is am even number (say 4), what should I return as midpoint (return 2 or 3?).

#naive approach will take O(n * k) where k is the longest subarray of zeroes
longest_subarray = float('-inf')
mid_point = None
n = [0,1,0,0,0,1,0,0,0,0]
'''
for i, ele in enumerate(n):

    if ele == 1: continue

    for j, ele in enumerate(i+1, n):
        if ele == 0 and j - i > longest_subarray:
            longest_subarray = j - i
            mid_point = (i + j) // 2
        elif j == 1:
            break
'''

#Better approach O(n)
stack = []
max_length = float('-inf')
i = 0
'''
while(i < len(n)):
    if n[i] == 0:
        stack.append(i)
    else:
        if stack:
            start = stack[0]
            end = stack.pop()
            length = end - start + 1

            if length > max_length:
                max_length = length
                mid_point = (start + end) // 2

            stack = []
    i += 1


#The follwing is a very important corner case
if stack:
    end = i - 1
    st_ind = stack[0]
    length = end - st_ind + 1

    if length > max_length:
        max_length = length
        mid_point = (st_ind + end) // 2

print ('max_length = ', max_length)
'''
#time: O(n)
#space: O(n) Since we are using stack. We can do this in O(1) space as well just by storing the start index of zero
#like below

#Even better approach
st_ind = None

while(i < len(n)):
    if n[i] == 0:
        if not st_ind: st_ind = i
    else:
        if st_ind != None:
            end = i - 1
            length = end - st_ind + 1

            if length > max_length:
                max_length = length
                mid_point = (st_ind + end) // 2

            st_ind = None
    i += 1

#The follwing is a very important corner case
if st_ind:
    end = i - 1
    length = end - st_ind + 1

    if length > max_length:
        max_length = length
        mid_point = (st_ind + end) // 2

print ('max_length = ', max_length)

#time: O(n)
#space: O(1)


'''
https://leetcode.com/discuss/interview-question/727705/Google-or-Phone-or-Given-N-sorted-arrays-find-k-smallest-elements
Google | Phone | Given N sorted arrays find k smallest elements

Sorted arrays:
[4 9 13 25]
[1 3 19 36]
[2 5 12 45]

K = 6

Ans: {1,2,3,4,5,9}

I gave O(NK) time complexity solution.
It could be done in better time complexity which is O(logNK)
'''
#As far as my thoughts and looking at the comments for this problem we cannot do it in O(log NK). If the rows and cols
#were sorted, we can do in log time https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/
#Questions to ask the interviewer, What should I do if I'm not able to form an output list of size k (this could happen
#when the total no of elems in all the lists is < k)
'''
Algo:
1 - Put all the list elements in a heap
2 - use a while loop to loop until you have k elements in your output list
'''

#approach 1
from heapq import heappush, heappop

def ksmall(arr, k):
  heap = []
  for i in range(len(arr)):
    for j in range(len(arr[i])):
      heappush(heap,arr[i][j])
  res = [heappop(heap) for _ in range(k)]
  return res

#time: O(Nm log Nm + k log Nm) N is the number of lists and m is the length of the longest list
#space: O(Nm)

#approach 2
inputs = [
    [4 9 13 25],
    [1 3 19 36],
    [2 5 12 45],
]
input_lists = []

#THis for loop will create a new array which will occupy space O(Nm)
for inp in inputs:
    new_list = deque()
    new_list.extend(inp)
    input_lists.append(new_list)

i = 0
while (i < len(inputs)): #This will not take extra space
    list_ele = inputs[i]
    list_ele.reverse()

i = 0
while (i < k):

    smallest_ele = float('+inf')

    for ind, my_list in enumerate(inputs):

        if my_list and my_list[-1] < smallest_ele:

            smallest_ele = my_list[0]
            smallest_ele_list = ind

    inputs[smallest_ele_list].pop()
    output_list.append(smallest_ele)

    i += 1

#Time: O(Nm + Nk) #N - no of lists, m - len of the longest list, k - input
#Space: O(1)

#approach3
import heapq
heap_list = heapq.heapify([])
len_arr = len(inputs[0]) #assuming all arrs are of same len

for inp in inputs:
    heapq.heappush(heap_list, (inp[0], inp, 1))

output_list = []
i = 0

while(i < k):
    tup = heapq.heappop(heap_list)
    output_list.append(tup[0])
    arr = tup[1]
    new_ind = tup[2]
    if new_ind < len_arr: #if arrays are of same length don't compute len(arr)
        new_tup = (arr[new_ind], arr, new_ind + 1)
        heapq.heappush(heap_list, new_tup)
    i += 1

#time: N log N + K log N
#space: N

Google interview question
https://leetcode.com/discuss/interview-question/736717/Google-or-Phone-or-Start-to-End-with-Safe-states

'''
I got a question similar to https://leetcode.com/problems/word-ladder/solution/
I gave a BFS based solution since I had seen the problem.

The Problem was described like so

Given a safe list: 001 010 100 101 111
Find a transformation from 000 to 111

Contraint: Change one char/bit at a time

Length of the String is variable and N < 100000, N being length of String.
'''
# You can convert 0 to 1 and 1 to 0. both ways are possible. When I first wrote the solution, I forgot to take into
# consideration the conversion from 1 to 0. Later I tried out an input 110 (assume 111 is the end state and not given
# in the word_dict, you might be able to reach the end state by just converting 0's to 1's

inp = '000'
str_len = len(inp)
word_dict = {'001', '010', '100', '101', '111'}
visited_dict = {}
zero_digits = 0

for digit in inp:

    if digit == '0': zero_digits += 1

def callback(my_str, zero_digits):
    global visited_dict
    ind = 0

    if not zero_digits:
        return True

    if my_str in visited_dict:
        return False

    while (ind < str_len):
        new_str = [char for char in my_str]
        new_str[ind] = '1' if new_str[ind] == '0' else '1'
        new_str = ''.join(new_str)

        if new_str in word_dict and new_str not in visited_dict:

            if callback(new_str, zero_digits - 1): return True

        ind += 1

    visited_dict[new_str] = False

is_possible = callback(inp, zero_digits)
print (is_possible)

#approach 2 BFS
queue = heapq.heapify([])
queue.heappush((0,inp))
all_combs = defaultdict(list)

for comb in word_dict:
    list_transformation = [char for char in my_str]
    all_combs_dict[comb] = list_transformation

while(queue):
    transformations, comb = heapq.heappop(queue)

    if ''.join(comb) == target:
        print ('min transforms needed = ', transformations)
        break

    for char in comb:
        if char == '1': comb[char] = 0
        else: comb[char] = 1

        new_comb = ''.join(comb)

        if new_comb in all_combs and new_comb not in visited_dict:
            visited_dict[new_comb] = True
            heapq.heappush((transformations + 1, new_comb))

# The following calc of time compl is wrong
# Time: 2 ^ n because each digit can be a 0 or 1. In the worst case, we are trying out all the possibilities to see
# if the target is reachable

# Following is the correct time complexity
# The above would have been correct if we don't have a safe list. But in this prob we are given a safe list of size W
# So, whatever combination we form should be one of the combs in safe list.
# We also have a visited dictionary. This will make sure that we process a combination ONLY ONCE.
# So we can be certain that the while loop will execute ONLY O(W) times.
# Time: O(W + W * M^2)


'''
Google interview question - Daily Coding Problem

An XOR linked list is a more memory efficient doubly linked list. Instead of each node holding next and prev fields, it holds a field named both, 
which is an XOR of the next node and the previous node. Implement an XOR linked list; it has an add(element) which adds the element to the end, and
a get(index) which returns the node at index.
'''

import _ctypes

class Node(object):
    def __init__(self, val, xor_id=None):
        self.val = val
        self.xor_id = xor_id

head_node = Node('head')
first_node = Node('1')
second_node = Node('2')
third_node = Node('3')
fourth_node = Node('4')
fifth_node = Node('5')

head_node.xor_id = id(first_node) ^ 0
first_node.xor_id = id(head_node) ^ id(second_node)
second_node.xor_id = id(first_node) ^ id(third_node)
third_node.xor_id = id(second_node) ^ id(fourth_node)
fourth_node.xor_id = id(third_node) ^ id(fifth_node)
fifth_node.xor_id = id(fourth_node) ^ 0

def di(obj_id):
    """ Inverse of id() function. """
    return _ctypes.PyObj_FromPtr(obj_id)


def add_element(head, val):
    first_node_loc = head.xor_id
    node = di(first_node_loc)
    prev_node = head

    while (node.xor_id != id(prev_node)):
        next_node_location = id(prev_node) ^ node.xor_id
        prev_node = node
        node = di(next_node_location)

    new_node = Node(val)
    node.xor_id = node.xor_id ^ id(new_node)
    new_node.xor_id = id(node)
    return new_node


def print_list(head):
    print ('\nEles in list are')
    first_node_loc = head.xor_id
    node = di(first_node_loc)
    prev_node = head
    print(node.val)

    while (node.xor_id != id(prev_node)):
        next_node_location = id(prev_node) ^ node.xor_id
        prev_node = node
        node = di(next_node_location)
        print (node.val)


def ele_at_ind(ind, head):
    print ('\nEle at index ', ind)
    i = 0
    first_node_loc = head.xor_id
    node = di(first_node_loc)
    prev_node = head

    while(i < ind):
        next_node_location = id(prev_node) ^ node.xor_id
        prev_node = node
        node = di(next_node_location)
        i += 1

    print(node.val)


print_list(head_node)
new_node = add_element(head_node, '6')
print_list(head_node)
ele_at_ind(3, head_node)

# time: O(n) for all the three functions. However since we support only adding ele at the END of the linked list, if we are using class, we can have
# a variable to hold the id of the last node. So, we can insert in O(1) time itself
# space: O(n) where n is the number of eles in inout. Apart from the Node data structure, we dont use any additional space.


'''
Daily coding problem - airbnb

Given a list of integers, write a function that returns the largest sum of non-adjacent numbers. Numbers can be 0 or negative.

For example, [2, 4, 6, 2, 5] should return 13, since we pick 2, 6, and 5. [5, 1, 1, 5] should return 10, since we pick 5 and 5.

Follow-up: Can you do this in O(N) time and constant space?
'''
# approach 0 - Worst way to approach recursion problems. A small mistake in the parameters you pass to your recursion func can lead to terrible
# run time
input_list = [2, 1, 5, 2, 7, 3, 2, 1, 5, 2, 7, 3]
memo = {}

def lps(index, curr_sum, prev_ele_picked):
    if index >= len(input_list):
        return curr_sum

    key = str(index) + str(prev_ele_picked) + str(curr_sum)

    if key in memo:
        print('in memo ', key)
        return memo[key]

    if prev_ele_picked:
        skip = lps(index + 1, curr_sum, prev_ele_picked=False)
        take = float('-inf')
    else:
        skip = lps(index + 1, curr_sum, prev_ele_picked=False)
        take = lps(index + 1, curr_sum + input_list[index], prev_ele_picked=True)

    memo[key] = max(skip, take)

    return memo[key]

print(lps(0, 0, False))
print(memo)

# time: 2 ^ n - Because you have the current sum variable as part of recursion. This is a bad way to approach recursion probs. If curr_sum was a
# Boolean value, your run time would have been (n ^ 3). Since curr_sum can be anything between 0 to sum(input_list), you cannot upper bound it with
# n^3. The following is a better approach

# approach 1
nums = [2, 4, 6, 2, 5]
nums = [5, 1, 1, 5]
expected_op = 13 # 2 + 6 + 5
# approach 1
pick_memo = {}
skip_memo = {}

def callback(index, prev_ele_picked):
    global memo

    if index >= len(nums):
        return 0

    if index in pick_memo and index in skip_memo:
        return max(pick_memo[index], skip_memo[index])

    if prev_ele_picked:
        skip = callback(index + 1, False)
        skip_memo[index] = skip
        return skip_memo[index]
    else:
        pick = nums[index] + callback(index + 1, True)
        skip = callback(index + 1, False)
        pick_memo[index] = pick
        skip_memo[index] = skip
        return max(pick_memo[index], skip_memo[index])

max_sum = callback(0, False)
print (pick_memo)
print (skip_memo)
print (max_sum)

# One careless mistake I commited was to just have one memo dictionary when we have 2 parameters determining the state of the recursion.
# We could have either
# 1 - solved it using a single dictionay whose keys are formed by (index + prev_ele_picked)
# 2 - Or having 2 dictionaries like the above
# time - O(n)
# space - O(n)

# approach 2 - time O(n) space O(1)
# refer dcm_7_17
# https://www.geeksforgeeks.org/maximum-sum-such-that-no-two-elements-are-adjacent/
arr[] = {5,  5, 10, 40, 50, 35}

  incl = 5
  excl = 0

  For i = 1 (current element is 5)
  incl =  (excl + arr[i])  = 5
  excl =  max(5, 0) = 5

  For i = 2 (current element is 10)
  incl =  (excl + arr[i]) = 15
  excl =  max(5, 5) = 5

  For i = 3 (current element is 40)
  incl = (excl + arr[i]) = 45
  excl = max(5, 15) = 15

  For i = 4 (current element is 50)
  incl = (excl + arr[i]) = 65
  excl =  max(45, 15) = 45

  For i = 5 (current element is 35)
  incl =  (excl + arr[i]) = 80
  excl =  max(65, 45) = 65

And 35 is the last element. So, answer is max(incl, excl) =  80


'''
Daily coding problem
Given a binary tree return the depest node in the tree
'''

# Not writing the answer as this is not a hard problem
# time: O(n)
# space: O(n)


'''
Daily Interview Pro - facebook question
Given a list of words, for each word find the shortest unique prefix. You can assume a word will not be a substring of another word (ie play and 

Example
Input: ['joma', 'john', 'jack', 'techlead']
Output: ['jom', 'joh', 'ja', 't']
'''
# Would have to change the logic slightly if we had to support substrings as well
input_words = ['joma', 'john', 'jack', 'techlead']
exp_op = ['jom', 'joh', 'ja', 't']


class Node():
    def __init__(self, val):
        self.val = val
        self.next = {}


trie_node = trie_main_root = Node('root')


def get_shortest_prefix(word, index, trie_node):  # joma, 0, r_n | joma, 1, j_n | joma, 2, o_n | joma, 3, m_n | joma, 4, a_n
    if index == len(word):
        return False, index

    char = word[index]
    trie_node = trie_node.next[char]
    child_has_branches, shortest_prefix_index = get_shortest_prefix(word, index + 1, trie_node)
    # F, 4 | F, 3 | T, 3 | T, 3 |

    if child_has_branches:
        return True, shortest_prefix_index

    if len(trie_node.next.keys()) > 1:
        return True, shortest_prefix_index
    else:
        return False, index


for word in input_words:  # time O(n * k)
    trie_node = trie_main_root
    index = 0
    while (index < len(word)):
        char = word[index]
        if char in trie_node.next:
            trie_node = trie_node.next[char]
        else:
            break

        index += 1

    while (index < len(word)):
        char = word[index]
        char_node = Node(char)
        trie_node.next[char] = char_node
        trie_node = char_node
        index += 1

for word in input_words:  # time O(n * k)
    print(word)
    _, shortest_prefix = get_shortest_prefix(word, index=0, trie_node=trie_main_root)
    print(word[0:shortest_prefix + 1])

# time O(n * k)
# space O(n * k) # We will have n * k nodes created in the worst case (when the first char of all input words are different).


'''
DIP - Google

Given a sorted list of numbers, and two integers low and high representing the lower and upper bound of a range, return a list of (inclusive) ranges 
where the numbers are missing. A range should be represented by a tuple in the format of (lower, upper).

Here's an example and some starting code:

def missing_ranges(nums, low, high):
  # Fill this in.
  
print(missing_ranges([1, 3, 5, 10], 1, 10))
# [(2, 2), (4, 4), (6, 9)]
'''
# print(missing_ranges([1, 3, 5, 10], 1, 10))
# Questions to ask interviewer here
# 1) Can i assume that the given inp list does not contain duplicates?
# 2) Can I assume that I will not be given empty list as input

nums = [1, 4, 7, 10]
nums = [3, 5, 7, 11]
nums = [2, 5, 7, 11]
nums = [4, 5, 7, 11]
nums = [0,1,2,20,21,22]
nums = [0,1,2,5]
lower = 3
upper = 10
missing_ranges = deque([])

# The following 3 if statements make life easier when handled before for loop

if nums[-1] < lower:
    missing_ranges.append((lower, upper))
    #return

if nums[0] > upper:
    missing_ranges.append((lower, upper))
    #return

if nums[0] > lower:
    start = lower
    end = nums[0] - 1
    missing_ranges.append((start, end))

for index, num in enumerate(nums): # 4
    if index == 0:
        continue

    if num < lower:
        continue

    prev_ele = nums[index - 1] # 1

    if num > lower and num > prev_ele + 1:
        start = prev_ele + 1 # if prev_ele + 1 >= lower else lower COMMENTED out this and following line to make the code more easy to understand
        end = num - 1 # if num - 1 <= upper else upper WE have added these checks after the while loop using 2 if statements

        missing_range = (start, end)
        missing_ranges.append(missing_range)


    if num >= upper:
        break

if missing_ranges[0][0] < lower:
    new_tup = (lower, missing_ranges[0][1])
    missing_ranges.popleft()
    missing_ranges.append(new_tup)

if missing_ranges[-1][1] > upper:
    new_tup = (missing_ranges[-1][0], upper)
    missing_ranges.pop()
    missing_ranges.append(new_tup)

if missing_ranges[-1][1] < upper:
    new_tup = (nums[-1] + 1, upper)
    missing_ranges.append(new_tup)

print (missing_ranges)

# time O(n)
# space O(1)


'''
DIP _ Airbnb

The power function calculates x raised to the nth power. If implemented in O(n) it would simply be a for loop over n and multiply x n times.
Instead implement this power function in O(log n) time. You can assume that n will be a non-negative integer.

Here's some starting code:

def pow(x, n):
  # Fill this in.

print(pow(5, 3))
# 125
'''

curr_pow = 1
desired_pow = 13 # test with 11, 13, 15
curr_val = calc_val =  5
pow_val_dict = {curr_pow: curr_val}

while(curr_pow * 2 <= desired_pow):
    calc_val *= calc_val
    curr_pow = curr_pow * 2
    pow_val_dict[curr_pow] = calc_val

remaining_power_to_be_raised = desired_pow - curr_pow

while (remaining_power_to_be_raised != 0):
    if curr_pow > remaining_power_to_be_raised:
        curr_pow = curr_pow // 2

    val_to_mul = pow_val_dict[curr_pow]
    calc_val *= val_to_mul
    remaining_power_to_be_raised -= curr_pow
    curr_pow = curr_pow // 2

print ('final ans = ', calc_val)

# time: log n
# space: O(k) where k is the desired power. In the worst case we might have a pow_val_dict of size k (eg: when k = 2)


'''
DIP - Apple

Given a list of words, and an arbitrary alphabetical order, verify that the words are in order of the alphabetical order.

Example:
Input:
words = ["abcd", "efgh"], order="zyxwvutsrqponmlkjihgfedcba"

Output: False
Explanation: 'e' comes before 'a' so 'efgh' should come before 'abcd'

Example 2:
Input:
words = ["zyx", "zyxw", "zyxwy"],
order="zyxwvutsrqponmlkjihgfedcba"

Output: True
Explanation: The words are in increasing alphabetical order

'''
'''
Possible questions
1 - Can I be ceratin that the order will not have the same char repeating at 2 diff places?
2 - If the same word comes twice in the words list one immediatly after the other. Is that valid?
'''

char_weight_dict = {}
words = ["abcd", "efgh"]
#words = ["zyx", "zyxw", "zyxwy"]
order="zyxwvutsrqponmlkjihgfedcba"
word_tup_list = []

for ind, char in enumerate(order):
    char_weight_dict[char] = ind

for word in words:
    char_list = []

    for char in word:
        char_list.append(char_weight_dict[char])

    word_tup_list.append(tuple(char_list))

# Approach 1 - inefficient approach, they are not asking you to sort the words RATHER they are asking you to CHECK if it's sorted
sorted_word_tuple_list = sorted(word_tup_list)
print (word_tup_list == sorted_word_tuple_list)

# Approach 2
for ind, word_val_tup in enumerate(word_tup_list):
    if ind == 0:
        continue

    if word_val_tup < word_tup_list[ind - 1]:
        print('Not sorted')

# Approach 2
# time: O(n * k) n is the num of words and k is the max num of char in any word OR O(N) where N is the total chars in all words
# space: O(M) M is the num of chars in order list

'''
DIP - Amazon
Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s.
If there isn't one, return 0 instead.

Example:
Input: s = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: the subarray [4,3] has the minimal length under the problem constraint.
'''

s = 7
nums = [2,3,1,2,4,3]
j = i = 0
window_sum = 0
len_nums = len(nums)
min_window_len = float('+inf')

while(i < len_nums):
    window_sum += nums[i]

    if window_sum >= s:
        while(j <= i and window_sum >= s):
            curr_win = i - j + 1
            min_window_len = min(min_window_len, curr_win)
            window_sum -= nums[j]
            j += 1

    i += 1

if min_window_len == float('+inf'):
    min_window_len = 0

print(min_window_len)

# time: O(n)
# space: O(1)


'''
DIP - Amazon
You are given an array of integers. Return the length of the longest consecutive elements sequence in the array.

For example, the input array [100, 4, 200, 1, 3, 2] has the longest consecutive sequence 1, 2, 3, 4, and thus, you should return its length, 4.

def longest_consecutive(nums):
  # Fill this in.

print longest_consecutive([100, 4, 200, 1, 3, 2])
# 4

Can you do this in linear time?
'''

nums = [100, 4, 200, 1, 3, 2, 6]
nums_dict = {}
longest_consecutive_seq = float('-inf')

for num in nums:
    nums_dict[num] = True

while(nums):
    num = nums.pop()

    if num not in nums_dict:
        continue

    curr_seq_len = 0

    if num - 1 in nums_dict:
        next_ele = num - 1

        while(next_ele in nums_dict):
            curr_seq_len += 1
            nums_dict.pop(next_ele)
            next_ele = next_ele - 1

    if num + 1 in nums_dict:
        next_ele = num + 1

        while(next_ele in nums_dict):
            curr_seq_len += 1
            nums_dict.pop(next_ele)
            next_ele = next_ele + 1

    nums_dict.pop(num)
    curr_seq_len += 1
    longest_consecutive_seq = max(longest_consecutive_seq, curr_seq_len)

print (longest_consecutive_seq)

# time: O(n)
# space: O(n)


'''
DIP - Facebook
Hi, here's your problem today. This problem was recently asked by Facebook:

Given a directed graph, reverse the directed graph so all directed edges are reversed.

Example:
Input:
A -> B, B -> C, A ->C

Output:
B->A, C -> B, C -> A
'''
'''
Questions for interviewer
1 - Can I assume that there will be no cycles in the graph?
'''

class Node:
  def __init__(self, value):
    self.adjacent = []
    self.value = value

a = Node('a')
b = Node('b')
c = Node('c')

a.adjacent += [b, c]
b.adjacent += [c]
graph_val_vertex_map = {
    a.value: a,
    b.value: b,
    c.value: c,
}
visited_dict = {}
initial_adjancency_count = {}
all_vertex_vals = graph_val_vertex_map.keys()

for vertex_val in all_vertex_vals:
    vertex_node = graph_val_vertex_map[vertex_val]
    initial_adjancency_count[vertex_val] = len(vertex_node.adjacent)

for vertex_val in graph_val_vertex_map.keys():
    vertex_node = graph_val_vertex_map[vertex_val]
    initial_adj_count = initial_adjancency_count[vertex_node.value]

    for adj_node in vertex_node.adjacent[:initial_adj_count]:
        adj_node.adjacent.append(vertex_node)

    vertex_node.adjacent = vertex_node.adjacent[initial_adj_count:]


print(graph_val_vertex_map)

# time: O(n * k) n is the num of vertices and k is the largest outdegree of any node in the graph
# space: O(n * k) Dict of size n where each key's value at most stores a list of size k. Or
# we can also say that it's O(n + e) Where n is the number of vertices and e is the num of edges
# https://stackoverflow.com/questions/33499276/space-complexity-of-adjacency-list-representation-of-graph


'''
DIP - Airbnb
Given a list of sorted numbers, and two integers k and x, find k closest numbers to the pivot x.

Here's an example and some starter code:

def closest_nums(nums, k, x):
 # Fill this in.

print(closest_nums([1, 3, 7, 8, 9], 3, 5))
# [7, 3, 8]
'''

'''
Questions for the interviewer
1 - If k > size of list, what should I do
2 - In cases where I have to pick 1 last element and I have 2 choices that are equi-distant from x, what should I pick 
(eg: x = 6, choices are 3 and 9). Should I pick 3 or 9?
'''
inds = [0, 1, 2, 3, 4]
nums = [1, 3, 7, 8, 10]


op_list = []
k = 3
x = 6

if nums[0] > x:
    print('ans is ', nums[:k + 1])
elif nums[-1] < x:
    print('ans is ', nums[-k:])

left = 0
right = len(nums) - 1

while(left < right):
    ele_left = nums[left]
    ele_right = nums[right]
    mid = (left + right) // 2
    ele_mid = nums[mid]

    if ele_mid < x:
        left = mid + 1
    elif ele_mid > x:
        right = mid - 1
    elif ele_mid == x:
        break

if nums[mid] == x:
    op_list.append(nums[mid])
    left = mid - 1
    right = mid + 1

if left == right:
    left = right - 1

while(len(op_list) < k):
    if left < 0 and right >=len(nums):
        break
    if left < 0:
        op_list.append(nums[right])
        right += 1
    elif right >= len(nums):
        op_list.append(nums[left])
        left -= 1
    else:
        if abs(nums[left] - x) < abs(nums[right] - x):
            op_list.append(nums[left])
            left -= 1
        else:
            op_list.append(nums[right])
            right += 1

print(op_list)

# time: log(n) + k ; Assuming we are given the len of the input list. Otherwise it will be O(n)
# space O(1)


'''
DIP - Apple
Given an array of numbers, determine whether it can be partitioned into 3 arrays of equal sums.

For instance,
[0, 2, 1, -6, 6, -7, 9, 1, 2, 0, 1] can be partitioned into:
[0, 2, 1], [-6, 6, -7, 9, 1], [2, 0, 1] all of which sum to 3
'''

nums = [0, 2, 1, -6, 6, -7, 9, 1, 2, 0, 1]

if len(nums) < 3:
    print('not possible')

left = 0
right = len(nums) - 1
left_sum = nums[0]
right_sum = nums[-1]
mid_sum = sum(nums) - left_sum - right_sum

while(left < right):
    if left_sum == right_sum == mid_sum:
        break

    elif left_sum > right_sum:
        mid_right_boundary_val = nums[right - 1]
        right_sum += mid_right_boundary_val
        mid_sum -= mid_right_boundary_val
        right -= 1

    elif right_sum > left_sum:
        mid_left_boundary_val = nums[left + 1]
        left_sum += mid_left_boundary_val
        mid_sum -= mid_left_boundary_val
        left += 1

print(nums[:left+1])
print(nums[left+1:right])
print(nums[right:])

# time O(n)
# space O(1)


'''
DIP - Apple
Given a sorted array, convert it into a binary search tree.

Can you do this both recursively and iteratively?
'''
'''
questions for the interviewer
1 - since the inp list is odd, we can take the mid elem as root. If its even like 6, should I take 3rd ele as root or 4th ele as root?
'''

class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


#Approach 1 - Recursive solution
input_arr = [-10, -3, 0, 5, 9, 15]
arr_len = len(input_arr)

def construct_bst(st, en):
    if st > en:
        return None

    mid = (st + en) // 2
    mid_val = input_arr[mid]
    root = Node(mid_val)
    root.left = construct_bst(st, mid - 1)
    root.right = construct_bst(mid + 1, en)
    return root

root = construct_bst(0, arr_len - 1)
print(root)

# time: O(n) In each recursive call we create a new node for an item in input array. We have n nodes which means we have made n recursive calls
# space: O(n) There will be n/2 calls in the recursion stack for the root node before it gets its left child. n/2 equates to n

# Approach 2 - Iterative solution

st = 0
en = arr_len - 1
mid = (st + en) // 2
root = parent = Node(input_arr[mid])
stack = [(parent, st, mid -1, 'left'), (parent, mid + 1, en, 'right')]

while(stack):
    parent, st, en, left_or_right = stack.pop()

    if st > en:
        continue

    mid = (st + en) // 2
    mid_node = Node(input_arr[mid])

    if left_or_right == 'left':
        parent.left = mid_node
    else:
        parent.right = mid_node

    tup_left = (mid_node, st, mid - 1, 'left')
    tup_right = (mid_node, mid + 1, en, 'right')
    stack.append(tup_left)
    stack.append(tup_right)

print(root)

# time: O(n) In each iteration of while loop we create a new node for an item in input array. We have n nodes which means we have made n iterations.
# You might think that we make more calls than that when you look at "if st > en" line. We ideally do not consider that as a call because we don't
# do any considerable work in that call. And in face that condition can be checked before lines "tup_left = (mid_node, st, mid - 1, 'left')" and
# "tup_right = (mid_node, mid + 1, en, 'right')" and can be avoided by not adding them to the stack. Not doing it here for simplicity
# space: O(n) There will be n/2 calls in the recursion stack for the root node before it gets its left child. n/2 equates to n


'''
DIP - LinkedIn
Given two rectangles, find the area of intersection.
'''
# Another related problem
# https://www.geeksforgeeks.org/total-area-two-overlapping-rectangles/ (or) https://leetcode.com/problems/rectangle-area/

# Refer rects_intersect_area.jpeg

class Rectangle():
  def __init__(self, min_x=0, min_y=0, max_x=0, max_y=0):
    self.min_x = min_x
    self.min_y = min_y
    self.max_x = max_x
    self.max_y = max_y

rect1 = Rectangle(0, 0, 3, 2)
rect2 = Rectangle(1, 1, 3, 3)

intersecting_rect_left_bottom_x = max(rect1.min_x, rect2.min_x)
intersecting_rect_left_bottom_y = max(rect1.min_y, rect2.min_y)
intersecting_rect_top_right_x = min(rect1.max_x, rect2.max_x)
intersecting_rect_top_right_y = min(rect1.max_y, rect2.max_y)

len_of_intersecting_rect = abs(intersecting_rect_left_bottom_x - intersecting_rect_top_right_x)
bredth_of_intersecting_rect = abs(intersecting_rect_left_bottom_y - intersecting_rect_top_right_y)

area = len_of_intersecting_rect * bredth_of_intersecting_rect
print(area)

# time: O(1)
# space: O(1)


'''
DIP - Microsoft
Given an array of heights, determine whether the array forms a "mountain" pattern. A mountain pattern goes up and then down.
print(Solution().validMountainArray([1, 2, 3, 2, 1]))
# True

print(Solution().validMountainArray([1, 2, 3]))
# False
'''
'''
Questions to ask the intervierer
1 - Can I assume that I don't have duplicates in the array. If we have duplicates in the array, the following binary search solution wont work
'''

arr = [1, 2, 3, 2, 1]
arr = [1, 2, 3, 2, 1, 0, -1]
arr_len = len(arr)

if arr_len < 3:
    print('no valid ans can be formed')

lo = 0
hi = arr_len - 1
mountain_peak_ele = None

while (lo <= hi):
    mid = (lo + hi) // 2

    if mid <= 0 or mid >= arr_len - 1:
        break

    mid_ele = arr[mid]
    prev_ele = arr[mid - 1]
    next_ele = arr[mid + 1]

    if mid_ele > prev_ele and mid_ele > next_ele:
        mountain_peak_ele = mid_ele
        break

    elif mid_ele > prev_ele and mid_ele < next_ele:
        lo = mid + 1

    elif mid_ele < prev_ele and mid_ele > next_ele:
        hi = mid - 1

    else:
        break

print(mountain_peak_ele)

# time O(log n)
# space O(1)


'''
DIP - Microsoft
Given a tree, the leaves form a certain order from left to right. Two trees are considered "leaf-similar" if their leaf orderings are the same.

For instance, the following two trees are considered leaf-similar because their leaves are [2, 1]:
    3
   / \
  5   1
   \
    2
    7
   / \
  2   1
   \
    2
'''
from collections import deque

class Node(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

t1 = Node(3)
t1.left = Node(5)
t1.right = Node(1)
t1.left.left = Node(6)
t1.left.right = Node(2)

t2 = Node(7)
t2.left = Node(2)
t2.right = Node(1)
t2.left.left = Node(6)
t2.left.right = Node(2)
# t2.left.right.left = Node(1) # un comment this line for False output

leaves_list = deque([])
leaf_similar = True

def callback(node, store_leaf_value):
    global leaves_list
    global leaf_similar

    if not node:
        return None

    callback(node.left, store_leaf_value)
    callback(node.right, store_leaf_value)

    if not node.left and not node.right: # leaf node
        if store_leaf_value: leaves_list.append(node.val)
        else:
            expected_leaf_value = leaves_list.popleft()

            if node.val != expected_leaf_value:
                leaf_similar = False

callback(t1, store_leaf_value=True)
callback(t2, store_leaf_value=False)

if leaves_list: leaf_similar = False

print(leaf_similar)

# time: O(n)
# space O(n)


'''
DIP - Amazon
Given a number like 159, add the digits repeatedly until you get a single number.

For instance, 1 + 5 + 9 = 15.
1 + 5 = 6.
'''

num = 159
curr_sum = 0

while(num):
    digit = num % 10
    curr_sum += digit

    if not num and curr_sum > 9:
        num = curr_sum

# 99 = 18 = 9 | len_dig = 2 | num_iter - 4
# 99991 = 37 = 10 = 1 | len_dig = 5 | num_iter - 9
# 9191 = 20 | len_dig = 4 | num_iter - 6
# time: Based on the above examples, we can upper bound O(n). In each case I can imagine the num of iterations to be less than 2n
# space: O(1)


'''
DIP - Facebook
A Perfect Number is a positive integer that is equal to the sum of all its positive divisors except itself.

For instance,
28 = 1 + 2 + 4 + 7 + 14

Write a function to determine if a number is a perfect number.
'''

import math

n = 2
i = 2

if n <= 0: print('not a perfect num')

fact_sum = 1 # 1 is a factor of every number

while(i <= math.sqrt(n)):
    if 28 % i == 0:
        print(i, ' ', 28 // i)
        fact_sum += i + (28 // i)

    i += 1

print(fact_sum)

if fact_sum == n:
    print('perfect number')

# time sqrt(n)
# space O(1)


'''
Given two binary trees that are duplicates of one another, and given a node in one tree, find that correponding node in the second tree.

For instance, in the tree below, we're looking for Node #4.

For this problem, you can assume that:
- There can be duplicate values in the tree (so comparing node1.value == node2.value isn't going to work).

Can you solve this both recursively and iteratively?
'''
from collections import deque

class Node:
  def __init__(self, val):
    self.val = val
    self.left = None
    self.right = None

#  1
# / \
#2   3
#   / \
#  4*  5
a = Node(1)
a.left = Node(2)
a.right = Node(3)
target_node = a.right.left = Node(4)
a.right.right = Node(5)

b = Node(1)
b.left = Node(2)
b.right = Node(3)
b.right.left = Node(4)
b.right.right = Node(5)

#print(findNode(a, b, a.right.left))
# 4

nth_recursion = 0
clone_node_found = False
clone_node = None

def traverse_path(node, find_clone_node):
    global clone_node_found
    global clone_node
    global nth_recursion

    if not node:
        return

    if find_clone_node == False and clone_node_found == False:
        nth_recursion += 1
        if target_node == node: clone_node_found = True

    elif find_clone_node == True and clone_node == None:
        nth_recursion -= 1
        if nth_recursion == 0: clone_node = node

    traverse_path(node.left, find_clone_node)
    traverse_path(node.right, find_clone_node)

traverse_path(a, False)
traverse_path(b, True)


#Iterative approach
queue = deque()
queue.append((a,b))
clone_node = None
clone_node_found = False

while(queue and clone_node_found == False):
    node_a, node_b = queue.popleft()
    node_a_val = node_a.val
    node_b_val = node_b.val

    if target_node == node_a:
        clone_node_found = True
        clone_node = node_b
        break

    if node_a.left: queue.append((node_a.left, node_b.left))
    if node_a.right: queue.append((node_a.right, node_b.right))

print(clone_node)
print('finish')


# Both approaches
# time: O(n)
# space O(n)


583. Deletion Distance
https://leetcode.com/problems/delete-operation-for-two-strings/

from collections import Counter


def deletion_distance(str1, str2):
    if not (str1) or not (str2):
        return max(len(str1), len(str2))

    dict_2 = Counter(str2)
    stack_2 = [char for char in str2]
    stack_2.reverse()
    dict_1 = Counter(str1)
    stack_1 = [char for char in str1]
    stack_1.reverse()
    dist_1 = get_distance(str1, dict_2, stack_2)
    dist_2 = get_distance(str2, dict_1, stack_1)
    # adzbcdefg
    # abcdefg
    print('d1 = ', dist_1)
    print('d2 = ', dist_2)
    return min(dist_1, dist_2)


def get_distance(src_str, dst_str_dict, stack):
    dist = 0

    for char in src_str:
        if stack and stack[-1] == char:
            ele = stack.pop()
            dst_str_dict[ele] -= 1
            if dst_str_dict[ele] == 0: dst_str_dict.pop(ele)

        else:

            if char in dst_str_dict:

                while (stack[-1] != char):
                    ele = stack.pop()
                    dst_str_dict[ele] -= 1

                    if dst_str_dict[ele] == 0: dst_str_dict.pop(ele)

                    dist += 1

                ele = stack.pop()
                dst_str_dict[ele] -= 1
                if dst_str_dict[ele] == 0: dst_str_dict.pop(ele)


            else:
                dist += 1

    dist += len(stack)
    return dist


print(deletion_distance('adzbcde', 'abcddvye'))

562. Leetcode - Longest Line of Consecutive One in Matrix
https://leetcode.com/problems/longest-line-of-consecutive-one-in-matrix/submissions/

# This problem does not require a BFS. Got confused by doing bfs here and proved out to be very inefficient. Was not able to use visited_dict
# visited_dict would have a kept a check on the time compl by keeping it to O(mn). Since visited dict cannot be used as a particular cell can be
# visited multiple times, the time comp went to O(mn * mnk) where the last mnk are directions we move from each cell.

So, a brute force mentod seems to be the best solution. refer the 1st sol in solutions tab


844. Backspace String Compare
https://leetcode.com/problems/backspace-string-compare/

# approach 1: Time and space O(m + n)
class Solution(object):
    def backspaceCompare(self, S, T):
        def build(S):
            ans = []
            for c in S:
                if c != '#':
                    ans.append(c)
                elif ans:
                    ans.pop()
            return "".join(ans)
        return build(S) == build(T)

#approach 2 - Time: O(m + n) Space: O(1)
class Solution(object):
    def backspaceCompare(self, S, T):
        def F(S):
            skip = 0
            for x in reversed(S):
                if x == '#':
                    skip += 1
                elif skip:
                    skip -= 1
                else:
                    yield x

        return all(x == y for x, y in itertools.izip_longest(F(S), F(T)))


222.Count Complete Tree Nodes
https://leetcode.com/problems/count-complete-tree-nodes/discuss/62088/My-python-solution-in-O(lgn-*-lgn)-time
'''
Given a complete binary tree, count the number of nodes.

Note:

Definition of a complete binary tree from Wikipedia:
In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. 
It can have between 1 and 2h nodes inclusive at the last level h.

Example:

Input: 
    1
   / \
  2   3
 / \  /
4  5 6

Output: 6
'''

class Solution:
    # @param {TreeNode} root
    # @return {integer}
    def countNodes(self, root):
        if not root:
            return 0
        leftDepth = self.getDepth(root.left)
        rightDepth = self.getDepth(root.right)
        if leftDepth == rightDepth:
            return pow(2, leftDepth) + self.countNodes(root.right)
        else:
            return pow(2, rightDepth) + self.countNodes(root.left)

    def getDepth(self, root):
        if not root:
            return 0
        return 1 + self.getDepth(root.left)


time: O(log n * log n)  # In each recursive call, we reduce the problem size by 2. This is a complex time compl calc. refer the comments in the above
# link to understand
space: O(h)

'''
829 -
https://leetcode.com/problems/consecutive-numbers-sum/

Given a positive integer N, how many ways can we write it as a sum of consecutive positive integers?

Example 1:

Input: 5
Output: 2
Explanation: 5 = 5 = 2 + 3
Example 2:

Input: 9
Output: 3
Explanation: 9 = 9 = 4 + 5 = 2 + 3 + 4
Example 3:

Input: 15
Output: 4
Explanation: 15 = 15 = 8 + 7 = 4 + 5 + 6 = 1 + 2 + 3 + 4 + 5
'''
import math

#approach 1
class Solution:
    def consecutiveNumbersSum(self, N: int) -> int:
        en_num = math.ceil(N / 2)
        nums = [i for i in range(en_num + 1)]
        i = len(nums) - 1
        j = i - 1
        num_ways = 0
        curr_sum = nums[i] + nums[j]

        while (i > 0 and j > 0):

            if curr_sum == N:
                num_ways += 1
                curr_sum -= nums[i]
                i -= 1
                j -= 1
                curr_sum += nums[j]

            elif curr_sum < N:
                j -= 1
                curr_sum += nums[j]

            else:
                curr_sum -= nums[i]
                i -= 1

        return num_ways + 1

N = 9366964
s = Solution()
#print(s.consecutiveNumbersSum(N))

time: O(n)

# approach 2
# https://leetcode.com/problems/consecutive-numbers-sum/discuss/129015/5-lines-C%2B%2B-solution-with-detailed-mathematical-explanation.
res = 0
i = 1

while (N > i * ((i - 1) // 2)):
    if ((N - i * (i - 1) / 2) % i == 0): ans += 1
    i+= 1

print(ans)

time: O(log n)

'''
Throttling gateway
https://leetcode.com/discuss/interview-question/518083/Citrix-or-OA-2020-or-Evaluating-Circuit-Expressions-or-Throttling-Gateway-or-Computing-Cluster-Quality
refer also throttling_gateway_citadel_OA.png if the above link does not work
'''
request_times = [1,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,11,11,11]
request_times = [4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10,10,11]
requests_in_queue = []
len_of_queue = 0
drops = 0

for curr_req_time in request_times:
    print('\n')
    print(drops)
    print(len_of_queue)
    print(requests_in_queue)

    if len_of_queue >= 3 and curr_req_time == requests_in_queue[-3]:
        drops += 1
        continue

    min_allowed_time_for_twentyth_req_from_last = curr_req_time - 9

    if len_of_queue >= 20:
        print(requests_in_queue[-20])
        print(min_allowed_time_for_twentyth_req_from_last)

    if len_of_queue >= 20 and requests_in_queue[-20] > min_allowed_time_for_twentyth_req_from_last:
        drops += 1
        continue

    min_allowed_time_for_sixtyth_req_from_last = curr_req_time - 59

    if len_of_queue >= 60 and requests_in_queue[-60] > min_allowed_time_for_sixtyth_req_from_last:
        drops += 1
        continue

    requests_in_queue.append(curr_req_time)
    len_of_queue += 1

print('\n', drops)


Mock Interview
'''
Input :
       7
      /  \
     12    2
    /  \    \
   11  49    5
  /         / \
 2         1   38
# """
#
#         3
#     5.     1
#   6.  2.  0  8
#      7 4

#         10
#     5.     15
#   4.  8.  11 18
#      7 9
#
#
# Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
# Output: 3
# Explanation: The Lowest Common Ancestor of nodes 5 and 1 is 3.
#      Assume: All the nodes are in the tree
#
# """

"""

1. Name of Function needs to be more accurate => Need to practice
2. The type of input or output needs to check with interviewer.
3.


Good points:
1. Ask the corner case of the problem
2.
"""
node_1 = 15  # smaller
node_2 = 18

# root 15
if root == None:
    return None

if root.val <= node_1.value and root.val <= node2.val:
    callback(root.right)
if root.val >= node_1.value and root.val >= node2.val:
    callback(root.left)

if root.val > node_1.val and root.val < node_2.val:  # 18 < 18
    lowest_c_a = node

if node_1_global == True and node2.val == node.val:
    l_c_a = node

if node_2_global == True and node1.val == node.val:
    l_c_a = node

if node.val == node_1:
    global_node_1_found = True
elif node.val == node_2:
    global_node_2_found = True

print(lca)
"""
  15
12  18

node_1 = 15
node_2 = 18

root = 15

"""

# ------------------
lowest_common_ancestor = None

node_1_found_global = False
node_2_found_global = False


def additional_checks(node, node_1_found, node_2_found):
    if node.val == node_1.value:
        if not node_2_found_global:
            return True, False
        else:  # node_2_found = True
            lowest_common_ancestor = node
            node_1_found_global = True
            return True, True

    if node.val == node_2.value:
        if not node_1_found_global:
            return False, True
        else:  # node_2_found = True
            lowest_common_ancestor = node
            node_2_found_global = True
            return True, True

    if node_1_found and node_2_found:
        lowest_common_ancestor = node
        return True, True


def callback(node):
    if not node:
        return True

    node_1_found, node_2_found = callback(node.left)
    node_1_found_left, node_2_found_left = additional_checks(node, node_1_found, node_2_found
    node_1_found, node_2_found = callback(node.right)
    node_1_found_right, node_2_found_right = additional_checks(node, node_1_found, node_2_found)

    #  return (node_1_found_left or node_1_found_right), (node_2_found_left, node_2_found_right)

    callback(root)
    print(lowest_common_ancestor.val)

----------------------------------------------------------------------------------------------------------------------------------------------------

Max sum path in tree from root to leaf
Input :
        7
      /  \
     12    2
    /  \    \
   11  49    5
  /         / \
 2         1   38

7 > 12 > 11 > 2
7 > 12 > 49


    1
  2  3

[2,1,3] => sum = 6
[1,3]  => sum = 4 (correct)

O/p:
[7, 12, 49]
Path
Goal:
1) The branch which has the largest sum and store the values of that branch (or vertical) in a list
if the largest [11,49,5] a path - root to the leaf

path root to leaf

All the nodes are positive:

traverse all the tree
BFS/DFS
max_sum = float('-inf')


# 0
#  -5
'''


# input : root: TreeNode
# output tree_result : list
class Tree:
    def __init__(self):
        self.max_sum = float('-inf')
        self.tree_result = []

    def max_branch(self, root):
        # DFS
        # corner case:
        if not root:
            return []
        current_sum = 0
        current_path = []
        self.dfs(root, 0, [])
        return self.tree_result

    def dfs(self, node, cur_sum, current_path):
        if not node: return
        # handle the corner case :
        # save the current node here
        current_path.append(node.val)
        cur_sum += node.val

        if not node.left and not node.right:
            # it is the leaf
            if self.max_sum < cur_sum:
                # update max_sum
                self.max_sum = cur_sum
                self.tree_result = current_path
            elif self.max_sum == cur_sum and len(self.tree_result) < len(current_path):  # O(max depth of the tree)
                self.tree_result = current_path
        # general case
        if node.left:
            self.dfs(node.left, cur_sum, current_path)
        if node.right:
            self.dfs(node.right, cur_sum, current_path)


Time
Complexity: O(numbers of nodes * O(max depth of the tree))
Space: O(max depth of the tree)


Test = Tree()
answer_list = Test.max_branch(root)


# Fast box delivery
#https://oss.1point3acres.cn/forum/202002/16/105240zg3icffzixuiqqjo.jpg!c

boxes = [3,4,2,1]
n = top_floor = len(boxes)
total_items_to_transport = sum(boxes)
time_needed = 0


def get_tup_list():
    global floor, distance_from_top_floor
    tup_list = []
    for floor, num_boxes in enumerate(boxes):
        distance_from_top_floor = top_floor - floor
        tup_list.append((num_boxes, distance_from_top_floor, floor))

    return tup_list


tup_list = get_tup_list()

# [(3,1), (2,2), (1,3)]

while(total_items_to_transport > 0):
    max_items, distance_from_top_floor, curr_floor = max(tup_list)
    time_needed += distance_from_top_floor

    for floor in range(curr_floor, top_floor):
        items_in_floor = boxes[floor]

        if items_in_floor > 0:
            boxes[floor] -= 1
            total_items_to_transport -= 1
            time_needed += 1

    tup_list = get_tup_list()



print(time_needed)

# time O(N * (n ^ 2))

'''
HackerRank

Starting with a 1-indexed array of zeros and a list of operations, for each operation add a value to each of the array element between two given 
indices, inclusive. Once all operations have been performed, return the maximum value in your array.

For example, the length of your array of zeros . Your list of queries is as follows:

    a b k
    1 5 3
    4 8 7
    6 9 1
Add the values of  between the indices  and  inclusive:

index->	 1 2 3  4  5 6 7 8 9 10
	[0,0,0, 0, 0,0,0,0,0, 0]
	[3,3,3, 3, 3,0,0,0,0, 0]
	[3,3,3,10,10,7,7,7,0, 0]
	[3,3,3,10,10,8,8,8,1, 0]
The largest value is  after all operations are performed.
'''
def arrayManipulation(n, queries):
    array = [0] * (n + 1)

    for query in queries:
        a = query[0] - 1
        b = query[1]
        k = query[2]
        array[a] += k
        array[b] -= k

    max_value = 0
    running_count = 0
    for i in array:
        running_count += i
        if running_count > max_value:
            max_value = running_count

    return max_value

# time: O(n)
'''
273. Integer to english words
https://leetcode.com/problems/integer-to-english-words/submissions/
'''

class Solution:
    def numberToWords(self, num: int) -> str:
        if num == 0:
            return "Zero"
        converted_str = ''
        i = 0
        units_dict = {
            0: 'Hundred',
            1: 'Thousand',
            2: 'Million',
            3: 'Billion'
        }

        while (num):
            part_of_num = num % 1000
            # print('\n', part_of_num)
            num = num // 1000
            unit_part_str = units_dict[i]
            num_str = self.convert_num_word(part_of_num)
            num_str_list = num_str.split()
            # print(num_str)

            if part_of_num > 99:
                num_str = num_str_list[0] + ' ' + units_dict[0] + ' ' + ' '.join(num_str_list[1:])

            if i != 0 and num_str.lstrip().strip():
                num_str = num_str.lstrip().strip() + ' ' + units_dict[i]

            # print(num_str)
            i += 1
            converted_str = (num_str + ' ' + converted_str).lstrip().strip()
            # print(converted_str)

        return converted_str.strip()

    def convert_num_word(self, num):
        num_to_words1 = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', \
                         6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', \
                         11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', \
                         15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', 19: 'Nineteen', 0: ''}

        num_to_words2 = ['', '', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']
        constructed_str = ''

        if num > 19:
            constructed_str = num_to_words1[num // 100] + ' '
            num = num % 100

            if num > 19:
                constructed_str += (num_to_words2[num // 10] + ' ' + num_to_words1[num % 10])
                return constructed_str.lstrip().strip()

            else:
                constructed_str += num_to_words1[num]
                return constructed_str.lstrip().strip()

        else:
            return num_to_words1[num]

# O(n) where n is the num of digits

'''
953: Verifying an Alien Dictionary
https://leetcode.com/problems/verifying-an-alien-dictionary/
'''
# The following is a brute force approach which seemed to be the best approach

class Solution(object):
    def isAlienSorted(self, words, order):
        order_index = {c: i for i, c in enumerate(order)}

        for i in xrange(len(words) - 1):
            word1 = words[i]
            word2 = words[i+1]

            # Find the first difference word1[k] != word2[k].
            for k in xrange(min(len(word1), len(word2))):
                # If they compare badly, it's not sorted.
                if word1[k] != word2[k]:
                    if order_index[word1[k]] > order_index[word2[k]]:
                        return False
                    break
            else:
                # If we didn't find a first difference, the
                # words are like ("app", "apple").
                if len(word1) > len(word2):
                    return False

        return True

# time O(N) where N is the total number of words in the input

'''
1249. Minimum Remove to Make Valid Parentheses
https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/

Given a string s of '(' , ')' and lowercase English characters. 

Your task is to remove the minimum number of parentheses ( '(' or ')', in any positions ) so that the resulting parentheses string is valid and 
return any valid string.

Formally, a parentheses string is valid if and only if:

It is the empty string, contains only lowercase characters, or
It can be written as AB (A concatenated with B), where A and B are valid strings, or
It can be written as (A), where A is a valid string.
'''

class Solution(object):
    def minRemoveToMakeValid(self, s):
        """
        :type s: str
        :rtype: str
        """
        #
        stack = []
        no_of_open_braces = 0

        for char in s:

            if char == '(':
                no_of_open_braces += 1
                stack.append('(')

            elif char == ')':
                if no_of_open_braces > 0:
                    no_of_open_braces -= 1
                    stack.append(')')

            else:
                stack.append(char)

        stack.reverse()
        s = ''.join(stack)
        # print s
        no_of_close_braces = 0
        stack = []

        for char in s:

            if char == ')':
                no_of_close_braces += 1
                stack.append(')')

            elif char == '(':
                if no_of_close_braces > 0:
                    no_of_close_braces -= 1
                    stack.append('(')

            else:
                stack.append(char)

        stack.reverse()
        # print ''.join(stack)
        return ''.join(stack)


'''
560. Subarray Sum Equals K
https://leetcode.com/problems/subarray-sum-equals-k/

'''

from collections import defaultdict


class Solution:
    def subarraySum(self, nums, k):
        if not nums:
            return

        res = 0
        running_sum = 0
        running_sums_dict = defaultdict(int)
        nums = [0] + nums
        # running_sums_dict[0] += 1
        #  [1,-1,1,1,1,3] k = 1
        # [0,1, 0,1,2,3,6]
        # [10,-10,10,2,4,-1] k = 5
        # [10,0,10,12,16,15]

        for i in range(0, len(nums)):
            running_sum += nums[i]
            running_sums_dict[running_sum] += 1
            print('\n', nums[i])
            print('running_sum = ', running_sum)
            print('running_sums_dict = ', running_sums_dict)
            diff = running_sum - k

            if diff in running_sums_dict:
                # if diff != 0: res += running_sums_dict[0]
                res += running_sums_dict[diff]

            print('res = ', res)
        return res

        '''
        for i in range(0, len(nums)):
            s = nums[i]
            if s == k:
                res += 1
            for j in range(i+1, len(nums)):
                s += nums[j]
                if s == k:
                    res += 1
        '''
        return res

s = Solution()
#print(s.subarraySum([10,-10,10,2,4,-1], 5))
print(s.subarraySum([1,-1,1,1,1,3], 1))


'''
Dutch Flag Partition 
5.1 EPI

The quicksort algorithm for sorting arrays proceeds recursively-it selects an element (the "pivot"), reorders the array to make all the elements 
less than or equal to the pivot appear first, followed by all the elements greater than the pivot. The two subarrays are then sorted recursively.
Implemented naively, quicksort has large run times and deep function call stacks on arrays with many duplicates because the subarrays may differ 
greatly in size. One solution is to reorder the array so that all elements less than the pivot appear first, followed by elements equal to the pivot,
followed by elements greater than the pivot.
'''

arr = [0,1,1,0,2,1,1,2] # 00112112
pivot = 2
i = k = 0
j = len(arr) - 1

while(k < j):
    if arr[k] < pivot:
        arr[i], arr[k] = arr[k], arr[i]
        i += 1
        k += 1
    elif arr[k] > pivot:
        arr[j], arr[k] = arr[k], arr[j]
        j -= 1
    else:
        k += 1

print(arr)

# time O(n)
# space O(1)

'''
EPI 5.3
Multiply two arbitrary precision numbers
'''

num_1 = [1,9,3,7,0,7,7,2,2]
num_2 = [-7,6,1,8,3,8,2,5,7,2,8,7]
num_1 = [9,9]
num_2 = [8,8]
res = [0] * (len(num_1) + len(num_2) + 1)

def mult_digit_with_num(num_1, num, res_st_ind):
    global res
    j = len(num_1) - 1
    carry = 0

    while(j >= 0):
        print('\n', j)
        print(num_1[j])
        prod = (num * num_1[j]) + carry
        prod_val = prod % 10
        carry = prod // 10
        print(prod_val)
        print(carry)
        res[res_st_ind] += prod_val
        res_st_ind -= 1
        j -= 1

    if carry: res[res_st_ind] += carry

i = len(num_1) - 1
res_st_ind = len(res) - 1
'''
mult(num_2, num_1[i], res_st_ind)
print(res)
mult(num_2, num_1[i - 1], res_st_ind - 1)
print(res)
'''
while(i >= 0):
    mult_digit_with_num(num_2, num_1[i], res_st_ind)
    i -= 1
    res_st_ind -= 1

print(res)
res.reverse()
carry = 0
print(res)
for ind, num in enumerate(res):
    val =  (num + carry) % 10
    carry = (num + carry) // 10
    res[ind] = val

res.reverse()
print(res)

'''
Advancing through an Array
EPI 5.4

Write a program which takes an array of n integers, where A[i] denotes the maximum you can advance from index l, and retums whether it is possible 
to advance to the last index starting from the beginning of the array.
'''

def can-reach-end(A):
    furthest_reach_so_far, last_index = 0, len(A) - 1
    i = 0

    while i <= furthest_reach_so_far and furthest_reach_so_far < last_index
        furthest-reach-so-far = nar(furthest-reach-so_far, AIi] + i)
        i+=1

    return furthest_reach_so_far >= last_index



'''
Robinhood interview OA question
Matrix problem
'''

mat = [
    [1,     1,      1],
    ['.',  '.',    '.'],
    ['.',   1,     '.'],
    ['#',  '.',     1],
    ['.',   1,     '.'],
    ['.',  '.',    '.'],
    ['.',  '.',    '.'],
    ['.',  '.',    '#'],
]

block_or_end = False
num_rows = len(mat)
num_cols = len(mat[0])

while(block_or_end == False):
    mat_copy = []
    for row in mat:
        new_row = list(row)
        mat_copy.append(new_row)

    print('\n mat_copy = ', )
    for row in mat_copy:
        print(row)

    if 1 in mat[-1]:
        block_or_end = True
        break

    for row in range(num_rows):

        if block_or_end:
            break

        for col in range(num_cols):
            if row == 0:
                mat[row][col] = '.'
                continue

            if mat[row][col] == '#':
                if mat_copy[row - 1][col] == 1:
                    block_or_end = True
                    break
                continue

            else:
                mat[row][col] = '.' if mat_copy[row - 1][col] == '#' else mat_copy[row - 1][col]

    print('\n mat = ', )
    for row in mat:
        print(row)

print('\n\nFinal state of matrix = \n')

for row in mat_copy:
    print(row)




class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

root = Node(1)
l1 = Node(2)
r1 = Node(5)
root.left = l1
root.right = r1
l2 = Node(3)
r2 = Node(4)
l1.left = l2
l1.right = r2

class FindLCA(object):
    def __init__(self):
        self.node_1 = None
        self.node_2 = None
        self.lca = None

    def get_lca(self, root, node_1, node_2):
        self.node_1 = node_1
        self.node_2 = node_2
        self.find_lca(root)
        return self.lca

    def find_lca(self, root):
        if not root:
            return None, None

        node_1_found_left, node_2_found_left = self.find_lca(root.left)
        node_1_found_right, node_2_found_right = self.find_lca(root.right)

        if root.val == self.node_1:
            if node_2_found_left or node_2_found_right:
                self.lca = root
                return True, True
            else:
                return True, node_2_found_left or node_2_found_right

        if root.val == self.node_2:
            if node_1_found_left or node_1_found_right:
                self.lca = root
                return True, True
            else:
                return node_1_found_left or node_1_found_right, True

        if not self.lca and node_1_found_left or node_1_found_right and node_2_found_left or node_2_found_right:
            self.lca = root

        return node_1_found_left or node_1_found_right, node_2_found_left or node_2_found_right


f = FindLCA()
lca = f.get_lca(root, 5, 2)
print(lca.val)

'''
199. Binary Tree Right Side View
https://leetcode.com/problems/binary-tree-right-side-view/submissions/
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

from collections import deque, defaultdict

class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

root = Node(1)
l1 = Node(2)
r1 = Node(5)
root.left = l1
root.right = r1
l2 = Node(3)
r2 = Node(4)
l1.left = l2
l1.right = r2

class Solution:
    def rightSideView(self, root):
        my_l = [(root, 0)]
        queue = deque(my_l)
        op_list = []

        level_dict = defaultdict(list)
        max_level = -1

        while (queue):
            node, level = queue.popleft()

            if not node:
                continue

            level_dict[level].append(node.val)
            queue.append((node.left, level + 1))
            queue.append((node.right, level + 1))

            if level > max_level:
                max_level = level

        for level in range(max_level + 1):
            op_list.append(level_dict[level][-1])

        return op_list

# time O(n)
# space O(n)


'''
543 Max diameter of tree
https://leetcode.com/problems/diameter-of-binary-tree/
refer 543_lc.jpg                  
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        max_dia = 0
        self.ans = 1

        def get_max_dia(node):
            global max_dia
            if not node:
                return 0

            max_depth_left = get_max_dia(node.left)
            max_depth_right = get_max_dia(node.right)
            # print ('\n node', node.val)
            # print ('left', max_depth_left)
            # print ('right', max_depth_right)
            self.ans = max(self.ans, max_depth_left + max_depth_right + 1)
            # print ('ans = ', self.ans)

            return 1 + max(max_depth_left, max_depth_right)

        max_dia = get_max_dia(root)
        # print(self.ans)
        return max_dia


'''
413 Arithmetic Slices
https://leetcode.com/problems/arithmetic-slices/
'''

class Solution:
    def numberOfArithmeticSlices(self, A: List[int]) -> int:
        if len(A) < 3:
            return 0

        diff_list = []
        nums = A

        for i in range(1, len(nums)):
            diff_list.append(nums[i] - nums[i - 1])

        # nums =      [1,2,3,4,5]
        # [-1,1,3,3,3,2,3,2,1,0]
        # diff_list =  [1,1,1,1]
        i = 0
        res = []

        while (i < len(diff_list)):
            j = i + 1

            while (j < len(diff_list) and diff_list[j] == diff_list[i]):
                if j - i >= 1: res.append((i, j + 1))
                j += 1

            i += 1

        # print(res)
        return len(res)

# time O(nk)
# space O(n ^ 2) for the inp [1,2,3,4,5,6], the len of res will be almost equal to n ^ 2


'''
560. Subarray Sum Equals K
https://leetcode.com/problems/subarray-sum-equals-k/
'''

from collections import defaultdict

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # [0,0,0]
        # {0:1,2,3,4}
        # res = 1 + 2 + 3 = 6
        running_sum = 0
        running_sums_dict = defaultdict(int)
        running_sums_dict[0] = 1
        res = 0

        for num in nums:
            running_sum += num

            if running_sum - k in running_sums_dict:
                res += running_sums_dict[running_sum - k]

            running_sums_dict[running_sum] += 1

        return res


'''
DRW - OA 
Integer V lies strictly between integers U and W if U < V < W or if U > V > W

A non empty zero indexed array A consisting of N integers is given.
A pair of indices (P, Q), where 0 <= P < Q < N, is said to have 'adjacent values'
if no value in the array lies strictly between values A[P] and A[Q],
and in addition A[P] != A[Q]

For example, in array A such that:
A[0] = 0
A[1] = 3
A[2] = 3
A[3] = 7
A[4] = 5
A[5] = 3
A[6] = 11
A[7] = 1

the following pairs of indices have adjacent values:
(0,7), (1,4), (1,7)
(2,4), (2,7), (3,4)
(3,6), (4,5), (5,7)

For example, indices 4 and 5 have adjacent values because the values a[4] = 5 and A[5] = 3 are different
and there is no value in array A that lies strictly between them
the only such value could be the number 4, which is not present in the array

Given two indices P and Q, their distance is defined as abs(P-Q)
where abs(X) = X for X>=0
and abs(X) = -X for X<=0
For example the distance between indices 4 and 5 is 1 because abs(4-5) = abs(5-4) = 1

Write a function that given a non-empty zero-indexed array A consisting of N integers
returns the maximum distance between indices of this array that have adjacent values
The function should return -1 if no adjacent indices exist

For example given array A such that:
A[0] = 1
A[1] = 4
A[2] = 7
A[3] = 3
A[4] = 3
A[5] = 5

the function should return 4 because:

indices 0 and 4 are adjacent because A[0] != A[4]
and the array does not contain any value that lies strictly between A[0] = 1 and A[4] = 3
the distance between these indices is abd(0-4) = 4
no other pair of adjacent indices that has a larger distance exists
Assume that

N is an integer within the range [1 .. 40,000]
each element of array A is an integer within the range [-2,147,483,648 to 2,147,483,647]
'''
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

A = [1, 4, 7, 3, 3, 5]
#A = [0, 3, 3, 7, 5, 3, 11, 1]
max_distance = -1
arr_dict = {}

for num in A:
    arr_dict[num] = True

for i in range(len(A)):
    if max_distance != -1:
        break
    for j in range(len(A) - 1, i, -1):
        is_adjacent = True

        if A[i] == A[j]:
            continue

        if A[i] < A[j]:
            low = A[i]
            high = A[j]
        else:
            low = A[j]
            high = A[i]

        for num in A:

            if num == A[i] or num == A[j]:
                continue

            if low < num < high:
                is_adjacent = False
                break

        if is_adjacent:
            max_distance =  j - i
            break

print(max_distance)

'''
Robinhood Arithmetic Progression question
'''
import math
def maxArithmeticProgress(a,b):
    factorOfDifference = set()
    # Take care of the case where len(a) is later
    if len(a) > 1:
        # Difference between the first and second element
        num = a[1] - a[0]
        # Continue to find difference between the elements in A and
        # find the gcd so that we can find the difference
        for i in range(2,len(a)):
            num = math.gcd(num, a[i]- a[i-1])
        # now that we found the difference, and need to find all the factors of that difference
        # num represents the smallest difference across A
        for i in range(1, num+1):
            if num % i == 0:
                factorOfDifference.add(i)
        # print(factorOfDifference)
        # now loop through the difference
        # and then compute the maxLengthArithmetricProgression
        # first add the first element in A to the answer list
        maxLength = -1
        for factor in factorOfDifference: # factor [1,2,4]
            cur = a[0] + factor
            length = 1
            # Continuously add elements that obey to the factors
            # and are in b or a, even after there is no more element in a to add.
            while cur in a or cur in b:
                length += 1
                cur += factor
            if cur >= a[-1]:
                # Actually traverse b on the left side
                # to see if a can be expanded on the left side
                cur = a[0] - factor
                while cur in b :
                    cur -= factor
                    length += 1
                maxLength = max(maxLength, length)
    return maxLength


a = [0, 4, 8, 16]
b = [0, 2, 6, 12, 14, 20]
print(maxArithmeticProgress(a,b))


'''
mock interview - chen
'''

root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

return 3

# [[ 5,3],[5,2,1],[-3,11]]

# [10,5,3,3], [10,5,3,-2], [10,5,2,1],[10,-3,11]

Input nums = [1,1,1], sum = 2
return 2


# [1,2,1,1] sum = 3
# res = 3
# dict = {0:1,1:1,3:1,4:1,5:1}
running_sums = [1, 2, 3]

dicionary = {0: 1, 1: 1, 2: 1, 3: 1}

res = 1

[10, 5, 3, 3]
{10: 1,
 15: 1,
 18: 1,
 21: 1}

for key in dictionary.keys():
    if key - 8 in dictionary:
        global_var_num_subsets += dictionary[key - 8]

sum_ = 15

# 10 -> 5 -> 3 -> 3 -> -2 ->root(3)

# Input nums = [1,2,2,1], sum = 3

subsets = [[10, 5, 3, 3], [10, 5, 3, -2], [10, 5, 2, 1], [10, -3, 11]]

num_subsets_add_to_k = 0  # 1 2 3
for subset in subsets:
    running_sum = 0
    running_sums_dict = default_dict(int)
    running_sums_dict[0] = 1
    # sum = 3
    for num in subset:  # [1,2,2,1,-3, 3] #{0:1, 1:1, 3:2, 5: 1, 6:1
        running_sum += num  # 1 3 5 6 3 6
        running_sums_dict[running_sum] += 1

        if running_sum - k in running_sums_dict:
            num_subsets_add_to_k += running_sums_dict[running_sum - k]

print(num_subsets_add_to_k)

# O(n) + O(num_branches) * O(longest_path)

'''
Amazon - OA
https://leetcode.com/discuss/interview-question/796241/Amazon-OA2-SDE-1(New-Grad)-2021-(Coding-2Questions-70mins)
'''

class Node:
    def __init__(self, val):
        self.val = val
        self.children = []

node_20 = Node(20)
node_12 = Node(12)
node_18 = Node(18)
node_11 = Node(11)
node_2 = Node(2)
node_3 = Node(3)
node_15 = Node(15)
node_8 = Node(8)

node_12.children = [node_11, node_2, node_3]
node_18.children = [node_15, node_8]
node_20.children = [node_12, node_18]

class MaxAvgAge():
    def __init__(self):
        self.max_avg_team_age = 0

    def calc_max_avg_age_every_team(self, node):
        if not node.children:
            return node.val, 1

        curr_team_age = node.val
        curr_team_size = 1

        for children in node.children:
            sub_team_avg_age, sub_team_size  = self.calc_max_avg_age_every_team(children)
            sum_age_sub_team = sub_team_avg_age * sub_team_size
            curr_team_age += sum_age_sub_team
            curr_team_size += sub_team_size

        curr_team_avg_age = curr_team_age / curr_team_size
        self.max_avg_team_age = max(self.max_avg_team_age, curr_team_avg_age)
        return curr_team_avg_age, curr_team_size

    def get_max_avg_team_age(self, root):
        self.calc_max_avg_age_every_team(root)
        return self.max_avg_team_age

maa = MaxAvgAge()
print(maa.get_max_avg_team_age(node_20))



'''
Amazon OA 
Turnstile
'''

arr_time = [0,0,1,5]
directions = [0,1,1,0]
arr_time = [0,1,1,3,3]
directions= [0,1,0,0,1]
'''
arr_time =   [0,1,1,3,6,6,6,7,7]
directions = [0,1,0,0,0,1,1,1,0]
arr_time =   [0,1,1,3,6,6,8,8]
directions = [0,1,0,0,0,1,1,0]
arr_time =  [0,1,1,3,3,3,3,3,3,3,6,8]
directions= [0,1,0,0,1,1,1,1,1,1,1,1]
'''
pass_times = [False for i in range(len(arr_time))]
curr_time = 0
prev_direction_time = ('exit',-1)
entry_line = [] # [0, 5]
exit_line = [] # [0, 1]

for index, time in enumerate(arr_time):
    if directions[index] == 0:
        entry_line.append((time, index))
    else:
        exit_line.append((time, index))

entry_line.reverse()
exit_line.reverse()

while(entry_line and exit_line):
    if entry_line[-1][0] <= curr_time and exit_line[-1][0] <= curr_time:
        if prev_direction_time[0] == 'exit':
            time, cust_ind = exit_line.pop()
        else:
            time, cust_ind = entry_line.pop()

        pass_times[cust_ind] = curr_time
        prev_direction_time = (prev_direction_time[0], curr_time)
        curr_time += 1

    elif entry_line[-1][0] == exit_line[-1][0]:
        prev_direction, last_used = prev_direction_time
        curr_time = arrival_time = entry_line[-1][0]

        if last_used == curr_time - 1: pass
        else: prev_direction = 'exit'

        if prev_direction == 'exit':
            arrival_time, cust_ind = exit_line.pop()
        else:
            arrival_time, cust_ind = entry_line.pop()

        prev_direction_time = (prev_direction, curr_time)
        pass_times[cust_ind] = arrival_time
        curr_time = curr_time + 1

    elif entry_line[-1][0] < exit_line[-1][0]:
        arrival_time, cust_ind = entry_line.pop()
        pass_times[cust_ind] = max(arrival_time, curr_time)
        prev_direction_time = ('enter', max(arrival_time, curr_time))
        curr_time = max(arrival_time, curr_time) + 1

    elif entry_line[-1][0] > exit_line[-1][0]:
        arrival_time, cust_ind = exit_line.pop()
        pass_times[cust_ind] = max(arrival_time, curr_time)
        prev_direction_time = ('exit', max(arrival_time, curr_time))
        curr_time = max(arrival_time, curr_time) + 1

final_line = entry_line + exit_line

while(final_line):
    arrival_time, cust_ind = final_line.pop()

    if arrival_time > curr_time:
        pass_times[cust_ind] = arrival_time
        curr_time = arrival_time + 1
    else:
        pass_times[cust_ind] = curr_time
        curr_time += 1

print(pass_times)
'''
Amazon OA
Max of min altitudes - matrix
https://leetcode.com/discuss/interview-question/383669/
'''


'''
Goldman - Shortest Path with wildcard
'''

class Node:
    def __init__(self, city_name):
        self.city_name = city_name
        self.neighbors = {}
        self.distance = float('+inf')


num_cities = 4
num_wild_cards = 1
city_name_node_mapping = {}
city_name_node_mapping[1] = Node(1)
home_city = city_name_node_mapping[1]
home_city.distance = 0
unvisited_cities = {}
available_paths = []

for road_str in roads:
    parts_of_str = road_str.split()
    src = int(parts_of_str[0])
    dst = int(parts_of_str[1])
    dist = int(parts_of_str[2])

    if src in city_name_node_mapping: src_node = city_name_node_mapping[src]
    else:
        src_node = Node(src)
        city_name_node_mapping[src] = src_node

    if dst in city_name_node_mapping: dst_node = city_name_node_mapping[dst]
    else:
        dst_node = Node(dst)
        city_name_node_mapping[dst] = dst_node

    src_node.neighbors[dst] = dist
    dst_node.neighbors[src] = dist
    unvisited_cities[src] = True
    unvisited_cities[dst] = True

unvisited_cities.pop(home_city)
unvisited_cities = unvisited_cities.keys()

def dfs(src_city, cost_list, dst_city):
    if src_city == dst_city:
        available_paths.append(cost_list)
        return

    src_city_node = city_name_node_mapping[city]
    for neighbor in src_city_node.neighbors:
        dfs(neighbor, cost_list + src_city_node[neighbor], dst_city)


while(unvisited_cities):
    available_paths = []
    dst_city = unvisited_cities.pop()
    dfs(1, [], dst_city)



'''
Goldman - OA
https://leetcode.com/discuss/interview-question/797932/Goldman-Sachs-OA-test-August-2020
https://leetcode.com/problems/height-checker/discuss/300472/Java-0ms-O(n)-solution-no-need-to-sort
'''
heights = [1,1,4,2,1,3]
height_freq = [0] * 100
curr_height = 0
res = 0

for h in heights:
    height_freq[h] += 1

for i in range(len(heights)):

    while(height_freq[curr_height] == 0):
        curr_height += 1

    if curr_height != heights[i]:
        res += 1

    height_freq[curr_height] -= 1

print(res)


'''
547 Friend Circle
https://leetcode.com/problems/friend-circles/
Goldman - OA
'''


class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:
        unvisited_students = {}
        num_students = len(M)
        groups = 0

        def do_dfs(student):
            unvisited_students.pop(student)
            student_connections = M[student]

            for curr_stu in range(num_students):
                if student_connections[curr_stu] == 1 and curr_stu in unvisited_students:
                    do_dfs(curr_stu)

        for student in range(num_students):
            unvisited_students[student] = True

        for student in range(num_students):
            if student in unvisited_students:
                groups += 1
                do_dfs(student)

        return groups


'''
1163: Last substring in Lexicographical order
https://leetcode.com/problems/last-substring-in-lexicographical-order/
https://leetcode.com/problems/last-substring-in-lexicographical-order/discuss/363662/Short-python-code-O(n)-time-and-O(1)-space-with-proof-and-visualization
Goldman - OA
'''

class Solution:
    def lastSubstring(self, s: str) -> str:
        i, j, k = 0, 1, 0
        n = len(s)
        while j + k < n:
            if s[i+k] == s[j+k]:
                k += 1
                continue
            elif s[i+k] > s[j+k]:
                j = j + k + 1
            else:
                i = max(i + k + 1, j)
                j = i + 1
            k = 0
        return s[i:]


'''
Dijkstra
'''

class Dijkstra:
    def __init__(self, num_vertices, adjacency_list):
        self.distances_from_source = [float('+inf')] * num_vertices
        self.visted_nodes = [False] * num_vertices
        self.num_visited_nodes = 0
        self.adjacency_list = adjacency_list

    def get_shortest(self):
        min_dist_so_far = float('+inf')
        min_dist_node = None

        for node_id, distance in enumerate(self.distances_from_source):
            if self.visted_nodes[node_id] == False and distance < min_dist_so_far:
                min_dist_so_far = distance
                min_dist_node = node_id

        return min_dist_node

    def dijkstra_shortest_dist_from_source(self):
        src_node_id = 0
        self.distances_from_source[src_node_id] = 0

        while(self.num_visited_nodes < len(self.adjacency_list)):
            curr_node = self.get_shortest()
            self.visted_nodes[curr_node] = True
            self.num_visited_nodes += 1

            for neighbor, connection_distance in enumerate(self.adjacency_list[curr_node]):
                if connection_distance != 0 and self.visted_nodes[neighbor] == False and \
                        self.distances_from_source[neighbor] > self.distances_from_source[curr_node] + connection_distance:
                    self.distances_from_source[neighbor] = self.distances_from_source[curr_node] + connection_distance

        return self.distances_from_source


adjacency_list = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
        [4, 0, 8, 0, 0, 0, 0, 11, 0],
        [0, 8, 0, 7, 0, 4, 0, 0, 2],
        [0, 0, 7, 0, 9, 14, 0, 0, 0],
        [0, 0, 0, 9, 0, 10, 0, 0, 0],
        [0, 0, 4, 14, 10, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 1, 6],
        [8, 11, 0, 0, 0, 0, 1, 0, 7],
        [0, 0, 2, 0, 0, 0, 6, 7, 0]
        ];
num_verices = len(adjacency_list)
d = Dijkstra(len(adjacency_list), adjacency_list)
print(d.dijkstra_shortest_dist_from_source())


'''
1041 - Robot bounded in circle
https://leetcode.com/problems/robot-bounded-in-circle/
Goldman - OA
Robot in circle
'''


class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        # north = 0, east = 1, south = 2, west = 3
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        # Initial position is in the center
        x = y = 0
        # facing north
        idx = 0

        for i in instructions:
            if i == "L":
                idx = (idx + 3) % 4
            elif i == "R":
                idx = (idx + 1) % 4
            else:
                x += directions[idx][0]
                y += directions[idx][1]

        # after one cycle:
        # robot returns into initial position
        # or robot doesn't face north
        return (x == 0 and y == 0) or idx != 0


'''
Amazon OA - Pagination
https://leetcode.com/discuss/interview-question/801590/amazon-oa2-pagination
'''
from collections import namedtuple

sort_parameter = 0
sort_order = 1
page_number = 1
num_items_per_page = 2
items = [
    ["item1", 10, 15],
    ["item2", 3, 4],
    ["item3", 17, 8]
]
tups_list = []

for item in items:
    if sort_parameter == 0:
        tup = (item[0], item[1], item[2])
    elif sort_parameter == 1:
        tup = (item[1], item[0], item[2])
    elif sort_parameter == 2:
        tup = (item[2], item[0], item[0])

    tups_list.append(tup)

if sort_order == 0:
    tups_list.sort()
else:
    tups_list.sort(reverse=True)

print(tups_list)
start_number_item = page_number * num_items_per_page
end_number = start_number_item + start_number_item
curr_item_num = 0
op_list = []

while(curr_item_num < len(tups_list) and curr_item_num < end_number):
    item = tups_list[curr_item_num]

    if curr_item_num >= start_number_item:
        if type(item[0]) == str: prod_name = item[0]
        else: prod_name = item[1]

        op_list.append(prod_name)

    curr_item_num += 1

print(op_list)

'''
Amazon - OA
Smallest Negative balance
'''

'''
1192 - Critical connections

https://leetcode.com/problems/critical-connections-in-a-network/

Amazon OA
'''
from collections import defaultdict

class FindCutVertices:
    def __init__(self, connection_list):
        self.connections_list = connection_list
        self.connections_dict = defaultdict(list)
        self.max_key = float('-inf')

        for con in self.connections_list:
            self.connections_dict[con[0]].append(con[1])
            self.connections_dict[con[1]].append(con[0])
            self.max_key = max(self.max_key, con[0], con[1])

        num_vertices = self.max_key + 1
        print(num_vertices)
        self.parent = [None] * num_vertices
        self.discovery_time = [None] * num_vertices
        self.lowest_possible_level = [None] * num_vertices
        self.time = 0
        self.parent[0] = -1
        self.aps = []
        self.critical_connections = []


    def dfs(self, u):
        self.discovery_time[u] = self.time
        self.lowest_possible_level[u] = self.time
        self.time += 1
        child = 0

        for adjacent_vertex in self.connections_dict[u]:
            if self.discovery_time[adjacent_vertex] == None:
                child += 1
                self.parent[adjacent_vertex] = u
                self.dfs(adjacent_vertex)
                self.lowest_possible_level[u] = min(self.lowest_possible_level[u], self.lowest_possible_level[adjacent_vertex])

                if self.parent[u] != None and self.lowest_possible_level[adjacent_vertex] >= self.discovery_time[u]:
                    self.aps.append(u)

                elif self.parent[u] == None and child > 1:
                    self.aps.append(u)

            elif adjacent_vertex != self.parent[u]:
                self.lowest_possible_level[u] = min(self.lowest_possible_level[u], self.discovery_time[adjacent_vertex])

        return self.aps

    def dfs_critical_connections(self, u):
        self.discovery_time[u] = self.time
        self.lowest_possible_level[u] = self.time
        self.time += 1
        child = 0

        for adjacent_vertex in self.connections_dict[u]:
            if self.discovery_time[adjacent_vertex] == None:
                child += 1
                self.parent[adjacent_vertex] = u
                self.dfs_critical_connections(adjacent_vertex)
                self.lowest_possible_level[u] = min(self.lowest_possible_level[u], self.lowest_possible_level[adjacent_vertex])

                if self.lowest_possible_level[adjacent_vertex] > self.discovery_time[u]:
                    self.critical_connections.append((u,adjacent_vertex))

            elif adjacent_vertex != self.parent[u]:
                self.lowest_possible_level[u] = min(self.lowest_possible_level[u], self.discovery_time[adjacent_vertex])

        return self.critical_connections


connection_list = [[0, 1], [0, 2], [1, 3], [2, 3], [2, 5], [5, 6], [3, 4]]
connection_list = [[0,1],[1,2],[2,0],[1,3]]
#connection_list = [[1,2],[1,3],[3,4],[1,4],[4,5]]
fcv = FindCutVertices(connection_list)
print(fcv.dfs(1)) # To find articulation points, pick any arbitrary point in the graph and run the algorithm
fcv = FindCutVertices(connection_list)
print(fcv.dfs_critical_connections(1)) # To find critical connections, pick any arbitrary point in the graph and run the algorithm


394. Decode String
https://leetcode.com/problems/decode-string/
'''
Given an encoded string, return its decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is 
guaranteed to be a positive integer.

You may assume that the input string is always valid; No extra white spaces, square brackets are well-formed, etc.

Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, 
there won't be input like 3a or 2[4].
'''

class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        op_str = ''

        for char in s:
            if char.isalpha() or char.isdigit() or char == '[':
                stack.append(char)
            else:
                aux_stack = []
                num_repeats_str = ''

                while (stack[-1].isdigit() == False):
                    aux_stack.append(stack.pop())

                while (stack and stack[-1].isdigit() == True):
                    num_repeats_str = stack.pop() + num_repeats_str

                num_repeats = int(num_repeats_str)
                aux_stack.pop()
                aux_stack.reverse()
                aux_stack = num_repeats * aux_stack
                stack.append(''.join(aux_stack))

        return ''.join(stack)

# Points to remember. DO not assume that the num of repeats is a single digit num. Remember you have to decode any substring and put them back into
# stack. eg: 3[a2[b]]. After decoding 2[b] as bb, you need to put bb into stack. stack becomes 3[abb] now

# time:
# space: O(n)

'''
528 Random Pick with Weight
https://leetcode.com/problems/random-pick-with-weight/
look at the diagram in solutions tab to understand better
'''
import random


class Solution:

    def __init__(self, w: List[int]):
        self.running_sum = 0
        self.running_sums_list = []

        for num in w:
            self.running_sum += num
            self.running_sums_list.append(self.running_sum)

    def pickIndex(self) -> int:
        random_search_num = self.running_sum * random.random()
        # print('\n', random_search_num)
        # print(self.running_sums_list)
        st = 0
        en = len(self.running_sums_list) - 1

        # for i, s in enumerate(self.running_sums_list):
        #    if s > random_search_num:
        #        print(i)
        #        break

        while (st < en):
            mid = (st + en) // 2

            if self.running_sums_list[mid] == random_search_num:
                return mid
            elif self.running_sums_list[mid] < random_search_num:
                st = mid + 1
            else:
                en = mid
        # print(st)
        return st

time: O(n)
space: O(n)

'''
843 - Guess the word
https://leetcode.com/problems/guess-the-word/discuss/133862/Random-Guess-and-Minimax-Guess-with-Comparison
'''
# haven't implemented the solution but understood the concept
