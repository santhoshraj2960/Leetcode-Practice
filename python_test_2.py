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
https://leetcode.com/problems/word-break/submissions/

Given a non-empty string s and a dictionary wordDict containing a list of non-empty
words, determine if s can be segmented into a space-separated sequence of one or 
more dictionary words.

Note:

The same word in the dictionary may be reused multiple times in the segmentation.
You may assume the dictionary does not contain duplicate words.
Example 1:

Input: s = "leetcode", wordDict = ["leet", "code"]
Output: true

Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
Output: false

class Solution(object):
    def wordBreak(self, s, wordDict):

        word_dict = {}
        visited_dict = {}
        for word in wordDict:
            word_dict[word] = True
            
        queue = [s]
        
        while(queue):
            #print queue
            curr_string = queue[0]
            queue = queue[1:]
            
            if not curr_string: #This means that we have successfully reached the 
            #end of ip string using words in the ip dict
                return True
            
            for word in wordDict:
                #print 'word = ', word
                if curr_string.startswith(word):
                    substr = curr_string[len(word):]
                    #print 'substr = ', substr
                    
                    if substr in visited_dict:
                        continue
                    
                    visited_dict[substr] = True
                    queue.append(curr_string[len(word):])
                    #print 'queue at end = ', queue
        
        return False


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


LC 53. Maximum subarray
https://leetcode.com/problems/maximum-subarray/
[34, -50, 42, 14, -5, 86]

Given this input array, the output should be 137. The contiguous subarray with the
largest sum is [42, 14, -5, 86].

Your solution should run in linear time.

#this is an example for greedy algo problem
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        curr_max = nums[0]
        curr_sum = nums[0]
        
        for num in nums[1:]:
            if curr_sum + num < num:
                curr_sum = num
            else:
                curr_sum += num
            curr_max = max(curr_sum, curr_max)
        
        return curr_max
#have a look at divide and conquer sol. just for reference in the link below
https://leetcode.com/problems/maximum-subarray/solution/

#the following approach will fail if you have list of only -ve numbers. Use grredy
#approach as above
max_poss_value = float('-inf')
#      [121, 87, 137,95, 81, 86]
my_l = [34, -50, 42, 14, -5, 86]
#      [34, -16, 26, 40, 35, 121]

#      [111, 77, 127,85, 71, 76, -10]
my_l = [34, -50, 42, 14, -5, 86, -10]
#      [34, -16, 26, 40, 35, 121, 111]

max_sum = float('-inf')
curr_sum = 0
ind = 0
while(ind < len(my_l)):
    if my_l[ind] < 0:
        max_sum = max(curr_sum, max_sum)

    curr_sum += my_l[ind]
    if curr_sum < 0:
        curr_sum = 0
    ind += 1



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


55. Jump Game
Given an array of non-negative integers, you are initially positioned at the first
index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

Example 1:

Input: [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
Example 2:

Input: [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum
             jump length is 0, which makes it impossible to reach the last index.


#BFS Approach O(n^2)
class Solution(object):
    def canJump(self, nums):
        queue = [0]
        my_l = nums
        visited_dict = {}
        while(queue):
            ind = queue[0]
            queue = queue[1:]
            jump_val = my_l[ind]

            if ind == len(my_l) - 1:
                return True
                #print 'end reached'
                break

            for i in range(1, jump_val + 1):
                if ind + i == len(my_l) - 1:
                    return True
                if ind + i in visited_dict or ind + i >= len(my_l):
                    continue
                visited_dict[ind+i] = True
                queue.append(ind + i)

        return False

#DP approach: O(n^2)
ind = len(nums) - 1
good_bad_index_list = [False] * (len(nums) - 1)
#An index is 'G' if we can reach the end from that index, else 'B'

while(index >= 0):
    max_jump = index + nums[index]
    
    for i in range(0, max_jump + 1):
        next_jump_index = index + nums[i]
        if next_jump_index >= len(nums) - 1:
            good_bad_index_list[index] = 'G'
            break
        elif good_bad_index_list[next_jump_index] == 'G':
            good_bad_index_list[index] = 'G'
            break

    index -= 1


#Greedy Solution O(n)
ind = len(nums) - 1
good_bad_index_list = [False] * (len(nums) - 1)
#An index is 'G' if we can reach the end from that index, else 'B'
last_good_index = len(nums) - 1

while(index >= 0):
    max_jump = index + nums[index]
    if max_jump >= last_good_index:
        last_good_index = index
    index -= 1    

return last_good_index == 0 #ret True if last good index is 0 else False


56. Leetcode
merge intervals
https://leetcode.com/problems/merge-intervals/submissions/

class Solution(object):
    def merge(self, intervals):
        new_intervals = []
        for interval in intervals:
            if interval[0] > interval[1]:
                smaller_ele = interval[1]
                larger_ele = interval[0]
            else:
                smaller_ele = interval[0]
                larger_ele = interval[1]
            tuple = (smaller_ele, larger_ele)
            new_intervals.append(tuple)
        
        intervals = sorted(new_intervals) #We were never told that the elements in
        # the interval are in sorted order
        merged_intervals = []
        
        for curr_interval in intervals:
            if not merged_intervals:
                merged_intervals.append([curr_interval[0], curr_interval[1]])
            else:
                last_interval = merged_intervals[-1]
                if last_interval[1] >= curr_interval[0]:
                    if last_interval[1] < curr_interval[1]: #special calse 
                    #eg:[[1,4], [2,3]]
                        merged_intervals[-1][1] = curr_interval[1]
                    
                else:
                    merged_intervals.append([curr_interval[0], curr_interval[1]])
                    
        return merged_intervals


57. Insert intervals
https://leetcode.com/problems/insert-interval/
Given a set of non-overlapping intervals, insert a new interval into the 
intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start
times.

Example 1:

Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
Example 2:

Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].

int_start_dict = {} 

class Solution(object):
    def insert(self, intervals, newInterval):
        #int_start_dict = {}
        op_intervals = []
        st = newInterval[0]
        en = newInterval[1]

        for ind, interval in enumerate(intervals):
            interval_st = intervals[0]
            if interval_st > st and insert_pos == False:
                insert_pos = ind - 1
            if insert_pos != False and interval_st > en:
                end_pos = ind - 1

        if insert_pos == -1:
            insert_pos = 0
        if end_pos == -1:
            end_pos = 0

        merged_int_added_to_output = False
        for ind,interval in enumerate(intervals):
            if ind in range(insert_pos, end_pos + 1):
                if not merged_int_added_to_output:
                    op_intervals.append(merged_int)
                    merged_int_added_to_output = True
                continue

            op_intervals.append(interval)

        return op_intervals

time: O(n)
space: O(1)

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

#Top down approach
memo = {}

def callback(rem_steps):
    global memo
    if rem_steps == 0:
        return 1
    if rem_steps < 0:
        return 0

    if rem_steps in memo:
        return memo[rem_steps]

    memo[rem_steps] = callback(rem_steps - 1) + callback(rem_steps - 2)
    #had a confusion in the above line if should do 1 + cb(..) + cb(..)
    #Substituted small values for rem_steps such as 1,2,3 to validate correctness
    return memo[rem_steps]

callback(2)


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
rest of the list has to be shifted).
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


91. Decode Ways
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

#approach 1: But this is a bad approach with O(2 ^ n) running time
2,2,6
Think of the problem as_ just a permutation of the commas in between the numbers.
There are 2 commas.
2 _ 2 _ 6
  0   0 -> 226 Invalid comb
  0   1 -> 22,6 Valid
  1   0 -> 2,26 Valid
  1   1 -> 2,2,6 Valid

#approach 2:
#There are only 2 possibilities at each index. Either take that number alone or take that
#number and the number next to it
string = '226'
string = '12'
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


572. Subtree of Another tree (Amazon - Kevin youtube)
https://leetcode.com/problems/subtree-of-another-tree/

#We can also do this problem with inorder traversal. Doing inorder traversal of both the
#trees and storing the values in 2 different lists and checking if the list of subtree
#is a sublist of parent inorder list. In that case we need to store the None value of the
#leaf node's left and right child seperately. Otherwise we wont get correct result

class Node(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

p_root = Node(3)
c_root = Node(4)
four_node = Node(4)
five_node = Node(5)
one_node = Node(1)
two_node = Node(2)
ten_node = Node(10)
two_node.left = ten_node #example 2 on leetcode
four_node.left = one_node
four_node.right = two_node
p_root.left = four_node
p_root.right = five_node

c_one_node = Node(1)
c_two_node = Node(2)
c_root.left =c_one_node
c_root.right = c_two_node

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

time: O(nm) - worst case we might end up checking if its a subtree of parent tree for_
every node of the parent tree. If n is the number of nodes in the parent tree and m is 
the number of nodes in the child tree, we will check if m nodes in the sub tree occur
in the parent tree for each node in parent tree whose size is n

space = O(nm) - If both parent and child tree are left sided or right sided trees, the
recursion stack will hold calls of size nm for every node in parent

#explanation for space and time compl.
eg: parent can be left sided tree of only 2 value for a height of 100 and_ child tree
can be a left sided tree of only 2 value for a height of 99. 


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
#Questions to ask before coding
#What should I return when there is no such combinations possible?
#Will there be duplicates in the input?
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

        if curr_seq_string in words_dict and len(remamining_words) < len(words) - 1:
            possible_concatenations.append(curr_seq)

        for index, word in enumerate(remamining_words):
            new_remaining_words = remamining_words[:index] + remamining_words[index + 1:]
            queue.append((curr_seq + [word], new_remaining_words))


for index, word in enumerate(words):
    do_bfs(word, words[:index] + words[index + 1:])

print(possible_concatenations)
#time: O(n! * n). the while loop in do_bfs will execute O(n!) times
#space: O(n!) In each iteration of the while loop, we build a fresh "curr_seq+ 
#new_remaining_words" variable which is equal to size n
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