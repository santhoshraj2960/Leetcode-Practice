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
import heapq
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





















# ------------------------------------------------------------------- Array block Binary search block ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------- End of Array block Binary search block ------------------------------------------------------------------------------
















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

# --------------------------------------------------------------------------- End of Tree block ---------------------------------------------------------------------------------------------















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