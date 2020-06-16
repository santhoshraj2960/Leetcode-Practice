
Learn all funs of itertools - very useful
Need to solve: (Have to solve more DPs)
Indexing in binary search related problems (bst from_ sorted_array)
all_comb of a string or 2 diff strings (with_ and without duplicates, etc.)
CTCI 8.1 - Triple step
form substrings of a given string from dict of words (incomplete)
Odd Even Jumps

Need more clarity:
Implement a bst from a sorted array (check dia in CTCI folder for more clarity)

Greedy problems to practice
Minimum Cost Tree From Leaf Values
Sum of Subarray Minimums
Online Stock Span
Score of Parentheses
Next Greater Element II
Next Greater Element I
Largest Rectangle in Histogram
Trapping Rain Water

Longest Increasing subseq
https://www.geeksforgeeks.org/longest-increasing-subsequence-dp-3/

Start to think through how you will be solving the problem before actually writing
the problem. Think about time and space complexities before writing down the prob.

Substrings of size K with K distin
s = "awaglknagawunagwkwagl", k = 4
Output: ["wagl", "aglk", "glkn", "lkna", "knag", "gawu", "awun", "wuna", "unag",
 "nagw", "agwk", "kwag"]

s = "awaglknagawunagwkwagl"
my_dict = {}
st_ind = 0
k = 4
op = []
for ind, item in enumerate(s):
    if ind - st_ind == k-1:
        print 'item =', item, 'my_dict = ', my_dict, 'ind = ', ind
        if item in my_dict and my_dict[item] < st_ind or not item in my_dict:
            op.append(s[st_ind: ind + 1])
            my_dict[item] = ind
            st_ind += 1
            print 'new_st_ind = ', st_ind
            continue
        
    if item in my_dict and my_dict[item] >= st_ind:
        st_ind = my_dict[item] + 1
        my_dict[item] = ind
        continue

    my_dict[item] = ind


Target value in a sorted matrix.
mat = [[10, 20, 30, 40],
       [15, 25, 35, 45],
       [27, 29, 37, 48],
       [32, 33, 39, 50]]
target = 29

max_r = len(mat)
max_c = len(mat[0])
row = 0
while(row < max_r and max_c > -1):
    top_right = mat[row][max_c-1]
    if target == top_right:
        print 'elem found at ', row, max_c-1
        break
    if target > top_right and row < max_r:
        row += 1
        continue
    if target < top_right and max_c > 0:
        max_c -= 1
        continue


Product Recommendations
https://leetcode.com/discuss/interview-question/414085/
numProducts = 5
repository = ["mobile", "mouse", "moneypot", "monitor", "mousepad"]
customerQuery = "mouse"
latest_res = []
all_res = []

for ind, item in enumerate(customerQuery):
    if ind < 2:
        continue
    latest_res = get_search_res(customerQuery[0:ind], latest_res)
    print latest_res
    all_res.append(latest_res)

def get_search_res(query, latest_res):
    res_list = []
    if latest_res:
        search_list = latest_res
    else:
        search_list = repository
    for item in search_list:
        if item.startswith(query):
            res_list.append(item)
    if len(res_list) > 3:
        res_list = sorted(res_list)[:3]
    return res_list


zombies in matrix
https://leetcode.com/discuss/interview-question/411357/

Given a 2D grid, each cell is either a zombie 1 or a human 0. Zombies can turn 
adjacent (up/down/left/right) human beings into zombies every hour. Find out how 
many hours does it take to infect all humans?

[[0, 1, 1, 0, 1],
 [0, 1, 0, 1, 0],
 [0, 0, 0, 0, 1],
 [0, 1, 0, 0, 0]]

zero_val_grids = []
num_rows = len(mat)
num_cols = len(mat[0])
new_mat = [[0]*num_cols for i in range(num_rows)]
for row in range(len(mat)):
    for col in range(len(mat[0])):
        if mat[row][col] == 1:
            continue
        else:
            get_time(row, col)

def get_time(row, col):
    time = 0
    my_list = [(row, col, time)]
    while(my_list):
        first_ele = my_list[0]
        print first_ele
        my_list = my_list[1:]
        row = first_ele[0]
        col = first_ele[1]
        time = first_ele[2]
        if mat[row][col] == 1:
            return time
        else:
            if row + 1 < num_rows:
                my_list.append((row+1, col, time+1))
            if col + 1 < num_cols:
                my_list.append((row, col+1, time+1))
            if row - 1 >= 0:
                my_list.append((row-1, col, time+1))
            if col - 1 >= 0:
                my_list.append((row, col-1, time+1))
        print my_list


form substrings of a given string from dict of words (incomplete)
#incomplete solution. I don't think you can use memo here
s = "catsanddog"
word_list = ["cat", "cats", "and", "sand", "dog"]
Output:
[
  "cats and dog",
  "cat sand dog"
]
def callback(word_list, s, sentance):
    if not s:
        return sentance
    
    for ind, ele in enumerate(s):
        if s[0:ind] in word_list:
            if s[:ind] in memo:
                return memo[s[:ind]]
            else:
                memo[s[:ind]] = callback(word_list, s[ind+1:], 
                    sentance + ' ' + s[:ind])


Tressure Island
mat = [
 ['O', 'O', 'O', 'O'],
 ['D', 'O', 'D', 'O'],
 ['O', 'O', 'O', 'O'],
 ['X', 'D', 'D', 'O']
]

queue = []
#entry in queue = (row, col, dist_covered_so_far)
row = 0
col = 0
queue.append((row, col, 0))
num_rows = len(mat)
num_cols = len(mat[0])
visited = [[row,col]] #you might get a question. What if I visit a particular 
#vertex second time from a different node and it happens to be the shortest path
#to the tressure. 
#Ans is it can't happen say we have a matrix like this
#o o  o  o o
#o o "o" D o
#o o  o  o X
# you can visit row2 col3 through following ways
#1) 00 -> 01 -> 02 -> 12 dist = 3
#2) 00 -> 01 -> 10 -> 12 dist = 3
#3) 00 -> 10 -> 11 -> 12 dist = 3
#4) 00 -> 01 -> 02 -> 03 -> 13 ->  12 dist = 5
#As you can see, the first time you visit a node through BFS is always the best 
#route when using BFS. The next avail routes could equal the best route but it 
#cannot beat the best possible route.
#It doesn't matter if you reach "12" by the ways (1 or 2 or 3). You anyways have
#reached "12" using the best avail route (i.e. route 1). So all that matters now
# is how you can reach the tressure (X) using best possible route from "12"

while(queue):
    print 'queue = ', queue
    element = queue[0]
    row = element[0]
    col = element[1]
    dist_covered_so_far = element[2]
    queue = queue[1:]
    variants = [[0,1], [1,0], [-1,0], [0,-1]]

    curr_row = row
    curr_col = col
    for var in variants:
        row = curr_row + var[0]
        col = curr_col + var[1]
        if [row, col] in visited:
            continue
        visited.append([row,col])
        if row <= -1 or row >= num_rows:
            continue
        if col <= -1 or col >= num_cols:
            continue
        if mat[row][col] == 'O':
            queue.append((row, col, dist_covered_so_far + 1))
        elif mat[row][col] == 'X':
            print "distance = ", dist_covered_so_far + 1
            queue = [] #setting queue to empty so that the outer while loop exits
            break


Copy List with Random Pointer
A linked list is given such that each node contains an additional random pointer 
which could point to any node in the list or null.

Return a deep copy of the list.

Example 1:

Input:
{"$id":"1","next":{"$id":"2","next":null,"random":{"$ref":"2"},"val":2},"random":{"$ref":"2"},"val":1}

Explanation:
Node 1 s value is 1, both of its next and random pointer points to Node 2.
Node 2 s value is 2, its next pointer points to null and its random pointer points
to itself. You can assume that all the values in the list are unique
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, next, random):
        self.val = val
        self.next = next
        self.random = random
"""
class Solution(object):
    
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        created_node_dict = {}
        
        def get_cloned_node(node):
            # If node exists then
            if node:
                # Check if its in the visited dictionary          
                if node in created_node_dict:
                    # If its in the visited dictionary then return the new node 
                    #reference from the dictionary
                    return created_node_dict[node]
                else:
                    # Otherwise create a new node, save the reference in the 
                    #visited dictionary and return it.
                    created_node_dict[node] = Node(node.val, None, None)
                    return created_node_dict[node]
            return None
        
        if not head:
            return head
        
        node = head
        new_node = Node(node.val, None, None)
        created_node_dict[node] = new_node
        
        while(node):
            new_node = get_cloned_node(node)
            new_node.next = get_cloned_node(node.next)
            new_node.random = get_cloned_node(node.random)
            
            node = node.next


Longest Plaindromic substring #aaaccc aabaa shjks
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        max_string = ''
        def check_pal(st, end, s):
            while(st >= 0 and end < len(s) and s[st] == s[end]):
                st -=1
                end += 1
            
            return s[st+1: end]
        
        if not s:
            return ""
        if len(s) == 1:
            return s
        
        for ind, ele in enumerate(s[:-1]):
            #if ind == 0:
            #    continue
            string = check_pal(ind, ind+1, s)
            #print string
            if len(string) > len(max_string):
                max_string = string
            string = check_pal(ind-1, ind+1,s)
            #print string
            if len(string) > len(max_string):
                max_string = string
            
        return max_string


parition labels:
https://leetcode.com/problems/partition-labels/
Input: S = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits 
S into less parts.

my_string = "ababcbacadefegdehijhklij"
max_pos_dict = {}
current_chars_dict = {}
substring = ''
substrings_list= []

for index, character in enumerate(my_string):
    max_pos_dict[character] = index

current_chars_dict[my_string[0]] = True
max_index_of_char_in_string = max_pos_dict[my_string[0]]

for index, character in enumerate(my_string):
    if character not in current_chars_dict and index > max_index_of_char_in_string:
        substrings_list.append(len(substring))
        substring = character
        current_chars_dict = {}
        current_chars_dict[character] = index
        max_index_of_char_in_string = max_pos_dict[character]
        continue
    else:
        substring += character
        if max_pos_dict[character] > max_index_of_char_in_string:
            max_index_of_char_in_string = max_pos_dict[character]

substrings_list.append(len(substring))


from collections import OrderedDict
class Solution:
    def partitionLabels(self, S: str) -> List[int]:
        my_dict = OrderedDict()
        for ind, char in enumerate(S):
            try:
                my_dict[char].append(ind)
            except:
                my_dict[char] = [ind]
            
        start_ind = 0
        end_ind = my_dict[S[0]][-1]
        part = []
        part_value = 1
        for key in S[1:]:
            if my_dict[key][0] > start_ind and my_dict[key][0] < end_ind and my_dict[key][-1] > end_ind:
                end_ind = my_dict[key][-1]
            elif my_dict[key][0] > end_ind:
                part.append(part_value)
                start_ind = my_dict[key][0]
                end_ind = my_dict[key][-1]
                part_value = 0
            
            part_value += 1
        part.append(part_value)
        return part


542. 01 Matrix

Input:
[[0,0,0],
 [0,1,0],
 [1,1,1]]

Output:
[[0,0,0],
 [0,1,0],
 [1,2,1]]

mat = [
[0,0,0],
[0,1,0],
[1,1,1],
]

max_dist_to_zero = 0
for row in range(len(mat)):
    for col in range(len(mat[0])):
        if mat[row][col] == 0:
            continue
        else:
            dist_to_zero = shortest_dist_to_zero(row, col, mat)
            max_dist_to_zero = max(max_dist_to_zero, dist_to_zero)

def shortest_dist_to_zero(row, col, mat):
    available_paths = [[0,1], [0,-1], [1,0], [-1,0]]
    queue = []
    dist = 0
    st = (row, col, dist)
    num_rows = len(mat)
    num_cols = len(mat[0])
    queue.append(st)
    while(queue):
        vertex_dist = queue[0]
        queue = queue[1:]
        row = vertex_dist[0]
        col = vertex_dist[1]
        dist = vertex_dist[2]
        if mat[row][col] == 0:
            return dist

        for path in available_paths:
            new_row = row + path[0]
            new_col = col + path[1]
            if (new_row < 0 or new_row >= num_rows or new_col < 0 
            or new_col >= num_cols):
                continue
            else:
                queue.append((new_row, new_col, dist+1))


root_s = s
root_t = t
visited = []
root_node_found = False
def cb(node):
    if not node or root_node_found:
        return
    else:
        if node == root_t:
            root_node_found = node
            return
        if not node in visited:
            visited.append(node)
            cb(node.left)
            if root_node_found:
                return
            cb(node.right)

Subtree of another sub tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
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
        print main_tree
        print sub_tree
        # have to check if subtree is a sublist of main tree
        # we can't check if a list is a sublist of another like we check for strings.
        # So we have to convert the list into a string and then compare both the lists
        main_tree = "_".join(str(i) for i in main_tree)
        main_tree = '_' + main_tree
        #Doing the following 2 steps for edge cases when maintree = [12], subtree = [2]
        '''
        main_tree = [12, 'Null', 'Null']
        sub_tree = [2, 'Null', 'Null']
        str v of main_tree = _12_Null_Null
        str v of sub_tree = _2_Null_Null
        '''
        sub_tree = "_".join(str(i) for i in sub_tree)
        sub_tree = '_' + sub_tree
        #print main_tree
        #print sub_tree
        return sub_tree in main_tree


Merge 2 sorted lists
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        prev_node = None
        head = None
        while(l1 or l2):
            if (l1 and l2):
                if l1.val < l2.val:
                    new_node = ListNode(l1.val)
                    l1 = l1.next
                else:
                    new_node = ListNode(l2.val)
                    l2 = l2.next
            elif not l2:
                new_node = ListNode(l1.val)
                l1 = l1.next
            elif not l1:
                new_node = ListNode(l2.val)
                l2 = l2.next
                
            if prev_node:
                prev_node.next = new_node
                prev_node = new_node
            else:
                head = new_node
                prev_node = new_node
                
        #node = head
        return head


https://leetcode.com/problems/path-with-maximum-minimum-value/

path with max score. First cell and last cell are excluded
mat = [
[6, 7, 8],
[5, 4, 2],
[8, 7, 6],
]

6, 7,  8
5, 4, "5" 
8, 7,  6

#let's take the above matrix try to explain the solution.
#all avail paths to reach last cell 6 are:
#1) 7 -> 8 -> 5  min_val_along_path = 5
#2) 7 -> 4 -> 5  min_val_along_path = 4
#3) 5 -> 4 -> 5  min_val_along_path = 4
#4) 5 -> 8 -> 7  min_val_along_path = 5
# Now the ans is the max of all the above min values i.e. 5.
#Let's take the "5" in the above matrix in pos [1][2]. 
# 5 can be reached either through 4 or 8. So, the min of paths going through 5 will
# be either 5 or the min value found so far along the paths (7 -> 8 or 7 -> 4 or
# 5 -> 4) which will be (5 or 4 or 4). But we also need to remember that the ques
# asks us to find the max of the  ^ min path values. So, we need the max of the vals
#(5 or 4 or 4) that we have found | above. So, the apt value at [1][2] is max(5,4,4)
#which is 5. 
#So, the formula we need to use is 
# cell_value = max(min(cell_value, val of cell above it), 
#                 (min(cell_value, val of cell left of it))
#i.e val of cell above it = [cell_row - 1][cell_col]
#    val of cell left of it = [cell_row][cell_col - 1]
#Note. Corner cases not mentioned above
mat = [
[6, 7, 8],
[5, 4, 2],
[8, 7, 6],
]
for row in range(len(mat)):
    for col in range(len(mat[0])):
        if ((row == 0 and col == 0) or (row == 1 and col == 0) or 
        (row == 0 and col ==1)):
            continue
        cell_val = get_val_for_cell(row, col)
        print row,col
        print cell_val
        mat[row][col] = cell_val

print mat


def get_val_for_cell(row, col):
    curr_val = mat[row][col]
    num_rows = len(mat)
    num_cols = len(mat[0])
    new_row = row - 1
    new_col = col - 1
    if row == num_rows - 1 and col == num_cols - 1:
        return max(mat[row-1][col], mat[row][col-1])
    elif row == 0:
        return min(curr_val, mat[row][col-1])
    elif col == 0:
        return min(curr_val, mat[row-1][col])
    else:
        return max(min(curr_val, mat[row][col-1]), 
            min(curr_val, mat[row-1][col]))


Favourite Geners:
Output: {  
   "David": ["Rock", "Techno"],
   "Emma":  ["Pop"]
}

userSongs = {  
   "David": ["song1", "song2", "song3", "song4", "song8"],
   "Emma":  ["song5", "song6", "song7"]
}

songGenres = {  
   "Rock":    ["song1", "song3"],
   "Dubstep": ["song7"],
   "Techno":  ["song2", "song4"],
   "Pop":     ["song5", "song6"],
   "Jazz":    ["song8", "song9"]
}


song_genre_rev = {}
for genre in songGenres:
    for song in songGenres[genre]:
        song_genre_rev[song] = genre #{'song1': 'Rock', ...}

user_fav_genre_dict = userSongs.fromkeys(userSongs, []) #{"David":[], "Emma":[]}

for user in userSongs:
    user_song_genre = songGenres.fromkeys(songGenres, 0) #{"Genre": 0}
    max_genre_count = 0
    user_fav_genre = []
    for song in userSongs[user]:
        genre = song_genre_rev[song]
        if genre in user_song_genre:
            user_song_genre[genre] += 1
        else:
            user_song_genre[genre] = 1

        if user_song_genre[genre] > max_genre_count:
            user_fav_genre = [genre]
            max_genre_count = user_song_genre[genre]
        elif user_song_genre[genre] == max_genre_count:
            user_fav_genre.append(genre)

    user_fav_genre_dict[user] = set(user_fav_genre)

print user_fav_genre_dict

Spiral matrix II
Input: 3
Output:
[
 [ 1, 2, 3 ],
 [ 8, 9, 4 ],
 [ 7, 6, 5 ]
]
use reference:
00 01 02 03
10 11 12 13
20 21 22 23
30 31 32 33

new_mat = [[False]*n for i in range(n)]
elem_to_fill = [i for i in range(1, (n*n)+1)]
r = 0 
c = 0
all_combs = [[0,1], [1,0], [0,-1], [-1,0]]
all_comb_index = 0
end = n
st = -1
while(elem_to_fill):
    #print 'elem_to_fill = ', elem_to_fill
    #print 'row , col = ', r, c
    #print 'st = ', st
    #print 'en = ', end
    #print 'new_mat = ', new_mat
    if (r == c and new_mat[r][c] != False): # moving inside mat from 00 -> 11 -> 22
        r = r + 1
        c = c + 1
        end -= 1
        st += 1
        new_mat[r][c] = elem_to_fill[0]
        all_comb_index = 0
        #print 'in else r, c = ', r, c
    else:
        new_mat[r][c] = elem_to_fill[0]
        
    if r + all_combs[all_comb_index][0] == end or r + all_combs[all_comb_index][0] \
     <= st:
        all_comb_index += 1
    if c + all_combs[all_comb_index][1] == end or c + all_combs[all_comb_index][1] \
     <= st:
        all_comb_index += 1

    r += all_combs[all_comb_index][0]
    c += all_combs[all_comb_index][1]
    elem_to_fill = elem_to_fill[1:]

print new_mat


Fibo using memo

memo = {}
def fibo(n, memo):
    if n == 0: return 0
    if n == 1: return 1

    if n in memo:
        return memo[n]
    else:
        print n
        memo[n] = fibo(n-1, memo) + fibo(n-2, memo)

    return memo[n]


CTCI 8.1 - Triple step

n_steps = {}
n_steps[1] = 1
n_steps[2] = 2
n_steps[3] = 4 (111,21,12,3)

n_steps[0] = 1 #We are setting this to 1 (instead of 0) to account for the first
#step we take eg: n_steps[4] = n_steps[4-1] + n_steps[4-2] + n_steps[4-3]
#n_steps[3] = 4 but to reach n_steps[3] we have taken a step. And thats why we set
#n_steps[0] = 1
total_steps = 10

while steps < total_steps:
    n_steps[steps] = n_steps[steps - 1]  + n_steps[steps - 2] + n_steps[steps - 3]
    steps += 1

****Frame sol using matrix DP - see your notebook for detailed sol****
Not using memozation (In efficient approach - still calc valu for rec. calls we have
    already calculated)

class poss_ways(object):
    poss_hops = [1,2,3]
    num_of_ways = 1
    def num_ways(self, n, routes):
        print n
        print self.num_of_ways
        if n == 0:
            self.num_of_ways += 1
            print 'num_of_ways = ', self.num_of_ways
            print 'routes = ', routes
            return
        elif n < 0:
            return False

        for i in poss_hops:
            print 'i = ', i
            is_hop_possible = self.num_ways(n - i, routes+[i])
            if not is_hop_possible:
                break
        return self.num_of_ways

p = poss_ways()
p.num_ways(3, [])


Unique twitter id sets 
https://leetcode.com/discuss/interview-question/376581/Twitter-or-OA-2019-or-Unique-Twitter-User-Id-Set
Input: [3,2,1,2,7]
op = [3,2,1,4,7]

max_val = my_l[0]
min_val = my_l[0]
my_dict = {}
dup = []
avail_keys = []
op_list = []
for ind, item in enumerate(my_l):
    try:
        my_dict[item].append(ind)
        dup.append(item)
    except:
        my_dict[item] = [ind]

    if item < min_val:
        min_val = item
    if item > max_val:
        max_val = item

for num in range(min_val, max_val + 1):
    if num in my_dict:
        continue
    avail_keys.append(num)

for item in dup:
    for occurance in range(len(my_dict[item]) - 1):
        if avail_keys:
            op_list.append(avail_keys[0])
            avail_keys = avail_keys[1:]
        else:
            op_list.append(op_list[-1] + 1)


Worker bike pairs 
https://leetcode.com/problems/campus-bikes/
distances = []     # distances[worker] is tuple of (distance, worker, bike) for 
#each worker_bike pair
for i, (x, y) in enumerate(workers):
    distances.append([])
    for j, (x_b, y_b) in enumerate(bikes):
        distance = abs(x - x_b) + abs(y - y_b)
        distances[-1].append((distance, i, j))#for each worker we are calculating
        #the distance to each bike and finally we are sorting by distance(sorting 
        #takes place below)
    distances[-1].sort(reverse = True)  # reverse so we can pop the smallest 
    #distance
#The problem is not fully solved.....
alloted_bikes_dict = {}
final_allotment = []

for dist in distances:
    all_bikes_closed_to_worker = dist
    for dist_worker_bike_tuple in all_bikes_closed_to_worker:
        worker = dist_worker_bike_tuple[1]
        bike = dist_worker_bike_tuple[2]
        if bike not in alloted_bikes_dict:
            alloted_bikes_dict[bike] = True
            final_allotment.append((worker, bike))
            break


Power Set - Generate all subsets of given set #there can be a lot of variations 
#of this problem
my_l = [1,2,3]
k = len(my_l)

def cb(my_l, length, op):
    if len(op) == length:
        print op
        return
    for ind, elem in enumerate(my_l):
        cb(my_l[ind + 1:], length, op + [elem])

for i in range(1, k):
    cb(my_l,i, [])


Implement a bst from a sorted array (check dia in CTCI folder for more clarity)
visited_nodes = []
def cb(st, en):
    mid = (st + en) / 2
    if st > en or mid in visited_nodes:
        return
    visited_nodes.append(mid)
    node = NewNode(mid)
    node.l = cb(st, mid - 1)
    node.r = cb(mid + 1, end)
    return node


Check if binary tree is balanced (check dia in CTCI folder for more clarity)

def get_height(node, h):
    if not node: return h

    left_h = get_height(node.l, h + 1)
    right_h = get_height(node.r, h + 1)

    if abs(left_h - right_h) > 1:
        print 'Tree unbalanced'

    height_of_node_in_tree = max(left_h, right_h)
    return height_of_node_in_tree


Discounted Price
Refer to images_python_test folder for question
https://leetcode.com/discuss/interview-question/378221/Twitter-or-OA-2019-or-Final-Discounted-Price
a = [9,7,3,3,4,5,8,4,1]
prices = [4, 6, 5, 4] #NOT OUTPUT - dont get confused
stack = []
temp = prices
for i in range(len(prices)):
    while len(stack) > 0 and prices[stack[-1]] >= prices[i]:
        poped = stack.pop()
        temp[poped] = prices[poped] - prices[i]
    stack.append(i)


Word Ladder
Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

Output: 5

Explanation: As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" ->\
"cog",
return its length 5.
Example 2:

Input:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

Output: 0

Explanation: The endWord "cog" is not in wordList, therefore no possible transformation.

if end_word not in word_list:
    return 0 #transformation not possible

q = []
q.append((begin_word, 0)) #word, num of transformation done so far
while(q):
    word_trans = q.pop()
    word = word_trans[0]
    num_trans = word_trans[1]

    if word == end_word:
        print 'num of trans needed = ', num_trans
    
    word_list.remove(word)
    my_dict = {}
    
    for char in word:
        try:
            my_dict[char] += 1
        except:
            my_dict[char] = 1

    for word in word_list:
        total_diff = 0
        word_cannot_be_counted = False
        for char in word:
            if char in my_dict and my_dict[char] > 0:
                my_dict[char] -= 1
            else:
                total_diff += 1

            if total_diff > 1:
                word_cannot_be_counted = True
                break
        
        if word_cannot_be_counted:
            continue
        else:
            q.append((word, num_trans + 1))


String transforms to another string 
https://leetcode.com/problems/string-transforms-into-another-string/discuss/355412/Python-simple-O(n)-with-explanation

class Solution:
    def canConvert(self, str1: str, str2: str) -> bool:
        if str1 == str2:
            return True
        m = {}
        for i in range(len(str1)):
            if str1[i] not in m:
                m[str1[i]] = str2[i]
            elif m[str1[i]] != str2[i]:
                return False
        return len(set(str2)) < 26 #Next, we check the number of unique characters 
        #in str2. If all 26 characters are represented, there are no characters 
        #available to use for temporary conversions, and the transformation is 
        #impossible. The only exception to this is if str1 is equal to str2, so we 
        #handle this case at the start of the function.

Binary tree with max path: # do in order trav and store the res in a list
# at every point where you have a -ve number, cut the list into 2 new lists
# Which ever partition has the max value, thats the ans

Count compete tree node:#go down the left sub tree n.l->n.l->n.l to find height
# now go down the path n.r->n.l to find the first path where h is not equal to the
# height we found in prev step. And untill you get this height, reduce the expected
# num of nodes by 1 in your base cond
https://leetcode.com/problems/count-complete-tree-nodes/

n = root
h = 0
h2 = 0

def cb_1(n, h):
    if not n:
        return
    cb(n.l, h + 1)

num_of_nodes_calculated = False
def cb_2(n, h2):
    if not n:
        if not h2 == h:
            num_nodes -= 1
            return
        
        if h2 == h:
            print 'num_nodes = ', num_nodes
            num_of_nodes_calculated = True
            return 
        
        if num_of_nodes_calculated:
            return

        ret_1 = cb_2(n.r, h2 + 1)
        ret_2 = cb_2(n.l, h2 + 1)

        if !(ret_1 and ret_2):
            return


cb(n, h)
num_nodes = (2 ** h) - 1
cb_2(n, h2)

knapsack problem: https://www.youtube.com/watch?v=8LusJS5-AGo
https://www.geeksforgeeks.org/0-1-knapsack-problem-dp-10/

Longest Increasing Path # have to write code

Odd Even Jumps: # have to write code

Activate Fountain

a = [1,2,3]

my_dict = {}
for i in range(1, len(a) + 1):
    my_dict[i] = -1



https://leetcode.com/discuss/interview-question/352458/
str_a = 'dcab'
str_b = 'aaa'


my_dict_b = {}

def get_smallest_char_count(str_a):
    my_dict_a = {}
    smallest_a = None

    for char in str_a:
        if not smallest_a:
            smallest_a = char
            my_dict_a[char] = 1
            continue
        elif char < smallest_a:
            smallest_a = char

        if char in my_dict_a:
            my_dict_a[char] += 1
        else:
            my_dict_a[char] = 1

    return my_dict_a[smallest_a]

#do the above for str_b and 


Odd Even Jumps:
https://leetcode.com/problems/odd-even-jump/discuss/217981/JavaC%2B%2BPython-DP-idea-Using-TreeMap-or-Stack

A = [5, 1, 3, 4, 2]
n = len(A)
next_higher, next_lower = [0] * n, [0] * n #[0,0,0,0,0]
stack = []
for a, i in sorted([a, i] for i, a in enumerate(A)):#[(1,1),(2,4),(3,2),(4,3),(5,1)]
    while stack and stack[-1] < i:
        next_higher[stack.pop()] = i #storing the index of the immediate next 
        #greater element
    stack.append(i)
# next_higher = [0, 4, 3, 0, 0]

stack = []
for a, i in sorted([-a, i] for i, a in enumerate(A)):
    while stack and stack[-1] < i:
        next_lower[stack.pop()] = i #storing the index of the immediate next lesser
        #element
    stack.append(i)

# next_higher = [0, 4, 3, 0, 0]
# next_lower = [3, 0, 4, 4, 0]
higher, lower = [0] * n, [0] * n
higher[-1] = lower[-1] = 1 #[0,0,0,0,1]

for i in range(n - 1)[::-1]:#[3,2,1,0]
    higher[i] = lower[next_higher[i]]#after 1st iter, [0,0,0,0,1]
    lower[i] = higher[next_lower[i]]#after 1st iter, [0,0,0,1,1]
return sum(higher)



Max Profits
https://www.algoexpert.io/questions/Max%20Profit%20With%20K%20Transactions

You are given an array of integers representing the prices of a single stock on 
various days (each index in the array represents a different day). You are also 
given an integer k, which represents the number of transactions you are allowed to
make. One transaction consists of buying the stock on a given day and selling it on
another, later day. Write a function that returns the maximum profit that you can 
make buying and selling the stock, given k transactions. Note that you can only 
hold 1 share of the stock at a time; in other words, you cannot buy more than 1 
share of the stock on any given day, and you cannot buy a share of the stock if 
you are still holding another share. Note that you also dont need to use all k 
transactions that you are allowed.

Sample input: [5, 11, 3, 50, 60, 90], 2
Sample output: 93 (Buy: 5, Sell: 11; Buy: 3, Sell: 90)

prices = [3,2,5,7,1,3,7]
k = 1

curr_min = prices[0]
curr_max = prices[0]
profit = []
total_profit = 0

for price in prices:
    if price < curr_min:
        profit.append(curr_max - curr_min)
        curr_min = price
        curr_max = price
    else:
        curr_max = max(curr_max, price)

profit.append(curr_max-curr_min)

print 'profit_list = ', profit
heapq.heapify_max(profit)
while(k > 0 and profit):
    total_profit += heapq.heappop(profit)
    k -= 1

print total_profit


Arithmetic Binary Tree (DIP)
Hi, here is your problem today. This problem was recently asked by Apple:

You are given a binary tree representation of an arithmetic expression. In this 
tree, each leaf is an integer value,, and a non-leaf node is one of the four 
operations: '+', '-', '*', or '/'.

Write a function that takes this tree and evaluates the expression.

Example:

#    *
#   / \
#  +    +
# / \  / \
#3  2  4  5

This is a representation of the expression (3 + 2) * (4 + 5), and should return 45.

- Use post order traversal and store elements in a list. Then do post order
  evaluation using stack


 Full Binary Tree(DIP)
 Hi, here is your problem today. This problem was recently asked by Google:

Given a binary tree, remove the nodes in which there is only 1 child, so that the 
binary tree is a full binary tree.

So leaf nodes with no children should be kept, and nodes _with 2 children should be
kept _as well.

- Do a BFS traversal and see if a node has just 1 child and ret that node 
  and its parent and remove that edge.

def remove_single_child_node(node):
    if not node:
        return None

    root.left = remove_single_child_node(node.left)
    root.right = remove_single_child_node(node.right)

    if (not node.left) or (not node.right):
        return None

Longest Increasing Subsequence (DIP) - DynProg, have to learn

Room scheduling (DIP):
Hi, here is your problem today. This problem was recently asked by Google:

You are given an array of tuples (start, end) representing time intervals for 
lectures. The intervals may be overlapping. Return the number of rooms that are 
required.

For example. [(30, 75), (0, 50), (60, 150)] should return 2.
 st_times = sorted(st_times) # [0,30,60]
 en_times = sorted(en_times) # [50,75,90]
 max_rooms = 0 
 curr_rooms_needed = 0
 en_time_pointer = 0
 for st in st_times:
    if st > en_times[en_time_pointer]: 
        curr_rooms_needed += 1
        max_rooms = max(max_rooms, curr_rooms_needed)
    elif st <= en_times[en_time_pointer]:
        curr_rooms -= 1
        en_time_pointer += 1


Compare Strings:
https://leetcode.com/discuss/interview-question/352458/

def get_freq_min_char(string):
    smallest_char = None
    smallest_char_count = 0
    for char in string:
        if not smallest_char or char < smallest_char:
            smallest_char = char
            smallest_char_count = 1
        else: smallest_char_count += 1
    return samllest_char_count

def solve(A, B):
    min_char_occurances_list = [0] * 10
    
    for word in A.split(','):
        min_char_occurances = get_freq_min_char(word)
        min_char_occurances_list[min_char_occurances] += 1
    print frequency_list
    #stop run the code and see
    output_list = []
    for word in B.split(','):
        min_char_occurances = get_freq_min_char(word)
        output_list.append(sum(min_char_occurances_list[:min_char_ocuurances]))
    
    print op_list


Largest Contiguous Subarray (defn of largest in this problem is the subarray that
 starts with_ the largest number)
https://leetcode.com/discuss/interview-question/352459/
Largest Subarray Length K

def compare_list(l1, l2):
    for ind,item in enumerate(l1):
        if item < l2[ind]:
            return l2
        elif item > l2[ind]:
            return l1

my_l = [1,4,3,2,5]
k = 4
current_max = [0] * k

while(len(my_l) >= k):
    l1 = my_l[0:k]
    print l1
    current_max = compare_list(current_max, l1)
    my_l = my_l[1:]



Max time:
https://leetcode.com/discuss/interview-question/396769/

input_time = "??:??"
output_time = ''

if input_time[0] != '?':
    output_time += input_time[0]
else:
    if input_time[1] == '?' or input_time[1] < 4:
        output_time += '2'
    else: output_time += '1'

if input_time[1] != '?':
    output_time += input_time[1]
else:
    if input_time[0] == 2: #change input to output and put 2 inside ''
        output_time += '3'
    else: output_time += '9'

output_time += ':'

if input_time[3] != '?':
    output_time += input_time[3]
else:
    output_time += '5'

if input_time[4] != '?':
    output_time += input_time[4]
else:
    output_time += '9'

Most booked hotel room
https://leetcode.com/discuss/interview-question/421787/

rooms = ["+1A", "+3E", "-1A", "+4F", "+1A", "-3E"]
rooms_dict = {}

for room in rooms:
    check_in_or_vacate = room[0]
    room_id = room[1:]
    if room_in in rooms_dict: #change in to id
        if check_in_or_vacate == '+':
            rooms_dict[room_id] += 1
    else:
        rooms_dict[room_id] = 1

    if room_dict[room_id] > max_bookings:
        max_bookings = room_dict[room_id]
        most_booked_room = room_id


watering plant
https://leetcode.com/discuss/interview-question/394347/
p = [2,4,5,1,2]

i = 0
j = len(p) - 1

orig_cap_1 = 5
orig_cap_2 = 7

curr_cap_1 = orig_cap_1
curr_cap_2 = orig_cap_2
no_of_times_filled = 1

while( i <= j):
    print 'i = ', i
    print 'j = ', j
    if i != j:
        if p[i] <= curr_cap_1:
            curr_cap_1 -= p[i]
            i += 1
        else:
            curr_cap_1 = orig_cap_1 - p[i]
            i += 1
            no_of_times_filled += 1
        if p[j] <= curr_cap_2:
            curr_cap_2 -= p[j]
            j -= 1
        else:
            curr_cap_2 = orig_cap_2 - p[j]
            j -= 1
            no_of_times_filled += 1
    else:
        if curr_cap_1 + curr_cap_2 > p[i]:
            break
        else:
            no_of_times_filled += 1
            break



Dynamo Rotations

from Collections import Counter

A = [2,1,2,4,2,2]
B = [5,2,6,2,3,2]

A = [3,5,1,2,3]
B = [3,6,3,3,4]

all_eles_a = Counter(A).most_common(len(A)) #[(2, 4), (1, 1), (4, 1)]
all_eles_b = Counter(B).most_common(len(B)) #[(2, 3), (3, 1), (5, 1), (6, 1)]
rot_possible = False

while(all_eles_a and all_eles_b):
    item = all_eles_a[0][0] if all_eles_a[0][1] >= all_eles_b[0][1] else\
    all_eles_b[0][0]
    ind = 0
    a_rot = 0
    b_rot = 0
    print 'item = ', item
    
    while(ind < len(A)):
        print 'ind = ', ind, A[ind], B[ind]
        print 'a_rot = ', a_rot
        print 'b_rot = ', b_rot
        if A[ind] != item and B[ind] != item:
            break
        elif A[ind] == item and B[ind] != item:
            b_rot += 1
        elif A[ind] != item and B[ind] == item:
            a_rot += 1
        ind += 1

    if ind == len(A):
        rot_possible = True
        print 'ele that can be formed = ', item
        print 'rot = ', min(a_rot, b_rot)
        break
    else:
        if item == all_eles_a[0][0]:
            all_eles_a = all_eles_a[1:]
        if item == all_eles_b[0][0]:
            all_eles_b = all_eles_b[1:]


Time to type a string
https://leetcode.com/discuss/interview-question/356477

keyboard = "abcdefghijklmnopqrstuvwxy", text = "cba" 

char_dict = {}
for index, char in keyboard: #enumerate missing
    char_dict[char] = index

previous_index = 0
time_needed_to_type = 0
for char in text:
    time_needed_to_type += my_dict[char] - previous_index #add abs()


Min number of chairs
https://leetcode.com/discuss/interview-question/356520
we could merge S and E into one list and add a flag to distinguish from attend time to leave time.
Input: S = [1, 2, 6, 5, 3], E = [5, 5, 7, 6, 8]
output: [(1, 1), (2, 1), (6, 1), (5, 1), (3, 1), (5, -1), (5, -1), (7, -1), (6, -1), (8, -1)] // 1 for come, -1 for leave

we need to sort the list by time, and place leave time before attend time if they are equal, after sort the list is:
[(1, 1), (2, 1), (3, 1), (5, -1), (5, -1), (5, 1), (6, -1), (6, 1), (7, -1), (8, -1)]

then we use a variable to count current guests num, another variable to record largest guests num
#not my appraoch
def cal_chairs(starts, ends):
    all = [(s, 1) for s in starts] + [(e, -1) for e in ends]
    all = sorted(all)
    num = 0
    largest = 0
    for pos, t in all:
        num += t
        if largest < num:
            largest = num
    return largest

#my approach
entry_times = sorted([1, 2, 6, 5, 3]) #[1,2,3,5,6]
exit_times = sorted([5, 5, 7, 6, 8])  #[5,5,6,7,8]

guests_present = 0
max_guests = 0

for index, entry_time in enumerate(entry_times):
    guests_present += 1
    for index_2, exit_time in enumerate(exit_times):
        if exit_time <= entry_time:
            guests_present -= 1
        else:
            break
    exit_times = exit_times[index_2:]
    max_guests = max(max_guests, guests_present)


k closest points to origin

Input: points = [[1,3],[-2,2]], K = 1
Output: [[-2,2]]
Explanation: 
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest K = 1 points from the origin, so the answer is just [[-2,2]].

import math
class Solution(object):
    def kClosest(self, points, K):
        """
        :type points: List[List[int]]
        :type K: int
        :rtype: List[List[int]]
        """
        dist_dict = {}
        for point in points:
            a = point[0]
            b = point[1]
            dist = a*a + b*b
            #dist = math.sqrt(dist)
            try:
                dist_dict[dist].append([a,b])
            except:
                dist_dict[dist] = [[a,b]]
                
       # print dist_dict
        sorted_dist = sorted(dist_dict.keys())
        op_list = []
        
        for dist in sorted_dist:
            #print dist_dict[dist]
            op_list += dist_dict[dist]
            if len(op_list) >= K:
                op_list = op_list[:K+1]
                break
        
        #print op_list
        
        return op_list



Roses:
https://leetcode.com/discuss/interview-question/334191

Given an array of roses. roses[i] means rose i will bloom on day roses[i]. Also 
given an int k, which is the minimum number of adjacent bloom roses required for a 
bouquet, and an int n, which is the number of bouquets we need. Return the earliest 
day that we can get n bouquets of roses.

Example:
Input: roses = [1, 2, 4, 9, 3, 4, 1], k = 2, n = 2
Output: 4

Explanation:
day 1: [b, n, n, n, n, n, b]
The first and the last rose bloom.

day 2: [b, b, n, n, n, n, b]
The second rose blooms. Here the first two bloom roses make a bouquet.

day 3: [b, b, n, n, b, n, b]

day 4: [b, b, b, n, b, b, b]
Here the last three bloom roses make a bouquet, meeting the required n = 2 
bouquets of bloom roses. So return day 4.

no_of_needed_boquets = 2
no_of_adjacacent_flowers_needed_for_single_boquet = k = 2

#Not the perfect solution but this is how it sh0uld work
def check_if_k_adjacent_roses_bloomes(i):
    no_of_adjacent_bloom_roses = 0
    i_copy_1 = i_copy_2 = i
    while(i > 0 and roses[i] == 'b'):
        i -= 1
        no_of_adjacent_bloom_roses += 1
        if no_of_adjacent_bloom_roses == k:
            break
    
    if no_of_adjacent_bloom_roses == k:
        while(i < i_copy_1): #these roses have already been picked for our bouquet
            roses[i] = 'already picked'
            i += 1
        no_of_needed_boquets -= 1
        return
    
    no_of_adjacent_bloom_roses = 0
    while(i_copy_1 < len(roses) and roses[i] == 'b'):
        i_copy_1 += 1
        no_of_adjacent_bloom_roses += 1
        if no_of_adjacent_bloom_roses == k:
            break
    
    if no_of_adjacent_bloom_roses == k:
        while(i_copy_1 < i_copy_2): #these roses have already been picked for our bouquet
            roses[i] = 'already picked'
            i_copy_1 += 1
        no_of_needed_boquets -= 1
        return 

while(no_of_needed_boquets > 0):
    i = 0
    while(i < len(roses)):
        if roses[i] == 1:
            roses[i] == 'b'
            check_if_k_adjacent_roses_bloomes(i)


Knapsack Theif problem

total_carriable_weight = 7
weight_value_dict = {1:1, 3:4, 4:5, 5:7}
sorted_available_weights = sorted(weight_value_dict.keys())
weight_val_matrix = [[-1 for i in range(total_carriable_weight+1)] \
for j in range(len(sorted_available_weights))]

for index, item_weight in enumerate(sorted_available_weights):
    for weight in range(0, total_carriable_weight + 1):
        
        if weight == 0: #you cannot carry any item coz all items have weight > 0
            weight_val_matrix[index][0] = 0 

        
        elif weight - item_weight == 0:
            if index == 0:
                weight_val_matrix[0][weight] = weight_value_dict[item_weight]
            else:
                weight_val_matrix[index][weight] = max(weight_value_dict \
                    [item_weight], weight_val_matrix[index-1][weight])
        
        elif weight - item_weight > 0:
            if index == 0:
                weight_val_matrix[0][weight] = weight_value_dict[item_weight]
            else:
                max_value_with_this_item = weight_value_dict[item_weight]+\
                    weight_val_matrix[index-1][weight - item_weight]
                
                max_value_without_this_item = weight_val_matrix[index-1][weight]
                
                weight_val_matrix[index][weight] = max(max_value_with_this_item, \
                max_value_without_this_item)

        elif weight - item_weight < 0:
            if index == 0:
                weight_val_matrix[0][weight] = weight_value_dict[item_weight]
            else:
                weight_val_matrix[index][weight] = weight_val_matrix[index-1] \
                [weight]


''' input
[
[-1, -1, -1, -1, -1, -1, -1, -1], 
[-1, -1, -1, -1, -1, -1, -1, -1],
[-1, -1, -1, -1, -1, -1, -1, -1], 
[-1, -1, -1, -1, -1, -1, -1, -1]
]
'''
'''output
[
[0, 1, 1, 1, 1, 1, 1, 1], 
[0, 1, 1, 4, 5, 5, 5, 5], 
[0, 1, 1, 4, 5, 6, 6, 9], 
[0, 1, 1, 4, 5, 7, 8, 9]
]
'''


Evaluate to a target 323, 3

target = 3
expression = '323'
expression_list = [int(i) for i in expression]
operations = ['*', '+', '-', '/']
all_operations_orders = []
#[['*', '*'], ['*', '+'], ['*', '-'], ['*', '/'], 
#['+', '*'], ['+', '+'], ['+', '-'], ['+', '/'],
#['-', '*'], ['-', '+'], ['-', '-'], ['-', '/'], 
#['/', '*'], ['/', '+'], ['/', '-'], ['/', '/']]


def cb(operations, operations_order):
    if len(operations_order) == 2:
        all_operations_orders.append(operations_order)
        return

    for operation in operations:
        cb(operations, operations_order + [operation])

for operation_order in all_operations_orders:
    res = expression_list[0]
    for index, operation in enumerate(operation_order):
        if operation == '*':
            res = res * expression_list[index + 1]
        if operation == '+':
            res = res + expression_list[index + 1]
        if operation == '-':
            res = res - expression_list[index + 1]
        if operation == '/':
            res = res / expression_list[index + 1]

    if res == target:
        print operation_order
        break

Minimum height trees
https://leetcode.com/problems/minimum-height-trees/
https://www.geeksforgeeks.org/roots-tree-gives-minimum-height/
#haven't run the code yet - need to run and see of it works fine
# we can also solve the prob using dfs
#BFS Apprach
n = 6
edges = [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]]
my_dictionary = {} #{0: [3], 1: [3], 2: [3], 3: [0, 1, 2, 4], 4: [3, 5], 5: [4]}
min_height = None

for edge in edges:
    key_1 = edge[0]
    key_2 = edge[1]
    if key_1 in my_dictionary:
        my_dictionary[key_1].append(key_2)
    else:
        my_dictionary[key_1] = [key_2]

    if key_2 in my_dictionary:
        my_dictionary[key_2].append(key_1)
    else:
        my_dictionary[key_2] = [key_1]

#BFS Apprach
for key in my_dictionary.keys():
    queue = [(key, 0)] #(key, height)
    visited_nodes = visited_nodes + [key]

    while(queue):
        node_height = queue.pop()
        node = node_height[0]
        height = node_height[1]
        children = my_dictionary[key]

        for child in children:
            if child in visited_nodes:
                continue
            else:
                visited_nodes.append(child)
                queue.append((child, height+1))

    if not min_height:
        min_height = height
        min_height_root = [key]
    elif height == min_height:
        min_height_root.append(key)
    elif height < min_height:
        min_height = height
        min_height_root = [key]

#DFS
for node in my_dictionary.keys():
    max_ht_keeping_node_as_root = 0
    visited_dict = {}
    max_ht = calc_max_ht(node, 0, visited_dict)

def calc_max_height(node, ht):
    visited_dict[node] = 'V'
    all_connected_vertices = my_dictionary[node]
    for vertex in all_connected_vertices:
        if not vertex in visited_dict:
            dummy_value = calc_max_height(vertex, ht+1, visited_dict)

    max_ht = max(max_ht, ht)
    return max_ht

Tilt of binary tree
https://www.geeksforgeeks.org/tilt-binary-tree/
https://leetcode.com/problems/binary-tree-tilt/
#need to run the code to verify

tilt = []
def cb(root):
    if not root: return 0

    l_sum_val = cb(root.left)
    r_sum_val = cb(root.right)

    tilt.append(abs(l_sum_val - r_sum_val))
    return root.val + l_sum_val + r_sum_val

total_tilt = sum(tilt)


Max binary tree
https://leetcode.com/problems/maximum-binary-tree/
#need to run the code to verify

input_list = [3,2,1,6,0,5]
def construct_tree(my_l):
    if not my_l:
        return None
    
    curr_max = my_l[0]
    max_index = 0
    
    for index, ele in enumerate(my_l):
        if ele > curr_max:
            curr_max = ele
            max_index = index

    root = NewNode(curr_max)
    left_c = cons_tree(my_l[0:max_index])
    right_c = cons_tree(my_l[max_index + 1:])
    root.left = left_c
    root.right = right_c
    return root


Equal tree partition
https://leetcode.com/problems/equal-tree-partition/
#need to run the code to verify

do a bfs and get the total sum of the tree as_ total_sum
then do a dfs to calculate the sum of each node from the leaf and_ check if_ it 
equals half the sum

def cb(root):
    if not root: return 0

    l_sum, remove_edge = cb(r.l)
    
    if remove_edge: 
        print 'the edge to be removed is ', root.val, root.l.val return
    r_sum, remove_edge = cb(r.r)
    
    if remove_edge: 
        print 'the edge to be removed is ', root.val, root.r.val return
    curr_sum = root.val + l_sum + r_sum

    if total_sum / 2 == curr_sum:
        rem_ed = True
        half_val_node = root
        return curr_sum, True

    return curr_sum, False


Print binary tree in a particular order
https://leetcode.com/problems/print-binary-tree/
#need to run the code to verify
input_:
Input:
      1
     / \
    2   5
   /   /
  3   8
 /     \
4       9
[["",  "",  "", "",  "", "", "", "1", "",  "",  "",  "",  "", "", ""]
 ["",  "",  "", "2", "", "", "", "",  "",  "",  "",  "5", "", "", ""]
 ["",  "3", "", "",  "", "", "", "",  "",  "8",  "",  "",  "", "", ""]
 ["4", "",  "", "",  "", "", "", "",  "",  "",  "",  "",  "", "", "9"]]
do a dfs and get the max height of tree as_ max_ht_of_tree

no_of_cols = (2**max_ht_of_tree) - 1 #above tree's height is 4
no_of_rows = max_ht_of_tree

matrix = [["" for col in range(no_of_cols)] for row in range(no_of_rows)]

#better solution
class Solution(object):
    def printTree(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[str]]
        """
        def get_height(node):
            return 0 if not node else 1 + max(get_height(node.left), get_height(node.right))
        
        def update_output(node, row, left, right):
            if not node:
                return
            mid = (left + right) / 2
            self.output[row][mid] = str(node.val)
            update_output(node.left, row + 1  , left   , mid - 1)
            update_output(node.right, row + 1 , mid + 1, right)
            
        height = get_height(root)
        width = 2 ** height - 1
        self.output = [[''] * width for i in xrange(height)]
        update_output(node=root, row=0, left=0, right=width - 1)
        return self.output

#NOT A GREAT SOLUTION BUT CORRECT SOLUTION
def form_matrix(node, row, pos):
    if not node:
        return

    mat[row][pos] = node.val
    current_height = row + 1 #considering root's height as 1
    relative_pos_for_children = 2 ** (total_height - current_height - 1)
    #for the 1st iteration, relative_pos_for_children is 4
    left_child_pos = pos - relative_pos_for_children #1st iter: 7 - 4 = 3
    right_child_pos = pos +  relative_pos_for_children #1st iter: 7 + 4 = 11
    form_matrix(node.left, row + 1, left_child_pos)
    form_matrix(node.left, row + 1, right_child_pos)

#The following solution is incorrect. Check what happens for 8 in the input tree
pos_of_root_of_tree = no_of_cols/2
inital_height = 0
q = [(root, pos_of_root_of_tree, inital_height)]

while(q):
    node, p, h = q.pop()
    matrix[h][p] = node.value
    if node.l: q.append((node.l, p - p/2, h+1))
    if node.r: q.append((node.r, p + p/2, h+1)) 


Distribute coins in a binary tree
https://leetcode.com/problems/distribute-coins-in-binary-tree/

#we are just calculating how many moves we have to make from each node in the tree
#recursively

class Solution(object):
    def distributeCoins(self, root):
        self.ans = 0

        def dfs(node):
            if not node: return 0
            L, R = dfs(node.left), dfs(node.right)
            self.ans += abs(L) + abs(R)
            return node.val + L + R - 1

        dfs(root)
        return self.ans

Coin Change
https://leetcode.com/problems/coin-change/
https://www.youtube.com/watch?v=1R0_7HqNaW0

DP problem
  0 1 2 3 4 5 6 7 8 9 10 11
1 0 1 2 3 4 5 6 7 8 9 10 11 -> min no of 1$ coins needed
2 0 1 1 2 2 3 3 4 4 5 5  6  -> min no of 2 and 1 $ coins needed
5 0 1 1 2 2 1 2 2 3 3 2  3  -> min no of 5, 2 and 1 $ coins needed


kth smallest element in a bst:
https://leetcode.com/problems/kth-smallest-element-in-a-bst/
https://www.youtube.com/watch?v=C6r1fDKAW_o&t=17s
inorder travesal and once your inorder list equals K, return that element
iterative approach
class Solution:
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        stack = []
        
        while True:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if not k:
                return root.val
            root = root.right


DeleteNodes and return forest. Like the question states, you just need to 
return the root nodes after deleting the nodes in the toDelete list
https://leetcode.com/problems/delete-nodes-and-return-forest/
https://www.youtube.com/watch?v=aaSFzFfOQ0o (time complexity and space comple is
    explained clearly)
You can also with BFS.
Just do dfs and whenever you delete a node, add its left and right child in the
new_roots dict_ and return True and capture that and set the parent's' left or 
right child to None

def delete_node(node):
    if not node:
        return False

    left_child_deleted = delete_node(node.left)
    right_child_deleted = delete_node(node.right)

    if left_child_deleted:
        node.left = None
    if right_child_deleted:
        node.right = None

    if node.value in to_delete_list:
        new_roots += [node.left, node.right]
        return True


Repeated DNA sequences
https://www.youtube.com/watch?v=0y7pU6PPrc4 
(time complexity explained clearly at 6:40)
https://leetcode.com/problems/repeated-dna-sequences/

Space complexity = 
Space complexity : O((N - L)L) to keep the hashset, that results in O(N) for the 
constant L = 10

my_dict = {}
output = []
while(i < len(string) - 9):
    if dna[i : i + 10] in my_dict:
        output.append(dna[i : i + 10])
    else:
        dna[i : i + 10] = True
    i += 1

Minimum cost to connect sticks
https://leetcode.com/problems/minimum-cost-to-connect-sticks/
https://www.youtube.com/watch?v=3dqR2nYElyw

my_sticks = [9,12,10,11,13,11]
cost = 0

1 - You have to constantly pick two sticks from_ the my_sticks list_ and_ combine them
and_ add it and update the 
2- cost += stick_1 + stick_2
3- Now we have to remove those 2 stick from_ the my_sticks and_ append the bigger_stick
in my_stics

heaps is the ideal ds to use here
my_sticks = heapq.heapify(my_sticks)
stick_1 = heap1.heap_pop()
stick_2 = heap1.heap_pop()
cost += stick_1 + stick_2
new_bigger_stick = stick_1 + stick_2
heapq.heap_push(my_sticks, new_bigger_stick)

heap_push and heap_pop are log n operations


Last stone weight
https://leetcode.com/problems/last-stone-weight/
https://www.youtube.com/watch?v=fBPS7PtPtaE

We have to use heapmax for this problem. It's similar to 
https://leetcode.com/problems/minimum-cost-to-connect-sticks/

#Remember to heapify the inputlist before the loop because time to heapify is O(n)
#and time to push and pop is log n

heap.heapify_max(my_l) #O(n)
while(len(my_l) > 1):
    big_elem_1 = heapq.heappop(my_l)
    big_elem_v = heapq.heappop(my_l)
    heapq.heappush(big_elem_1 - big_elem_2)

return my_l[0]

Reverse vowels of a string
https://leetcode.com/problems/reverse-vowels-of-a-string/
https://www.youtube.com/watch?v=1NXs_idViuQ
#this is a working solution
eg: leetcode -> leotcede
Use 2 ptr approach. 
1st ptr is at the start of the string
2nd ptr is at the end of the string
vowels_dict = {'a':1, 'e':1, 'i':1, 'o':1, 'u':1}
i_p = 'leetcode'
rev_vow_str = i_p
p1 = 0
p2 = len(i_p) - 1
while(p1 < p2):
    print 'p1 = ', p1
    print 'p2 = ', p2
    if not i_p[p1] in vowels_dict:
        p1 += 1
    if not i_p[p2] in vowels_dict:
        p2 -= 1
    if i_p[p1] in vowels_dict and i_p[p2] in vowels_dict:
        rev_vow_str = rev_vow_str[0:p1] + i_p[p2] + rev_vow_str[p1+1:p2] + i_p[p1]\
        + rev_vow_str[p2+1:]
        p1 += 1
        p2 -= 1
    print rev_vow_str



Minimul path sum
https://www.youtube.com/watch?v=ItjZdu6jEMs
https://leetcode.com/problems/minimum-path-sum/

When you have keywords like "minimize" and_ stuff, its a good indication that
you can do it with_ dynamic programming.
i_p = 
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]

o_p_stage_1 = 
[
  [1,4,5]
]

o_p_stage_2 = 
[
  [1, 4, 5],
  [2, 7, 6] my_mat[row][col] += min(my_mat[row-1][col], my_mat[row][col - 1])
]

o_p_stage_3 = 
[
  [1, 4, 5],
  [2, 7, 6] my_mat[row][col] += min(my_mat[row-1][col], my_mat[row][col - 1])
  [6, 8, 7]
]

my_mat[tot_rows - 1][tot_cols - 1] is_ the final answer


https://leetcode.com/problems/partition-equal-subset-sum/
Use my appraoch similar to the one desc here
#have to submit on leetcode and see if it works for all cases
https://www.geeksforgeeks.org/partition-problem-dp-18/ (check out the matrix 
    at the bottom of this page and learn the time complexity as_ well)
intrerchange rows and cols

my_l = [1, 5, 5, 7, 10]
my_l = sorted(my_l)
sum_of_eles = sum(my_l)
if sum_of_eles % 2 != 0: print 'not possible'

target = sum_of_eles/2

my_mat = [[True for i in range(target + 1)] for i in range(len(my_l))]

for row in my_mat:
    row[0] = 0

for row in my_mat:
    print row

for index_list in range(len(my_l)):
    element = my_l[index_list]
    print 'elem = ', element
    for col in range(target + 1):
        print 'col = ', col
        if col == 0:
            continue
        else:
            if element - col == 0:
                print 'ele - col = 0'
                continue #Because by default all cells are set to True
            elif element > col:
                print 'elem > col'
                if index_list > 0:
                    my_mat[index_list][col] = my_mat[index_list - 1][col]
                else:
                    my_mat[index_list][col] = False
            elif col > element:
                print 'col > elem'
                difference = col - element
                if index_list > 0 and my_mat[index_list - 1][difference] == True:
                    my_mat[index_list][col] = True
                else:
                    my_mat[index_list][col] = False
            print 'my_mat[index_list][col] = ', my_mat[index_list][col]



    0    1     2      3     4     5     6     7     8     9    10     11
1-[True, True, True, True, True, True, True, True, True, True, True, True]
2-[True, True, True, True, True, True, True, True, True, True, True, True]
3-[True, True, True, True, True, True, True, True, True, True, True, True]
4-[True, True, True, True, True, True, True, True, True, True, True, True]
5-[True, True, True, True, True, True, True, True, True, True, True, True]
7-[True, True, True, True, True, True, True, True, True, True, True, True]



https://leetcode.com/problems/binary-tree-right-side-view/
https://www.youtube.com/watch?v=jCqIr_tBLKs

Do a BFS and_ store the values inside a dictionary whose keys denote levels of 
the tree and_ the values are a list_ of values in_ the tree.

my_d = {}

def cb(root, h):
    if not root:
        return 
    
    if not h in my_d:
        my_d[h] = root.val
    
    cb(root.right, h+1)
    cb(root.left, h+1)


Output will be
For each key get the last value of the list_ stored in the value part of that key

my_d[0] = [0]
my_d[1] = [1, 2]
my_d[2] = [3,4,5,6]
       0
   1       2
3    4   5   6
 
o_p = 0, 2, 6


Subsets
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
    

Valid Palindrome II
https://leetcode.com/problems/valid-palindrome-ii/
https://www.youtube.com/watch?v=L_74qbyPHXE
Input: "abca"
Output: True
Explanation: You could delete the character 'c'.

input_string = "mal bayaba lam"
You cannot solve this problem with_ a single for_ loop that iterates half way 
through the list_ and_finds out if_ a string is pal or not. See this case

input_string = "mal bayaba lam"

if you use a single for_ loop, the above case would fail

test inputs: "mal bayaba lam", 'malayablam', 'malbayalam'

def check_palindrome(input_string, is_callback):
    print 'input_string = ', input_string
    print 'is_callback = ', is_callback
    index = 0
    string_len = len(input_string)
    while(index < string_len / 2):
        if input_string[index] == input_string[string_len - index -1]:
            index += 1
            continue
        elif not is_callback:
            print 'is_callback = ', is_callback
            is_palindrome = check_palindrome(input_string[index + 1:\
                string_len - index], True)
            if is_palindrome:
                print 'pal can be made'
                return True
            is_palindrome = check_palindrome(input_string[index:\
                string_len - index - 1], True)
            if is_palindrome:
                print 'pal can be made'
                return True
        else:
            return False

    return True


DIP
You are given an array of integers, and an integer K. Return the subarray which
 sums to K. You can assume that a solution will always exist.

my_l = [1, 3, 2, 5, 7, 2] 
k = 14
sub_arr = [2, 5, 7]


#running_sums_dict = {1: 0, 4: 1, 6: 2, 11: 3, 18: 4, 20: 5}
#running_sums_dict.keys() prints [1, 4, 6, 11, 18, 20]
sums = 0
running_sums_dict = {}
for index, ele in enumerate(my_l):
    sums += ele
    running_sums_dict[sums] = index

for ele in running_sums_dict.keys():
    if ele - k in running_sums_dict:
        print 'ele = ', ele
        print 'ele - k = ', ele - k
        index_1 = running_sums_dict[ele]
        index_2 = running_sums_dict[ele - k]
        if index_1 > index_2: print my_l[index_2+1:index_1 + 1]
        else: print my_l[index_1+1:index_2 + 1]

DIP
You are given an array of integers. Return the length of the longest consecutive
 elements sequence in the array.

For example, the input array [100, 4, 200, 1, 3, 2] has the longest consecutive
 sequence 1, 2, 3, 4, and thus, you should return its length, 4.

my_l = [100, 4, 200, 1, 3, 2]
#longest conseq sequence is [1,2,3,4]
#len of longeset conseq seq is 4
len_max_seq = 0
my_ele_dict = {}

for ele in my_l:
    my_ele_dict[ele] = True

while(my_ele_dict):
    main_ele = my_ele_dict.keys()[0]
    print 'ele = ', main_ele
    len_seq = 1
    
    if main_ele + 1 in my_ele_dict:
        ele = main_ele + 1
        
        while(ele in my_ele_dict):
            len_seq += 1
            my_ele_dict.pop(ele)
            ele = ele + 1
    
    if main_ele - 1 in my_ele_dict:
        ele = main_ele - 1
        
        while(ele in my_ele_dict):
            len_seq += 1
            my_ele_dict.pop(ele)
            ele = ele - 1

    len_max_seq = max(len_max_seq, len_seq)
    print 'len_max_seq = ', len_max_seq
    
    my_ele_dict.pop(main_ele)


DIP
You are given an array of integers. Return all the permutations of this array.
https://www.geeksforgeeks.org/write-a-c-program-to-print-all-permutations-of-a-given-string/

print permute([1, 2, 3])
# [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]

all_permuted_lists = []
my_l = [1, 2, 3]

#The below solution (cb_1) will work only for lists without duplicate elements. The
#line if len(set(permuted_list)) == len(my_l) will never succeed when there are dups
#in your input list
def cb_1(my_l, permuted_list):
    if len(permuted_list) == len(my_l):
        if len(set(permuted_list)) == len(my_l): 
            all_permuted_lists.append(permuted_list)
        return

    for index, ele in enumerate(my_l):
        cb(my_l, permuted_list + [ele])

#The below solution will take into account the duplicate element cases also. This 
#is also the more efficient one
my_main_l = [1, 2, 3]
def cb_2(my_l, permuted_list):
    print 'my_l = ', my_l
    if len(permuted_list) == len(my_main_l):
        #if len(set(permuted_list)) == len(my_l): 
        all_permuted_lists.append(permuted_list)
        return

    for index, ele in enumerate(my_l):
        #cb(my_l, permuted_list + [ele])
        cb(my_l[:index] + my_l[index + 1:], permuted_list + [ele])


DIP
You are given the root of a binary tree. Find the path between 2 nodes that 
maximizes the sum of all the nodes in the path, and return the sum. The path 
does not necessarily need to go through the root.

https://leetcode.com/problems/binary-tree-maximum-path-sum/submissions/

class Node:
  def __init__(self, val):
    self.val = val
    self.left = None
    self.right = None

def maxPathSum(root):
  # Fill this in.

# (* denotes the max path)
#       *10
#       /  \
#     *2   *10
#     / \     \
#   *20  1    -25
#             /  \
#            3    4


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def __init__(self):
        self.max_value = None
    
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        def cb(root):
            if not root:
                return 0
            print 'root = ', root
            l_max_val = cb(root.left) #max val of left sub tree
            print 'l_max_val = ', l_max_val
            r_max_val = cb(root.right) #max val of right sub tree
            print 'r_max_val = ', r_max_val

            to_return_max_value = max([root.val, root.val + l_max_val, \
                root.val + r_max_val]) #any path going to root (10) can be distinct paths
            # from the leaf nodes (i.e 20 or 1)

            if self.max_value: #We check if 20 -> 2 -> 1 is the max path in tree
                self.max_value = max([root.val, root.val + l_max_val, \
                    root.val + r_max_val, root.val + l_max_val + r_max_val, \
                    self.max_value])
            else:
                self.max_value = max([root.val, root.val + l_max_val, \
                    root.val + r_max_val, root.val + l_max_val + r_max_val])

            return to_return_max_value
        
        cb(root)
        print 'max_value = ', self.max_value
        return self.max_value


DIP
Given a sorted list of positive numbers, find the smallest positive number that 
cannot be a sum of any subset in the list.
https://www.geeksforgeeks.org/find-smallest-value-represented-sum-subset-given-array/

NP Completeness problem

Example:
Input: [1, 2, 3, 8, 9, 10]
Output: 7
Numbers 1 to 6 can all be summed by a subset of the list of numbers, but 7 
cannot.

def findSmallest(nums):
  # Fill this in.

print findSmallest([1, 2, 3, 8, 9, 10])

matrix = [[for col in range(input_list[-1] ** 2)] for row in input_list]
input_dict = {}


DIP
Given a number of integers, combine them so it would create the largest number.

Example:
Input:  [17, 7, 2, 45, 72]
Output:  7 72 45 2 17
def largestNum(nums):
  # Fill this in.

print largestNum([17, 7, 2, 45, 72])
# 77245217

input_list = [17, 7, 2, 45, 72]
max_len_num = 0
my_dictionary = {}
for num in input_list:
    if len(str(num)) > max_len_num:
        max_len_num = len(str(num))

#max_len_num = 2 This is the max no of digs of any num in input_list

for num in input_list:
    new_num = str(num)
    last_digit_num = new_num[-1]
    
    if len(new_num) < max_len_num:
        while(len(new_num) < max_len_num):
            new_num += last_digit_num
    
    if new_num in my_dictionary:
        my_dictionary[int(new_num)].append(str(num))
    else:
        my_dictionary[int(new_num)] = [str(num)]

print my_dictionary
#{72: ['72'], 17: ['17'], 77: ['7'], 22: ['2'], 45: ['45']}
#The assumption is 7 > [76, 75, 74,..1] but 7 < 78
#To maximize a number we have to make sure that all the max digits should come 
#first. At the same time we can't change the numbers in input list.
#So we are modifying the all the numbers so that they are max (eg: we modify 7 to
#77 because all numbers in the list less than 77 should appear after 7 in o/p)
#Our dict will be modified_val: [original_values]

sorted_keys = sorted(my_dictionary.keys())[::-1]
#[77, 72, 45, 22, 17]
largest_val_str = ''
for key in sorted_keys:
    largest_val_str += ''.join(my_dictionary[key])
# 77245217



DIP
Given a string, rearrange the string so that no character next to each other are 
the same. If no such arrangement is possible, then return None.

Example:
Input: abbccc
Output: cbcbca
def rearrangeString(s):
  # Fill this in.

print rearrangeString('abbccc')
# cbcabc
#We can use heaps here in this problem to improve time comple N(log N)
#current col is n**2 (log N)
#haven't solved using heaps. Solve later. 
b ccc ad 
bc ac dc
from collections import Counter
my_str = 'cccaaaaabbbbb'
c = Counter(my_str).most_common(len(my_str))
temp_c = []
for item in c:
    temp_c.append((item[1], item[0]))

c = sorted(temp_c, reverse=True)
temp_c = []
output_str = ''

while(c):
    print 'begin for c = ', c
    for item in c[:2]:
        count = item[0] - 1
        char = item[1]
        output_str += char
        if count > 0:
            temp_c.append((count, char))
        
    temp_c += c[2:]
    print 'end for temp_c = ', temp_c
    c = sorted(temp_c, reverse=True)
    temp_c = []
    print 'end for c = ', c
    if len(c) == 1 and c[0][0] > 1:
        print 'not_possible'
        break


DIP
Given an array of characters with repeats, compress it in place. The 
length after compression should be less than or equal to the original array.

Example:
Input: ['a', 'a', 'b', 'c', 'c', 'c']
Output: ['a', '2', 'b', 'c', '3']
Here is a starting point:

class Solution(object):
  def compress(self, chars):
    # Fill this in.

print Solution().compress(['a', 'a', 'b', 'c', 'c', 'c'])
# ['a', '2', 'b', 'c', '3']

input_list = ['a', 'a', 'b', 'c', 'c', 'c']
index = 1
len_inp = len(input_list)

iterating_character = input_list[0]
no_of_occurances = 1

while(index < len_inp):
    if input_list[index] == iterating_character:
        no_of_occurances += 1
    else:
        input_list.append(iterating_character)
        input_list.append(no_of_occurances)
        iterating_character = input_list[index]
        no_of_occurances = 1

    index += 1

input_list.append(iterating_character)
input_list.append(no_of_occurances)

input_list = input_list[len_inp:]


DIP
Given an array of integers, arr, where all numbers occur twice except one number 
which occurs once, find the number. Your solution should ideally be O(n) time and 
use constant extra space.

https://www.geeksforgeeks.org/find-element-appears-array-every-element-appears-twice/
Learn XOR operations

Example:
Input: arr = [7, 3, 5, 5, 4, 3, 4, 8, 8]
Output: 7
class Solution(object):
  def findSingle(self, nums):
    # Fill this in.

nums = [1, 1, 3, 4, 4, 5, 6, 5, 6]
print(Solution().findSingle(nums))
# 3


DIP 
Fibonacci series

fib = [0,1]
n = 5
i = 2

while(i < n):
    new_ele = fib[i - 1] + fib[i - 2]
    fib.append(new_ele)
    i += 1


DIP
Given a list of numbers of size n, where n is greater than 3, find the maximum 
and minimum of the list using less than 2 * (n - 1) comparisons.

Here is_ a start:

def find_min_max(nums):
  # Fill this in.

print find_min_max([3, 5, 1, 2, 4, 8])
# (1, 8)
if my_l[0] < my_l[1]:
    curr_min = my_l[0]
    curr_max = my_l[1]
else:
    curr_min = my_l[1]
    curr_max = my_l[0]

index = 2
while(index < len(input_list)):
    if input_list[index] < curr_min:
        curr_min = input_list[index]
    elif input_list[index] > curr_max:
        curr_max = input_list[index]
    index += 1


DIP
A k-ary tree is a tree with k-children, and a tree is symmetrical if the data of 
the left side of the tree is the same as_ the right side of the tree.\

Here_is an example of a symmetrical k-ary tree.
'''
        4
     /     \
    3        3
  / | \    / | \
9   4  1  1  4  9
'''
left_root = root.left
right_root = root.right

def bfs(root):
    # Do a bfs and returns all values in tree

left_values = sorted(bfs(left_root))
right_values = sorted(bfs(right_root))
return left_values == right_values
#check if left_values and right_values are same (use any approach)

Follow up (similar question)
https://www.geeksforgeeks.org/symmetric-tree-tree-which-is-mirror-image-of-itself/
The foll is a valid tree
The sol given in the geekforgeek links is also good
     1
   /   \
  2     2
 / \   / \
3   4 4   3

The foll is not
    1
   / \
  2   2
   \   \
   3    3
#left_tree_list = {2LnR3}
#right_tree_list = {2LnR3}
#now replace L with R and R with L in the left list and 
#check if the left and right list are same. If same, they are symmetrical

My Sol: #the below sol is not accurate. Will fail in some cases
Do a bfs and store the values in a dictionary with levels as keys
[0]: [1]
[1]: [2, 2]
[2]: [None, 3, None, 3]

for each of the values in the dictionary,
1- Have a start and end pointer at index 0 and -1
2- iterate till start_ptr > end_ptr
3- in each iteration check if left_ptr and right_ptr point to same values


DIP
The h-index is a metric that attempts to measure the productivity and 
citation impact of the publication of a scholar. The definition of the h-index 
is if a scholar has at least h of their papers cited h times.

Given a list of publications of the number of citations a scholar has, find their
h-index.

Example:
Input: [3, 5, 0, 1, 3]
Output: 3
Explanation:
There are 3 publications with 3 or more citations, hence the h-index is 3.

input_list = [3, 5, 0, 1, 3]
hindex_dict = {}

for hindex in input_list:
    if inp in hindex_dict:
        hindex_dict[hindex] = hindex_dict[hindex] + 1
    else:
        hindex_dict[hindex] = 1

total_publications = len(input_list)
h_index_val = 0
while(total_publications >= 0):
    h_index_val += input_dict[total_publication]
    if h_index_val == total_publications:
        print h_index_val
        break
    total_publications -= 1


DIP
Starting at index 0, for an element n at index i, you are allowed to jump at 
most n indexes ahead. Given a list of numbers, find the minimum number of jumps to
reach the end of the list.

Example:
Input: [3, 2, 5, 1, 1, 9, 3, 4]
Output: 2
Explanation:

The minimum number of jumps to get to the end of the list is 2:
3 -> 5 -> 4

input_list = [3, 2, 5, 1, 1, 9, 3, 4]
index_list = [0, 1, 2, 3, 4, 5, 6, 7]

max_jump_l = [3, 3, 7, 4, 5, 8, 8, 8]

input_list = [3, 2, 5, 1, 1, 9, 3, 4]
max_jumps_l = []

for index, max_allowed_jump in enumerate(input_list):
    max_jumps_l.append(index + max_allowed_jump)

index = 0
jumps = 0

while(index < len(input_list)):
    max_allowed_jump = max_jumps_l[index]
    
    if index + max_allowed_jump >= len(input_list):
        jumps += 1
        break
    
    window = max_jumps_l[index + 1:max_allowed_jump + 1]
    profitable_jump = max(window)
    index += profitable_jump
    jumps += 1


DIP
Two words can be 'chained' if the last character of the first word is the same _as
the first character of the second word.

Given a list of words, determine if there is a way to 'chain' all the words in a 
circle.

Example:
Input: ['eggs', 'karat', 'apple', 'snack', 'tuna']
Output: True
Explanation:
The words in the order of ['apple', 'eggs', 'snack', 'karat', 'tuna'] creates a 
circle of chained words.
#This is an extremly diff or time consuming problem
https://www.geeksforgeeks.org/given-array-strings-find-strings-can-chained-form-circle/


GeekForGeeks
https://www.geeksforgeeks.org/word-ladder-set-2-bi-directional-bfs/
Word Ladder – Set 2 ( Bi-directional BFS )
Input: Dictionary = {POON, PLEE, SAME, POIE, PLEA, PLIE, POIN}
start = “TOON”
target = “PLEA”
Output: 7
TOON -> POON –> POIN –> POIE –> PLIE –> PLEE –> PLEA

word_list = ["POON", "PLEE", "SAME", "POIE", "PLEA", "PLIE", "POIN"]
start = "TOON"
target = "PLEA"
target_found = False

def word_diff(word_1, word_2):
    word_1_dict = {}
    word_2_dict = {}
    word_diff = 0

    for char in word_1:
        if char in word_1_dict:
            word_1_dict[char] = word_1_dict[char] + 1
        else:
            word_1_dict[char] = 1

    for char in word_2:
        if char in word_1_dict and word_1_dict[char] > 0:
            word_1_dict[char] -= 1
        else:
            word_diff += 1

    if word_diff > 1:
        return False
    else:
        return True

queue = []
queue.append(start)
while(queue):
    curr_word = queue.pop()
    if curr_word == target:
        print 'word has been reached'
        break
    for word in word_list:
        if word_diff(curr_word, word):
            print 'next_word = ', word
            queue.append(word)
            word_list.remove(word)



DIP
Hi, here iss your problem today. This problem was recently asked by Google:

Given a string with a certain rule: k[string] should be expanded to string k times.
So for example, 3[abc] should be expanded to abcabcabc. 
Nested expansions can happen, so 2[a2[b]c] should be expanded to abbcabbc.

print decodeString('2[a2[b]3[cd]ef]')
# abbcdabbcd

output_str = ''

def get_new_string(new_string):
    eval_string = ''
    open_braces = 1
    new_string = new_string[1:]
    
    while(open_braces > 0):
        print 'new_string = ', new_string
        print 'open_braces = ', open_braces
        char = new_string[0]
        if char == '[':
            open_braces += 1
        elif char == ']':
            open_braces -= 1
        if not open_braces == 0:
            eval_string += char
        new_string = new_string[1:]

    return eval_string

#'2[a2[b]3[cd]ef]' a bb cd cd cd ef
def decode_string(my_string):
    global output_str

    if not my_string:
        return

    open_braces = 0
    for index, char in enumerate(my_string):
        if char.isalpha():
            output_str += char
            print 'output_str = ', output_str
        elif char in ['[', ']']:
            continue
        else:
            repeat_times = int(char)
            continue_until_index = 
            new_string = get_new_string(my_string[index+1:])
            #stripped_string = new_string.split(']')[-1] #'cd' the one after ']'
            #new_string = ']'.join(new_string.split(']')[:-1])
            print 'current_string = ', my_string
            print 'index = ', index
            print 'new_string = ', new_string
            print 'output_str = ', output_str
            print '\n\n'
            while(repeat_times > 0):
                decode_string(new_string)
                repeat_times -= 1
            
            #print 'output_str end of while = ', output_str
            break

decode_string('2[a2[b]cd]')
class Solution(object):
    def decodeString(self, s):
        stack = []; curNum = 0; curString = ''
        for c in s:
            if c == '[':
                stack.append(curString)
                stack.append(curNum)
                curString = ''
                curNum = 0
            elif c == ']':
                num = stack.pop()
                prevString = stack.pop()
                curString = prevString + num*curString
            elif c.isdigit():
                curNum = curNum*10 + int(c) #MULTIPLYING BY 10 IF WE GET 2 or MORE
                #dig NUMS
            else:
                curString += c
        return curString
# currNum = 2, 2; currString = 'b'
#s = ['', 2,a,2,]


DIP
Hi, here is your problem today. This problem was recently asked by Google:

Given a binary tree, remove the nodes in which there is only 1 child, so that the 
binary tree is a full binary tree.

So leaf nodes with no children should be kept, and nodes with_ 2 children should be
kept as_ well.
https://www.geeksforgeeks.org/given-a-binary-tree-how-do-you-remove-all-the-half-nodes/

# Given this tree:
#     1
#    / \ 
#   2   3
#  /   / \
# 0   9   4

# We want a tree like:
#     1
#    / \ 
#   0   3
#      / \
#     9   4

def remove_half_nodes(root):
    if not root:
        return None

    root.left = remove_half_nodes(root.left)
    root.right = remove_half_nodes(root.right)

    if (root.left and root.right) or (not left_node and not right_node):
        return root
    else:
        if root.left and not root.right:
            new_root = root.left
            temp = root
            root = None
            del(temp)
            return new_root
        else:
            new_root = root.right
            temp = root
            root = None
            del(temp)
            return new_root


Practice problem
Delete a node in BST
https://leetcode.com/problems/delete-node-in-a-bst/
*****************NEED TO PRACTICE THIS**************
root = [5,3,6,2,4,null,7]
key = 3

    #    5
    #   / \
    #  3   6
    # / \   \
    #2   4   7
#   /  \
#  1   2.5
#      /
#    2.25
replace 3 with 2.5 (its inorder predecessor)

#    5
#   / \
#  4   6
# /     \
#2       7

Another valid answer is [5,2,6,null,4,null,7].

#    5
#   / \
#  2   6
#   \   \
#    4   7

    #     5
    #   /   \
    #  3     6
    # / \   / \
    #2   4 5.5 7
#   / \
#  1   2.5

class Solution:
    def successor(self, root):
        """
        One step right and then always left
        """
        root = root.right
        while root.left:
            root = root.left
        return root.val
    
    def predecessor(self, root):
        """
        One step left and then always right
        """
        root = root.left
        while root.right:
            root = root.right
        return root.val
        
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root:
            return None
        
        # delete from the right subtree
        if key > root.val:
            root.right = self.deleteNode(root.right, key)
        # delete from the left subtree
        elif key < root.val:
            root.left = self.deleteNode(root.left, key)
        # delete the current node
        else:
            # the node is a leaf
            if not (root.left or root.right):
                root = None
            # the node is not a leaf and has a right child
            elif root.right:
                root.val = self.successor(root)
                #THE FOLLOWING LINE RECURSIVELY DELETES THE NODES AND STABILIZES 
                #OUR BST
                root.right = self.deleteNode(root.right, root.val)
            # the node is not a leaf, has no right child, and has a left child    
            else:
                root.val = self.predecessor(root)
                #THE FOLLOWING LINE RECURSIVELY DELETES THE NODES AND STABILIZES 
                #OUR BST
                root.left = self.deleteNode(root.left, root.val)
                        
        return root


DIP
Hi, here is_ your problem today. This problem was recently asked by Twitter:

Given a list of integers, return the bounds of the minimum range that must be 
sorted so that the whole list would be sorted.
https://www.geeksforgeeks.org/minimum-length-unsorted-subarray-sorting-which-makes-the-complete-array-sorted/

Example:
Input:   [1, 7, 9, 5, 7, 8, 10] #[1, 7, 8, 5, 9, 6, 10]
sort_inp=[1, 5, 7, 7, 8, 9, 10]
Output: (1, 5)
Explanation:
The numbers between index 1 and 5 are out of order and need to be sorted.

input_list = [1, 7, 8, 5, 9, 6, 10]
input_list = [1, 7, 9, 5, 7, 8, 10] #
input_list = [10, 12, 20, 30, 25, 40, 32, 31, 35, 50, 60]
input_list = [0, 1, 15, 25, 6, 7, 30, 40, 50]
contradicting_curr_min = None
contradicting_curr_max = None
prev_num = None
for ind, num in enumerate(input_list):
    if not prev_num:
        prev_num = num
        continue    
    elif num > prev_num:
        prev_num = num
        continue
    if contradicting_curr_min:
        contradicting_curr_min = min(num, contradicting_curr_min)
        contradicting_curr_max = max(prev_num, contradicting_curr_max)
    else:
        contradicting_curr_min = num
        contradicting_curr_max = prev_num
    prev_num = num

min_index_range = None
max_index_range = None
#[1, 7, 8, 5, 9, 6, 10]
for ind, num in enumerate(input_list):
    if not min_index_range and num > contradicting_curr_min:
        min_index_range = ind
    if not max_index_range and num > contradicting_curr_max:
        max_index_range = ind - 1

print min_index_range
print max_index_range


Hi, here is your problem today. This problem was recently asked by Apple:

You are given the root of a binary tree. You need to implement 2 functions:

1. serialize(root) which serializes the tree into a string representation
2. deserialize(s) which deserializes the string back to the original tree that it
 represents

  # Fill this in.

#     1
#    / \
#   3   3
#  / \   \
# 2   5   7
#          \
#           3

#We do a BFS and populate the compressed_list. n stands for None
compressed_list = [1l3r3, 3l2r5, 3lnr7, 2lnrn, 5lnrn, 7lnr3, 3lnrn]

root = Node(item[0])
left_child = Node(item[3])
right_child = Node(item[3])
root.left = left_child
root.right = right_child

items_dict = {}
if left_val == right_val:
    items_dict[left_val] = [left_child, right_child]
else:
    items_dict[left_val] = [left_child]
    items_dict[right_val] = [right_child]

#items_dict = {'3':[Node(3), Node(3)]}

for item in compressed_list[1:]:
    node_val = item[0]
    node = items_dict[node_val][0]#taking out the first node with value of 3
    items_dict[node_val] = items_dict[node_val][1:] #poping out the first node with
    #value of 3 as we will be using it now.
    #The other node with the same value of 3 will be used in the forth coming
    #iterations
    left_val = item[3]
    right_val = item[5]
    
    if not left_val == 'n': 
        left_child = Node(left_val)
        node.left = left_child
        
        if left_val in items_dict:
            items_dict[left_val].append(left_child)
        else:
            items_dict[left_val] = [left_child]
    
    if not right_val == 'n': 
        right_child = Node(right_val)
        node.right = right_child
        
        if right_val in items_dict:
            items_dict[right_val].append(right_child)
        else:
            items_dict[right_val] = [right_child]


Hi, here is your problem today. This problem was recently asked by Apple:

You are given a binary tree representation of an arithmetic expression. In this
tree, each leaf is an integer value,, and a non-leaf node is one of the four 
operations: '+', '-', '*', or '/'.

Write a function that takes this tree and evaluates the expression.

#    *
#   / \
#  +    +
# / \  / \
#3  2  4  5


def eval_expression(root):
    if not root:
        return None

    left_eval_exp = eval_expression(root.left)
    right_eval_exp = eval_expression(root.right)

    if not left_eval_exp and not right_eval_exp:#leaf node
        return root.val
    else:
        if root.val == '+':
            return left_eval_exp + right_eval_exp
        if root.val == '*':
            return left_eval_exp * right_eval_exp
        if root.val == '-':
            return left_eval_exp - right_eval_exp
        if root.val == '/':
            return left_eval_exp / right_eval_exp


Hi, here is your problem today. This problem was recently asked by Microsoft:

Given a time in the format of hour and minute, calculate the angle of the hour and
minute hand on a clock.

def calcAngle(h, m):
  # Fill this in.

print calcAngle(3, 30)
# 75
print calcAngle(12, 30)
# 165

total_clock_angle = 360
total_hours = 12
each_uni_move_represents = 360 / 12
mins_represented_in_hours = input_minute / 5 #there are 60 mins/hr. 60 is the max 
#min and 12 is the max hr. you can get 12 from 60 by div 60 by 5.
rough_angle = abs(hr - mins_represented_in_hours) * each_unit_move_represents
#if the inp time is 3:30, rough angle with be 90. 
correct_angle = rough_angle - ((mins_represented_in_hours / total_hours)*30)
#We account for the move in hours hand as the min progresses. We are dividing 
#the 360 deg of a circle into 12 parts where each part represents 30 deg move(360/12
#). Between each part (eg: move from 1 to 2, the 30 degress is split equally). If 
#the time is 1:30, mins_represented_in_hours would be 6. 6/12 = 0.5 which means that
#the hour hand has half passed 1 to reach 2. So, we have to subtract (0.5 * 30),
#Where 30 is the angle representing each unit move in the clock.


DIP
Hi, here is your problem today. This problem was recently asked by Microsoft:

You are given an array of integers. Return the length of the longest increasing 
subsequence (not necessarily contiguous) in the array.

Example:
[0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
[8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]

The following input should return 6 since the longest increasing subsequence is 
0, 2, 6, 9 , 11, 15.

{10, 22, 9, 33, 21, 23, 50, 41, 60, 80}

[9, 10, 21, 22, 23 33, 41, 50, 60, 80]


DIP
Hi, here is your problem today. This problem was recently asked by Amazon:

You are given an array of integers. Return an array of the same size where the 
element at each index is the product of all the elements in the original array 
except for_ the element at that index.

For example, an input of [1, 2, 3, 4, 5] should return [120, 60, 40, 30, 24].

You cannot use division in this problem.


print products([1, 2, 3, 4, 5])
# [120, 60, 40, 30, 24]

inp_l = [1, 2, 3, 4, 5]
#fwd_l = [1, 1, 2, 6, 24]
rev_inp_l = [5,4,3,2,1]
#rev_l =    [1,5,20,60,120]

#res_l = [120,60,40,30,24]
fwd_l = []
res_l = []

for ind, item in enumerate(inp_l):
    if ind == 0:
        fwd_l.append(1)
    else:
        fwd_l.append(fwd_l[-1] * inp_l[ind-1])

inp_l.reverse() #[5 ,4,3,2,1]

for ind, item in enumerate(inp_l):
    if ind == 0:
        rev_l.append(1)
    else:
        rev_l.append(rev_l[-1] * rev_l[ind-1])


DIP
Hi, here is your problem today. This problem was recently asked by Facebook:

Given a sorted list of numbers, return a list of strings that represent all of the
consecutive numbers.

Example:
Input: [0, 1, 2, 5, 7, 8, 9, 9, 10, 11, 15]
Output: ['0->2', '5->5', '7->11', '15->15']

input_list = [0, 1, 2, 5, 7, 8, 9, 9, 10, 11, 15]
output_list = []

for index, ele in enumerate(input_list):
    if index == 0:
        start_element = str(ele)
    elif ele - 1 != prev_element and ele != prev_element:
        output_string = str(start_element) + '->' + str(prev_element)
        output_list.append(output_string)
        start_element = str(ele)
    prev_element = ele

output_list.append(str(prev_element) + '->' + str(ele))


Hi, here is your problem today. This problem was recently asked by Google:

You are given an array of tuples (start, end) representing time intervals for 
lectures. The intervals may be overlapping. Return the number of rooms that are 
required.

For example. [(30, 75), (0, 50), (60, 150)] should return 2.

input_list = [(30, 75), (0, 50), (60, 150)]
start_times =sorted([times[0] for times in input_list])#[0,30,40,50,60,80]
end_times =sorted([times[1] for times in input_list])#[50,52,55,75,77,150]

start_times = [0,30,40,50,60,80]
end_times = [50,52,55,75,77,150]
start_time_ptr = 0
end_time_ptr = 0

max_rooms_needed = 0
curr_rooms_req = 0

while(start_time_ptr < len(start_times)):
    curr_rooms_req += 1    
    
    while(end_times[end_time_ptr] <= start_times[start_time_ptr]):
        curr_rooms_req -= 1
        end_time_ptr += 1
    
    max_rooms_needed = max(max_rooms_needed, curr_rooms_req)
    start_time_ptr += 1


DIP
Hi, here is your problem today. This problem was recently asked by Google:

You are given a stream of numbers. Compute the median for each new element .

Eg. Given [2, 1, 4, 7, 2, 0, 5], the algorithm should output 
[2, 1.5, 2, 3.0, 2, 2, 2]
https://leetcode.com/problems/find-median-from-data-stream/solution/
https://www.geeksforgeeks.org/median-of-stream-of-integers-running-integers/

import bisect
input_list = [2, 1, 4, 7, 2, 0, 5]
sorted_input_list = []
output_list = []

  
for element in input_list:
    bisect.insort(sorted_input_list, element)
    len_sort_inp_list = len(sorted_input_list)
    mid_of_list = len_sort_inp_list / 2

    if len_sort_inp_list % 2 == 0:
        median = float(sorted_input_list[mid_of_list] + \
        sorted_input_list[mid_of_list - 1]) / float(2)
    else:
        median = sorted_input_list[mid_of_list]
    
    output_list.append(median)


DIP
Hi, here is your problem today. This problem was recently asked by AirBNB:

Given a list of words, group the words that are anagrams of each other. (An 
anagram are words made up of the same letters).

Example:

Input: ['abc', 'bcd', 'cba', 'cbd', 'efg']
Output: [['abc', 'cba'], ['bcd', 'cbd'], ['efg']]

from collections import Counter
word_list = ['abc', 'bcd', 'cba', 'cbd', 'efg']
anagram_dict = {}
output_list = []

for word in word_list:
    character_occurances = Counter(word).items()
    key = ''
    
    for item in character_occurances:
        key += item[0] + str(item[1])
    
    key = ''.join(sorted(key))
    print key
    
    if key in anagram_dict:
        anagram_dict[key].append(word)
    else:
        anagram_dict[key] = [word]

for value in anagram_dict.values():
    output_list.append(value)


DIP
https://leetcode.com/problems/valid-parenthesis-string/

Hi, here is your problem today. This problem was recently asked by Uber:

You are given a string of parenthesis. Return the minimum number of parenthesis 
that would need to be removed in order to make the string valid. "Valid" means 
that each open parenthesis has a matching closed parenthesis.

Example:

input_string = "()())()"

The following input should return 1.

")("

input_string = "()()()**))"
input_string = '**(('
#the following is the sol for the leetcode problem mentioned above
class Solution(object):
    def checkValidString(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if not s:
            return True
        
        input_string = s
        stack = []

        for index, char in enumerate(input_string):
            if char == '(':
                stack.append(str(index) + char)
            elif char == ')':
                if stack: stack.pop()
                else: 
                    #print 'ivalid string'
                    return False
            elif char == '*':
                stack = [str(index) + char] + stack
            #print stack
        
        if not stack:
            return True
        
        stack_list = []
        bracket_list = []
        
        for item in stack:
            if '*' in item:
                stack_list = [item] + stack_list #[14*, 17*, 48*]
            else:
                bracket_list.append(item)#[34(, 47(]
                
        if not bracket_list:
            return True
        
        bracket_list.reverse()
        for item in bracket_list:
            if not stack_list:
                return False
            ele = stack_list.pop()
            #print ele
            #print item
            if int(ele.split('*')[0]) > int(item.split('(')[0]):
                continue
            else:
                return False
            
        return True


Hi, here is your problem today. This problem was recently asked by LinkedIn:

Given a 2-dimensional grid consisting of 1's (land blocks) and 0's (water blocks),
count the number of islands present in the grid. The definition of an island is 
_as follows:
#1.) Must be surrounded by water blocks.
#2.) Consists of land blocks (1's) connected to adjacent land blocks (either 
#vertically or horizontally).
Assume all edges outside of the grid are water.

Example:
Input: 
10001
11000
10110
00000

Output: 3

no_of_islands = 0
visited_cells_dict = {}
all_directions = [(0,1), (1,0), (0,-1), (-1,0)]

def do_bfs(row, col):
    q = []
    q.append((row,col))
    while(q):
        #do bfs and append cells with 1 to queue and add visited cells to 
        #visited_cells_dict
    no_of_islands += 1
    return

for row in range(len(matrix)):
    for col in range(len(row)):
        if matrix[row][col] == 1 and not str(row)+str(col) in visited_cells_dict:
            do_bfs(row,col)


Hi, here is your problem today. This problem was recently asked by Twitter:

You are given the root of a binary tree. Find and return the largest subtree of 
that tree, which is a valid binary search tree.

# Fill this in.

#     5
#    / \
#   6    7
#  /   /   \
# 2   4     9
#      \    /
#       5  8
min_level = 0 #Min level->Min level = max height
bst_root_with_max_ht = None
def dfs(root, level):
    if not root:
        return None

    curr_max_left, _ = dfs(root.left, level+1)
    _, curr_min_right = dfs(root.right, level+1)

    if not root.value < curr_min_right or not root.value > curr_max_left:
        print 'Tree unbalanced from here ', root.value
    else:
        if level < min_level:
            bst_root_with_max_ht = root

    if not curr_max_left:
        return root.value, min(curr_min_right, root.value)
    
    if not curr_min_right:
        return max(curr_max_left, root.value), root.value
    
    if not curr_max_left and curr_min_right:
        return root.value, root.value
    
    return max(curr_max_left, root.value), min(curr_min_right, root.value)


Hi, here is your problem today. This problem was recently asked by Apple:

You are given the root of a binary tree, along with two nodes, A and B. Find and 
return the lowest common ancestor of A and B. For this problem, you can assume that
each node also has a pointer to its parent, along with its left and right child.

class TreeNode:
  def __init__(self, val):
    self.left = None
    self.right = None
    self.parent = None
    self.val = val


def lowestCommonAncestor(root, a, b):
  # Fill this in.

#   a
#  / \
# b   c
#    / \
#   d*  e*

node_1 = A
node_2 = B

node_1_parent_dict = {}

while(node_1):
    if node_1.parent == node_2:
        return node_2
    else:
        node_1_parent_dict[node_1.parent] = True
    node_1 = node_1.parent

while(node_2):
    if node_2.parent == node_1:
        return node_1
    elif node_2 in node_1_parent_dict:
        return node_2
    else:
        node_2 = node_2.parent


Hi, here is your problem today. This problem was recently asked by Twitter:

Given an array, nums, of n integers, find all unique triplets (three numbers, a, b
, & c) in nums such that a + b + c = 0. Note that there may not be any triplets 
that sum to zero in nums, and that the triplets must not be duplicates.

Example:
Input: nums = [0, -1, 2, -3, 1]
Output: [0, -1, 1], [2, -3, 1]

input_nums = sorted([[0, -1, 2, -3, 1]])#[-4,-4,-3,-2,-2,-1,0,1,2,2,5,7,8]
input_nums_dict = {}
triplets = []

for index, num in input_nums:
    if num > 0:
        break

    neg_int_ptr = index
    pos_int_ptr = len(input_nums) - 1
    while(pos_int_ptr >= 0):
        positive_integer = input_nums[pos_int_ptr]
        difference = -(num + positive_integer)# -4 + 8 = 4 (we need a -4 to get 0)
        if difference == num and input_nums_dict[num] > 1:
            triplets.append(num,num,positive_integer)#(-4,-4,8)
        
        elif difference == positive_integer and \
        input_nums_dict[positive_integer] > 1:
            triplets.append(num,positive_integer, positive_integer)#(-4,2,2)

        elif difference in input_nums_dict:
            triplets.append(num, positive_integer, difference)

        pos_int_ptr -= 1

return set(triplets)


Hi, here is your problem today. This problem was recently asked by Apple:

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

order="zyxwvutsrqponmlkjihgfedcba"
order_dict = {}

for index, char in enumerate(order):
    order_dict[char] = index

for index, word in enumerate(words[:-1]):
    next_word = words[index + 1]
    min_len_word = min(len(word), len(next_word))
    index = 0
    while(index < min_len_word):
        char_1 = word[index]
        char_2 = next_word[index]
        if order_dict[char_1] <= order_dict[char_2]:
            continue
        else:
            print "not in order"
            break


DIP
Hi, here is your problem today. This problem was recently asked by Apple:

Given an array with n objects colored red, white or blue, sort them in-place so 
that objects of the same color are adjacent, with the colors in the order red, 
white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and 
blue respectively.

Note: You are not suppose to use the library sort function for this problem.

Can you do this in a single pass?

Example:
Input: [2,0,2,1,1,0] #[2,0,2,1,0,2,1,0]
Output: [0,0,1,1,2,2]

input_list = [2,0,2,1,0,2,1,0]
index = 0
len_input_list = len(input_list)
zero_index = len(input_list)
two_index = len(input_list) * 2 - 1
input_list.extend([1] * len_input_list)

while(index < len_input_list):
    print input_list[index]
    if input_list[index] == 0:
        input_list[zero_index] = 0
        zero_index += 1
    elif input_list[index] == 2:
        input_list[two_index] = 2
        two_index -= 1
    index += 1

print input_list[len_input_list]


DIP ****************SOLVE LATER*************
https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/solution/
Hi, here is your problem today. This problem was recently asked by Microsoft:

You are given the preorder and inorder traversals of a binary tree in the form of 
arrays. Write a function that reconstructs the tree represented by these traversals.

Example:
Preorder: [a, b, d, e, c, f, g]
Inorder:  [d, b, e, a, f, c, g]
#albrc 

The tree that should be constructed from these traversals _is:

#    a
#   / \
#  b   c
# / \ / \
#d  e f  g


DIP
Hi, here is your problem today. This problem was recently asked by Microsoft:

A unival tree is a tree where all the nodes have the same value. Given a binary tree, return the number of unival subtrees in the tree.

For example, the following tree should return 5:

   0
  / \
 1   0
    / \
   1   0
  / \
 1   1

The 5 trees are:
- The three single '1' leaf nodes. (+3)
- The single '0' leaf node. (+1)
- The [1, 1, 1] tree at the bottom. (+1)

Do a DFS and return is_unival for ro

no_of_unival_trees = 0
def count_unival_trees(root, is_unival):
    if not root:
        return True
    
    is_unival_left = count_unival_trees(root.left)
    is_unival_right = count_unival_trees(root.left)

    if not is_unival_left or not is_unival_right:
        return False

    if root.val == root.left.val == root.right.val and is_unival_left and\
     is_unival_right:
         no_of_unival_trees += 1
         is_unival = True
    
    elif not root.left and not root.right:
        no_of_unival_trees += 1
        is_unival = True
    
    elif root.left and not root.right and root.val == root.left.val and \
    is_unival_left:
        no_of_unival_trees += 1
        is_unival = True
    
    elif root.right and not root.left and root.val == root.right.val and \
    is_unival_right:
        no_of_unival_trees += 1
        is_unival = True
    
    else:
        is_unival = False

    return is_unival

Hi, here is your problem today. This problem was recently asked by Amazon:

You are given a string s, and an integer k. Return the length of the longest 
substring in s that contains at most k distinct characters.

For instance, given the string:
aabcdefff and k = 3, then the longest substring with 3 distinct characters would be
defff. The answer should be 5.

#this shouldn't be hard. Just make use of dictionary
from collections import OrderedDict

window_dict = OrderedDict({})
window_begin_index = 0
my_string = 'aabcdefff'
max_window_size = 0

for index, char in enumerate(my_string):
    if char in window_dict:
        window_dict.move_to_end(char)
        window_dict[char] = index
    elif len(window_dict.keys()) < window_size:
        window_dict[char] = index
        window_dict.move_to_end(char)
    else:
        window_size = index - window_begin_index
        to_be_removed_key = window_dict.keys()[0]
        window_begin_index = window_dict[to_be_removed_key] + 1
        window_dict.pop(to_be_removed_key)


DIP
Hi, here is your problem today. This problem was recently asked by Amazon:

Given a binary tree, return all values given a certain height h.

#Do a dfs and use a dict so store values at diff heights


DIP
Hi, here is your problem today. This problem was recently asked by Twitter:

Given an array of integers of size n, where all elements are between 1 and n 
inclusive, find all of the elements of [1, n] that do not appear in the array. 
Some numbers may appear more than once.

Example:
Input: [4,5,2,6,8,2,1,5]
Output: [3,7]

For this problem, you can assume that you can mutate the input array.


DIP
Hi, here is your problem today. This problem was recently asked by Facebook:

You are given the root of a binary search tree. Return true if it is a valid binary
search tree, and false otherwise. Recall that a binary search tree has the property
that all values in the left subtree are less than or equal to the root, and all
values in the right subtree are greater than or equal to the root.

#    5
#   / \
#  3   7
# / \ /
#1  8 4
#      \
#       6


#The following approach wont work for the foll case. The reason is we are trying to
#do a bottom up approach in the below solution. We check for bst cond after the 
#recursive call returns. 

#Rather what we should be doing is do a top down approach where we pass the values
#of the predecessor roots down the tree (just 2 boundaries it should be in).
#See the second solution for understanding how this is implemented.

#However, the easiest way is to do inorder traversal of the tree and see if i+1
#index is greater than i th index

#     3
#    /
#   1
#  / \ 
# 0   2
#      \
#       4
#First solution (wrong - won't work for all cases)
def check_bst(root):
    if not root:
        return None, None

    left_max, _ = check_bst(root.left)
    _, right_min = check_bst(root.right)

    if left_max and root.value < left_max:
        print 'Not a bst'
        return False

    if right_min and root.value > right_min:
        print 'Not a bst'
        return False
    
    if left_max and right_min:
        return right_min, left_max
    elif left_max:
        return root.value, left_max
    elif right_min:
        return right_min, root.value
    else:
        return root.value, root.value

#second solution (correct solution)
def helper(node, lower = float('-inf'), upper = float('inf')):
    if not node:
        return True
    
    val = node.val
    if val <= lower or val >= upper:
        return False

    if not helper(node.right, val, upper):
        return False
    if not helper(node.left, lower, val):
        return False
    return True

Hi, here is your problem today. This problem was recently asked by Facebook:

You are given an array of integers. Return the smallest positive integer that is 
not present in the array. The array may contain duplicate entries.

For example, the input [3, 4, -1, 1] should return 2 because it is the smallest 
positive integer that does not exist in the array.

Your solution should run in linear time and use constant space.

O(n) and O(1)
https://www.geeksforgeeks.org/find-a-repeating-and-a-missing-number/
The following method is assuming
Given an unsorted array of size n. Array elements are in the range from 1 to n

input_list = [7, 3, 4, 5,-4, -3, 5,-5, 6, 2]
index = 0
while(index < len(input_list)): # removing the -ve nums from list
    if input_list[index] < 0:
        if index == len(input_list) - 1:
            input_list = input_list[:-1]
            break
        input_list = input_list[:index] + input_list[index + 1:]
        continue
    index += 1

#input_list becomes [7, 3, 4, 5, 5, 6, 2]
max_num = max(input_list)
#sum_of_range = (max_num * (max_num + 1)) / 2
index = 0

while(index < len(input_list)):
    new_index = abs(input_list[index]) - 1
    ele_at_new_index = input_list[new_index]
    
    if ele_at_new_index < 0:
        print 'new_index is a repeating number. \
        Do not consider this ', ele_at_new_index
        index += 1
        continue
    else:
        input_list[new_index] = -input_list[new_index]
        index += 1

for ind, ele in enumerate(input_list):
    if ele > 0:
        print 'the missing ele is ', ind + 1
        break


DIP
Hi, here is your problem today. This problem was recently asked by Google:

A look-and-say sequence is defined _as the integer sequence beginning with a single
digit in which the next term is obtained by describing the previous term. An example
is easier to understand:

Each consecutive value describes the prior value.

1      #
11     # one 1's
21     # two 1's
1211   # one 2, and one 1.
111221 # #one 1, one 2, and two 1's.
312211
13112221

Your task is, return the nth term of this sequence.

print 1
print 11
n = 5
prev_output = str(11)
i = 2
while(i < n):
    current_output = ''
    prev_ele = prev_output[0]
    count = 1
    i_2 = 1

    while i_2 < len(prev_output):
        if prev_output[i_2] == prev_ele:
            count += 1
        else:
            current_output += (str(count) + str(prev_ele))
            prev_ele = prev_output[i2]
            count = 1
        
        i_2 += 1
    
    current_output += (str(count) + str(prev_ele))
    print current_output
    prev_output = current_output
    i += 1

from collections import Counter
my_string = 'aabbccc'

def max_poss_pal_substring(my_string):
    if not my_string:
        return None

    character_dict = {}
    single_char_key = None
    output_string = ''
    
    for char in my_string:
        if char in character_dict:
            character_dict[char] += 1
        else:
            character_dict[char] = 1

    for key in character_dict.keys():
        if character_dict[key] == 1:
            character_dict.pop(key)
            single_char_key = key
            continue
        elif character_dict[key] % 2 == 1:
            character_dict[key] -= 1
            single_char_key = key

        no_of_occurances = character_dict[key]
        additional_chars = ''.join([key for i in range(no_of_occurances)])
        output_string = additional_chars[0:no_of_occurances / 2] + output_string + \
        additional_chars[0:no_of_occurances / 2] #adding half to the front of output
        #and adding other half to the end of output

    if single_char_key:
        output_len = len(output_string)
        output_string = output_string[0:output_len/2] + single_char_key + \
        output_string[output_len/2:]

    return output_string

max_poss_pal_substring('babcadbbb') 
#returns bbadabb
max_poss_pal_substring('malayalam') 
#returns lmaayaaml
max_poss_pal_substring('abc') 
#returns b



def no_of_jumps(remaining_steps, memo):
    if remaining_steps < 0:
        return 0
    elif remaining_steps == 0:
        return 1
    elif remaining_steps == 1:
        return 1
    elif remaining_steps == 2:
        return 2

    if remaining_steps in memo:
        return memo[remaining_steps]
    
    no_of_poss_three_jumps = no_of_jumps(remaining_steps - 3, memo)
    no_of_poss_two_jumps = no_of_jumps(remaining_steps - 2, memo)
    no_of_poss_one_jumps = no_of_jumps(remaining_steps - 1, memo)

    memo[remaining_steps] = no_of_poss_three_jumps + no_of_poss_two_jumps + \
        no_of_poss_one_jumps

    return memo[remaining_steps]


Minimum Cost Tree From Leaf Values
https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/

Question:
Given an array arr of positive integers, consider all binary trees such that:

- Each node has either 0 or 2 children;
- The values of arr correspond to the values of each leaf in an in-order traversal of the 
tree.  (Recall that a node is a leaf if and only if it has 0 children.)
- The value of each non-leaf node is equal to the product of the largest leaf value in its 
left and right subtree respectively.

Among all possible binary trees considered, return the smallest possible sum of the values
of each non-leaf node.  It is guaranteed this sum fits into a 32-bit integer.

**The tricky part is the following line
- The values of arr correspond to the values of each leaf in an in-order traversal of the 
tree.  (Recall that a node is a leaf if and only if it has 0 children.)

It indirectly says that you can pick only the immediate left neighbour or the immediate
right neighbour of any element when constructing your tree

Lets review the problem again.


The problem can translated as_ following:
Given an array A, 
1 initialize sum = 0
2 find the min ele 'e' in the array
3 pick the smallest neighbour(immediate left or immediate right) of 'e', sat 'e1'
4 multiply e and e1 and add it to sum. Take out e from array.
5 Continue steps 2 - 4 untill no elem remains in the array

Solution 1
With the intuition above in mind,
the explanation is short to go.

We remove the element form the smallest to bigger.
We check the min(left, right),
For each element a, cost = min(left, right) * a

Solution 1
Time O(N^2)
Space O(N)
    def mctFromLeafValues(self, A):
        res = 0
        while len(A) > 1:
            i = A.index(min(A))
            res += min(A[i - 1:i] + A[i + 1:i + 2]) * A.pop(i)
        return res

Solution 2 Time - O(n) space O(n)
#keep in mind 1 thing. the ele at index 0 of stack is infinity
class Solution(object):
    def mctFromLeafValues(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        A = arr
        res = 0
        stack = [float('inf')]
        for a in A:
            while stack[-1] <= a:
                mid = stack.pop()
                res += mid * min(stack[-1], a)
            stack.append(a)
        print stack
        print res
        while len(stack)  >2: #keep in mind 1 thing.The ele at ind 0 is infinity
            res += stack.pop() * stack[-1]
        return res


Segregate Odd even num -Mathworks internship
def segregateEvenOdd(arr): 
  
    # Initialize left and right indexes 
    left,right = 0,len(arr)-1
  
    while left < right: 
  
        # Increment left index while we see 0 at left 
        while (arr[left]%2==0 and left < right): 
            left += 1
  
        # Decrement right index while we see 1 at right 
        while (arr[right]%2 == 1 and left < right): 
            right -= 1
  
        if (left < right): 
              # Swap arr[left] and arr[right]*/ 
              arr[left],arr[right] = arr[right],arr[left] 
              left += 1
              right = right-1


DP
https://www.hackerearth.com/practice/algorithms/dynamic-programming/introduction-to-dynamic-programming-1/tutorial/
Number of ways to count to N with 1,3,5
dp = {}
dp[0] = dp[1] = dp[2] = 1 #2 = 1 + 1
dp[3] = 2 # 1+1+1 or 3+0
dp[4] = 2 #1+1+1+1 or 3+1
dp[5] = 2
n = 5
i= 6

while(i < n):
    dp[i] = dp[i - 5] + dp[i - 3] + dp[i -1]
    print 'i = ', i
    print 'dp = ', dp[i]
    i += 1


DP
https://www.hackerearth.com/practice/algorithms/dynamic-programming/introduction-to-dynamic-programming-1/tutorial/
p1=2, p2=3, p3=5, p4=1, p5=4.
my_l = [2, 3, 5, 1, 4] #imagine how dp works for [8,2,3,5,1,4]
#Let's say you compute 84,24,34....31
#Insetad of choosing 8 in the first place you pick 4 so, it 24. If you recollect
#from the prev step, 24 has already been computed. So, use the mem result
mem = {}

def max_profits(year, result, st, end):
    global mem

    if st > en:
        result += my_l[st] * year
        return result

    if str(st) + str(en) in mem:
        return mem[str(st) + str(en)]

    prof_1 = max_profits(year + 1, result + my_l[st] * year, st + 1, en)
    prof_2 = max_profits(year + 1, result + my_l[st] * year, st, en - 1)
    mem[str(st) + str(en)] = max(prof_1, prof_2)
    return mem[str(st) + str(en)]


valid sudoku board
def get_next_grid_starting_point(grid_starting_point):
    row = grid_starting_point[0]
    col = grid_starting_point[1]
    new_col = col + 3
    
    if new_col >= 9:
        new_row += 3
        new_col = 0
    return (new_row, new_col)

def check_grid(grid_starting_point):
    row = grid_starting_point[0]
    col = grid_starting_point[1]
    grid_list = []
    for comb in all_cells_grid:
        new_row = row + comb[0]
        new_col = col + comb[1]
        val = matrix[new_row][new_col]
        if val in rows_dict[new_row] or val in rows_dict[new_col] or\
        val in grid_list:
            return False
        else:
            rows_dict[new_row].append(value)
            rows_dict[new_col].append(value)
            grid_list.append(value)

rows_dict = {}
cols_dict = {}
grid_dict = {}
all_cells_grid = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
total_grids = 9
current_grid = 0
grid_starting_point = (0,0)

while(current_grid < total_grids):
    is_valid_grid = check_grid(grid_starting_point)
    new_row, new_col = get_next_grid_starting_point(grid_starting_point)
    current_grid += 1


DIP
Hi, here is your problem today. This problem was recently asked by Google:

You are given the root of a binary tree. Return the deepest node (the furthest node 
_from the root).

Example:

    a
   / \
  b   c
 /
d

The deepest node in this tree is d at depth 3.

#Do a dfs and count the number of times you go deep
#Have a max counter as global variable and update it if the height is greater than
#the max height
max_height = 0
def get_max_ht(node, height):
    if not node:
        return

    get_max_ht(node.left, height + 1)
    get_max_ht(node.right, height + 1)
    max_height = max(height, max_height)


DIP
Hi, here is your problem today. This problem was recently asked by AirBNB:

Given two strings A and B of lowercase letters, return true if and only if we can 
swap two letters in A so that the result equals B.

Example 1:
Input: A = "ab", B = "ba"
Output: true

Example 2:

Input: A = "ab", B = "ab"
Output: false

Example 3:
Input: A = "aa", B = "aa"
Output: true

Example 4:
Input: A = "aaaaaaabc", B = "aaaaaaacb"
Output: true

Example 5:
Input: A = "", B = "aa"
Output: false

if len(str_a) != len(str_b):
    print 'not possible'

no_of_mismatch = 0
while(i < len(str_a)):
    if str_a[i] != str_b[i]:
        no_of_mismatch += 1
        if no_of_mismatch >= 2:
            print 'no_possible'
            break


DIP
Hi, here is your problem today. This problem was recently asked by Uber:

You have a landscape, in which puddles can form. You are given an array of 
non-negative integers representing the elevation at each location. Return the amount
of water that would accumulate if it rains.

For example: [0,1,0,2,1,0,1,3,2,1,2,1] should return 6 because 6 units of water can 
get trapped here. See the comment below the diagram to understand better.

           X               
       X███XX█X              
     X█XX█XXXXXX                   
#  [010210132121]

input_list = [0,1,0,2,1,0,1,3,2,1,2,1]
max_left = []
max_right = []
units_of_water = 0

max_left.append(input_list[0])
for land in input_list:
    if land > max_left[-1]:
        max_left.append(land)
    else:
        max_left.append(max_left[-1])

input_list.reverse()
max_right.append(input_list[0])
for land in input_list:
    if land > max_left[-1]:
        max_right.append(land)
    else:
        max_right.append(max_right[-1])

input_list.reverse()
max_right.reverse()
for index, land in enumerate(input_list):
    units_of_water += min(max_left, max_right) - land



Leetcode - Google
https://leetcode.com/problems/search-in-rotated-sorted-array/
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.

Your algorithms runtime complexity must be in the order of O(log n).

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
def search(self, nums, target):
        if not nums:
            return -1

        low, high = 0, len(nums) - 1
        
        while low <= high:
            mid = (low + high) / 2
            if target == nums[mid]:
                return mid

            if nums[low] <= nums[mid]:
                if nums[low] <= target and nums[mid] >= target:
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                if target >= nums[mid] and target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1

        return -1


Leetcode - Google
https://leetcode.com/problems/text-justification/

Example 1:

Input:
words = ["This", "is", "an", "example", "of", "text", "justification."]
maxWidth = 16
Output:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]
Example 2:

Input:
words = ["What","must","be","acknowledgment","shall","be"]
maxWidth = 16
Output:
[
  "What   must   be",
  "acknowledgment  ",
  "shall be        "
]
Explanation: Note that the last line is "shall be    " instead of "shall     be",
             because the last line must be left-justified instead of fully-justified.
             Note that the second line is also left-justified becase it contains only one word.
Example 3:

Input:
words = ["Science","is","what","we","understand","well","enough","to","explain",
         "to","a","computer.","Art","is","everything","else","we","do"]
maxWidth = 20
Output:
[
  "Science  is  what we",
  "understand      well",
  "enough to explain to",
  "a  computer.  Art is",
  "everything  else  we",
  "do                  "
]

#need to check if the code works fine

input_words = ["This", "is", "an", "example", "of", "text", "justification."]
stack = []
output_line = ''
len_curr_line = 0
max_width = 16

while(input_words):
    len_curr_line += len(input_words[0] + ' ')
    if len_curr_line - 1 <= max_width:
        stack.append(input_words[0] + ' ')
        input_words = input_words[1:]
    else:
        len_curr_line -= len(input_words[0] + ' ') - 1#-1 is for the last space
        no_of_spaces_needed = max_width - len_curr_line
        space_between_each_char = no_of_spaces_needed / len(stack) 
        miscillaneous_spaces_needed = no_of_spaces_needed % len(stack) 
        
        for ele in stack:
            output_line += ele + ''.join([' ' for i in range(space_between_each_char)])
            if miscillaneous_spaces_needed:
                output_line += ' '
                miscillaneous_spaces_needed -= 1
        
        output_line.strip()
        output_line += '\n'
        len_curr_line = 0
        stack = []

for ele in stack:
    output_line += ele

print output_line


Leetcode - google
Next greatest time
https://leetcode.com/problems/next-closest-time/solution/
Given a time represented in the format "HH:MM", form the next closest time by 
reusing the current digits. There is no limit on how many times a digit can be 
reused.

Input: "19:34"
Output: "19:39"

Input: "23:59"
Output: "22:22"

My sol
import itertools
from datetime import datetime

class Solution(object):
    def nextClosestTime(self, time):
        """
        :type time: str
        :rtype: str
        """
        if time[0] == time[1] == time[3] == time[4]:
            return time[0] + time[1] + ':' + time[3] + time[4]
        
        orig_time = time
        time = time.replace(':','')
        
        all_combs = itertools.product(time,repeat=4)
        all_combs = [item for item in all_combs]
        curr_min = None
        
        for mal_formed_str in all_combs:
            time_str = mal_formed_str[0] + mal_formed_str[1] + ':' + mal_formed_str[2] + mal_formed_str[3]
            if time_str == orig_time:
                continue
            try:
                time_obj = datetime.strptime(time_str,'%H:%M')
                #print time_obj
                time_diff = (time_obj - datetime.strptime(orig_time,'%H:%M')).total_seconds()
                if time_diff < 0:
                    continue
                #print time_diff
                if not curr_min:
                    curr_min_time = time_str
                    curr_min = time_diff
                elif  time_diff < curr_min:
                    curr_min_time = time_str
                    curr_min = time_diff
            except:
                continue
        
        if not curr_min:
            for mal_formed_str in all_combs:
                time_str = mal_formed_str[0] + mal_formed_str[1] + ':' + mal_formed_str[2] + mal_formed_str[3]
                if time_str == orig_time:
                    continue
                try:
                    time_obj = datetime.strptime(time_str,'%H:%M')
                    #print time_obj
                    time_diff = abs((time_obj - datetime.strptime(orig_time,'%H:%M')).total_seconds())
                    #print time_diff
                    if not curr_min:
                        curr_min_time = time_str
                        curr_min = time_diff
                    elif  time_diff > curr_min:
                        curr_min_time = time_str
                        curr_min = time_diff
                except:
                    continue
            
        return curr_min_time
            


Recommended sol
class Solution(object):
    def nextClosestTime(self, time):
        ans = start = 60 * int(time[:2]) + int(time[3:])
        print start
        elapsed = 24 * 60
        allowed = {int(x) for x in time if x != ':'}
        for h1, h2, m1, m2 in itertools.product(allowed, repeat = 4):
            hours, mins = 10 * h1 + h2, 10 * m1 + m2
            if hours < 24 and mins < 60:
                cur = hours * 60 + mins
                print hours, mins
                print cur
                cand_elapsed = (cur - start) % (24 * 60)
                print cand_elapsed
                if 0 < cand_elapsed < elapsed:
                    ans = cur
                    elapsed = cand_elapsed

        return "{:02d}:{:02d}".format(*divmod(ans, 60))


Leetcode - Google
Diamenter of a binary tree
https://leetcode.com/problems/diameter-of-binary-tree/

Calculate the left and right height of tree using dfs. Sum the height and return

Leetcode - Google
https://leetcode.com/problems/string-transforms-into-another-string/submissions/
class Solution(object):
    def canConvert(self, str1, str2):
        """
        :type str1: str
        :type str2: str
        :rtype: bool
        """
        my_d = {}
        word_len = len(str1)
        
        index = 0
        while(index < word_len):
            #print index
            key_1 = str1[index]
            val_1 = str2[index]
            #print key_1
            #print val_1
            if key_1 in my_d and my_d[key_1] != val_1:
                return False
            
            my_d[key_1] = val_1
            index += 1
        
        if str1 == str2:
            return True
        return len(set(str2)) < 26 
            

Leetcode - Google
Confusing Numbers
https://leetcode.com/problems/confusing-number-ii/submissions/

Example 1:

Input: 20
Output: 6
Explanation: 
The confusing numbers are [6,9,10,16,18,19].
6 converts to 9.
9 converts to 6.
10 converts to 01 which is just 1.
16 converts to 91.
18 converts to 81.
19 converts to 61.
Example 2:

Input: 100
Output: 19
Explanation: 
The confusing numbers are [6,9,10,16,18,19,60,61,66,68,80,81,86,89,90,91,98,99,100].

import itertools
class Solution(object):
    def confusingNumberII(self, N):
        """
        :type N: int
        :rtype: int
        """
        ******* Good Solution ******
        valid = [0,1,6,8,9]
        mapping = {0: 0,1: 1,6: 9,8: 8, 9: 6}

        self.count = 0

        def backtrack(v, rotation,digit):
            if v: 
                if v != rotation: 
                    self.count += 1  
            for i in valid: 
                if v*10+i > N:
                    break 
                else:
                    backtrack(v*10+i, mapping[i]*digit + rotation, digit*10)
        
        backtrack(1,1, 10)
        backtrack(6,9,10)
        backtrack(8,8,10)
        backtrack(9,6,10)

        return self.count
        
        ******* My Solution ******
        confusing_numbers_str = '01689'
        all_permutations = itertools.product(confusing_numbers_str, repeat=len(str(N)))
        confusing_numbers_dict = {0:0,1:1,6:9,8:8,9:6}
        confusing_numbers = 0
        
        for permutaion in all_permutations:
            orig_num = num = int(''.join(permutaion))
            if num > N:
                continue
            new_num = 0
            
            while(num):
                dig = num % 10
                if dig in confusing_numbers_dict:
                    new_num = new_num * 10 + confusing_numbers_dict[dig]
                num = num / 10
                
            if orig_num != new_num:
                #print 'num = ', num
                #print 'new_num = ', new_num
                confusing_numbers += 1

        return confusing_numbers


Google - leetcode
https://leetcode.com/problems/expressive-words/submissions/

Input: 
S = "heeellooo"
words = ["hello", "hi", "helo"]
Output: 1
Explanation: 
We can extend "e" and "o" in the word "hello" to get "heeellooo".
We can't extend "helo" to get "heeellooo" because the group "ll" is not size 3 or more.

import itertools
class Solution(object):
    def expressiveWords(self, S, words):
        """
        :type S: str
        :type words: List[str]
        :rtype: int
        """
        char_groupbys = itertools.groupby(S)
        split_word_tuples = []
        extend_words = 0
        
        for item in char_groupbys:
            ele = item[0]
            no_of_occurances = len(list(item[1]))
            split_word_tuples.append((ele, no_of_occurances))
        
        for word in words:
            char_groupbys = itertools.groupby(word)
            can_be_extended = True
            
            for ind, item in enumerate(char_groupbys):
                char = item[0]
                no_of_occurances = len(list(item[1]))
                
                if char != split_word_tuples[ind][0] or no_of_occurances > split_word_tuples[ind][1]:
                    can_be_extended = False
                    break
                
                if split_word_tuples[ind][1] < 3:
                    if no_of_occurances != split_word_tuples[ind][1]:
                        can_be_extended = False
                        break
            
            if can_be_extended and ind + 1 == len(split_word_tuples):
                extend_words += 1
                
        return extend_words


Step number generation
def construct_tree(root): #8 89
    if root > input_range:#1000
        return
    
    least_sig_dig = root % 10 #8 9
    if least_sig_dig != 0:
        left_child = (root * 10) + least_sig_dig - 1 # 898
    if least_sig_dig != 9:
        right_child = (root * 10) + least_sig_dig + 1 #89 
    print left_child
    print right_child
    construct_tree(left_child)
    construct_tree(right_child)


for i in range(1,10): #0-9
    print i
    construct_tree(i)
    
num = '121'
step_number = True
while(num > 9):#121, 12, 1 
    last_dig = int(str(num)[-1])  #1, 2
    second_last_digit = int(str(num)[-2]) #2 1
    if abs(last_dig - second_last_dig) == 1:
        num = int(str(num)[0:-1]) #12
        continue
    else:
    step_number = False
        break

 
O(nlog(n))

Dynamic programming tutorial wine problem
https://www.hackerearth.com/practice/algorithms/dynamic-programming/introduction-to-dynamic-programming-1/tutorial/
# refer recursion tree dia (memo_rec_wine.jpeg)
#                                   (0,4)
#                                    /   \
#                               (1,4)            (0,3)
#                             /       \
#                        (2,4)         (1,3)
#                        /    \       /     \ 
#                    (3,4)     (2,3) (2,3)   (1,2)
#       --------------/\        /  \
#    (4,4)             (3,3)  (3,3) (2,2)
#    /\                 /\
#(5,4)(4,3)         (4,3)(3,2)

wine_price = [2, 3, 5, 1, 4] # refer recursion tree dia (memo_rec_wine.jpeg)
#indices   = [0, 1, 2, 3, 4]
curr_profit = 0
memo = {}

#we don't pass year as an argument to the function because
# - if start and year are indices x and y, No matter what combination you had 
#   chosen in your previous step, year would the same
# eg: let's say my start and end are 2 and 3, no matter how you reached this ind
#     (could be (0,1,4) or (0,4,1) or (4,0,1) you have completed 3 years because
#      3 inputs have already been picked), the curr year is 4.
# - Now our sub problem is, given year as 4, should we pick wine at index 2 (which
#   is 5 or should we pick wine at index 3 (which is 1)

def maximize_profit(start, end):
    global curr_profit
    if start > end:
        return 0
    
    print 'st, end = ', start, '\t', end
    key = str(start) + str(end)#LINE ADDED FOR MEMO
    if key in memo:#LINE ADDED FOR MEMO
        return memo[key]#LINE ADDED FOR MEMO

    year = len(wine_price) - (end - start)
    print 'year = ', year
    profit_1 = year * wine_price[start] + maximize_profit(start + 1, end)
    profit_2 = year * wine_price[end] + maximize_profit(start, end - 1)
    print 'profit_1, profit_2 = ', profit_1, '\t', profit_2

    memo[key] = max(profit_1, profit_2)#LINE ADDED FOR MEMO
    #return max(profit_1, profit_2)
    return memo[key]#LINE ADDED FOR MEMO

maximize_profit(0, len(wine_price) - 1)


Class problem (Algorithms)
words = ['BOT', 'HEAR', 'EAR', 'A', 'HEART', 'HAND', 'AND', 'SATURN', 'SPIN', 'IN']
string = 'BOTHEARTHANDSATURNSPIN'
memo = {}

def is_splittable(start, end):
    with_word = False
    if start >= len(string):
        return True
    if end > len(string):
        return False
    
    key = str(start) + str(end)
    if key in memo and memo[key][0] == 'visited':
        return memo[key]
    
    memo[key] = ['visited', False]
    if string[start:end] in words:
        print 'string = ', string[start:end]
        with_word = is_splittable(end, end + 1)#this line cannot be 
        #is_s..(start, start + 1) because, let's consider 'spin'. We have found
        #that spin is a word, we need not check if 'pin' or 'in' is a word because
        #in order for 'pin' to be considered for a valid_splittable word, 's' should
        #be a valid splittable word. 
        #Let's for a moment assume that 's' is a valid splittable word.
        #what would have happened is, in the foll line
        #if string[0:1] (which is 's') in word: would be True
        #Now we would have started a recursion in the following line
        #with_word = is_splittable(start+1, end + 1) which would then tell us if
        #'s' and 'pin' is a valid split
        #Since 's' by itself do not make a meaningful word, we don't have to 
        #force ourselves to check if a group of characters after s is vaild.
        #Therefore we can only start a new recursion like below
        #with_word = is_splittable(start+1, end + 1) 
        #only if string[start:end] is a valid word
    without_word = is_splittable(start, end + 1)
    memo[key][1] = with_word or without_word
    return memo[key][1]

is_splittable(0,0)
#string[0:4]
#'BOTH'
#>>> memo['04']
#['visited', False]
#>>> memo['03']
#['visited', True]



Shortest way to form a string
https://leetcode.com/problems/shortest-way-to-form-string/

Example 1:

Input: source = "abc", target = "abcbc"
Output: 2
Explanation: The target "abcbc" can be formed by "abc" and "bc", which are subsequences of source "abc".
Example 2:

Input: source = "abc", target = "acdbc"
Output: -1
Explanation: The target string cannot be constructed from the subsequences of source string due to the character "d" in target string.
Example 3:

Input: source = "xyz", target = "xzyxz"
Output: 3
Explanation: The target string can be constructed as follows "xz" + "y" + "xz".

#xyzxa  xyzx
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
                continue
            else:
                source_dict[char] = True
                
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


DIP
Hi, here is your problem today. This problem was recently asked by Facebook:

Given an array and an integer k, rotate the array by k spaces. 
Do this without generating a new array and without using extra space.

Here is an example and some starter code

def rotate_list(nums, k):
  # Fill this in.


a = [1, 2, 3, 4, 5]
rotate_list(a, 2)
print(a)
#[3, 4, 5, 1, 2] this is anti clock wise rotation. see also clockwise_rotate(a, 2)
arr = [1, 2, 3, 4, 5]
k = 2
if k > len(arr):
    k -= (k % len(arr))

while(k > 0):
    temp = arr[0]
    arr.pop(0)
    arr.append(temp)
    k -= 1

print arr


a = [1,2,3,4,5]
clockwise_rotate(a, 2) #below is method 1: O(n) time O(1) space
print a
#[4,5,1,2,3]
arr = [1, 2, 3, 4, 5]
k = 2
len_arr = len(arr)
if k > len_arr:
    k -= (k % len_arr)

while(k > 0):
    temp = arr[-1]
    arr.pop(len_arr - 1)
    arr.insert(0, temp)
    k -= 1

print arr

#method 2: When we rotate an array by k, all the 'n' elements in the array will shift
#by k spaces
arr = [1, 2, 3, 4, 5]
k = 2
len_arr = len(arr)
if k > len_arr:
    k -= (k % len_arr)

curr_index = 0
temp = arr[curr_index]#1 

while(len_arr > 0):
    curr_index = (curr_index + k) % len(arr) #2 | 4 | 1 | 3 | 0
    temp, arr[curr_index] = arr[curr_index], temp #[1,2,1,4,5]; temp = 3 | [1,2,1,4,3];
    #temp = 5 | [1,5,1,4,3] temp=2 | [1,5,1,2,3] temp=4 | [4,5,1,2,3] temp=1
    len_arr -= 1 #4 3 2 1 0


Hi, here is your problem today. This problem was recently asked by Google:

Given a node in a binary search tree (may not be the root), find the next largest 
node in the binary search tree (also known as_ an inorder successor). The nodes in 
this binary search tree will also have a parent field to traverse up the tree.

node_found = False
target = 2

def inorder(root):
    global node_found
    global target
    if not root:
        return
    
    inorder(root.left)
    #print root.value
    if node_found:
        print root.value
        node_found = False
        return
    if root.value == target:
        node_found = True
    inorder(root.right)



Hi, here is your problem today. This problem was recently asked by Google:

Given a binary tree, find and return the largest path from root to leaf.

Just do a dfs and find out



Hi, here is your problem today. This problem was recently asked by AirBNB:

Given a phone number, return all valid words that can be created using that phone 
number.

For instance, given the phone number 364
we can construct the words ['dog', 'fog'].

Here is a starting point:

letter_maps = {
    1: [],
    2: ['a', 'b', 'c'],
    3: ['d', 'e', 'f'],
    4: ['g', 'h', 'i'],
    5: ['j', 'k', 'l'],
    6: ['m', 'n', 'o'],
    7: ['p', 'q', 'r', 's'],
    8: ['t', 'u', 'v'],
    9: ['w', 'x', 'y', 'z'],
    0: []
}

valid_words = ['dog', 'fish', 'cat', 'fog']
num = 364
op = ['dog', 'fog']

def bfs(num):
    global letter_maps
    global valid_words
    dig = num % 10
    rem_num = num / 10
    queue = []
    for letter in letter_maps[dig]:
        queue.append((letter, rem_num))

    while(queue):
        first_elem = queue[0]
        queue = queue[1:]
        characters_so_far = first_elem[0]
        rem_num = first_elem[1] / 10
        dig = first_elem[1] % 10

        for letter in letter_maps[dig]:
            if rem_num > 0:
                queue.append((letter+characters_so_far, rem_num))
            else:
                formed_word = letter+characters_so_far
                if formed_word in valid_words: print formed_word

bfs(num)


Hi, here is your problem today. This problem was recently asked by Uber:
***** YOU KNOW HOW TO SOLVE IT USING BOTTOM UP APPR. TRY SOLVING USING
PROFESSOR METHOD, RECURSE -> MEMOIZE -> DP *****
Given a list of possible coins in cents, and an amount (in cents) n, return
the minimum number of coins needed to create the amount n. If it is not possible 
to create the amount using the given coin denomination, return None.

Here is an example and some starter code:
  
print(make_change([1, 5, 10, 25], 36))
# 3 coins (25 + 10 + 1)
coins = [25, 10, 7, 2]
n = 12
memo = {}

def make_change(n, comb):
    if n == 0:
        print comb
        return True

    if n < 0:
        return False

    if n in memo:
        return memo[n]

    print 'n = ',n 
    for coin in coins:
        is_poss = make_change(n - coin, comb + [coin])

        if is_poss:
            memo[n] = is_poss#we are setting memo here in this lline itself and 
            #not at the end of loop coz the value of is_poss will be the True
            #only if the last coin value (in your for loop) is included in the 
            #comb. 
            #Whereas here even if one of the coins(in the for loop) contribute to
            #a result that can make n, it means that it is true
            print memo
            #print n

    if not n in memo:
        memo[n] = False
    
    return memo[n] #got confused because initally i was not returning from here
    #and my memo was
    #{1: True, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False, 8: False, 9: False, 10: True, 11: False}
    #After returning from here my memo became
    #{1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True, 9: True, 10: True, 11: True}


DIP
Hi, here is your problem today. This problem was recently asked by LinkedIn:

Given a list of numbers and an integer k, partition/sort the list such that the all
numbers less than k occur before k, and all numbers greater than k occur after the
number k.

def partition_list(nums, k):
  # Fill this in.

print(partition_list([2, 2, 2, 5, 2, 2, 2, 2, 5], 3))
# [2, 2, 2, 2, 2, 2, 2, 2, 5]

#same as quick sort partition around the pivot approach
#have 2 ptrs i and j. i traverses the list. 
#j will be at index 0. We swap ele at index j and increment j only if ele pointed 
#to by i is lesser than pivot


DIP - Swap Every Two Nodes in a Linked List
Hi, here is your problem today. This problem was recently asked by Twitter:

Given a linked list, swap the position of the 1st and 2nd node, then swap the 
position of the 3rd and 4th node etc.

curr_node = head
fwd_node = curr_node.next

#The foll. while loop will just swap the values and not the nodes themselves
while(fwd_node != None):
    fwd_node.val, curr_node.val = curr_node.val, fwd_node.val
    curr_node = fwd_node.next
    fwd_node = curr_node.next

#the following block of code will actually reverse the pos of nodes
llist = Node(1, Node(2, Node(3, Node(4, Node(5)))))
curr_node = llist
prev_node = None
while(curr_node != None):
    fwd_node = curr_node.next
    if fwd_node == None:
        prev_node.next = curr_node
        break
    tmp = fwd_node.next
    fwd_node.next = curr_node
    
    if prev_node:
        prev_node.next = fwd_node
    else:
        head = fwd_node

    prev_node = curr_node
    curr_node = tmp

node = head
while(node):
    print node.value
    node = node.next


Filter Binary Tree Leaves
Hi, here is your problem today. This problem was recently asked by Twitter:

Given a binary tree and an integer k, filter the binary tree such that its leaves 
do not contain the value k. Here are the rules:

- If a leaf node has a value of k, remove it.
- If a parent node has a value of k, and all of its children are removed, remove it.

# Fill this in.

#     1
#    / \
#   1   1
#  /   /
# 2   1
n5 = Node(2)
n4 = Node(1)
n3 = Node(1, n4)
n2 = Node(1, n5)
n1 = Node(1, n2, n3)

print(filter(n1, 1))
#     1
#    /
#   1
#  /
# 2

k = 13
def filter_node(node):
    global k
    if not node:
        return None

    node.left = filter_node(node.left)
    node.right = filter_node(node.right)

    if node.value == k and not node.left and not node.right:
        return None
    else:
        return node#forgot to add this return initially and every node
        #except the root was removed or the link was broken


Leetcode
32. Longest Valid Parentheses
https://leetcode.com/problems/longest-valid-parentheses/submissions/
See down for a better approach

class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        stack = []
        max_len = 0
        
        for char in s:
            if char == '(': #append open brackets to stack
                stack.append(char)
            elif char == ')':
                curr_seq_len = 0
                matching_open_brace_found = False
                
                while(stack):
                    ele = stack.pop()
                    if ele == '(':
                        curr_seq_len += 2
                        matching_open_brace_found = True
                        break
                    elif ele == '-':
                        stack.append('-')
                        break
                    else:
                        curr_seq_len += ele
                
                if stack and type(stack[-1]) == int:
                    stack[-1] = stack[-1] + curr_seq_len
                elif not matching_open_brace_found:
                    stack.append(curr_seq_len)
                    stack.append('-')#handy in ex: ) ()() ) ()() (
                    #we need '-' to say the stack that the first '()()' and the
                    #second '()()' are seperated by ')' which is invalid
                else:
                    stack.append(curr_seq_len)
        
        for ele in stack:
            if ele == '(' or ele =='-':
                continue
            elif ele > max_len:
                max_len = ele
        
        return max_len

Better approach (java) Try understanding this solution with the following ip
')()(()(())'
class Solution {
public:
    int longestValidParentheses(string s) {
        int n = s.length(), longest = 0;
        stack<int> st;
        for (int i = 0; i < n; i++) {
            if (s[i] == '(') st.push(i);
            else {
                if (!st.empty()) {
                    if (s[st.top()] == '(') st.pop();
                    else st.push(i);
                }
                else st.push(i);
            }
        }
        if (st.empty()) longest = n;
        else {
            int a = n, b = 0;
            while (!st.empty()) {
                b = st.top(); st.pop();
                longest = max(longest, a-b-1);
                a = b;
            }
            longest = max(longest, a);
        }
        return longest;
    }
};
The workflow of the solution is as_ below.

Scan the string from beginning to end.
If current character is '(',
push its index to the stack. If current character is ')' and the
character at the index of the top of stack is '(', we just find a
matching pair so pop from the stack. Otherwise, we push the index of
')' to the stack.
After the scan is done, the stack will only
contain the indices of characters which cannot be matched. Then
let s use the opposite side - substring between adjacent indices
should be valid parentheses.
If the stack is empty, the whole input
string is valid. Otherwise, we can scan the stack to get longest
valid substring as described in step 3.
