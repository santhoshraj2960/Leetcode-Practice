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