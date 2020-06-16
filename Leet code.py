my_s = """
1
good
thing:
I
go
off
schedule
very
rarely.I
plan
things
very
well.

Other
Coding
websites:
https: // medium.com / coderbyte / the - 10 - best - coding - challenge - websites -
for -2018 - 12b57645b654

Job
site and leetcode
replica:
https: // www.swecareers.com / company - profiles

Leetcode
expl:
- Kevin
Naughton
Jr
- Nick
White
- Back
to
back
swe

Interview
Prep
https: // www.youtube.com / watch?v = lDTKnzrX6qU

All
google
tech
roles
https: // www.youtube.com / watch?v = XEhZFtq0xTk

Google
resume
writing
tips:
https: // youtu.be / zHcCaBfsc2s

Why
nosql
for big data?
https: // analyticsindiamag.com / nosql - vs - sql - database - type - better - big - data - applications /
nosql
has
no
schema
requirements:
Say
tomorrow
you
insert
a
webpage
into
your
table and it
does
not have
meta
tags,
but
you
have
a
restriction
that
it
has
to
be
present(You
did
not set
null = True)
Your
insert
will
fail.To
resolve
this
what
should
you
do?
set
null = True
for that col and_ do migrations.This is_ just an example

Distribute
data
storage and distributed
query
execution
are
a
problem(yet
possib)
https: // blog.yugabyte.com / what - is -distributed - sql /

When
sql and nosql?
Need
acid(Bank
database) - sql
otherwise
nosql(web
crawler
database, where
data
consiste
- cy is not the
highest
priority)

One
trick
you
can
use is.Tell
the
interviewer
Theoritically
I
can
see
this
should
work.But
the
code
might
need
some
minor
fixes and as_
I
go
through
the
code and explain
it
to
you, I
can
discover
those
hidden
issues and fix
them.Is
that
fine?

Join
the
meeting
atleast
5
mins
before
it
starts.Nortonlifelock is an
example.
Ran
out
of
time
as_
I
joined
the
call
on
time
but
due
to
technical
difficulty, lost
6
mins and as_
a
result
was
not able
to
go
to
the
second
question
Think
about
the
corner
cases.Singly
linked
list
median
finding
when
only
1
node in list
Handle
them
before
your
interviewer
asks
you
to
handle.
When
the
given
problem is not hard, the
interviewer
he
interviewer is going
to
be
looking
heavily
at
how
careful
you
are.Did
you
check
for error conditions?

- Think
out
loud.Speak
as_
you
think
because
the
interviewer
will
know
your
thought
process.NEED
TO
WALK
THE
INTERVIEWER
THROUGH
THE
PROCESS
- Connect
your
phone
to
laptop and get
hotspot
from phone.Wifi driver

issue
- Explain
your
algo if possible
before
starting
to
write
code
so
you
may
know if you
need
to
tweak
your
algo

Behavioral
qustions
for which we need correct answers?
Have
you
ever
had
conflict and if so, how
did
you
handle
it?
Sophia
example, https: // www.youtube.com / watch?v = J49el9Fl8lM & t = 680
s, sat
down and weigh
the
pros and cons.If
unable
to
solve, escalete
to
manager
If
you
didnt
meet
a
deadline, wht
would
you
do?

In
design
based
questions, make
use
of
api
gateway( or) load
balancer
which
distributes
the
load
to
different
servers(n)
based
on
the
amount
of
cpu
usage.Make
use
of
n
queues,
the
processes
running
on
each
of
the
server
will
pick
the
tasks
from the respective

queues.
We
also
need
to
learn
about
the
task
scheduling
algorithms
like
round - robin, etc

Mock
Kevin
youtubers
presentation
of
interview
questions

Get
your
algorithm
precise and expalin
it
shortly
to
your
interviewer
before
writing
code
Implementing
your
algo
should
go
as_smooth_as
possible
Be
sure
to
return your
answer if you
are
using
a
func
at
the
end
of
the
func.Be
sure
to
increment
'i' in
while (i < len(something)) loops.
Space
complexity
can
go
exponential
https: // aonecode.com / google - coding - interview - questions / Word - Break - Combinations

# Common questions to ask before coding
for problems like "Given a set of numbers(integers, values)", the first question to ask
the
interviewer is "Does it include negative values"
as_well?
What
should
I
return when
there is no
such
combinations
possible?
Will
there
be
duplicates in the
input?
Is
the
input
sorted?

Try
to
avoid
at
all
costs
using
dictionary or something else to
solve
topological
sort
related
problems
https: // aonecode.com / facebook - coding - interview - questions / Minimum - Time - to - Complete - Tasks

adding
elems
to
a
set
directly
can
be
done
by
s = set()
s.add(5)
s.remove(5)
printing
elems
of
a
set:
for item in s:
    print
    s

Python
pointer - id(variable_name)
gives
the
memory
location
of
this
variable.Needed
this in the
problem
"Find cycles in tree"
Dont
remove
any
elem
from a list

when
iteration
it
with_
a
for_
loop.
l = [1, 2, 3, 4]
for item in l:
    print(item)
    if item == 2:
        l.remove(2)
# prints 1,2,4

We
can
sometimes
use
set
for solving a problem.Refer Maximum Length of a Concatenated
String
with Unique Characters.See link under runtime
set
Union - s | t - O(len(s) + len(t))
set
intersection - s & t - O(min(len(s), len(t))

Need
to
be
more
careful
when
reading
a
question.eg.
1239
leetcode.The
question
asks
"Maximum Length of a Concatenated String with Unique Characters" and not
"Maximum Length of a Concatenated CONTINUOS String with Unique Characters"

When
handling
questins
like
"Is subtree of another tree", we
got
to
be
a
bit
more
careful
when
doing
the
solution
by
tree
traversal
algos and storing
the
values in a
list and
checking if child
list is sublist
of
parent
list.We
need
to
store
the
none
val
of
left
and right
child
of
leaf
nodes in the
list
for correct results.

        If you want to travel directions in matrix based problems, use something like this
directions =[(0, 1), (1, 0), (0, -1), (-1, 0)]

dequeue - Double ended queue where insertion and deletion from_both ends can be done in_
O(1) time.
from_ collections import dequeue
d = dequeue()
d.extend([2, 3, 4, 5, 6])
d.appendleft(1)
d.popleft()  # prints 1
In lists, appendleft can be done like
l =[1] + l  # but this is (n) operation
eg of application.Sliding window maximum

Virtual functions:
    https: // www.geeksforgeeks.org / virtual - function - cpp /
              https: // docs.microsoft.com / en - us / cpp / cpp / virtual - functions?view = vs - 2019
If
you
declare
2
functions
of
the
same
name in both
base and derived
class_.When
you
add
virtual
keyword
before
the
function
name in the
base
class_
then
this
function is a
virtual
function.
    A
virtual
function is a
member
function
that
you
expect
to
be
redefined in derived
classes.
    When
you
refer
to
a
derived
class_
object
using
a
pointer or a
reference
to
the
base
class_, you
can
call
a
virtual
function
for that object and_ execute the derived class_'s
version of the function.
When the virtual fun in base class_ or the function in the derived class_ has different
arguments, the purpose of virtual functions is lost.It becomes method overloading


Virtual destructors
Deleting a derived class_object using a pointer to a base_class, that has a non-virtual
destructor results in undefined behavior.To correct this situation, the base_class
should be defined with a virtual destructor.
https://
    www.geeksforgeeks.org / virtual - destructor /

    4
oop
concepts - Inheritance, Polymorphism, encapsulation, Abstraction
Polymorphism
https: // medium.com / @ shanikae / polymorphism - explained - simply - 7294
c8deeef7
Method
overriding is dynamic
Polymorphism
Method
overloading is static
Polymorphism

Diff
bet
Abstraction and enacapsulation:
https: // javarevisited.blogspot.com / 2017 / 04 / difference - between - abstraction - and -encapsulation - in -java - oop.html
In
encapsulation
the
motive is to
secure
data
by
declaring
it
private.
    Bank
account
balance, b_a_b - should
be
updated
only
by
transaction
method.
    The
variable
b_a_b
should
be
declared
private
so
that
no
other
classes
other
than
the
current
_class
can
update
this
variable.

    On
the
other
hand
Abstraction
is_
hiding
the
underlying
functionality
as_
its
not needed
for the person who is_ using it.Eg - person using hashmap need not_ know the underlying
implementaion of how its done (using tree or graph or bst or stack).He needs to know to
just how to use it and thats all.

Abstraction:
    Hiding
undelying
functionality
of
the
class_
to
make
sure
the
person
using
the
class_
doesnt
get
perplexed
about
the
implementaion
https: // stackify.com / oop - concept - abstraction /

          enacapsulation
Process
of
bundling
all
the
methods and attributes
exclsively
specific
to
a
class_.
    Sensitive(attributes and functions)
should
be
declared in private and made
accessible
only
the
other
public
methods
of
the
same
class_.
https: // stackify.com / oop - concept -
for -beginners - what - is -encapsulation /

    https: //
    techdifferences.com / difference - between - multiprocessing - and -multithreading.html
Multiprocessing
- Multiprocessing
adds
CPUs or processors
to
increase
computing
power.
- Multiple
processes
are
executed
concurrently.
- Creation
of
a
process is time - consuming and resource
intensive.

    Multithreading
- Multithreading
creates
multiple
threads
of
a
single
process
to
increase
computing
power.
- Multiple
threads
of
a
single
process
are
executed
concurrently.
- Creation
of
a
thread is economical in both
sense
time and resource.
- threads
belonging
to
the
same
process
has
to
share
the
belongings
of
that
process
like
code, data, and system
resources.
- eg: A
word
processor, displays
graphic, responds
to
keystrokes, and at
the
same
time, it
continues
spelling and grammar
checking.You
do
not have
to
open
different
word
processors
to
do
this
concurrently.It
does
get
happen in a
single
word
processor
with the help of
multiple threads.

Math fact
a composite number must have a factor less than the square root of that number.Otherwise,
the number is prime.So if you check for_ prime,
for i in range(2, math.sqrt(n)):

    2 - ptr
approach
can
also
be
used in these
tkind
of
probs
https: // leetcode.com / problems / remove - duplicates -
from

-sorted - array /
https: // leetcode.com / problems / remove - element /
          the
reson
behind
using
2
ptr is we
need
to
solve
it in O(1)
space

Recursion
problems
will
require
for loop inside the recursive function itself sometimes
eg: https: // leetcode.com / problems / restore - ip - addresses /
              got
to
familiarize
with problems like these that were discussed in class_

For Recursion problems, if you find difficulty in finding time compl.quickly draw a
recursion tree and
try to derive a relationship between the number of non Leaf nodes
and the
input
size and upper
bound
it
eg: https: // leetcode.com / problems / decode - ways /

python
list - pop() is O(1), but
pop(index) is O(n)(since
the
whole
rest
of
the
list
has
to
be
shifted).O(n ^ 2) is the
time
comp
of
following
prob
https: // leetcode.com / problems / remove - duplicates -
from

-sorted - array - ii /
https: // wiki.python.org / moin / TimeComplexity
Except
push, pop, get, set and len, all
list
operations
are
non
constant(i.e
not O(1))
dict.keys() in python
2 is O(n), in python
3, its
O(1)
If
you
slice
a
list
like
l[0:i] + l[i + 1:], its
O(h) + O(k), where
h and k
are
len
of
the
slices.But
s + k = n - 1.
So
we
have
to
consider
this
slice
as_
O(n)

Python
appending
an
element
does
not create
new
memory.append
happens in -place
Python
prepending
an
element
to
array in -place
arr.insert(0, 'value')
See
clockwise
rotate
of
'rotate the array by k spaces'
problem.see
methods
1 and 2

dictionary
subsets
res = subset.items() <= superset.items()  # method 1
res = all(superset.get(key, None) >= val for key, val in subset.items())  # method 2
Dictionary
comparisions
are
tuple
comparisions(key_1, val_1) > (key_2, val_2)

bitmasks
are
a
foolproof
way
of
soving
certain
types
of
problems, especially
permutations
eg: subsets - https: // leetcode.com / problems / subsets /

Permuatation and combination
https: // www.mathsisfun.com / combinatorics / combinations - permutations.html

I
find
it
easier
to
calculate
the
time
complexity
by
logical
reasoning
rather
than
simply
going
with the below formula.
eg: https: // leetcode.com / problems / subsets / solution / (see approach 2 on LC),
Restore
IP
addresses.
Note
there is a
difference
between
subsets(1
st
problem
above) and
https: // aonecode.com / google - coding - interview - questions / Word - Break - Combinations
because
the
order
the
each
number
occurs
matter
whereas in the
word
break
combinations
problem
the
words
can
occur in any
order.Calculation
run
times
of
these
2
probs
need
to
be
done
differently
Try
to
understand
how
you
came
up
with the time compl for_above probs.

For
permutations, time
complexity is
nPr = n! / (n - r)!, The
reason
this is nPr
we
would
have
made
nCr
recursive
calls
to
have
computed
all
the
results
space is same
as_time
because, we
need
to
store
nPr
number
of
values in result
variable

For
combination
problems, time
complexity is
nCr = n! / r! *(n - r)! (We just multiply 1 / r! to nPr to get nCr)
space is same
as_
time
because
you
have
to
store
nCr
nnumber
of
values in result
variable
https: // leetcode.com / problems / combinations / solution /

In
sliding
window
problems, think
of
solving
the
problem
using
a
window
like
approach, shortening
the
window and expanding
the
window.Using
dictionary
could
also
be
an
approach
but
sometimes
it
may
get
complicated
eg: Minimum
sliding
window

New
technique
to
solve
problem:
Use
stack
with the following prop.
- You
cannot
place
a
bigger
element
on
top
of
a
smaller
element.This
statement is
not valid.In
histogram
you
cannot
place
small
elem
on
top
of
a
bigger
element.
- In
some
cases( or may
be
most
cases) you
will
store
indices in the
stack
instead
of
the
element
itself.For
ex: In
Trapping
Rain
Water, LC
problem, We
need
to
know
how
far
the
verticals
walls( or bars) are
apart.This is the
breadth
of
the
rectangle and we
need
it
calc
the
area.Ref - dia
lc_stack_general.png
- How
to
determine
which
of
the
following
scenarios
a
problem
can
be
solved in
1 - You
cannot
place
a
big
ele
on
top
of
small
ele
2 - You
cannot
place
small
ele
on
top
of
big
ele
Draw
the
bar
chart
diagram in you
notebook.Think
which
one(small
bar or big
bar)
you
need in later
parts
to
solve
the
problem.

For
ex.in_
trapping
rain
water
you
want
to
keep
the
big
bars(height
5) in your
stack
untill
a
bar
bigger
than
that
comes in.
This
implies
that
bars
of
height <= 5
can
occur
anywhere
after
5 in the
stack.
The
thought
process
behind
this
reasoning is.Lets
say
we
get
bars
of
size
3 and 2
after
5.
5
will
act
as_
left
boundary and water
can
still
be
stored
on
top
of
3 and 2.
So, 5
can
act
as_
a
left
boundary
for all bars whose height is_ less than 5.
If
we
get
a
new_bar
of
size >= 5, 5
can
never
act
as_
left
boundary
because
now
we
must
have
new_bar(ht >= 5)
as_
the
left
boundary.

In
another
ex.Think
of
Largest
Rectangle in Histogram
problem.
Contrasting
to
trapping
rain
water, here
we
need
to
keep
the
small
bars(height
2) in
your
stack
untill
a
bar
smaller
than
than
comes in.
This
implies
that
bar
of
height >= 2
can
occur
anywhere
after
2 in the
stack.
The
thought
behind
this
reasoning is same
as_
above.
2
will
act
as_
a
left
boundary
for
    all
bars
that
occur
on
the
right
hand
side
of
it(which
are >= 2).

In
both
these
problems, we
can
notice
one
thing.
- If
a
smaller
bar in the
bar
chart
can
bound
the
larger
bars, we
need
to
keep
the
smaller
bars in the
stack
as_
long
as_
possible.That is till
a
bar
smaller
than
that
comes in.In
that
case, Largest
Rectangle in Histogram

curr_ele = inp[curr_ind]
while (curr_ele <= inp[stack[-1]]):
    stack_top_index = stack.pop()
    height_stack_top_index = inp[stack_top_index]
    width = curr_ind - stack_top_index
    area = height_stack_top_index * width

- If
a
taller
bar in the
bar
chart
can
bound
the
samller
bars, we
need
to
keep
the
taller
bars in the
stack
as_
long
as_
possible.That is till
a
bar
taller
than
that
comes in.In
that
case, trapping
rain
water

while (curr_ele >= inp[stack[-1]])

New
idea
for problem solving
    Having
dummy
nodes in linked
list
related
problem
to
easily
solve
the
problem.
eg: 83: Remove
Duplicates
from Sorted List

In
Matrix
related
problems, a
wierd
thing
happens, check
the
transpose
problem
when
we
assign
tran_mat[i][j] = mat[i][j]
it
becomes[[1, None], [1, None], [1, None]]
which is wrong
it
should
have
become[[1, None], [None, None], [None, None]]
which is correct
This
problem
happens
because
of
internal
memory
allocation
reasons
https: // stackoverflow.com / questions / 240178 / list - of - lists - changes - reflected - across - sublists - unexpectedly
** ** ** ** ** Solution
for the above prob ** ** ** ** ** ** ** **
mat = [[False] * len_row_col for _ in range(len_row_col)]

For
matrix
problems, In
order
to
get
"n'th"
cell in a
matrix.
cell = mat[n / num_cols][n % num_cols]
[0, 1, 2]
[3, 4, 5]
[6, 7, 8]
- You
can
apply
the
above
formula in sudoku
validator and search
2
d
matrix and also
search
a
flat
2
d
matrix

- usually
word
search
related
problems
are
best
when
done
using
tries

There
are
problems
where
you
have
to
populate
your
result
array
from the end

eg:
https: // leetcode.com / problems / merge - sorted - array
If
populating
from the front

doesnt
work, populate
from end.Not clear

how
to
decide
which
probs
require
populating
from rear.If you

think
of
a
solid
solution
before
starting
your
answer, you
will
know if you
should
start
from_
front or rear

Learn
what is a
PriorityQueue
A
PriorityQueue is what is called
a
binary
heap.It is only
ordered / sorted in the
sense
that
the
first
element is the
least.In
other
word, it
only
cares
about
what
is in the
front
of
the
queue, the
rest
are
"ordered"
when
needed

Merge
intervals - FB
question
https: // leetcode.com / problems / merge - intervals /
Do
NOT
ASSUME
your
intervals
are
sorted and do
not ASSUME
that
the
0
th
ele in
a
particular
interval is smaller
than
the
1
st
ele
of
the
next
interval.
You
cannot
also
assume
that[[1, 4], [2, 3]]
wont
exist.Though
2 is less
than
4,
you
cannot
assume
that
the
3
will
be
greater
than
4(coz
its
not)

Consider
using
2
pointers in a
single
loop.
1
ptr
might
get
incremented and the
other
might
get
decremented
eg:
https: // leetcode.com / problems / 3
sum - closest /
https: // leetcode.com / problems / 4
sum /

Sometimes
the
best
possible
sol
to
solve
the
problem
could
be
to
use
nested
loops
eg: https: // leetcode.com / problems / jump - game - ii /

When
trying
to
memoize
you
need
an
n - dimensional
array
where
n is the
no
of
paramenters
the
recursive
function
absolutely
needs.
eg: smooth_shuffle
needs
4 - d
array
when
you
want
to
memoize
your
algo

binary
search and getting
the
index
of
an
elem in a
sorted
array


def find_element(self, nums, target, pos):
    if not nums:
        return -1

    mid = len(nums) / 2
    if nums[mid] == target:
        return pos + mid  # print 'target found = ', target
    elif nums[mid] > target:
        return self.find_element(nums[0:mid], target, pos)
    else:
        return self.find_element(nums[mid + 1:], target, pos + mid + 1)


binary
search
when
using
indices


def binary_search(st, en, search_num):
    if st > en:
        return False

    mid = (st + en) / 2
    row_num = mid / num_cols
    col_num = mid % num_cols

    if mat[row_num][col_num] == search_num:
        return True

    if mat[row_num][col_num] < search_num:
        # Important point ******mid + 1*******
        return binary_search(mid + 1, en, search_num)
    else:
        # Important point ******mid - 1*******
        return binary_search(st, mid - 1, search_num)


if you do not have "mid + 1" or "mid - 1" and instead substitute just "mid", if the
element is not there, in some
edge
cases
it
leads
to
inifinite
loop.
eg: seach
a
2
d
matrix
This
holds
true
only in the
second
approach, where
you
are
only
passing
indices.When
you
are
passing
nums
array
itself
as_ in approach
1
you
have
to
do
arr[st:mid] and arr[mid + 1:]

building
a
trie
cannot
be
done
using
recursion.You
have
to
use
while and for_
loop
comb
eg: shortest
unique
prefix

Returning
at
the
end
of
the
function(especially
when
you
are
doing
recursion) is
very
important.
eg: filter
binary
tree and Given
a
list
of
possible
coins in cents, and an
amount

In
linked
list
problems
try not to have more than 2 pointers to nodes in your code
eg: Swap
Every
Two
Nodes in a
Linked
List
First
i
planned
on
having
fwd_ptr = node.next.next.next
The
above
approach
would
have
made
my
while loop more complicated
Later
I
came
up
with an approach to just settle for_ one node and it has worked
fine.

NP
concept:
for sorted pos int array eg arr =[1, 2, 3, 6, ...] say i = 6
If
elements
from

0
to(i - 1)
can
represent
1
to ‘res - 1(which is 5)’, then
elements
from

0
to
i
can
represent
from

1
to ‘res + arr[i] – 1’ be
adding
‘arr[i]’ to
all
subsets
that
represent
1
to ‘res

Greedy
solutions:
- Go
through
the
input
elems
one
at
a
time.
- Make
a
decision
on
whether
you
have
to
pick
the
element or not and move
fwd
- eg: Minimum
Cost
Tree
From
Leaf
Values, Find
the
maximum
sum
of
all
possible

contiguous...
Bit
shift
operation:
- right
shift
divides
a
num
by
2
- left
shift
multiplies
a
num
by
2

To
find
the
duplicates in a
list and missing
elements
refer
You
are
given
an
array
of
integers.Return
the
smallest
positive
integer
the...
in python_test.py
XOR
operations:
1 ^ 1 = 0
1 ^ 2 = 2 ^ 1
res = 7 ^ 3 ^ 5 ^ 4 ^ 5 ^ 3 ^ 4
print
res  # prints 7 as all other eles occur EXACTLY twice
0 ^ 0 = 0
0 ^ 1 = 1
1 ^ 0 = 1
1 ^ 1 = 0

x ^ 0
s = x, x ^ 1
s = ~x, x ^ x = 0, x & 0
s = 0, x & 1
s = x, x & x = x,
x | 0
s = x, x | 1
s = 1
s, x | x = x

Think
of
running
sums
approach
as_
one
of
the
approaches
to
solving
a
prob
You
are
given
an
array
of
integers.
eg: You
are
given
an
array
of
integers, and an
integer
K.Return
the
subarray.

In
Dyn
Prog
problems
like
how
many
ways
can
you
add
to
N
with_
1, 3, 5
we
have
to
define
the
base
cases
properly
to
get
the
proper
result.
This is a
bottom
up
approach
eg:
Number
of
ways
to
count
to
N
with 1, 3, 5
    dp = {}
dp[0] = dp[1] = dp[2] = 1  # 2 = 1 + 1
dp[3] = 2  # 1+1+1 or 3+0
dp[4] = 2  # 1+1+1+1 or 3+1
dp[5] = 3  # 1+1+1+1+1 or 3+1+1 or 5+0
n = 10
i = 6  # We have to make sure that all numbers from and after 6 can be substituted
# in the formula dp[i] = dp[i - 5] + dp[i - 3] + dp[i -1]

while (i < n):
    dp[i] = dp[i - 5] + dp[i - 3] + dp[i - 1]
    print
    'i = ', i
    print
    'dp = ', dp[i]
    i += 1

The
above
sol is a
bottom - up
approach

- When
you
have
keywords
like
"minimize"
and_
stuff, its
a
good
indication
that
you
can
do
it
with_
dynamic
programming.
eg: https: // leetcode.com / problems / minimum - path - sum /

For
dynamic
programming(memoization), calculating
run
time
should
be
done
line in
pg133
of
CTCI.For
Recursion
run
time
calc, follow
pg132
of
CTCI.

Sometimes
you
have
to
change
the
postion
of
left and right
for binary tree args
in problems
like
this in recursive
calls
update_output(node.left, row + 1, left, mid - 1)
update_output(node.right, row + 1, mid + 1, right)
# Question: Print binary tree in a particular order

DP
cannot
be
used
when
you
have
keywords
like
Continuous in your
problem
statement
You
cannot
use
Dynamic
programming
for_
probs
when
the
problem
asks
for_
a
continuious
subarray
within
an
array
like
- You
are
given
an
array
of
integers.Find
the
maximum
sum
of
all
possible
continuious
subarrays
of
the
array

Time
complexity
of
binary
tree
traversals(inor, preor, postor) or using
dfs
to
traverse
a
binary
tree is O(n) and the
space
complexity is O(n)
because
at
a
given
time
you
will
store
only
O(max_height)
calls in stack
for recursion purpose and_ if_ the
tree is left
sided
tree, we
will
sore
n
recursive
calls in the
stack

Graph
Time and space:
Undirected
graphs and directed
graphs:
dfs
time: O(V + E) - You
visit
all
vertices and edges
exactly
once
dfs
sapce: O(V)
In
your
recursion
stack
Lets
say
we
have
5
vertices and each
of
those
vertices
are
connected
to
every
other
vertex
We
will
have
n
vertices and n - 1
edges
going
out
of
each
of
those
vertices.Since
we
represent
the
neighbors
of
a
vertex in adjacency
list( or matrix), we
need
to
allocate
space
for edges as_ well.So, it will be n ^ 2 space for_ this prob
https: // aonecode.com / facebook - coding - interview - questions / Minimum - Time - to - Complete - Tasks

New
concept: itertools
itertools
has
lot
of
other
libraries
to
find
permutation, combinations,
catesian_product and a
lot
more
all_functions: https: // docs.python.org / 3 / library / itertools.html

eg:
import itertools

tree = [1, 0, 1, 1, 4, 1, 4, 1, 2, 3]
for ele, v in itertools.groupby(tree):
    ...
print
'ele = ', k
...
print
'v = ', list(v)
...
ele = 1
v = [1]
ele = 0
v = [0]
ele = 1
v = [1, 1]
ele = 4
v = [4]
ele = 1
v = [1]
ele = 4
v = [4]
ele = 1
v = [1]
ele = 2
v = [2]
ele = 3
v = [3]

>> > blocks = [(k, len(list(v)))
               for k, v in itertools.groupby(tree)]
>> > blocks
[(1, 1), (0, 1), (1, 2), (4, 1), (1, 1), (4, 1), (1, 1), (2, 1), (3, 1)]

python
bisect
module:  # when you want to insert an element into a sorted array and
still
keep
it
sorted, make
use
of
bisect
module.This
can
be
helpful
for questions
    where
you
have
a
continuous
input
stream
of
numbers.
eg: You
are
given
a
stream
of
numbers.Compute
the
median
for each new element
Python
comes
with a bisect module whose purpose is to find a position in list where
an
element
needs
to
be
inserted
to
keep
the
list
sorted.time
comp
to
insert is
log
n
as_
it
uses
binary
search

import bisect


def insert(list, n):
    bisect.insort(list, n)
    return list


list = [1, 2, 4]
n = 3

print(insert(list, n))
Output:
[1, 2, 3, 4]

sum([True, False, True])  # Outputs 2

- In
tree
problems
your
first
priority
of
approach
should
be
DFS or BFS.If
you
can
not think
of
a
viable
sol
with dfs or bfs, go for_ inorder, preorder or postorder

In
tree
problems,
>> > root = Node(1)
>> > root.left = Node(2)
>> > root.left.left = Node(4)
>> > new_node = root.left.left
>> > new_node.data
4
>> > node = root.left
>> > node.data
2
>> > node = new_node
>> > node.data
4
>> > root.left.data  # root.left still points to the old val(2) and not new val(4)
2
If
you
do
not explicitly
change
the(left or right)
pointer
of
root( or node) to
the
new_node, the
root( or node) will
still
be
pointing
to
the
old
node
So, you
have
to
do
this
>> > root.left = new_node
application
of
this
concept is in python_test.py in the
foll
problem
"Given a binary tree, remove the nodes in which there is only 1 child"

List
difference is not possible
but
set
difference is possible
>> > [1, 2, 3, 4] - [1, 2, 3]  # is not possible
TypeError: unsupported
operand
type(s)
for - 'list' and_ 'list'
>> > set([1, 2, 3, 4]) - set([1, 2, 3])
set([4])

- list
comprehension
with_
if_
else_
'|'.join([ele if ele in ['X', 'O'] else '-' for ele in row])  # both if and else
'|'.join([ele for ele in row if ele in ['X', 'O']])  # only if

- Binary
search
time
complexity = O(log
n)  # you are searching in a list
- Searching in BST: time_comlexity(worst
case) = O(n)  # you are searching in a tree
avg and best
case - (log n)
https: // www.geeksforgeeks.org / complexity - different - operations - binary - tree - binary - search - tree - avl - tree /
- Deletion
from BST: time_complexity = O(n)
https: // www.geeksforgeeks.org / binary - search - tree - set - 2 - delete /
- Bottomup
approaches in trees
can
be
done
ONLY
by
DFS(cannot
be
done
with_
bfs)
eg: https: // www.youtube.com / watch?v = aaSFzFfOQ0o
- Top
down
approaches
can
also
be
implemented
using
DFS.Eg:
Return
true if it is a
valid
binary
search
tree
- Try
to
make
use
of
every
single
detail in the
question.You
can
solve
the
problem
easily and in a
better
way.There
could
be
some
hints
at
the
end
of
the
question
like
this
one(compare
strings)
https: // leetcode.com / discuss / interview - question / 352458 /
See
the
python
ans
to
understand
how
he
has
made
use
of
the
clue in the
question

One
important
thing in matrix
problems is:
1 - Dont
bother
about
the
values in the
matrix
2 - Draw
your
matrix in your
note and then
see
how
row and col
numbers
change
according
to
the
question
00
01
10
11

Counters
from collections import Counter

c = Counter('1110')
print
c  # op: Counter({'1': 3, '0': 1})
print
c.get('1')  # op: 3
print
c['1']  # op:3

A = [1, 2, 1, 2, 4, 2, 2, 4]
Counter(A).most_common(3)
[(2, 4), (1, 2), (4, 2)]

Heaps
import heapq

li = [5, 7, 9, 1, 3]
heapq.heapify(li)

heapq.heappush(li, 4)

print(heapq.heappop(li))

Min and Max
heap:
>> > import heapq
>> > listForTree = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
>> > heapq.heapify(listForTree)  # for a min heap
heapq.heappop(listForTree)  # prints 1
>> > heapq._heapify_max(listForTree)  # max heap
heapq.heappop(listForTree)  # prints 15

- heap
push and heap
pop
are(log
n) operations.For
building
a
heap, the
O(n) = n
https: // www.geeksforgeeks.org / time - complexity - of - building - a - heap /

- Heaps
do
not store
values in ascending
order.
Rather
heappop()
outputs
the
elemets in ascending
order.The
order
of
elements in
a
list
which
has
been
heapifies is not ascending

How
to
write
Class and objects


class Person(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def test(self):
        print
        'test called'


if __name__ == '__main__':
    p1 = Person("John", 36)
    print(p1.name)
    print(p1.age)
    p1.test()

if __name__ == '__main__':  # what does this mean
    https: // stackoverflow.com / questions / 419163 / what - does - if -name - main - do

Zip
for creating dict of 2 linked lists
https: // realpython.com / python - zip - function /

OrderedDict  # Python 3 Stores the keys in the order we save them
from collections import OrderedDict

1.
d = OrderedDict.fromkeys('abcde')
# can also be initialized as
# d = OrderedDict()
d.move_to_end('b')
d.keys()  # odict_keys(['a', 'b', 'c', 'd', 'e'])
list(d.keys())  # ['a', 'b', 'c', 'd', 'e']
print
''.join(d.keys())  # 'acdeb'
d.move_to_end('b', last=False)
print
''.join(d.keys())  # 'bacde'
d.pop('key')  # remove a key from dict

d[b] = 1
d[a] = 2
d[c] = 5
print
d[(list(d.keys())[2])]  # 2

Graph / tree
concepts:
1.
For
problems in which
the
operations
we
do
on
a
tree
are
related
to
levels
(i.e.horizontal levels)
of
the
tree or finding if there is an
edge
bet
2
nodes, use
BFS
approach
2.
For
problems
related
to
vertical
levels(height)
of
tree
try using dfs
3.
you
can
mostly
solve
any
graph or tree
problem
using
the
following
concepts:
3.1.Recursion
3.2.Dictionary
3.3.BINARY
Tree
traversal
algos - Inorder, preorder and post
order - These
traversals
work
only
for binary trees.FOr other type of trees, you can use
BFS or DFS
3.4.Queue
3.5.Stack

For
some
of
the
problems
like
4.6( in -order
successor) in CTCI, We
need
to
come
up
with a pseudocode or small rules when trying to come up
with a solution

2.
Removing
element
from list

animal = ['cat', 'dog', 'rabbit', 'guinea pig', 'dog', 'dog']
animal.remove('rabbit')
print
animal  # ['cat', 'dog', 'guinea pig', 'dog', 'dog']
animal.remove('dog')  # Removes only the 1st occurance of dog
print
animal  # ['cat', 'guinea pig', 'dog', 'dog']
animal.pop(2)( or) del animal[2, 3]  # deletes element at index 2 time - O(n)
print
animal  # ['cat', 'dog', 'dog']
del animal[2, 10]  # will remove element at index 2. This will not throw
# index error
print
animal  # ['cat', 'dog']

0.1 - Make
the
code
WORK
ON
PAPER
PEN
first.If
you
can
get
this
you
are
70 % complete
Dictionary
1.
Dictionary
lookups
are
faster
than
list
lookups
because
dictionaries
are
hash
tables and hash
tables
are
indexed
3.
In
order
to
delete
a
key
from a dictionaty

use
my_dict.pop(key)
7.
create
a
new
dict
with the keys of an exisiting dict
b = a.fromkeys(a, 'my_value')  # 'my_value' can be anythong (0 or [] or None)

12: If
you
know
you
are
going in the
wrong
direction
while soving a problem,
immediately
stop
the
approach and think
about
a
diff
approach.If
you
are
not getting
any
approach in mind, just
leave
the
problem and move
on.
13: Always
try to solve the problem first even if the approach you are using is
not the
best
approach(note: you
should
not proceed if you
are
solving
using
worst
approach,
for ex: using
two
for_loops
unless
it
's the only approach).
And
once
you
are
getting
the
output
even
with a non best approach,
later
try to optimise your solution
17: Handle
edge / corner
cases
a_s
much
a_s
possible

----------------------------------------------------------------------------------------

leftshit: 128 << 1  # prints 256
rightshit: 128 >> 1  # prints 64
- If
we
shift
all
the
bits
left
one
place, discard
the
leftmost
bit, and insert
a
zero
on
the
right, the
result is equal
to
multiplying
the
num
by
2
because
binary
representaions
are
2 ** n.Think
logically, you
will
get
how
its
happening
eg: 23 - 00010111
after
shiting
1
bit
left
46 - 00101110

# the following para is only for understanding. Don't think too much on implementing
Signed
right
shift - will
do
a
right
shift
preserving
the
sign(left
most
digit)
(if the left most dig in a signed bin representaion is 1, its a -ve num and vice
 versa)
For
example,
if binary representation of number is 10….100, then right shifting it
by
2
using >> will
make
it
11…….1.

Think if you
can
solve
using
Dyn
prog
1 - Bottom
up or top
down
2 - memoization
3 - knapsack or matrix
approach - While
using
matrix
approach, we
may
have
to
fill
rows
first(we
might
have
to
fill
it
from_
0
to
n or from_
n
to
0) or cols(we
might
have
to
fill
it
from_
0
to
n or from_
n
to
0).Look
at
the
problem and decide
how
you
are
going
to
fill
your
matrix.Professors
approach, solve
with recursion and
    then
see
where
the
recurrance is happening and then
convert
it
to
DP and see
how
to
fill in your
rows or cols

infinity
can
be
represented
as_
float('inf') or float('-inf')

Dynamic
Prog

Refer - Dynamic
programming
tutorial
wine
problem
- Follow
Prof.Kashthuri
approach.Come
up
with a recursive algo and then use
memoization or dynamic
prog
approach.Pass
indices
such
as_
start_ind, end_ind
to
your
recursive
function
so
that
it
will
be
easy
for you to memoize the
results
- https: // www.quora.com / How - do - I - figure - out - how - to - iterate - over - the - parameters - and -
write - bottom - up - solutions - to - dynamic - programming - related - problems / answer / Michal - Danil % C3 % A1k?srid = 3
OBi & share = 1
- https: // www.hackerearth.com / practice / algorithms / dynamic - programming / introduction -
to - dynamic - programming - 1 / practice - problems / algorithm / win - the - game /

Time
complexity
sample
questions
https: // www.geeksforgeeks.org / practice - questions - time - complexity - analysis /
- When
you
are
iterating
upto
100
times and you
index is incremented
by
a
factor
(multiplication or division)(not addition or subtraction)
of
some
number, x
(say 2, 3, 4, 5, ..etc).

while (i < 100):  # time comp of this loop is (log n)
    print
    i
    i = i * 2

while (i < 100):  # time comp of this loop is O(n)
    print
    i
    i += 2

- we
can
access
a
global_
var
without
doing
anything
_from
within
a
fun
but if you
want
to
change
the
val
of
this
global_
var
from inside of

your
fun, you
need
to
use
the
keyword
"global"
followed
by
the
variable
name
to
tell
the
fun
that
you
want
to
use
the
global var

global_var = 5


def fun():
    print
    global_var  # will print 5
    global_var += 3  # will throw an exception
    global global_var
    global_var += 3  # will update the valur to 8


- Time
complexity
of
inbuilt
aggregation
funs
like
sum(my_l), count(my_l) is O(n)

https: // leetcode.com / discuss / career / 451590 / google - summer - 2020 - swe - intern - offer - very - helpful - resources

Programming
concepts:

Solve
hackerrank
challenges
https: // www.hackerrank.com / interview / interview - preparation - kit?h_l = domains & h_r = hrw & utm_source = hrwCandidateFeedback

When
you
are
using
the
feature
function
inside
another
function in python,
a
strange
thing
that
occurs is that
you
can
use
access
list
objects
of
the
parent
function in child
function
but
string
objects
are
not accessible

Make
a
wise
decision
when
it
comes
to
choosing
BFS or DFS.Remeber
the
Facebook
friends
example.Think
through
your
mind
completely
before
getting
into
an
approach.
Finding
shortest
distance
bet
2
points is always
a
GIVEN
to
use
BFS

Kepp in mind
how
you
visualize
a
graph:
[
    [0, 0, 0],
    [0, 0, 0],  # if r1c1 was 1 path 2 wont happen
    [0, 0, 0]
]
Remember
the
robot, obstacle
problem
https: // leetcode.com / problems / unique - paths - ii /
Visualize
how
the
robot
can
go
through
the
grid in the
foll.code


def distinct_path(row, col):
    if row == total_num_of_rows - 1 and col == total_num_of_cols - 1:
        self.total_distinct_routes_to_dest += 1
    elif row < total_num_of_rows - 1 and my_matrix[row][col] != 1:
        distinct_path(row + 1, col)
    elif row < total_num_of_cols - 1 and my_matrix[row][col] != 1:
        distinct_path(row, col + 1)


Here
the
robot
goes
through
the
foll
paths:
1 - r0c0 -> r1c0 -> r2c0 -> r2c1 -> r2c2
2 -              -> r1c1 -> r2c1 -> r2c2
3 -                      -> r1c2 -> r2c2
4 -      -> r0c1 -> r1c1 -> r2c1 -> r2c2 and it
goes
on....

3 - When
dealing
with diagonal manipulations (Swapping elems along primary or
secondary diagonal).Write a 3 * 3 matrix with_ just row_num col_num.Place your
pen
along
the
diagonal and imagine
cutting
the
paper
along
the
diag
into
2
pieces.(Note.
None
of
the
pieces
should
contain
the
diag
elems).Once
you
have
the
2
pieces, interchange
their
pos
along
the
diag and write
the
appropriate(new)
row_num
col_num
seperately.

Reload
module
without
leaving
terminal
import imp
import tic_tac_toe

tic_tac_toe = imp.reload(tic_tac_toe)
t = tic_tac_toe.TicTacToe()
t.print_tic_tac_toe_matrix()

Use
the
concept
of
stacks and queues in solving
a
problem if you
are
finding
it
hard in figuring
out
the
sol

What
should
you
do in interview:
- Try
running
through
many
problems
on
pen and paper
coz
writing
code
take
too
much
time(GOOGLE
tech
lead).
-- Write
_class(object) and def_methods(self, arguments) if
there is a
possibility
to
just
show
interviewer
that
you
know
Obj
Ori
Prog.
-- Mostly
the
QUALITY
OF
CODE
MATTERS.Prefer
to
use
iterative
solution
as_much
as_possible
coz
that is most
preferred
as_you
might
not know
the
stack
size
the
problem
needs in real
world
problems.
-- https: // www.youtube.com / watch?v = dIrS31CCITM
- use
comments in the
code  # function to check if the str is a palindrome)

Hard
problems
are
very
rare.Do
not waste
time
trying
to
write
code
for them.
    You
    can
    write
    alog or pseudo
    code in your
    paper
    but
    DO
    NOT
    SOLVE(MORE
    TIME
    IS
    NEEDED
    AND
    YOU
    ARE
    WASTING
    TIME)

    How
    are
    dictionaries
    stored in memory
    https: // stackoverflow.com / questions / 327311 / how - are - pythons - built - in -dictionaries - implemented

    If
    you
    want
    to
    break
    a
    line
    into
    miltiple
    lines
    What is the
    line? You
    can
    just
    have
    arguments
    on
    the
    next
    line
    without
    any
    problems:

    a = dostuff(blahblah1, blahblah2, blahblah3, blahblah4, blahblah5,
                blahblah6, blahblah7)
    Otherwise
    you
    can
    do
    something
    like
    this:

    if a == True and \
            b == False

    From
    your
    example
    line:

    a = '1' + '2' + '3' + \
        '4' + '5'

    List
    Concepts:

    Sorting
    of
    list
    of
    tuples
    a = [(2, 3), (6, 7), (3, 34), (24, 64), (1, 43)]
    >> > sorted(a)
    [(1, 43), (2, 3), (3, 34), (6, 7), (24, 64)]
    >> > a = [(2, 3), (6, 7), (3, 34), (24, 64), (1, 43), (1, 42)]
    >> > sorted(a)
    [(1, 42), (1, 43), (2, 3), (3, 34), (6, 7), (24, 64)]

    1.
    my_list = [1, 2, 3]
    my_list.append(4)  # doesn't return the list. returns None
    my_list + [4]  # returns a list object
    [5] * 4
    will
    output[5, 5, 5, 5]

YOU
CANNOT
DELETE
ITEM
FROM
A
LIST
WHILE
ITERATING
THROUGH
IT
using
for loop, instead
    use
    while loop
18: Quick
sort:
Implementation: https: // www.geeksforgeeks.org / quick - sort /
video: https: // www.youtube.com / watch?v = PgBzjlCcFvc
19: How
to
approach
an
interview
question
pg: 62
of
book
CTCI
20: string or character
matching is a
costly
process.Try
using
ord(char)
in string
matching
problems
like
https: // github.com / careercup / CtCI - 6
th - Edition - Python / tree / master / Chapter1 / 1
_Is % 20U
nique
21: Matrix
representation in Python
my_matrix = [
    [1, 2, 3, 4, 0],
    [6, 0, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 0, 18, 19, 20],
    [21, 22, 23, 24, 25]
]
if you want to add elements to the first row of an EMPTY matrix do
output_matrix.append([])
for i in range(5):
    output_matrix[0].append(i)
Initializing
a
matrix:
matrix = [[0] * num_cols for i in range(num_rows)]

22: We
know
how
to
add
2
number
starting
from the units

digit.you
have
2
variables
sum = 0 and carry = 0 and solve
the
problem
What
iff
you
are
FORCED
to
add
2
number
from the top

most
digit
like
https: // github.com / careercup / CtCI - 6
th - Edition - Python / blob / master / Chapter2 / 5
_Sum_Lists.py
we
have
to
use
the
formula
res = res * 10 + num_1 + num_2
23: There
are
mainly
3
ways
to
handle
LinkedList
problems.
- Brute
Force(Use
nested
for loop) - inefficient
- Hash
table(Complexity - best or good
Depending
on
the
problem)
Try
to
get
a
result
using
Hash
table
approach
first
- 2
Pointer(Mostly
the
best).Diff
approaches
are
like
the
following
- two
fast and slow
pointes
on
the
same
list
- two
fast and slow
pointers
on
2
diff
lists

26: Start
using
set
data
structure
when
ever
there is a
need
Palindrome
prob


class Solution(object):
    largest_palindrome = ''

    def longestPalindrome(self, s):
        res = ""
        for i in xrange(len(s)):
            # odd case, like "aba"
            tmp = self.helper(s, i, i)
            if len(tmp) > len(res):
                res = tmp
            # even case, like "abba"
            tmp = self.helper(s, i, i + 1)
            if len(tmp) > len(res):
                res = tmp
        return res

    # get the longest palindrome, l, r are the middle indexes
    # from inner to outer
    def helper(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1;
            r += 1
        return s[l + 1:r]


Interview
preps
https: // www.quora.com / How - can - I - prepare - myself - to - be - a - software - engineer - at - Google
https: // www.freecodecamp.org / news / how - to - get - a - software - engineer - job - at - google - and -other - top - tech - companies - efa235a33a6d /

geeksforgeeks
important
links
https: // www.geeksforgeeks.org / fundamentals - of - algorithms /
https: // practice.geeksforgeeks.org / courses / dsa - self - paced?utm_source = geeksforgeeks & utm_medium = referral & utm_campaign = GfG_Right_Top_DSA_SP_Info
https: // www.geeksforgeeks.org / recursion /
https: // practice.geeksforgeeks.org / tag - page.php?tag = recursion & isCmp = 0
https: // www.geeksforgeeks.org / commonly - asked - data - structure - interview - questions - set - 1 /

8.
I
have
seen
so
far
follows
the
following
approach:
THINK
OF
RECURSSION
IN
A
PROBABILISTIC
OR
COMBINATIONS
PROBLEM.
YOU
MIGHT
IN
MOST
CASES
HAVE
TO
FILL
IN
THE
BLANK('_')
with a possible
list
of
values.

All
you
have
to
do is find
those
MISSING
BLANKS and Find
the
POSSIBLE
VALUES
that
you
have
to
fill
into
these
missing
blank and iterate
through
these
possible
values.Your
loop
should
be
such
that
you
iterate
to
the
next
possible
value and callback
your
fn.


def callback(expected_length, possible_options, binary_string):
    if len(binary_string) == expected_length:
        possible_options.append(binary_string)
        return

    for i in range(0, 2):
        callback(expected_length, possible_options, binary_string + str(i))


def generate_all_comb(word, n, mystr):
    if n == 0:
        print
        mystr
        return
    for letter in word:
        # you can't do mystr = mystr + letter in this line because my_str value should
        # **ONLY** be generated DYNAMICALLY (like in the next line)
        generate_all_comb(word, n - 1, mystr + letter)
    return


8.1
how
to
draw
recursion
diagram
while solving problem
https: // www.geeksforgeeks.org / recursion /
8.2
For
these
kind
of
problems
https: // www.geeksforgeeks.org / print - all - combinations - of - given - length /
We
need
to
use
probabilistic
approach.Lets
say
k = 3
i / p = ['a', 'b']
o / p = '_ _ _'(In
each
of
the
dashes
we
will
have
either
'a' or 'b')
Loop
through
you
inputs and populate
each
dash
with AN input and call your
fun
recursively.End
case
will
be
when
your
output
string is of
length
k
8.3
https: // www.geeksforgeeks.org / print - all - possible - expressions - that - evaluate - to - a - target /
Most
of
the
problems
related
to
binary
numbers
fall
under
this
category

8.3
.1
First
approach - applicable
when
your
list
of
possibilities is large
For
problems
where
the
spots('_')
you
need
to
fill
are
missing
char in a
str
and you
have
a
set
of
possible
values
that
will
go
into
the
spots('_').
Something
similar
to
8.2
but
slightly
different
k = 3
i / p = ['a', 'b']
Question
condition
can
be
like - Fill
diff
combs
of
ip in given
string
"d _ _ e_ f"
o / p = 'd _ _e _ f'(In
each
of
the
dashes
we
will
have
either
'a' or 'b')
In
this
case, forget
"d,e and f".Just
focus
on
the
3
dashes("_").
So
your
Qus is
k = 3
i / p = ['a', 'b']
o / p = '_ _ _'(In
each
of
the
dashes
we
will
have
either
'a' or 'b')
First
get
all
possible
combinations and then
substitute
these
combinations in your
actual
input
'd _ _e _ f'
8.3
.2
Second
approach(Non
standard
approach) -
applicable
when
your
list
of
possibilities is small


def callback(my_str):
    if my_str.count('_') == 0:
        all_possible_combinations.append(my_str)
        return
    for index, char in enumerate(my_str):
        if char == '_':
            callback(my_str[0:index] + '0' + my_str[index + 1:])
            callback(my_str[0:index] + '1' + my_str[index + 1:])


8.4
we
might
not have
to
use
for loop every time.Rather we can index our
input
like
the
following


def a_to_i(mystr, val):  # PROGRAM IS UNTESTED
    if not my_str:
        return val

    a_to_i(mystr[1:], (val * 10) + (ord(mystr[0]) - ord('0')))


8.5.How
to
generate
all
comb
of
2
diff
words(no
2
elements
of
the
same
word
can
be in a
combination)
output = []
letter_comb = ''
my_string_list = ['abc', 'def']


def backtrack(my_string_list, letter_comb):
    if not my_string_list:
        output.append(letter_comb)
        return

    for letter in my_string_list[0]:
        backtrack(my_string_list[1:], letter_comb + letter)


Leet
code
Sols: https: // github.com / qiyuangong / leetcode


def push(value):
    new_node = Node(value)
    new_node.next = self.head
    self.next = new_node


def add_lists(l1, l2):
    carry = 0
    while l1:
        added_number = 0
        if carry:
            added_number = added_number + carry
            carry = 0

        first_value = l1.value
        second_value = l2.value

        added_number += first_value + second_value

        if carry added_number / 10 == 0:


def fill_dict(char, row_counter, my_dict):
    my_dict[row_counter].append(char)

    num_of_rows = 3
    string = 'paypalishiring'
    row_counter = 1
    col_counter = 1

    my_dict = {}

    for i in range(1, num_of_rows + 1):
        my_dict[i] = []

    for char in string:
        if row_counter == 1 or increment_cyle:
            increment_cyle = True
            if decrement_cycle:
                decrement_cycle = False
            my_dict = fill_dict(char, row_counter, my_dict)
            row_counter += 1

        if row_counter == num_of_rows or decrement_cycle:
            decrement_cycle = True
            if increment_cyle:
                increment_cyle = False
                row_counter -= 1
                continue
            my_dict = fill_dict(char, row_counter, my_dict)
            row_counter -= 1


def generate_all_duplets(nums):
    duplets = []

    for index, num in enumerate(nums):
        for index_2, num_2 in enumerate(nums[index:]):
            duplets.append(num, num_2)

    return duplets


smallest_word = my_list[0]
for word in my_list:
    if len(word) < len(smallest_word):
        smallest_word = word


def check_if_all_elements_start_with_smallest_word(smallest_word, my_list):
    for word in my_list:
        if word.startswith(smallest_word):
            continue
        else:
            return False
    return True


while (smallest_word):
    if check_if_all_elements_start_with_smallest_word(smallest_word, my_list):
        print
        smallest_word
        break
    else:
        smallest_word = smallest_word[:-1]

start_pointer = 0
end_pointer = len(my_list) - 1
max_area = 0

while (True):
    if start_pointer == end_pointer:
        break

    curr_area = min(my_list[start_pointer], my_list[end_pointer]) * (
                end_pointer - start_pointer)

    if curr_area > max_area:
        max_area = curr_area

    if my_list[
        start_pointer] > my_dict = {2:'abc', 3: 'def', 4: 'ghi', 5: 'jkl', 6: 'mno', 7: 'pqrs', 8: 'tuv', 9: 'wxyz'}

    all_individual_combs = []

    while (digits):
        num = digit % 10
        all_individual_combs.append(my_dict[num])
        digit = digit / 10
        my_list[end_pointer]:
    end_pointer -= 1
else:
    start_pointer += 1

    my_dict = {2: 'abc', 3: 'def', 4: 'ghi', 5: 'jkl', 6: 'mno', 7: 'pqrs', 8: 'tuv',
               9: 'wxyz'}

    all_individual_combs = []

    while (digits):
        num = digits % 10
        all_individual_combs.append(my_dict[num])
        digits = digits / 10

    comb_0 = all_individual_combs[0]
    comb_1 = all_individual_combs[1]
    len_of_string_to_be_formed = len(all_individual_combs)  # assuming all i/p to be <100
    my_list = []

    for letter in comb_0:
        generate_all_combinations(letter, comb_1, len_of_string_to_be_formed)

output = []
letter_comb = ''
my_string_list = ['abc', 'def']


def backtrack(my_string_list, letter_comb):
    if not my_string_list:
        output.append(letter_comb)
        return

    for letter in my_string_list[0]:
        backtrack(my_string_list[1:], letter_comb + letter)


my_var = "1011_0011_01"


def backtrack(num, num_len):
    if not '_' in num:
        print
        num
        return num

    for digit in num:
        if digit == '_':
            backtrack(num.replace('_', '0', 1), num_len)
            backtrack(num.replace('_', '1', 1), num_len)


def backtrack(my_word_list, len_expected, formed_word=''):
    if len(formed_word) == len_expected:
        my_combination_list.append(formed_word)
        return

    for letter in my_word_list[0]:
        backtrack(my_word_list[1:], len_expected, formed_word=formed_word + letter)


my_combination_list = []
my_word_list = ['abc', 'def']

backtrack(my_word_list, len(my_word_list))


def a_to_i(my_str, val):
    print
    'val = ', val
    if not my_str:
        return val

    return a_to_i(my_str[1:], val * 10 + (ord(my_str[0]) - ord('0')))


_
_
_
_
_
_


def form_bin_str_len_n(expected_length, possible_options, binary_string):
    if len(binary_string) == expected_length:
        possible_options.append(binary_string)
        return

    for i in range(0, 2):
        print
        'i = ', i
        form_bin_str_len_n(expected_length, possible_options, binary_string + str(i))
        print
        'returned possible_options = ', possible_options, binary_string


# https://www.geeksforgeeks.org/print-all-possible-expressions-that-evaluate-to-a-target/
# Print all possible expressions that evaluate to a target
# Not fully complete - optimization needs to be done
import operator


def print_all_strings_k_length(self, string, k, formed_word):
    if formed_word.count('(') > k / 2 or formed_word.count(')') > k / 2:
        return

    if len(formed_word) == k:
        all_combination.append(formed_word)
        return

    for char in string:
        print_all_strings_k_length(string, k, formed_word + char)


def stack_check(combination):
    close_brace_count = 0
    open_brace_count = 0
    for char in combination:
        if char == '(':
            open_brace_count += 1
        else:
            close_brace_count += 1
            if close_brace_count > open_brace_count:
                return False
    return True


proper_combinataions = []
all_combination[:] = [combination for combination in all_combination if
                      stack_check(combination)]
for combination in all_combination:
    if not stack_check(combination):
        print
        combination
        all_combination.remove(combination)
        continue

    # proper_combinataions.append(combination)

input_string = '123'
input_string = '_'.join([char for char in input_string])
k = input_string.count('_')
combinations_string = '+-*'
ops = {"+": operator.add, "-": operator.sub, '*': operator.mul}
all_combination = []
all_possible_input_string_variations = []

print_all_strings_k_length(combinations_string, k, '')

for combination in all_combination:
    formed_string = input_string
    for variation in combination:
        pos = formed_string.find('_')
        formed_string = formed_string[0:pos] + variation + formed_string[pos + 1:]
    all_possible_input_string_variations.append(formed_string)


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

    def get_data(self):
        return self.data

    def get_next(self):
        return self.next_node


slow_ptr = fast_ptr = head
result_node_head = None
next_greatest_element = 0

while (slow_ptr.next):
    print
    'fast_ptr = ', fast_ptr.val
    print
    'slow_ptr = ', slow_ptr.val
    if fast_ptr.val > slow_ptr.val:
        next_greatest_element = fast_ptr.val
        # result_node, result_node_head = self.insert_new_element(next_greatest_element, result_node_head, result_node)
        # result_node, result_node_head = self.insert_new_element(next_greatest_element, result_node_head, result_node)
    if next_greatest_element or not fast_ptr.next:
        slow_ptr = slow_ptr.next
        print
        'next_greatest_element = ', next_greatest_element
        # insert node here
        next_greatest_element = 0
        fast_ptr = slow_ptr
    fast_ptr = fast_ptr.next

# insert 0 at the end


for idx, x in enumerate(nums):
    while st and st[-1][1] < x:
        a, b = st.pop()
        cache[a] = x
    st.append((idx, x))

result = [-1] * len(nums)
for idx, x in enumerate(nums):
    if idx in cache:
        result[idx] = cache[idx]

print
result

Quick
Sort

my_list = [10, 80, 30, 90, 40, 50, 70]
copy_list = my_list


def place_pivot_in_appropriate_pos(pivot_index):
    pos_of_last_found_smallest_ele = -1
    pos_of_pivot = pivot_index
    pivot_element = my_list[pos_of_pivot]

    for index, element in enumerate(copy_list):
        if element < pivot_element:
            pos_of_last_found_smallest_ele += 1
            temp = my_list[pos_of_last_found_smallest_ele]
            my_list[pos_of_last_found_smallest_ele] = element
            my_list[index] = temp
            current_pivot_pos = index
            print
            'ins if', my_list

    print
    pos_of_last_found_smallest_ele
    # if pos_of_last_found_smallest_ele >= pivot_index:
    temp = my_list[pos_of_last_found_smallest_ele + 1]
    my_list[pos_of_last_found_smallest_ele + 1] = pivot_element
    my_list[pos_of_pivot] = temp
    return pos_of_last_found_smallest_ele


for pivot_index, pivot_element in enumerate(copy_list):
    place_pivot_in_appropriate_pos(pivot_index)

Intersection
of
linked
list

One
important
thing
to
note is when
2
linked
lists
intersect, all
the
elements
after
the
intersection
point
will
be
same
as_
the
intersecting
node
s
next_
part
can
only
hold
the
address
to
ONE
NODE
which
is_
next
to
it.

1->2->3->4->5->6(list_1)
7->8
_ | (list_2)
As
you
can
see
list_1 and list_2
intersect
at
the
Node
5 and the
next_
part
of
node
5
can
only
point
to
1
node
which
is_
6

Do
not get
confused if list_2
would
be
something
like
7->8->5->11->12.
5
cannot
point
to
11
because
it
is_
already
pointing
to
6.

Traverse
through
both
the
nodes
individually.Subtract
the
length
of
the
smallest
list_
from_
the
longest
one
and_
Start
to
traverse
the
longer
list_
from_
the
start
untill
you
reach
the
point
where
both
the
lists
will
be
of
same
size.
Now
start
traversing
both
the
lists
together
in_
each
iteration.If
the
node
pointed
to
by
both
the
lists is same, that
is_
your
intersection
point
"""

for line in my_s.split('\n\n'):
    print(line)