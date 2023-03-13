#!/usr/bin/env python
# coding: utf-8

# In[1]:


def maxval(a, b, c):
    """
    compute the maximum among three numbers

    Arguments:
        a, b, c: integers
    Return:
        max_of_three: integer
    """
    if a > b and a > c:
        return a
    elif a < b and b > c:
        return b
    elif c > a and c > b:
        return c
    elif a > b and a < c:
        return c
    elif b > a and b < c:
        return c
    elif b > c and c<a:
        return a
    else:
        c > a and c<b
        return b


# In[7]:


def dim_equal(A,B):
    if len(A) == len(B):
        return 'True'
    else:
        return 'False'


# In[6]:


A = [[1,2],[3,4]]
B = [[5,6],[7,8]]

len(A) == len(B)


# In[8]:


dim_equal([[1,2],[3,4]],[[5,6],[7,8]])


# In[9]:


def first_three(L):
    maxi = []
    mini = []
    for i in range(L):
        
        if l[i]>maxi:
            maxi = l[i]
            i = i+1
            
        


# In[33]:



def first_three(L):
    x=[]

    while (len(l)>0):
        min = l[0]
        for i in range(len(l)):
            if l[i]<min:
                min= l[i]
        x.append(min)
        l.remove(min)
    return (x[-1],x[-2],x[-3])


# In[35]:


first_three(1,2,3,4,5)


# In[22]:


numbers = input().split(' ')  # Accept input as a space-separated sequence
integers = [int(float(num)) for num in numbers]
print(integers)
# Convert each element to an integer
output = ",".join(str(num) for num in integers)  # Convert the list to a comma-separated string
print(output)  # Print the output


# In[37]:


def first_three(L):
    """
    computes the first three maximums

    Argument:
        L: list
    Return:
        fmax, smax, tmax: three integers
    """
    x=[]

    while (len(l)>0):
        min = l[0]
        for i in range(len(l)):
            if l[i]<min:
                min= l[i]
        x.append(min)
        l.remove(min)
    return (x[-1],x[-2],x[-3])


# In[38]:


L = [1,2,3,4,5]
first_three(L)


# In[45]:


l = [1.0,3.7,8.9,5.5,1.9,6.3,0.1,9.9]
a = []

for i in range(len(l)-1):
    if float((l[i])) < float((l[i+1])):
        if i not in a:
            a.append(i)
print(a)
      
    


# In[56]:


l = [1.0,3.7,8.9,5.5,1.9,6.3,0.1,9.9]

x=[]
while len(l) > 0:
        min = l[0]
        for i in range(len(l)):
            if (l[i]) < (min):
                min = (l[i])
        x.append(min)
        l.remove(min)
print(x)


# In[90]:


def get_range(L):
    """
    compute the range of a list L

    Argument:
        L: list
    Return:
        range: float
    """
    x= []
    while (len(L)) > 0:
        mini = [10.0]
        for i in range(len(L)):
            if ((L[i])) < mini[0]:
                mini = L[i]
            x.append(mini)
            L.remove(mini)
        range_of = (L[-1]-L[0])
    return (range_of)


# In[91]:


L = [1.0,3.7,8.9,5.5,1.9,6.3,0.1,9.9]
get_range(L)


# In[95]:


l = [1.0,3.7]
print(l[0])


# In[93]:


mini = [10.0]
l[0] < mini[0]


# In[96]:


def get_range(nums):
    min_val = nums[0]
    max_val = nums[0]
    for num in nums:
        if num < min_val:
            min_val = num
        elif num > max_val:
            max_val = num
    return max_val - min_val


# In[97]:


nums = [1.0,3.7,8.9,5.5,1.9,6.3,0.1,9.9]
get_range(nums)


# In[98]:


def get_Range(nums):
    min_val = nums[0]
    max_val = nums[0]
    for i in nums:
        if min_val > i:
            min_val = i
        elif i > max_val:
            max_val = i
    return max_val - min_val


# In[99]:


nums = [1.0,3.7,8.9,5.5,1.9,6.3,0.1,9.9]
get_Range(nums)


# In[107]:


def is_perfect(n):
    n = 6
    a= []
    b = 0
    for i in range(1,n):
        if n % i == 0:
            a.append(i)
    for i in range(len(a)):
        b += a[i]
    if b == n:
        return 'True'
    else:
        return 'False'


# In[108]:


is_perfect(6)


# In[128]:


alpha = 'abcdefghijklmnopqrstuvwxyz'
d1 = 'dog'
d2 = 'cat'
disc1 = []
disc2 = []
distance = 0
for i in range(len(alpha)):
    for j in range(len(d1)):
        if d1[j] == alpha[i]:
            disc1.append(i)
for i in range(len(alpha)):
    for j in range(len(d2)):
        if d2[j] == alpha[i]:
            disc2.append(i)
for i in range(len(disc1)):
    for j in range(len(disc2)):
        distance = abs((disc1[i])+(disc2[j]))


# In[129]:


print(disc1)
print(disc2)
print(distance)


# In[134]:


alpha = 'abcdefghijklmnopqrstuvwxyz'
d1 = 'dog'
d2 = 'cat'
disc1 = []
disc2 = []
distance = 0
# for i in range(len(alpha)):
#     for j in range(len(d1)):
#         if d1[j] == alpha[i]:
#             disc1.append(i)
# for i in range(len(alpha)):
#     for j in range(len(d2)):
#         if d2[j] == alpha[i]:
#             disc2.append(i+1)
for i in range(len(alpha)):
    for j in range(len(d2)):
        


# In[147]:


alphabet = 'abcdefghijklmnopqrstuvwxyz'
d1 = 'dog'
d2 = 'cat'
a = []
b = []
for i in range(len(d1)):
    for k in range(len(alphabet)):
        if d1[i] == alphabet[k]:
            a.append(k+1)
for k in range(len(alphabet)):
    for i in range(len(d2)):
        if d2[i] == alphabet[k]:
            b.append(k)
            
#         for j in range(len(d2)):
#             if d1[i] == alphabet[k]:
#                 a.append(k)
#             if d2[j] == alphabet[k]:
#                 b.append(k)
print(a)
print(b)


# In[151]:


def distance(d1,d2):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    a = []
    b = []
    distance = 0
    for i in range(len(d1)):
        for k in range(len(alphabet)):
            if d1[i] == alphabet[k]:
                a.append(k+1)
    for i in range(len(d2)):
        for k in range(len(alphabet)):
            if d2[i] == (alphabet[k]):
                b.append(k+1)
    for i in range(len(a)):
        for j in range(len(b)):
            if len(a) != len(b):
                return -1
            else:
                distance = abs(a[i]+b[j])
                return distance


# In[152]:


d1 = 'dog'
d2 = 'cat'
distance(d1,d2)


# In[100]:


alphabet = 'abcdefghijklmnopqrstuvwxyz'
distance= 0
a = []
b = []
c = []
d1 = 'dog'
d2 = 'cat'
result_list=[]
for i in range(len(d1)):
    for k in range(len(alphabet)):
        if d1[i] == alphabet[k]:
            a.append(abs(k+1))
for i in range(len(d2)):
    for k in range(len(alphabet)):
        if d2[i] == (alphabet[k]):
            b.append(abs(k+1))
            
for i in range(len(a)):
    result_list.append(abs(a[i]-b[i]))

sum = 0
for i in result_list:
    sum = sum + i
    distance = sum


# In[5]:


1+14+13


# In[96]:


list1 = [1, 2, 3, 4, 5]
list2 = [5, 4, 3, 2, 1]
result_list = []

for i in range(len(list1)):
    result_list.append(abs(list1[i] - list2[i]))

print(result_list)
sum = 0
for i in result_list:
    sum = sum+i
print(sum)


# In[10]:


def some_function(word):
    space = ' ' # there is a single space between the quotes
    if space in word:
        return False
    # both letters 'A' and 'Z' are in upper case
    if not('A' <= word[0] <= 'Z'):
        return False
    for i in range(1, len(word)):
        # both letters 'a' and 'z' are in lower case
        if not('a' <= word[i] <= 'z'):
            return False
    return True


# In[77]:


def poly(L, x_0):
    psum = 0
    n = len(L)
    for i in range(n):
        psum = psum + L[i] * (x_0 ** i)
    return psum


# In[78]:


poly([1, 2, 3], 5)


# In[51]:


print(A[1])


# In[81]:


def poly_zeros(L, a, b):
    zeros = [ ]
    for x in range(a, b + 1):
        if poly(L, x) != 0:
            zeros.append(x)
    return zeros


# In[86]:


poly_zeros([2, -3, 1], 0, 4)


# In[85]:


def poly_zeros(L, a, b):
    zeros = [ ]
    for x in range(a, b + 1):
        if poly(L, x) == 0:
            zeros.append(x)
    return zeros


# In[88]:


a = [[1,2],[3,4],[5,6]]
print(len(a[0]))


# In[89]:


matrix = list(map(int,input().split()))
for i in range(len(matrix)):
    n = int(input()).split(' ')
print(matrix)


# In[92]:


alphabet = 'abcdefghijklmnopqrstuvwxyz'
d1 = 'dog'
d2 = 'cat'

for i in range(len(alphabet)):
    


# In[93]:


l = [1,2,3]
a = []
for i in l:
    a.append(-i)
print(a)


# In[103]:


def distance(word_1, word_2):
    """
    compute distance between two words

    Arguments:
        word_1, word_2: strings
    Return:
        word_distance: int
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    word_distance= 0
    a = []
    b = []
    c = []
    result_list=[]
    if len(word_1) != len(word_2):
        return -1
    for i in range(len(word_1)):
        for k in range(len(alphabet)):
            if word_1[i] == alphabet[k]:
                a.append(abs(k+1))
    for i in range(len(word_2)):
        for k in range(len(alphabet)):
            if word_2[i] == (alphabet[k]):
                b.append(abs(k+1))
                
    for i in range(len(a)):
        result_list.append(abs(a[i]-b[i]))
    
    sum = 0
    for i in result_list:
        sum = sum + i
        distance = sum
    return(distance)


# In[105]:


distance('dog','dogt')


# In[115]:


matrix = [[1,2,3],[4,5,6]]
transpose = []
rows = len(matrix)
cols = len(matrix[0])

transpose = [[0 for j in range(rows)] for i in range(cols)]
for i in range(rows):
    for j in range(cols):
        transpose[j][i] = matrix[i][j]
for row in transpose:
    print(row)


# In[111]:


print(transpose)


# In[114]:


# Define the matrix
matrix = [[1, 2, 3],
          
          [7, 8, 9]]

# Get the dimensions of the matrix
rows = len(matrix)
cols = len(matrix[0])

# Create an empty matrix to store the transposed matrix
transpose = [[0 for j in range(rows)] for i in range(cols)]

# Loop through the original matrix and copy the values to the transpose matrix
for i in range(rows):
    for j in range(cols):
        transpose[j][i] = matrix[i][j]

# Print the original matrix and the transposed matrix
print("Original matrix:")
for row in matrix:
    print(row)
    
print("Transposed matrix:")
for row in transpose:
    print(row)


# In[122]:


def transpose(mat):
    """
    compute the transpose of the matrix

    Argument:
        mat: list of lists
    Return:
        mat_trans: list of lists
    """

    A_trans = []
    rows = len(mat)
    cols = len(mat[0])
    
    A_trans = [[0 for j in range(rows)] for i in range(cols)]
    for i in range(rows):
        for j in range(cols):
            A_trans[j][i] = mat[i][j]
    for row in A_trans:
          print(row)


# In[123]:


mat = [[1,2],[3,4]]
transpose(mat)


# In[147]:


def transpose(mat):
    """
    compute the transpose of the matrix

    Argument:
        mat: list of lists
    Return:
        mat_trans: list of lists
    """

    mat_trans = []
    rows = len(mat)
    cols = len(mat[0])
    
    mat_trans = [[0 for j in range(rows)] for i in range(cols)]
    for i in range(rows):
        for j in range(cols):
            mat_trans[j][i] = mat[i][j]
#     for row in mat_trans:
        return (mat_trans)


# In[148]:


mat = [[1,2],[3,4],[5,6]]
transpose(mat)


# In[151]:


def transpose_matrix(matrix):
    """
    Transposes the given matrix without using the built-in `transpose()` function.
    """
    # Get the number of rows and columns in the matrix
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Create a new matrix with the number of columns and rows flipped
    transposed_matrix = [[0 for j in range(num_rows)] for i in range(num_cols)]

    # Iterate over the original matrix and copy values to the new matrix
    for i in range(num_rows):
        for j in range(num_cols):
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix


# In[153]:


matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

transposed_matrix = transpose_matrix(matrix)

print(transposed_matrix)


# In[162]:


matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

sum_row = 0

for i in range(len(matrix)):
    sum_row = sum_row+matrix[i]
    
print(sum_row)


# In[195]:


def sum_row(mat):
    row = len(mat)
    col = len(mat[0])
    
    row_sums = []
    
    for i in range(row):
        row_sum = 0
        for j in range(col):
            row_sum += mat[i][j]
        row_sums.append(row_sum)
    return row_sums
    
def sum_col(mat):
    row = len(mat)
    col = len(mat[0])
    col_sums = []
    for j in range(col):
        col_sum = 0
        for i in range(row):
            col_sum += mat[i][j]
        col_sums.append(col_sum)
    return col_sums

def sum_diagonal(mat):
    num_rows = len(mat)
    num_cols = len(mat[0])

    diagonal_sum = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if i == j:
                diagonal_sum += matrix[i][j]

    return diagonal_sum


# In[198]:


mat = [[1,2,3],[4,5,6],[7,8,9]]
print(sum_row(mat))
print(sum_col(mat))
print(sum_diagonal(mat))


# In[199]:





# In[196]:


sum_diagonal(mat)


# In[206]:



row = len(mat)
col = len(mat[0])
    
row_sums = []
col_sums = []
for i in range(row):
    row_sum = 0
    for j in range(col):
        row_sum += mat[i][j]
    row_sums.append(row_sum)
    
for j in range(col):
    col_sum = 0
    for i in range(row):
        col_sum += mat[i][j]
    col_sums.append(col_sum)
    
num_rows = len(mat)
num_cols = len(mat[0])

diagonal_sum = 0
for i in range(num_rows):
    for j in range(num_cols):
        if i == j:
            diagonal_sum += matrix[i][j]
for i in range(len(row_sums)):
    for i in range(len(col_sums)):
        if i == diagonal_sum:
            print('True')
        else:
            print('False')
            

               
   


# In[207]:


# Define a matrix
matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

# Create a new matrix with swapped rows and columns
transposed_matrix = []
for i in range(len(matrix[0])):
    row = []
    for j in range(len(matrix)):
        row.append(matrix[j][i])
    transposed_matrix.append(row)

# Print the transposed matrix
for row in transposed_matrix:
    print(row)


# In[210]:


def transpose(mat):
    mat_trans = []
    for i in range(len(mat[0])):
        row=[]
        for j in range(len(mat)):
            row.append(mat[j][i])
        mat_trans.append(row)
    for row in mat_trans:
        return row


# In[211]:


mat = [[1,4,7],[2,5,8],[3,6,9]]
transpose(mat)


# In[221]:


# Define a matrix
matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]
def transpose_matrix(matrix):
# Create a new matrix with swapped rows and columns
    transposed_matrix = []
    for i in range(len(matrix[0])):
        row = []
        for j in range(len(matrix)):
            row.append(matrix[j][i])
        transposed_matrix.append(row)

    # Print the transposed matrix
        for row in transposed_matrix:
            print(row)


# In[222]:


matrix = [[1,4,7],[2,5,8],[3,6,9]]
transpose(matrix)


# In[224]:


l = [1,2,3,4,5]
l.pop(2)


# In[1]:


input_seq = input("Enter a sequence of words: ")

freq = {}

for word in input_seq.split():
    if word in freq:
        freq[word] += 1
    else:
        freq[word] = 1

print("Frequency dictionary:", freq)


# In[4]:


n = int(input())
n_str = str(n)
number = { '1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine','0':'zero''}

for i in n_str:
    print(number[digit])


# In[5]:


def is_key(D, key):
    """
    Determine the presence of key in D

    Arguments:
        D: dict
        key: could be of any type
    Return:
        bool
    """
    for i in D:
        return True
    else:
        return False

def value(D, key):
    """
    Get the value corresponding to the key in D

    Arguments:
        D: dict
        key: could be of any type
    Return:
        result: depends on the dict; refer problem statement
    """
    for i in D:
        return D[key]
    if i not in D:
        return 'None'


# In[10]:



seq = input().split(',')    
real_dict = {chr(i):[] for i in range(ord('a'),ord('z')+1)


for word in seq:
             real_dict[word[0]].append(word)


# In[12]:


seq = input().split(',')
real_dict = {chr(i):[] for i in range(ord('a'),ord('z')+1)}
for word in seq:
    real_dict[word[0]].append(word)


# In[24]:


seq = input().split(',')
real_dict = {chr(i):[] for i in range(ord('a'),ord('z')+1)}
for word in seq:
    real_dict[word[0]].append(word)

print(real_dict)


# In[43]:


seq = input().split(',')

real_dict = {chr(i):[] for i in range(ord('a'),ord('z')+1)}
for i in seq:
    real_dict[i[0]].append(i)
print(real_dict)
my_list = ['a','b','c','d','e','f','g','j','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

for item in my_list:
    if item in real_dict and real_dict[i] is not None:
        print(item,real_dict[item])
    
        


# In[44]:


def fibo(n):
    if n == 1 or n == 2:
        return 1
    return fibo(n - 1) + fibo(n - 2)


# In[45]:


fibo(5)


# In[46]:


fibo(10)


# In[4]:


def fibo(n):
    if n ==1 or n ==2:
        return 1
    return fibo(n-1)+fibo(n-2)


# In[5]:


fibo(10)


# In[7]:


pw, valid = input(), False  # initialize valid to False
# first condition is being checked here
if 8 <= len(pw) <= 32:
    # second condition is being checked here
    if 'a' <= pw[0] <= 'z' or 'A' <= pw[0] <= 'Z':
        # third and four conditions are being checked here
        if not('/' in pw or '\\' in pw or '=' in pw or "'" in pw or '"' in pw or ' ' in pw):
            valid = True
print(valid)


# In[1]:


def fibo(n):
    if n ==1 or n ==2:
        return 1
    else:
        return (fibo(n-1)+fibo(n-2))


# In[2]:


fibo(3)


# In[3]:


fibo(10)


# In[8]:


def get_column(mat,col):
    col_list = []
    m = len(mat)
    for row in range(m):
        col_list.append(mat[row][col])
    return col_list
    row_list = []
    n = len(mat)
    for col in range(n):
        row_list.append([col][row])
    return row_list
        


# In[9]:


get_column([[1, 2], [3, 4]], 0)


# In[6]:


def get_column(mat, col):
    col_list = [ ]
    m = len(mat)
    for row in range(m):
        col_list.append(mat[row][col])
    return col_list

def get_row(mat, row):
    row_list = [ ]
    n = len(mat[0])
    for col in range(n):
        row_list.append(mat[row][col])
    return row_list


# In[7]:


get_column([[1, 2], [3, 4]], 0)


# In[14]:


def two_level_sort(scores):
    """
    Perform a two-level sort

    Argument: 
        scores: list of tuples, (string, integer)
    Return:
        result: list of tuples (string, integer)
    """
    # sort by marks in ascending order
#     scores.sort(key=lambda x: x[1])
    sorted(scores)
    # create a dictionary to store students with the same marks
    
    marks_dict = {}
    for name, marks in scores:
        if marks not in marks_dict:
            marks_dict[marks] = []
        marks_dict[marks].append(name)
    # sort the names within each group of same marks
    for marks in marks_dict:
        marks_dict[marks].sort()
    # create a list of tuples sorted in two levels
    sorted_scores = []
    for marks in sorted(marks_dict.keys()):
        for name in marks_dict[marks]:
            sorted_scores.append((name, marks))
    return sorted_scores


# In[15]:


[('Harish', 80), ('Ram', 90), ('Harshita', 80)]


# In[16]:


print(scores)


# In[28]:


def insert(L,x):
    
    m = []
    inserted = False
    for i in range(len(L)):
        if x > L[i]:
            inserted = False
            m.append(i)
        else:
            inserted = True
            m.append(x)
    return m
        


# In[24]:


def inser(L,x):
    for i in range(len(L)):
        if x > L[i]:
            continue
        else:
            L.append(x)
    return L


# In[36]:


L = [1,2,3,5]
x = 4
insert(L,x)


# In[23]:


print(inser)


# In[35]:


def insert (L,x):
    out_L = []
    inserted = False
    for elem in L:
        if (not inserted) and (x<elem):
            out_L.append(x)
        out_L.append(elem)
    
#     if (not inserted):
#         out_L.append(x)
    return out_L


# In[45]:


def factor (n):
    F = set()
    for i in range(1,n+1):
        if n%i == 0:
            F.add(i)
    return F

def common_factor(a,b):
    a_f = factor(a)
    b_f = factor(b)
    return a_f.intersection(b_f)
def factor_upto (n):
    D = dict()
    for i in range(1,n+1):
        D[i] = factor(i)
    return D


# In[40]:


factor(10)


# In[46]:


a = 10
b=25
common_factor(a,b)


# In[48]:


n = 4
factor_upto(n)


# In[53]:


L = input().split(',')
real_dict = dict()

for word in L:
    start = word[0]
    if start not in real_dict:
        real_dict[start] = []
    real_dict[start].append(word)
print(real_dict)


# In[52]:


L = input().split(',')

real_dict = dict()

for word in L:
    start = word[0]
    if start not in real_dict:
        real_dict[start] = [ ]
    real_dict[start].append(word)
print(real_dict)


# In[57]:


def group_by_city(scores_dataset):
    
    for student in scores_dataset:
        cities = dict()
        city = student['city']
        name = student['Name']
        if city not in cities:
            cities[city].append(name)
    return cities

def busy_cities(scores_dataset):
    cities = group_of_city(score_dataset)
    
    busy = []
    maxpop = 0
    if city in cities:
        if len(cities[city]) > maxpop:
            maxpop = len(cities[city])
            busy=[city]
        elif len(cities[city]) == maxpop:
            busy.append(city)
    return busy


# In[58]:


200
group_by_city(scores_dataset)


# In[59]:


n = int(input())

matrix = []

for i in range(n):
    l = []
    for num in input().split(' '):
        l.append(int(num))
        matrix.append(l)
print(matrix)

    


# 

# In[60]:


n = int(input())

matrix = [ ]
for i in range(n):
    L = [ ]
    for num in input().split(' '):
        L.append(int(num))
    matrix.append(L)
print(matrix)


# In[67]:


n = int(input())
I = []
for i in range(n):
    row= []
    for j in range(n):
        
        if i == j:
            row.append(1)
        else:
            row.append(0)
    I.append(row)

for i in range(n):
    for j in range(n):
        if j != (n-1):
            print(I[i][j],end=',')
        else:
            print(I[i][j])


# In[73]:


n = int(input())
matrix = []

for i in range(n):
    row = []
    for x in range(n):
        row.append(int(x))
    matrix.append(row)
s = int(input())
for row in range(n):
    for col in range(n):
        matrix[row][col] *= s
        
for row in range(n):
    for col in range(n):
        if col != (n-1):
            print(matrix[row][col], end=' ')
        else:
            print(matrix[row][col])
            
        
    
        


# In[74]:



n = int(input())
matrix = [ ]
for i in range(n):
    row = [ ]
    for x in input().split(' '):
        row.append(int(x))
    matrix.append(row)
s = int(input())

for row in range(n):
    for col in range(n):
        matrix[row][col] *= s

for row in range(n):
    for col in range(n):
        if col != n - 1:
            print(matrix[row][col], end = ' ')
        else:
            print(matrix[row][col])


# In[87]:


n = int(input())

A = []
for i in range(n):
    row = []
    for x in input().split(','):
        row.append(int(x))
    A.append(row)

B = []
for i in range(n):
    row = []
    for x in input().split(','):
        row.append(int(x))
    B.append(col)

C = []
for i in range(n):
    row = []
    for j in range(n):
        row.append(0)
    C.append(row)

for i in range(n):
    for j in range(n):
        C[i][j] = A[i][j]+B[i][j]
        
        if C[i][j] != n-1:
            print(C[i][j],end = ',')
        else:
            print(C[i][j])


# In[82]:



n = int(input())
matrix = [ ]
for i in range(n):
    row = [ ]
    for x in input().split(' '):
        row.append(int(x))
    matrix.append(row)
s = int(input())

for row in range(n):
    for col in range(n):
        matrix[row][col] *= s

for row in range(n):
    for col in range(n):
        if col != n - 1:
            print(matrix[row][col], end = ' ')
        else:
            print(matrix[row][col])


# 

# In[97]:


def solution(L):
    sorted_L = [ ]
    while L != [ ]:
        max_elem = L[0]
        for elem in L:
            if elem > max_elem:
                max_elem = elem
        L.remove(max_elem)
        sorted_L.append(max_elem)
    return sorted_L


# In[98]:


L = [1.1,2.2,3.3]
solution(L)


# In[100]:


# Create a dictionary to store the team names and their respective scores
teams = {"CSK": 0, "DC": 0, "KKR": 0, "MI": 0, "PK": 0, "RR": 0, "RCB": 0, "SH": 0}

# Get the details of the matches played
matches = int(input("Enter the number of matches played: "))

# Loop through each match and update the scores of the teams
for i in range(matches):
    print("Enter details of match", i+1)
    team1, team2, result = input().split()
    if result == "1":
        teams[team1] += 2
    elif result == "2":
        teams[team2] += 2
    else:
        teams[team1] += 1
        teams[team2] += 1

# Sort the teams based on their scores and names
sorted_teams = sorted(teams.items(), key=lambda x: (-x[1], x[0]))

# Display the IPL points table
print("IPL Points Table")
print("----------------")
print("Team\t\tPoints")
print("----------------")
for team, points in sorted_teams:
    print(team, "\t\t", points)


# In[101]:


n = input()
flag = False
for i in range(len(n)):
    if n[i] == 'o' or n[i] == 'l':
        flag == True
        break
    else:
        flag == False
if flag == False:
    print("No mistake")
else:
    for i in range(len(n)):
        if n[i] == 'l':
            n[i].replace(1)


# In[105]:


n = '987o35l7o4'

for i in range(len(n)):
    if n[i] == 'o':
        n[i].replace('o','0')
        


# In[124]:


m = '987o35l7o4'
for i in m:
    if i == 'o':
        i.replace('o','0')
        print(i,end='')
    elif i == 'l':
        i.replace('l','1')
        print(i,end='')
    else:
        print(i,end='')
        


# In[125]:


number = input("Enter a ten-digit number: ")

mistakes = 0
correct_number = ""

for digit in number:
    if digit == "0":
        mistakes += 1
        correct_number += "o"
    elif digit == "1":
        mistakes += 1
        correct_number += "l"
    else:
        correct_number += digit

if mistakes == 0:
    print("No mistakes")
else:
    print(f"{mistakes} mistake(s)")
    print(correct_number)


# In[130]:


m = [[1, 0], [0, 1]]

for i in (m):
    for j in (m):
        if i == j:
            print('Diagonal')
        else:
            print('not')


# In[132]:


m = [[1, 2,3], [4, 1,6],[7,8,1]]

for i in m:
    for j in m:
        if i == j:
            print(i)
        


# In[1]:


def freq_to_words(words):
    """
    Get the collection of all words that have a given frequency

    Argument
       words: list of strings
    Return: 
       result: dictionary 
           key: integer
           value: list of strings
    """
    freq_dict = {}
    
    for word in words:
        if word in freq_dict:
            freq_dict[word] += 1
        else:
            freq_dict[word] = 1
        
    word_freq_dict = {}
    word_freq_dict= { }
    
    for word,freq in freq_dict.items():
        if freq in word_freq_dict:
            word_frq_dict[freq].append(word)
        else:
            word_freq_Dict[freq] = [word]
    return word_freq_dict
    for word,freq in freq_dict.items():
        if freq in word_freq_dict:
            word_freq_dict[freq].append(word)
        else:
            word_freq_dict[freq] = [word]
    return word_freq_dict
    


# In[2]:


# Basic idea:
# (1) First get the transpose of the matrix.
# (2) Reverse the rows of the transposed matrix.
# You have done matrix transpose in week-5.
# You also know how to reverse a list.
# Put these two things together.
def get_column(mat,col):
    col_list = [ ]
    m = len(mat)
    for row in range(m):
        col_list.append(mat[row][col])
    return col_list
def get_column(mat, col):
    col_list = [ ]
    m = len(mat)
    for row in range(m):
        col_list.append(mat[row][col])
    return col_list

def transpose(mat):
    m, n = len(mat), len(mat[0])
    mat_trans = [ ]
    for i in range(n):
        mat_trans.append(get_column(mat, i))
    return mat_trans

def rotate(mat):
    # Get transpose
    mat_trans = transpose(mat)
    rotated_mat = [ ]
    m, n = len(mat_trans), len(mat_trans[0])
    for i in range(m):
        row = [ ]
        for j in range(n):
            # reverse the row of the transpose
            row = [mat_trans[i][j]] + row
        rotated_mat.append(row)
    return rotated_mat


# In[1]:


n = len(mat)
 m = len(mat[0])
 result = [[0] * n for i in range(m)]
 for i in range(n):
     for j in range(m):
         result[j][n-i-1] = mat[i][j]
 return result

n = len(mat)
m = len(mat[0])
result = [[0]*n for i in range(m)]
for i in range(m):
 for j in range(m):
     result[j][n-j-1] =mat[i][j]
 return result


# In[4]:


def number_to_list(num):
    # Convert the number to a string and get its length
    num_str = str(num)
    num_len = len(num_str)
    
    # Initialize an empty list to store the result
    result = []
    
    # Iterate over the digits of the number and append the corresponding value to the result
    for i, digit in enumerate(num_str):
        value = int(digit) * 10 ** (num_len - i - 1)
        result.append(value)
    
    return result


# In[5]:


num = 123
number_to_list(num)


# In[6]:


def number_to_list(num):
    # Convert the number to a string and get its length
    num_str = str(num)
    num_len = len(num_str)
    
    # Initialize an empty list to store the result
    result = []
    
    # Loop over each digit of the number
    for i in range(num_len):
        # Calculate the value of the current digit
        digit_value = int(num_str[i]) * 10 ** (num_len - i - 1)
        
        # Append the digit value to the result
        result.append(digit_value)
    
    return result


# In[27]:


n = 5

for i in range(1,n+1):
    for j in range(1,i+1):
        print(j,end='')
        if j != i:
            print(",",end='')
    print()
for i in range(n-1,0,-1):
    for j in range(1,i+1):
        print(j,end='')
        if j != i:
            print(',',end='')
    print()


# In[29]:


def check0(L):
    if len(L) == 0:
        return 0
    if L[0] == 0:
        return 1
    else:
        return check0(L[1:len(L)])
        


# In[31]:


L = [1,2,3,4,5,6]
check0(L)


# In[ ]:




