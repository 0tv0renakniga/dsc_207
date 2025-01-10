# -*- coding: utf-8 -*-
"""week1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KFAPIrGUuW0LWMqsWFWD-au0JYQdV6D2

# Week 1 Study Notebook

# UC San Diego OMDS DSC 207

## Check your Python skills while using Colab notebooks.

### Installing Otter-Grader

Otter-Grader is a powerful tool that helps you check your work before submitting assignments. To get started with Otter-Grader, you need to install it within this Colab notebook.

To install Otter-Grader, simply run the below code cell.

Note that questions 1-3 and 5-9 are autograded, while questions 4 and 10 will be graded manually.
"""

# Commented out IPython magic to ensure Python compatibility.
# DO NOT MODIFY
# %pip install otter-grader

files = "https://github.com/dsc207rfall2023/fa23course-dsc207r/raw/main/assignments/Week%201/tests.zip"
!wget $files && unzip -o tests.zip

# Initialize Otter
import otter
grader = otter.Notebook()

"""### Loading Python modules"""

import pandas as pd
import numpy as np

# ADD ADDITIONAL PYTHON MODULES BELOW

"""### 1.  Write a program to swap a string and an int variable and print their data types to verify the swapping operation

* Initialize 2 variables my_int = 5 , and my_str = "Hi"
* Swap these variables such that my_int now holds the value of my_str and vice versa
* Verify this by printing their data types

_Points:_ 5
"""

my_int = ...
my_str = ...

grader.check("q1")

"""### 2. Write a Python Program to Calculate the length of the diagonal of a rectangle

* Initialize the 2 sides of the rectangle a and b to 5 and 6
* Calculate the length of the diagonal using the formula - diagonal = √(a^2 + b^2)

_Points:_ 5
"""

def calc_diag(a, b):
    ...

grader.check("q2")

"""### 3. Write a Python Program to Check if a Number is divisible by 3

* Initialize a variable that takes input from user
* Check and return True or False if the number is divisible by 3
* If the input is not an int, return False

_Points:_ 10
"""

def is_divisible_by_3(num):
    ...

grader.check("q3")

"""<!-- BEGIN QUESTION -->

### 4. Write a Python Program to Check if a number has exactly 3 digits (no zeros allowed at the beginning and the number can be assumed to be positive)

* Initialize a variable that takes input from user(input will be given as a string)
* Check and print true if the number has exactly 3 digits(no zeros allowed at the beginning) else print false

_Points:_ 10
"""

num = ...

"""<!-- END QUESTION -->

### 5. Write a Python Program to Check Whether a String is Palindrome or Not

* Initialize a string
* Lower case it to allow easy comparison
* Check using if-else statements if the string is a palindrome
* A palindrome is a string which is same read forward or backwards.

_Points:_ 10
"""

def if_palindrome(x):
    ...

grader.check("q5")

"""### 6.  **Print numbers divisible by 5 between 4 and 30, inclusive**

* Iterate through the numbers in the for loop and notice the way range has been used to generate numbers in the for loop
* Produce a list and print containing only the numbers divisible by 5 using if statement

_Points:_ 5
"""

sol = ...

grader.check("q6")

"""### 7. Write a Python Program to print the first n numbers of the Fibonacci series starting from 0

* Use a loop and return a list of first n numbers of the Fibonacci series: 0,1,1,2,3,....


_Points:_ 15
"""

def fibonacci(n):
    ...

grader.check("q7")

"""### 8. Write a Python Program that defines a function to Remove Vowels From a String

* You can initialize a string of your choice
* Define the function
* Use a for loop to return an updated string with the vowels removed.

_Points:_ 10
"""

def remove_vowels(input_str):
    ...

grader.check("q8")

"""### 9. Write a program to define a function that takes two numbers as arguments and returns the sum of all numbers between the arguments passed (exclusive, i.e. excluding the limits).

* You can define a range of your choice
* Define the function
* Use a for loop to add numbers within the range(exclusive)

_Points:_ 10
"""

def sum(a, b):
    ...

grader.check("q9")

"""<!-- BEGIN QUESTION -->

### 10. Define your own Python function!

#### Part A
In Python, functions are blocks of code that only run when they are called. Functions are programmed to complete a specific task. You can input data, called parameters, into functions. Functions can also return data as a result.

Write a function with the following components:

* Pass a list as the function parameter. The list can be composed of any data type (int, string, float, etc.)
* At least one conditional (if/else/elif)
* One loop (ex. for, while, do-while)
* Returns one or more values. This value can be a list or a single element.


Ensure the following:
* Your program contains all the components listed above.
* You test the function by calling it outside of the function.
* Your program runs without any errors.

_Points:_ 10
"""



"""#### Part B
Record a video, no more than 3 minutes long.

* Explain the function at a high level. We want you to explain the flow of code, including the order in which the lines of code are executed.
* Explain at least two of the required components. For example, you could describe your conditional statement and what your loop does.

You can reference the example video here: https://drive.google.com/file/d/1LragCuaHO_eVMm1EDfFv7wqQcrC86NuW/view?usp=sharing

Upload the recorded video to your drive and paste the shareable link in the cell below.

Note: Ensure that we are able to view the video.

_Points:_ 10
"""

#

"""<!-- END QUESTION -->

### Submission

Before you submit your notebook, ensure that you've run all the cells sequentially to display images, graphs, and outputs correctly (if the outputs are missing we might not be able to award points for those parts). Take the time to review your solutions and evaluate using the public test cases. Once you're satisfied, save your notebook by navigating to "File" in the Colab Notebook menu and selecting "Save" or using the keyboard shortcut (usually Ctrl + S or Cmd + S).

To submit, export your notebook as an .ipynb file. Ensure it's saved with the same name as the assignment. Then, upload this .ipynb file to Gradescope. Remember to submit the correct version.

Please save your work before exporting it, and if you encounter any technical issues or have questions about the submission process, reach out to the course staff for assistance.

---

To double-check your work for questions 1-3 and 5-9, the cell below will rerun all of the autograder tests.
"""

grader.check_all()