# ML Bootcamp
##### An introductory bootcamp to Python, Machine Learning, and ScikitImage.

Hi! Welcome to ML bootcamp. I created this resource for all of you to get a quick introduction to Machine Learning and all the other tools/applications we'll be using! 

A quick few notes before I dive in. As we're working with Machine Learning this semester - more specifically, Computer Vision - we're going to be coding in Python. For reasons I won't list here, working in a high-level language like Python grants you alot more flexibility, making coding much easier! In addition, we're going to be relying on some frameworks that are industry-standard, and they're all in Python. All of this goes to say that Python is the way to go for anything machine learning! If you're not familiar with Python, no worries! It's much easier to know a low-level language like C++ and then go on to learn Python, so I have no reason to think that all of you won't be able to catch on fairly quickly! 

Tutorial 1: [Basics of Python](https://colab.research.google.com/drive/1fsSr3XKjdiorIWPp7b5QTQ06phcdG0Aa?usp=sharing)
https://colab.research.google.com/drive/1fsSr3XKjdiorIWPp7b5QTQ06phcdG0Aa?usp=sharing

Tutorial 2: [Numpy and Data Processing](https://colab.research.google.com/drive/1aaznBQvQMmOsxS0hFBEeGbvmWeR9whZf?usp=sharing)
https://colab.research.google.com/drive/1aaznBQvQMmOsxS0hFBEeGbvmWeR9whZf?usp=sharing

Tutorial 3: [OpenCV](https://colab.research.google.com/drive/1eekb4H00iNaHXi_6OAwyELcdi64wWaNm?usp=sharing)
https://colab.research.google.com/drive/1eekb4H00iNaHXi_6OAwyELcdi64wWaNm?usp=sharing


## Table of Contents  

* [Introduction to Machine Learning](#introduction-to-machine-learning)
  * [Machine Learning Overview](#what-is-machine-learning)
  * [Computer Vision](#computer-vision)
  * [Introduction to OpenCV](#opencv)
* [IDE Setup - Pycharm](#ide-setup---pycharm) 
  * [Installation](#installation) 
    * [Python](#python)
    * [Pycharm](#pycharm)
  * [Project setup](#project-setup)
  * [Using your IDE](#using-your-ide) 
* [Miscellaneous](#miscellaneous)<br/>
_Not done yet! Will fill in stuff as we go along!_
  * [Python](https://www.python.org/)
  * [Git Commands](https://confluence.atlassian.com/bitbucketserver/basic-git-commands-776639767.html)
  
# [Introduction to Machine Learning](#introML)
## What is Machine Learning?
The official definition: The use and development of computer systems that are able to learn and adapt without following explicit instructions, by using algorithms and statistical models to analyze and draw inferences from patterns in data.

Unofficially, the way I think about it is simply using data to predict future events/outcomes. For example, if you consistently wake up at 9:30 AM +- 30 minutes for a year, I'm going to predict that on January 1st of the next year, you're going to wake up at 9:30 AM. Bam, I just applied the very concepts that ML is based on. Machine Learning encompasses a very broad set of subfields ranging from Deep Learning to Computer Vision. For this project, we're going to focusing on Computer Vision and potentially some Natural Language Processing, so that's what I'm going to be focusing on here!

Let's talk about the basics of ML. Consider a 2d graph of data points that plot the grade of EECS 183 students against the average day that they start projects. We can then use an algorithm as a way of finding a function that maps to the data well (meaning the difference between each data point and the function's prediction is minimized). In basic cases, the function may look something like this. This is a basic case of [**linear regression**](https://www.ibm.com/topics/linear-regression#:~:text=Resources-,What%20is%20linear%20regression%3F,is%20called%20the%20independent%20variable.).

![image](https://user-images.githubusercontent.com/42425774/193395816-c1375cf2-7c29-4814-8b9f-0ac57dcf27a4.png)

_Of course, the data may also be mapped best with other more complex functions: quadratic, polynomial, log, exponential etc._

Or in some cases, you're looking at differentiating data. For example, university admissions. I want to predict whether or not someone gets into UMich based on their height. Let's say the blue data points represent people who didn't get in. Oh, sorry to all of the giants, but it seems like you have a greater chance of being rejected. We can then find a function to map a decision boundary (in this case, a [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)) to split the data so that if a data point comes in, we can predict the outcome (e.g. a 4 foot tall person applies - are they above the boundary line or below? If above, we'll predict that they get in!). This is called a [**logistic regression algorithm**](https://www.ibm.com/topics/logistic-regression), but don't worry too much about this. I'm just giving you some context for how machine learning can apply to all different situations.
![image](https://user-images.githubusercontent.com/42425774/193395949-37b12e0f-f1cc-4c9d-8df8-d40efb881723.png)

This was just a quick intro to the idea of what ML can do! If you're interested in pursuing ML in a more standard format (a.k.a courses), I would consider `Calculus I/II/(Potentially III)-> Discrete Math -> Linear Algebra -> Data Structures and Algorithms -> Machine Learning` to be a pretty standard course set. Like so many other technical disciplines, Machine Learning is very much based on math, specifically matrices and optimization. Thus, Lin Alg and Calculus are pretty crucial! The other courses - Discrete Math and DSA - are just general CS courses that you should take anyway...

## Computer Vision
Focusing in on Computer Vision (CV), it's the study of using computers to derive data/patterns out of visualizations (e.g. images, videos etc.) At a **very** basic level and focusing on how we'll be using CV, CV works by dividing an image into a grid. Conveniently, screens themselves are grids. Grids of pixels. CV algorithms then can analyze the colour of every pixel and develop patterns based on the change in these pixels as pictures changed. This image describes the process well...
![image](https://user-images.githubusercontent.com/42425774/193396391-7de26584-848a-41a6-b337-e62a00c38a32.png)

If I gave the computer hundreds of thousands of pictures of the same person in different situations, we could potentially train the model well enough to the point where it could detect whether or not a person is the person the model was trained to detect. What's a potential application of this? You probably guessed it, facial recognition.

An important distinction to make here... 
**Supervised Learning** - A computer is given data that it knows the answer too. For example, I gave the computer hundreds of thousands of pictures of ONLY Mark Schlissel. It now knows what Mark Schlissel looks like. Then I give it ONLY pictures of Mary Sue Coleman, then Santa Ono etc. Then, it uses the data it collected to predict who's who later.
**Unsupervised Learning** - A computer is given data without knowing the answer. I.e. I gave the computer pictures of all the past UMich presidents and didn't tell it who was who. It would have to determine patterns in the dataset to be able to tell which pictures represented the same people.

Unsupervised learning is generally a harder way to approach things. Luckily, we'll be focusing on supervised learning.

## OpenCV
Everything above this point was a theoretical approach to understanding ML/CV. We're going to be focusing on application, which means it isn't _mandatory_ to understand the concepts behind them. We'll be using a library called [OpenCV](https://opencv.org/), an ML framework originally designed by Intel. It has built in functions for pretty much everything. It's amazing. We're just going to be bringing together all these functions that have been made for us to create a really cool end product.

### Setting up OpenCV on Pycharm.
In the top bar click on `Pycharm->Preferences`. Then, go into `Project: [Project Name] -> Python Interpreter`. From here, click on the little `+` at the top of the packages panel. A popup panel will show up displaying a list of all available libraries you can add to your project. You'll want to type in `opencv-python` into the top textbox. Then, click on the option that's labelled exactly that (`opencv-python`) and click install package. Now you'll have access to opencv functions!

### Autocomplete isn't working!
Pycharm's support for autocomplete for OpenCV is really bad. It just doesn't work. So to resolve this, here's a workaround that should work!
1. In PyCharm, go to `Pycharm -> Preferences -> Python Interpreter`.
2. Click on the gear symbol next to the interpreter path and select `SHOW ALL`.
3. Click on that icon that looks like a folder tree (on the top bar). It should be last icon (at least in my version of Pycharm it is).
4. Click on the "+" icon.
5. Navigate to <your_project_path>\venv\lib\<python3.10 (or whatever python version you have)\site-packages\cv2
6. Click `open` then `OK`. 
7. Autocomplete _should_ be working now!

# IDE Setup - Pycharm
This section is optional. Choosing an IDE is a personal choice; I just _recommend_ that you use Pycharm. Jetbrains seems to have top notch IDEs for pretty much any language and Pycharm is no exception. I think it's one of the more powerful IDEs out there, but if you prefer something else, feel free to go with that! If you have no preference or haven't coded in Python before, Pycharm could be a good place to start. 

_As a quick note, the following tutorial and screenshots are all on Mac. If you have a Windows, you should be able to follow along without any issues, but just note that things may look a little different. But as usual, if you encounter any issues that you can't fix, feel free to reach out!_

## Installation
To start, you'll want to install [Python](https://www.python.org/downloads/). At the date of creation of this tutorial, the latest stable release of Python is 3.10.7. If you already have Python installed, I would recommend upgrading to 3.10.7 if you don't already have it. If you don't, it's not a HUGE deal, but 3.10.7 supports adding data type labels to parameters, return types etc. that make code more readable and easier to understand. 

One **GREAT** thing about Python from a setup standpoint is that you don't have to go through the process of setting up a compiler. Python's interpreter combines the interpreter and compiler, meaning that there's much less setup needed than a language like C++. The disadvantage of this is that in comparison to other languages, Python is often **significantly slower**. If you've ever compared runtimes between C++ code and Python code, C++ is almost always significantly faster.


_As a side note, if you're interested in learning more about Python as a general language, they have an [FAQ section](https://docs.python.org/3/faq/general.html) that has some interesting info!_

### Python
1. Go through the installation module and just sticking with all the default options is fine!
2. Once you're done that, either navigate to the installation directory or if you're on Mac, it should just pop up in finder. It should look something like the image below...

![image](https://user-images.githubusercontent.com/42425774/193393173-a291a535-d5b8-4c74-bb8f-ec8f41333263.png)

3. **IMPORTANT** At this point, you'll want to open up `Python Launcher`. Once you do that, the following panel will pop up...
![image](https://user-images.githubusercontent.com/42425774/193393233-d7ffe1c1-a200-4fb4-a6c7-4e7c5ffd9a62.png)

4.[REMEMBER THE INTEPRETER PATH. In the image, that's the highlighted path `/usr/local/bin/python3`. This path will be important later... But at this point, you don't need to do anything other than remembering that path! You can go ahead and close everything up](#rememberStep).

### Pycharm
1. Diving into [Pycharm](https://www.jetbrains.com/pycharm/), they have two versions of the IDE: Community and Professional. For 99% of us, the community version is enough! It's the free version that's slightly less powerful. The professional version costs around $250 to use for a year. BUT, as students, we get it for free! So if you're interested in either exploiting UMich's resources or having an unecessarily more powerful IDE, feel free to go with the Professional version.
2. **If you decide to go with the Professional version, read this step.** **If not, skip this step and go onto step 3.** You'll have to [apply](https://www.jetbrains.com/shop/eform/students) for an education license from Jetbrains if you're interested in the Professional version. All you have to do is navigate to that link and enter your university email address among other information. Once you submit, you should receive an email from Jetbrains verifying your identity and then prompting you to make an account. Just follow the steps it gives and create an account. After this is done, keep in mind that you'll have to renew the licenese every year, but that's like a 30 second process. If I remember correctly, all it prompts you to do is login to your account lol.  
2. Go ahead and click on download for whatever version you prefer installing. Just follow the installation module. This step shouldn't be too hard. After it installs, the welcome screen should look similar to this: 

![image](https://user-images.githubusercontent.com/42425774/193393601-ab31106a-4829-4b0f-be7d-3a9b7cbe699a.png)

## Project Setup
1. Once you have the welcome screen open, you'll want to click on `New Project`. Something like the following should pop up. Stick with the `Pure Python` option.
![image](https://user-images.githubusercontent.com/42425774/193393671-d7707e7e-86c3-4d5e-8e96-397416ff8234.png)
2. You can change the project location (the first bar at the top of the panel) to wherever is convenient. _Pay attention to the next few steps! They're important to get right and if you mess up, it can be time-consuming and annoying to fix!
3. You'll want to focus on this section of the panel. The first two options - `New environment using` and `Location` shouldn't need to change. The third box: `Base interpreter` will require some work. Click on the 3 dots beside this box.
![image](https://user-images.githubusercontent.com/42425774/193393710-7b39ab59-5e14-4ad0-a2a9-9e655a4823a9.png)
4. Remember when I asked that you remember the path of your Python interepreter? You'll want to navigate there. For example, for me, it was `/usr/local/bin/python3`. Once you navigate there, you'll want to select the Python exectuable titled your python version. For example `Python 3.10`. Select that as your interpreter.
![image](https://user-images.githubusercontent.com/42425774/193394172-6e9ffd3f-69a8-4494-8ca5-921201d8e57d.png)
5. Go ahead and click create and you should be all good to go. It should look something like this! Click the big green run button in the top right to run the given script `main.py`. Hopefully, it should output `Hi, Pycharm.`. If it does, you've setup everything correctly! If not, let me know and we can work through any issues!
![image](https://user-images.githubusercontent.com/42425774/193394305-0b084442-f2d7-45a9-b009-81936502ebf8.png)

## [Using your IDE](#pycharmUse)
I'll leave a bunch of random tips here so that you can get a sense of how Pycharm works! 

### Creating a file/directory
1. Right click on the directory you want to create the file or subdirectory in on the left sidebar navigator. This menu will pop up. Go ahead and click `New -> file/directory`. If you want to create a **Python file** navigate down to `Python file`. For a text file, json file etc. just click the general `File`. And of course, if you want to create a directory, just click `Directory`. _As a general side note, if you ever want to manipulate a file/directory (e.g. rename, copy and paste it somewhere else, etc.), just right click on the object on the **left sidebar navigator** and you'll have access to these options!_
![image](https://user-images.githubusercontent.com/42425774/193394344-ad58ff7b-0aad-4ae1-9013-a63b98f7ca51.png)

### Running Code
**WARNING: STAY AWAY FROM THE BIG GREEN BUTTON IN THE TOP RIGHT CORNER AND THE RUN BUTTON IN THE CONSOLE**

That warning is a little dramatic. But the reason I warn you from using it is that it doesn't always run the file you want to run. It'll run whatever file is selected in the textbox to the left of the button. It won't automatically change based on the file your currently viewing, so if you're like me and always forget to change this box to reflect the file you want to run, there's an easier way!

Go ahead and right click anywhere in the coding area of the file you're currently in. A menu will pop up. Just click run from there! That way, you'll **ALWAYS** run the file you want to run.

![image](https://user-images.githubusercontent.com/42425774/193395009-871765df-00b6-4619-8d00-351c12c8fd13.png)

### Debugging Code
The process I recommend for running the debugging feature is the same as for running code. Right click in the coding panel of the file you want to debug, and just click debug! To setup a **breakpoint**, left click in between the line number and the code.
![image](https://user-images.githubusercontent.com/42425774/193395178-075df7ab-a065-4db0-b2b1-d447bdffd814.png)

### Refactoring
The power of refactoring is pretty crazy. Through the refactoring menu, you can add parameters to functions, change function signatures etc. I'll only write out how to use it to change variable names, but feel free to explore these other refactoring features too!  

One of the most useful tools is _refactoring_. Let's say you've declared a variable `a = 1`. Let's say you reference this variable about a thousand times in a file. Then you realize that you've broken EECS 183 style grading rules and that naming a variable `a` was a terrible idea. How do you change the variable name?

1. You'll want to highlight the variable name at one point where it's referenced. Right click on it, then follow `Refactor -> Rename`. Go ahead and type in a better variable name and magically you'll see all uses of the variable changing to match the new name!
![image](https://user-images.githubusercontent.com/42425774/193394662-c4384e64-9bde-45a5-bbf5-a3878c224604.png)

_This process also generally works for function names, file names, etc._

### Reformatting code
As an EECS 183 Grader, a part of me **dies** everytime I see code that isn't formatted AT ALL. It's just unreadable. Fortunately, Pycharm has a way of automatically formatting your code!

Navigate to the top bar, under `Code -> Reformat Code`. Alternatively, memorize the keyboard shortcut! `Cmd + Option + L` for Macs.

