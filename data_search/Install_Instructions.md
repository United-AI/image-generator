# Data Searcher
---
Instructions on how to install and use the Data Searcher

## Installation Tutorial

### Step 1: Python libraries
Make sure that python 3.9 is installed on your Computer.
If it is not, install is from here:  
https://www.python.org/downloads/

In the 'data_search' folder, enter the following command:
```
pip install -r requirements.txt
```

### Step 2: Install Google Chrome and the corresponding Chrome Driver
You can download Chrome here:
https://www.google.com/chrome/

You can download the corresponding Driver here:
https://chromedriver.chromium.org/downloads
Put the Driver in a location that you know the path to.

This is how you find which version of Chrome you downloaded:
The three dots in the top right -> help -> about google chrome -> it states the version there

## Setup and usage

### The Setup file
The Setup.py file has 3 values that you need to enter.

self.DRIVER_PATH is the path to where you stored the chromedriver
self.amount_images is the number of images you want to download.
self.Search_Term is the Term you want to enter into google images to find the images you will download.

### Running the code
Once you have filled out the Setup.py file you can run the data_searcher.py file.
First you will have to wait a bit while it finds the images. You will then get a window that shows you an image and has a 'yes' and a 'no' button.
Pressing a button will save the image in the corresponding folder (the 'yes' or the 'no' folder). These folders are found in the 'images'/'your search term' folder.
You will be shown the amount of images you asked for and then the window will close and the program finish.
