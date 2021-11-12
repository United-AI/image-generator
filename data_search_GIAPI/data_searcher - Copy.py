import time
import tkinter as tk
from Setup import Setup
from google_images_search import GoogleImagesSearch
from io import BytesIO
from PIL import Image
import hashlib

def saveyes():
    global yesorno
    global yesornofolder
    yesorno = True
    yesornofolder = "yes"

def saveno():
    global yesorno
    global yesornofolder
    yesorno = True
    yesornofolder = "no"

root = tk.Tk()
canvas = tk.Canvas(root, width=300,height= 300, bg='#565656')
button1 = tk.Button(root, text = "yes", command = saveyes, anchor = "w")
button1.configure(width = 10, activebackground = "#4EF943", relief = 'raised')
button1_window = canvas.create_window(75, 300, anchor='s', window=button1)
button2 = tk.Button(root, text = "no", command = saveno, anchor = "w")
button2.configure(width = 10, activebackground = "#F94343", relief = 'raised')
button2_window = canvas.create_window(225, 300, anchor='s', window=button2)
canvas.pack()
yesorno = False
yesornofolder = ""

gis = GoogleImagesSearch(Setup.Dev_API_key, Setup.prject_cx)

def get10(query,folderpath='./images',num=10):
    gis.search({'q': query, 'num': num})
    for raw_image in gis.results():
        # here we tell the BytesIO object to go back to address 0
        my_bytes_io.seek(0)
        # take raw image data
        raw_image_data = raw_image.get_raw_data()
        # this function writes the raw image data to the object
        image.copy_to(my_bytes_io, raw_image_data)
        # or without the raw data which will be automatically taken
        # inside the copy_to() method
        image.copy_to(my_bytes_io)
        # we go back to address 0 again so PIL can read it from start to finish
        my_bytes_io.seek(0)
        # create a temporary image object
        temp_img = Image.open(my_bytes_io)
        image = Image.open(image_file).convert('RGBA')
        while not yesorno:
            img = image.resize((200, 200))
            tkimage = ImageTk.PhotoImage(img)
            canvas.create_image(50,0, anchor='nw', image=tkimage)
            canvas.update()
        yesorno = False
        raw_image.resize(500, 500)
        raw_image.download(folder_path+"/"+yesornofolder)

start = time.time()
get10(Setup.Search_Term,Setup.DRIVER_PATH)
print(time.time()-start)
