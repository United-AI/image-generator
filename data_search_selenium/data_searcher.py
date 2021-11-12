from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import selenium
import requests
from PIL import Image, ImageTk
import time
import os
import io
import hashlib
import tkinter as tk
from Setup import Setup
# This is the path I use
# DRIVER_PATH = '.../Desktop/Scraping/chromedriver 2'
# Put the path for your ChromeDriver here
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')

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

def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep_between_interactions:int=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)    
    
    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)
        
        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        
        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls    
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src') and actual_image.get_attribute('src')[-4:] != '.svg':
                    image_urls.add(actual_image.get_attribute('src'))
            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(30)
            return
            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls

def persist_image(folder_path:str,url:str):
    global yesorno
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGBA')
        while not yesorno:
            img = image.resize((200, 200))
            tkimage = ImageTk.PhotoImage(img)
            canvas.create_image(50,0, anchor='nw', image=tkimage)
            canvas.update()
        yesorno = False
        file_path = os.path.join(folder_path+"/"+yesornofolder,hashlib.sha1(image_content).hexdigest()[:10] + '.png')
        with open(file_path, 'wb') as f:
            image.save(f, "PNG")
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")


def search_and_download(search_term:str,driver_path:str,target_path='./images',number_images=5):
    target_folder = os.path.join(target_path,'_'.join(search_term.lower().split(' ')))
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        os.makedirs(target_folder+"/yes")
        os.makedirs(target_folder+"/no")

    with webdriver.Chrome(executable_path=driver_path, chrome_options=options) as wd:
        res = fetch_image_urls(search_term, number_images, wd=wd, sleep_between_interactions=0.25)
        
    for elem in res:
        persist_image(target_folder,elem)
    root.destroy()

start = time.time()
search_and_download(Setup.Search_Term,Setup.DRIVER_PATH,number_images=Setup.amount_images
print(time.time()-start)
