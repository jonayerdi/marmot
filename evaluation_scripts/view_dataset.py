from argparse import ArgumentParser
from os.path import split
import tkinter as tk

from PIL import ImageTk, Image  

from utils.dataset import Dataset

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 500
CANVAS_WIDTH = WINDOW_WIDTH
CANVAS_HEIGHT = WINDOW_HEIGHT - 20

canvas = [None]
text = [None]
step_slider = [None]
images = [None]
image = [None]
images_index = [0]

def set_image(filename):
    image[0] = ImageTk.PhotoImage(Image.open(filename))
    canvas[0].delete(tk.ALL)
    text[0].delete(1.0, 2.0)
    text[0].insert(1.0, split(filename)[1])
    #canvas[0].create_text(WINDOW_WIDTH/2,WINDOW_HEIGHT-20, text=split(filename)[1])
    canvas[0].create_image(
        0, 
        0, 
        anchor=tk.NW, 
        image=image[0],
    )

def step_change(index):
    index = int(index)
    images_index[0] = index
    step_slider[0].set(index)
    set_image(images[0][index])

def onclick_left():
    step_change((images_index[0] - 1) if images_index[0] > 0 else len(images[0]) - 1)

def onclick_right():
    step_change((images_index[0] + 1) % len(images[0]))

def view_dataset(data_dir):
    images[0] = Dataset(data_dir=data_dir, events_file=None).images
    print(f'Loaded {len(images[0])} images')

    window = tk.Tk()
    window.title(f'{data_dir}')
    window.geometry(f'{WINDOW_WIDTH}x{WINDOW_HEIGHT+47}')
    window.resizable(False, False)
    window.configure(background='#AAAAAA')

    canvas[0] = tk.Canvas(window, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
    canvas[0].configure(background='#DDDDDD')
    canvas[0].grid(row=0, column=0, columnspan=1)

    text[0] = tk.Text(window, height=1, width=78, borderwidth=0)
    text[0].grid(row=1, column=0, columnspan=1)
    #text[0].configure(state='disabled')
    #text[0].configure(inactiveselectbackground=text[0].cget("selectbackground"))

    step_slider[0] = tk.Scale(window, from_=0, to=len(images[0])-1, resolution=1, orient=tk.HORIZONTAL, length=WINDOW_WIDTH-150)
    step_slider[0].grid(row=2, column=0, columnspan=1)
    step_slider[0].configure(command=step_change)

    button_left = tk.Button(window, text="Prev", command=onclick_left)
    button_left.grid(row=2, column=0, sticky=tk.W)
    button_right = tk.Button(window, text="Next", command=onclick_right)
    button_right.grid(row=2, column=0, sticky=tk.E)

    set_image(images[0][images_index[0]])

    tk.mainloop()

def main(args):
    parser = ArgumentParser(description='Dataset viewer')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, required=True)
    params = parser.parse_args(args=args)
    view_dataset(
        data_dir=params.data_dir,
    )

if __name__ == '__main__':
    import sys
    main(args=sys.argv[1:])
