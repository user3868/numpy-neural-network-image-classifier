import os
import shutil
import tkinter as tk
from PIL import Image, ImageTk

class ImageLabelerApp:
    def __init__(self, root, source_folder, dest_folder, prefix, load_step=30):
        self.root = root
        self.source_folder = source_folder
        self.dest_folder = dest_folder
        self.prefix = prefix
        self.load_step = load_step
        self.image_files = [f for f in os.listdir(self.source_folder) if f.endswith('.png')]
        self.loaded_images_count = 0
        self.check_vars = []

        self.root.title('Image Labeler')
        self.create_widgets()
        self.load_images(initial=True)

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, borderwidth=0)
        self.frame = tk.Frame(self.canvas)
        self.vsb = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((4,4), window=self.frame, anchor="nw", tags="self.frame")

        self.frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        self.label_button = tk.Button(self.root, text='Label', command=self.label_images)
        self.label_button.pack()

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)

    def on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        if self.canvas.yview()[1] == 1.0:  # If scrolled to the bottom
            self.load_images()

    def load_images(self, initial=False):
        start = self.loaded_images_count
        end = start + self.load_step
        if initial:
            end = self.load_step

        for idx, filename in enumerate(self.image_files[start:end], start=start):
            image_path = os.path.join(self.source_folder, filename)
            img = Image.open(image_path)
            img.thumbnail((100, 100))  # Adjust thumbnail size
            photo = ImageTk.PhotoImage(img)

            chk_var = tk.IntVar()
            chk_button = tk.Checkbutton(self.frame, image=photo, variable=chk_var)
            chk_button.image = photo
            chk_button.grid(row=idx // 5, column=idx % 5)  # Adjust grid layout
            self.check_vars.append((chk_var, filename))

        self.loaded_images_count = end

    def label_images(self):
        if not os.path.exists(self.dest_folder):
            os.makedirs(self.dest_folder)

        for chk_var, filename in self.check_vars:
            if chk_var.get() == 1:
                new_name = f'{self.prefix}_{filename}'
                shutil.copy(os.path.join(self.source_folder, filename), os.path.join(self.dest_folder, new_name))

def main():
    source_folder = 'similarityImage'
    dest_folder = 'labelImage'
    prefix = '0'  # You can change the prefix as needed

    root = tk.Tk()
    app = ImageLabelerApp(root, source_folder, dest_folder, prefix)
    root.mainloop()

if __name__ == "__main__":
    main()