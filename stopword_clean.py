import os
import tkinter
from tkinter import filedialog
from tkinter import messagebox
from clean_text import clean_text

print("Select the input folder containing the text files.")
root = tkinter.Tk()
root.withdraw()
input_folder = filedialog.askdirectory(title="Select Input Folder")
print("Select the output folder.")
output_folder = "cleaned_data"
root.destroy()
if not input_folder or not output_folder:
    print("Input or output folder invalid.")
    exit()


os.makedirs(output_folder, exist_ok=True)
if os.listdir(output_folder):
    messagebox.showerror("Folder Not Empty",f"The folder '{output_folder}' is not empty.\nPlease select an empty folder.")
    exit()



for root_dir, sub_dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.endswith('.txt'):
            input_path = os.path.join(root_dir, filename)
            rel_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()

            cleaned_text = clean_text(text)
            print(cleaned_text + '\n')

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

