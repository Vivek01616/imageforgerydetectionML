import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter import filedialog
import tensorflow as tf
import numpy as np
from PIL import ImageChops, ImageEnhance
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
np.random.seed(2)
global model
model = tf.keras.models.load_model("model_casia_run1.h5")
from PIL.ExifTags import TAGS

def get_image_metadata(image_path):
    with Image.open(image_path) as img:
        metadata = img._getexif()
        if metadata:
            for tag_id in metadata:
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'Software':
                    return metadata[tag_id]
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image
image_size = (128, 128)
def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0
def upload_image():
    global filepath
    filepath = filedialog.askopenfilename()
    if not filepath.endswith(('.jpg', '.jpeg', '.png','.tif')):
        messagebox.showerror("Error", "Invalid Image format. Please upload a JPG, JPEG, or PNG image.")
        return
    image=convert_to_ela_image(filepath,90)
    image.save(r"C:\Users\naniv\Desktop\final_year_proj\test.jpg")
    messagebox.showinfo("Image Uploaded", "Image uploaded successfully")
def prediction():
    image_path = 'test.jpg'
    image = prepare_image(image_path)
    image = image.reshape(-1, 128, 128, 3)
    y_pred = model.predict(image)
    y_pred_class = np.argmax(y_pred, axis = 1)[0]
    return y_pred_class
    
def test_image():
    new_window = tk.Toplevel(root)
    new_window.title("Uploaded Image")
    get_value=prediction()
    if get_value == 0:
        label = tk.Label(new_window, text="The Uploaded Image is Tampered",font=("Arial", 25))
    else:
        label = tk.Label(new_window, text="The Uploaded Image is Normal image",font=("Arial", 25))
    label.pack()
    uploaded_image = Image.open(filepath)
    tk_image = ImageTk.PhotoImage(uploaded_image)
    image_label = tk.Label(new_window, image=tk_image)
    image_label.image = tk_image
    image_label.pack()
    ela_window = tk.Toplevel(root)
    ela_window.title("ELA Image")
    ela_image = convert_to_ela_image(filepath, 90)
    ela_tk_image = ImageTk.PhotoImage(ela_image)
    ela_label = tk.Label(ela_window, image=ela_tk_image)
    ela_label.image = ela_tk_image
    ela_label.pack()
    image_path = 'test.jpg'
    image = prepare_image(image_path)
    image = image.reshape(-1, 128, 128, 3)
    y_pred = model.predict(image)
    new_window = tk.Toplevel(root)
                
    new_window.title("Predicted Image Confidence")
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.pie([y_pred[0][0]*100, y_pred[0][1]*100], labels=['Tampered', 'Normal'], autopct='%1.1f%%')
    image_path = filepath
    image_format = image_path.split('.')[-1].lower()
    if image_format in ['jpg', 'jpeg', 'png', 'tiff']:
        software = get_image_metadata(image_path)
        if get_value == 0:
            if software:
                ax.set_title(f"Tampered by {software}")
            else:
                ax.set_title("Image confidence")
        else:
            ax.set_title("Image confidence")
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
def generate_model():
     model.summary()
def exit_app():
    root.destroy()

root = tk.Tk()
root.title("Image Loader")
root.configure(bg="sky blue")
root.geometry("600x500")
font = ('times', 16, 'bold')
heading = tk.Label(root, text='Fake Image Identification')
heading.config(bg='lavenderblush', fg='DarkOrchid1')
heading.config(font=font)
heading.config(height=3, width=120)
heading.place(x=100,y=5)
heading.pack()
font1 = ('times', 14, 'bold')
model_button = tk.Button(root, text="Generate image Train & Test Model Summary", font=font1, command=generate_model)
model_button.place(x=100,y=100) 
model_button.config(font=font1)
upload_button = tk.Button(root, text="Upload Test Image", font=font1, command=upload_image)
upload_button.place(x=100,y=150) 
upload_button.config(font=font1)
test_button = tk.Button(root, text="Classify Picture In Image", font=font1, command=test_image)
test_button.place(x=100,y=200)
test_button.config(font=font1)
exit_button = tk.Button(root, text="Exit", font=("Arial", 9), command=root.quit)
exit_button.place(x=100,y=250) 
exit_button.config(font=font1)


root.mainloop()