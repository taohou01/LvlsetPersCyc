from collections import namedtuple
import mainModule
import filtration
import tkinter as tk
from tkinter import messagebox
from tkinter import PhotoImage

# This is for open-open (summer)
def runScript(file_details):
    ##################################################
    # Set any necessary flags here
    # mainModule.ply_flags['flag1'] = False
    ##################################################

    return mainModule.runMain(file_details.name,\
                file_details.direction, file_details.skip_vertices)

# This is for closed-open(winter 2024)
def closedOpenStep1(file_details):
    mesh = runScript(file_details)
    return filtration.run_closed_open_step1(file_details.name, mesh)

# This is for closed-open(summer 2024)
def closedOpenStep2(file_details, number_of_intervals):
    ##################################################
    # Set any necessary flags here
    # filtration.ply_flags['flag1'] = False
    # mainModule.ply_flags['flag1'] = True
    # mainModule.ply_flags['flag2'] = True
    # filtration.ply_flags['flag1'] = True
    # filtration.ply_flags['flag2'] = False
    # filtration.ply_flags['flag5'] = False
    # filtration.ply_flags['flag6'] = True
    filtration.enable_checks = True
    ##################################################

    mainModule.ply_flags['flag5']=False
    result = closedOpenStep1(file_details)
    mainModule.ply_flags['flag5']=True
    filtration.run_closed_open_step2(result, number_of_intervals)

def populate_mesh_files():
    fileslist = []
    skip_no_vertices = []

    # 'skip_vertices' is a list of vertices that explicitly specifies 
    # the indices of vertices that are not critical
    script_file = namedtuple('script_file', ['desc', 'name', 'direction', 'skip_vertices'])

    loop10 = script_file(desc="Manifold mother & child", \
                         name="670_loop10.off", \
                         direction=(3387, 3684), \
                         skip_vertices=[6872, 8627])
    fileslist.append(loop10)

    doublehelix = script_file(desc="Manifold double helix", \
                              name="24.off", \
                              direction=(317, 509), \
                              skip_vertices=skip_no_vertices)
    fileslist.append(doublehelix)

    botijo = script_file(desc="Botijo", \
                         name="Botijo.off", \
                         direction=(6405, 17727), \
                         skip_vertices=skip_no_vertices)
    fileslist.append(botijo)

    sixSeventyFE = script_file(desc="Non manifold mother & child", \
                               name="670_FE_final.off", \
                               direction=(101805, 165951), \
                               skip_vertices=skip_no_vertices)
    fileslist.append(sixSeventyFE)

    kitten = script_file(desc="Non manifold kitten", \
                         name="366_kitten_final.off", \
                         direction=(58826, 85439), \
                         skip_vertices=skip_no_vertices)
    fileslist.append(kitten)

    snake_loop = script_file(desc="Snake Loop", \
                             name="snake_loop.off", \
                             direction=(1544, 1199), \
                             skip_vertices=skip_no_vertices)
    fileslist.append(snake_loop)

    return fileslist

def build_GUI():
    mesh_files = populate_mesh_files()

    def on_submit():
        # Get the selected item from the listbox
        selected_items = choices_listbox.curselection()
        if selected_items:
            selected_index = selected_items[0]
            print(f'Executing the script on selected mesh file: {mesh_files[selected_index].name}')

            # Change the background color of the selected item
            choices_listbox.config(bg='black')  # Reset all items to default background
            choices_listbox.itemconfig(selected_index, {'bg': 'green'})

            number_of_intervals = int(entry.get())
            if not is_valid_positive_integer(number_of_intervals):
                number_of_intervals = 1
            
            selected_mode = radio_value.get()

            for i, var in enumerate(check_vars[0:6], start=0):
                mainModule.ply_flags[f"flag{i + 1}"] = var.get()
    
            for i, var in enumerate(check_vars[7:13], start=7):
                filtration.ply_flags[f"flag{i - 6}"] = var.get()

            ########################################################################
            msg = f"Processing the {mesh_files[selected_index].name} file\n\n\
                Selected mode: {selected_mode}"
            messagebox.showinfo("Input", msg)

            # CALL THE DESIRED FUNCTION HERE
            if selected_mode == "open-open":
                closedOpenStep1(mesh_files[selected_index])
            if selected_mode == "closed-open":
                closedOpenStep2(mesh_files[selected_index], number_of_intervals)
            ########################################################################
            
        else:
            print("No file selected.")

    def center_window(window, width, height):
        # Get the screen width and height
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        # Calculate the x and y coordinates for the Tk window
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)

        # Set the window's position and size
        window.geometry(f'{width}x{height}+{int(x)}+{int(y)}')

    class ToolTip:
        def __init__(self, widget, text):
            self.widget = widget
            self.text = text
            self.tooltip_window = None
            self.widget.bind("<Enter>", self.show_tooltip)
            self.widget.bind("<Leave>", self.hide_tooltip)

        def show_tooltip(self, event=None):
            # Create a tooltip window when hovering
            if self.tooltip_window or not self.text:
                return
            x, y, _, _ = self.widget.bbox("insert")
            x += self.widget.winfo_rootx() + 25
            y += self.widget.winfo_rooty() + 25

            self.tooltip_window = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)  # Remove window decorations
            tw.wm_geometry(f"+{x}+{y}")  # Position the tooltip

            label = tk.Label(tw, text=self.text, background="blue", relief="solid", borderwidth=1)
            label.pack()

        def hide_tooltip(self, event=None):
            # Destroy the tooltip window when no longer hovering
            tw = self.tooltip_window
            self.tooltip_window = None
            if tw:
                tw.destroy()
    
    # Create the main window
    root = tk.Tk()
    root.title("Optimal Levelset Persistent Cycles for Closed-Open Interval")

    label = tk.Label(root, text="SELECT THE MESH FILE TO RUN")
    label.pack(pady=3)

    # Set the desired window size
    window_width = 600
    window_height = 430

    # Center the window on the screen
    center_window(root, window_width, window_height)

    # List of choices
    choices = [cur_file.desc + "  ===   " + cur_file.name for cur_file in mesh_files]

    # Create a listbox to display the choices
    choices_listbox = tk.Listbox(root, height=len(choices), width=40, justify=tk.CENTER)
    choices_listbox.pack(pady=10)

    # Insert choices into the listbox
    for choice in choices:
        choices_listbox.insert(tk.END, choice)

    # Variable to store the selected radio button option
    radio_value = tk.StringVar()
    radio_value.set("closed-open")  # Set default selection

    # Create a frame to hold the radio buttons
    radio_frame = tk.Frame(root)
    radio_frame.pack(pady=10)

    # Add radio buttons inside the frame and place them side by side
    radio1 = tk.Radiobutton(radio_frame, text="Closed-Open", variable=radio_value, value="closed-open")
    radio1.pack(side=tk.LEFT, padx=10)  # Padding to add space between buttons

    radio2 = tk.Radiobutton(radio_frame, text="Open-Open", variable=radio_value, value="open-open")
    radio2.pack(side=tk.LEFT, padx=10)

    # Add a label to prompt the user
    label = tk.Label(root, text="# of closed open intervals to produce cycles:")
    label.pack(pady=10)

    # Add the entry (text field)
    entry = tk.Entry(root, width=30)
    entry.pack(pady=5)

    # Label for prompt
    label = tk.Label(root, text="Select output ply flags:")
    label.pack(pady=10)

    # Create a list to hold variables for each checkbox
    check_vars = [tk.BooleanVar() for _ in range(13)]

    # Create two frames for two rows of checkboxes
    checkbox_frame1 = tk.Frame(root)
    checkbox_frame1.pack(pady=5)

    checkbox_frame2 = tk.Frame(root)
    checkbox_frame2.pack(pady=5)

    # ToolTip text for each checkbox
    tooltips = [
        f"OO: Output triangles containing critical vertices in red",
        f"OO: Output triangles containing any critical values in red",
        f"OO: Output non-compatible triangles with blue color",
        f"OO: Output cross edges to a text file",
        f"OO: Output even and odd sets from open-open step 10",
        f"OO: Output sources and sinks for open-open",
        f"OO: Output final cross edges for open-open",
        f"CO: Output the simplicial complex after each forward or backward arrow",
        f"CO: Output k_1 and k_2 described in the Summer 2024 document",
        f"CO: Update 'index' in filtration.py to print any connected component",
        f"CO: Update 'index' in filtration.py to print the boundary of any connected component",
        f"CO: Output only the connected components that have the edge sigma_beta_minus_1",
        f"CO: Output the cross edges Z_j in Step 5 of Closed Open Step 2"
    ]

    # Add 7 checkboxes to the first row (checkbox_frame1)
    for i in range(7):
        cb = tk.Checkbutton(checkbox_frame1, text=f"flag {i + 1}", variable=check_vars[i])
        cb.pack(side=tk.LEFT, padx=5)
        ToolTip(cb, tooltips[i])  # Attach ToolTip to each checkbox

    # Add 6 checkboxes to the second row (checkbox_frame2)
    for i in range(7, 13):
        cb = tk.Checkbutton(checkbox_frame2, text=f"flag {i + 1}", variable=check_vars[i])
        cb.pack(side=tk.LEFT, padx=5)
        ToolTip(cb, tooltips[i])  # Attach ToolTip to each checkbox

    # Load an image for the info icon (You can replace this with a real icon file)
    # info_image = PhotoImage(width=16, height=16)  # Placeholder empty image for demonstration
    # Alternatively, you can load an actual image file like this:
    info_image = PhotoImage(file='infoicon3.png')

    # Create an info button with the image
    info_button = tk.Button(root, image=info_image)
    info_button.pack(side="right", anchor="ne", padx=2, pady=2)

    # Attach a ToolTip to the button
    msg = "This program was developed using \n" \
        " Python v3.11 \n" \
        " Numpy v1.26.4 \n"  \
        " PyMaxflow v1.3.0 \n" \
        " Gudhi Simplex Tree v3.10.1"
    ToolTip(info_button, msg)

    # Create a Submit button
    submit_button = tk.Button(root, text="CONTINUE", command=on_submit, width=20, height=5)
    submit_button.pack(pady=1)

    # Run the application
    root.mainloop()

def is_valid_positive_integer(s):
    try:
        num = int(s)
        return num > 0
    except ValueError:
        return False

if __name__ == "__main__":
    build_GUI()