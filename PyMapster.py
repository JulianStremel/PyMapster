import os
import json
import struct
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button,Slider

def list_to_slices(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def is_ascending(lst):
    return all(lst[i] < lst[i+1] for i in range(len(lst)-1))

def is_descending(lst):
    return all(lst[i] > lst[i+1] for i in range(len(lst)-1))


# render an image of the table in the console by printing the values in the table in a grid format as their integer 
def render_table(offset, tablesize, byte, bytes_read):
    # Helper function to split list into slices
    table_string = f"Table at offset: {offset}\n"

    def list_to_slices(lst, size):
        return [lst[i:i+size] for i in range(0, len(lst), size)]

    # Helper function to draw a single row of values with padding and borders
    def draw_row(values, col_width):
        nonlocal table_string
        # Top border for each square
        top_border = "+" + "+".join(["-" * col_width for _ in values]) + "+"
        table_string += top_border + "\n"

        # Center the values inside each square
        row = "|"
        for val in values:
            val_str = str(val).center(col_width)
            row += val_str + "|"
        table_string += row + "\n"

    # Bottom border
    def draw_bottom(col_count, col_width):
        nonlocal table_string
        bottom_border = "+" + "+".join(["-" * col_width for _ in range(col_count)]) + "+"
        table_string += bottom_border + "\n"

    # Start rendering the table
    col_width = 12  # You can adjust this based on the width of numbers you expect
    for i in range(tablesize):
        # Convert the values from bytes
        values = [int.from_bytes(b''.join(i), "little") for i in list_to_slices(bytes_read[offset+(i*tablesize):offset+((i+byte)*tablesize)], byte)]
        draw_row(values, col_width)
    draw_bottom(tablesize, col_width)
    return table_string  # Return the entire table as a string

# Function to convert bytes into a pandas DataFrame and visualize it as a 3D table
def render_3d_table(offset, tablesize, byte, bytes_read):
    # Helper function to split list into slices
    def list_to_slices(lst, size):
        return [lst[i:i + size] for i in range(0, len(lst), size)]

    # Extract the table data
    table_data = []
    for i in range(tablesize):
        values = [int.from_bytes(b''.join(i), "little") for i in list_to_slices(bytes_read[offset + (i * tablesize):offset + ((i + byte) * tablesize)], byte)]
        table_data.append(values)

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(table_data)

    # Display the DataFrame (2D table)
    print(df)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Get the dimensions of the table
    x = np.arange(df.shape[0])
    y = np.arange(df.shape[1])
    X, Y = np.meshgrid(x, y)

    # Flatten the DataFrame to get Z values for plotting
    Z = df.values.flatten()

    # Plot each point as a bar
    ax.bar3d(X.flatten(), Y.flatten(), np.zeros_like(Z), 1, 1, Z, shade=True)

    # Set labels and title
    ax.set_xlabel('X axis (Rows)')
    ax.set_ylabel('Y axis (Columns)')
    ax.set_zlabel('Values')
    ax.set_title('3D Representation of Table')

    plt.show()

# Function to convert bytes into a pandas DataFrame and visualize it as a 3D plane
def render_3d_plane(offset, tablesize, byte, bytes_read):
    tb_x = tablesize
    tb_y = tablesize
    byte = byte
    stepsize = 1
    mode = "int"

    # Helper function to split list into slices
    def list_to_slices(lst, size):
        return [lst[i:i + size] for i in range(0, len(lst), size)]

    # Extract the table data
    def extract_data(offset,tb_x,tb_y,byte):
        table_data = []
        for i in range(tb_x):
            match mode:
                case "int":
                    values = [int.from_bytes(b''.join(i), "little") for i in list_to_slices(bytes_read[offset + (i * tb_y):offset + ((i + byte) * tb_y)], byte)]
                case "float":
                    values = [struct.unpack("f",b''.join(i))[0] for i in list_to_slices(bytes_read[offset + (i * tb_y):offset + ((i + byte) * tb_y)], byte)]
            table_data.append(values)
        return pd.DataFrame(table_data)

    # Create a figure for the 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Function to update the plot based on the offset
    def update_plot(offset,tb_x,tb_y,byte):
        ax.clear()
        df = extract_data(offset,tb_x,tb_y,byte)

        # Get the X, Y dimensions of the table
        x = np.arange(df.shape[0])
        y = np.arange(df.shape[1])
        X, Y = np.meshgrid(y, x)

        # The Z values will be the actual data from the DataFrame
        Z = df.values

        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

        # Set labels and title
        ax.set_xlabel('X axis (Columns)')
        ax.set_ylabel('Y axis (Rows)')
        ax.set_zlabel('Values')
        ax.set_title(f"OFFSET: {offset} | TABLE SIZE: {tb_x}x{tb_y} | DATA SIZE: {byte*8} bits")

        plt.draw()

    # Initial plot rendering
    update_plot(offset,tb_x,tb_y,byte)

    # Add button functionality to adjust the offset
    def next_offset(event):
        nonlocal offset
        offset += stepsize  # Change this to adjust the step size
        update_plot(offset,tb_x,tb_y,byte)

    def previous_offset(event):
        nonlocal offset
        offset = max(0, offset - stepsize)  # Prevent negative offsets
        update_plot(offset,tb_x,tb_y,byte)

    def incerase_x(event):
        nonlocal tb_x
        tb_x += 1
        update_plot(offset,tb_x,tb_y,byte)
    
    def decrease_x(event):
        nonlocal tb_x
        tb_x -= 1
        update_plot(offset,tb_x,tb_y,byte)
    
    def incerase_y(event):
        nonlocal tb_y
        tb_y += 1
        update_plot(offset,tb_x,tb_y,byte)
    
    def decrease_y(event):
        nonlocal tb_y
        tb_y -= 1
        update_plot(offset,tb_x,tb_y,byte)
    
    def set_8bit(event):
        nonlocal byte
        byte = 1
        update_plot(offset,tb_x,tb_y,byte)
    
    def set_16bit(event):
        nonlocal byte
        byte = 2
        update_plot(offset,tb_x,tb_y,byte)
    
    def set_32bit(event):
        nonlocal byte
        byte = 4
        update_plot(offset,tb_x,tb_y,byte)

    def set_offset(event):
        nonlocal offset
        offset = int(event)
        update_plot(offset,tb_x,tb_y,byte)

    def set_8x8(event):
        nonlocal tb_x,tb_y
        tb_x = 8
        tb_y = 8
        update_plot(offset,tb_x,tb_y,byte)
    
    def set_16x16(event):
        nonlocal tb_x,tb_y
        tb_x = 16
        tb_y = 16
        update_plot(offset,tb_x,tb_y,byte)
    
    def set_32x32(event):
        nonlocal tb_x,tb_y
        tb_x = 32
        tb_y = 32
        update_plot(offset,tb_x,tb_y,byte)

    def set_32x100(event):
        nonlocal tb_x,tb_y
        tb_x = 32
        tb_y = 100
        update_plot(offset,tb_x,tb_y,byte)

    def set_100x1000(event):
        nonlocal tb_x,tb_y
        tb_x = 100
        tb_y = 1000
        update_plot(offset,tb_x,tb_y,byte)

    def set_int(event):
        nonlocal mode
        mode = "int"
        update_plot(offset,tb_x,tb_y,byte)
    
    def set_float(event):
        nonlocal mode
        mode = "float"
        update_plot(offset,tb_x,tb_y,byte)

    def set_stepsize(event):
        nonlocal stepsize
        stepsize = int(event)
        update_plot(offset,tb_x,tb_y,byte)

    # Add buttons for navigation
    axprev = plt.axes([0.1, 0.02, 0.1, 0.05])
    axnext = plt.axes([0.21, 0.02, 0.1, 0.05])

    # Add buttons for X-axis control
    axinc_x = plt.axes([0.1, 0.1, 0.1, 0.05])
    axdec_x = plt.axes([0.21, 0.1, 0.1, 0.05])

    # Add buttons for Y-axis control
    axinc_y = plt.axes([0.1, 0.18, 0.1, 0.05])
    axdec_y = plt.axes([0.21, 0.18, 0.1, 0.05])

    # Add buttons for bit selection
    ax8bit = plt.axes([0.41, 0.02, 0.1, 0.05])
    ax16bit = plt.axes([0.52, 0.02, 0.1, 0.05])
    ax32bit = plt.axes([0.63, 0.02, 0.1, 0.05])

    # Add buttons for switching between int and float
    axint = plt.axes([0.74, 0.02, 0.1, 0.05])
    axfloat = plt.axes([0.85, 0.02, 0.1, 0.05])

    # Add buttons for table size selection
    ax8x8 = plt.axes([0.41, 0.1, 0.1, 0.05])
    ax16x16 = plt.axes([0.52, 0.1, 0.1, 0.05])
    ax32x32 = plt.axes([0.63, 0.1, 0.1, 0.05])
    ax32x100 = plt.axes([0.74, 0.1, 0.1, 0.05])
    ax100x1000 = plt.axes([0.85, 0.1, 0.1, 0.05])

    # Add slider for offset
    axoffset = plt.axes([0.1, 0.85, 0.7, 0.03])
    axstepsize = plt.axes([0.1, 0.25, 0.6, 0.03])

    # Create buttons and slider objects
    bnext = Button(axnext, 'Next')
    bprev = Button(axprev, 'Previous')
    binc_x = Button(axinc_x, 'Increase X')
    bdec_x = Button(axdec_x, 'Decrease X')
    binc_y = Button(axinc_y, 'Increase Y')
    bdec_y = Button(axdec_y, 'Decrease Y')
    b8bit = Button(ax8bit, '8 bit')
    b16bit = Button(ax16bit, '16 bit')
    b32bit = Button(ax32bit, '32 bit')
    b8x8 = Button(ax8x8, '8x8')
    b16x16 = Button(ax16x16, '16x16')
    b32x32 = Button(ax32x32, '32x32')
    b32x100 = Button(ax32x100, '32x100')
    b100x1000 = Button(ax100x1000, '100x1000')
    baxint = Button(axint, 'INT')
    baxfloat = Button(axfloat, 'FLOAT')

    soffset = Slider(axoffset, 'Offset', 0, len(bytes_read), valinit=offset, valstep=100)
    sstepsize = Slider(axstepsize, 'Step Size', 1, 100, valinit=stepsize, valstep=1)

    # Button and slider event connections
    bnext.on_clicked(next_offset)
    bprev.on_clicked(previous_offset)
    binc_x.on_clicked(incerase_x)
    bdec_x.on_clicked(decrease_x)
    binc_y.on_clicked(incerase_y)
    bdec_y.on_clicked(decrease_y)
    b8bit.on_clicked(set_8bit)
    b16bit.on_clicked(set_16bit)
    b32bit.on_clicked(set_32bit)
    b8x8.on_clicked(set_8x8)
    b16x16.on_clicked(set_16x16)
    b32x32.on_clicked(set_32x32)
    b32x100.on_clicked(set_32x100)
    b100x1000.on_clicked(set_100x1000)
    baxint.on_clicked(set_int)
    baxfloat.on_clicked(set_float)

    soffset.on_changed(set_offset)
    sstepsize.on_changed(set_stepsize)

    plt.show()

# Function to find tables in a binary file
def find_tables(file):
    with open(file, "rb") as infile:
        result = {"file":"",
                "8bit": {"ascending": [], "8by8": []},
                "16bit": {"ascending": [], "8by8": []},
                "32bit": {"ascending": [], "8by8": []}}
        tablesize = 16
        bytes_read = []
        index = -1
        size = os.path.getsize(file)

        result["file"] = file

        if os.path.exists("result.json"):
            try:
                with open("result.json", "r") as f1:
                    result = json.load(f1)
                    if result["file"] == file:
                        for _ in tqdm(range(size), desc="reading file", unit="byte"):
                            data = infile.read(1)
                            if not data:
                                break
                            bytes_read.append(data)
                        render_3d_plane(result["32bit"]["8by8"][0], 8, 4, bytes_read)
                        return
            except Exception as e:
                pass

        for _ in tqdm(range(size), desc="reading file", unit="byte"):
            data = infile.read(1)
            if not data:
                break
            # read bytes into a list
            bytes_read.append(data)
            index += 1

            # 8 bit
            #if index >= (tablesize):
            #    # convert bytes to int
            #    tmp = [int.from_bytes(b''.join(i), "little") for i in list_to_slices(bytes_read[index-(tablesize):index], 1)]
            #    if is_ascending(tmp):
            #        result["8bit"]["ascending"].append(index)

            # 16 bit
            if index >= (2*tablesize):
                # convert bytes to int
                tmp = [int.from_bytes(b''.join(i), "little") for i in list_to_slices(bytes_read[index-(2*tablesize):index], 2)]
                if is_ascending(tmp):
                    result["16bit"]["ascending"].append(index)

            # 32 bit
            if index >= (4*tablesize):
                # convert bytes to int
                tmp = [int.from_bytes(b''.join(i), "little") for i in list_to_slices(bytes_read[index-(4*tablesize):index], 4)]
                if is_ascending(tmp):
                    result["32bit"]["ascending"].append(index)
            
        
            
        # searching for 8 by 8 tables in 8 bit
        #tmp = (sorted(result["8bit"]["ascending"]+result["8bit"]["descending"]))
        #res1 = []
        #f.write("8 bit tables\n")
        #for value in tqdm(tmp, desc="finding 8 by 8 tables (8bit)", unit="index"):
        ##   if value+1 in tmp:
        #       res1.append(value)
        #       f.write(render_table(value + 2, 8, 1, bytes_read))
        #result["8bit"]["8by8"] = res1

        # searching for 8 by 8 tables in 16 bit
        tmp = result["16bit"]["ascending"]
        res2 = []
        for value in tqdm(tmp, desc="finding 8 by 8 tables (16bit)", unit="index"):
            if value+2 in tmp:
                res2.append(value)
        result["16bit"]["8by8"] = res2

        # searching for 8 by 8 tables in 32 bit
        tmp = result["32bit"]["ascending"]
        res3 = []
        for value in tqdm(tmp, desc="finding 8 by 8 tables (16bit)", unit="index"):
            if value<140000:
                continue
            if value+4 in tmp:
                res3.append(value)

        result["32bit"]["8by8"] = res3

        with open("result.json", "w") as fileout:
            json.dump(result, fileout)

        render_3d_plane(result["32bit"]["8by8"][0], 8, 4, bytes_read)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find tables in a binary file")
    parser.add_argument("file", type=str, help="The binary file to analyze")

    args = parser.parse_args()
    find_tables(args.file)