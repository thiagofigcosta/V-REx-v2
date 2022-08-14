from pynput import mouse
import json
import time
import sys

x_coords = []
y_coords = []

filename='graph_from_mouse.json'
verbose = True

def finish():
    global x_coords, y_coords, filename
    x_min, x_max = x_coords[:2]
    x_coords=x_coords[2:]
    for i in range(len(x_coords)):# min max norm
        x_coords[i] = max(x_coords[i], x_min)
        x_coords[i] = min(x_coords[i], x_max)
        x_coords[i] = float(x_coords[i]-x_min)/float(x_max-x_min)
    y_max,y_min = y_coords[:2]
    y_coords=y_coords[2:]
    for i in range(len(y_coords)): # min max norm
        y_coords[i] = max(y_coords[i], y_min)
        y_coords[i] = min(y_coords[i], y_max)
        y_coords[i] = 1-float(y_coords[i]-y_min)/float(y_max-y_min)
    
    output = {
        'x_coords':x_coords,
        'y_coords':y_coords,
    }
    output_json = json.dumps(output, indent=4)
    with open(filename, 'w') as f:
        f.write(output_json)
    sys.exit()

def on_click(x, y, button, pressed):
    global x_coords, y_coords, verbose
    if button == mouse.Button.left:
        if not pressed: # released
            if verbose:
                type_of_capture = 'Graph point Capture'
                if len(x_coords)==0:
                    type_of_capture = 'Left Bottom Corner (0) Capture'
                elif len(x_coords)==1:
                    type_of_capture = 'Right Top Corner (1) Capture'
                print(f'{type_of_capture} at: x: {x}, y: {y}')
            x_coords.append(x)
            y_coords.append(y)
    elif button == mouse.Button.right:
        if verbose:
            print(f'Right click, finishing Capture...')
        finish()
            

print('Click on the left bottom corner of the graph, then')
print('click on the right top corner of the graph, then')
print('click on the graph points, finally, to finish,')
print('click with the right button')
print()
print()

print('Sleeping for 2 seconds...')
time.sleep(2)
print('Starting capture...')
listener = mouse.Listener(on_click=on_click)
listener.start()
listener.join()