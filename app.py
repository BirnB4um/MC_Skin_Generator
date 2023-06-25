
import os
import onnxruntime as ort
from PIL import Image
import random
from time import sleep
from minepi import Skin
import asyncio
from tkinter.filedialog import askopenfilename, asksaveasfilename
import threading
import numpy as np
import sys
import pygame
from keyboard import is_pressed as key_is_pressed
from time import time

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 2
session_options.inter_op_num_threads = 2

seed = int(time())
np.random.seed(seed)
random.seed(seed)

pixel_lookup_table = [(8,0), (9,0), (10,0), (11,0), (12,0), (13,0), (14,0), (15,0), (16,0), (17,0), (18,0), (19,0), (20,0), (21,0), (22,0), (23,0), (8,1), (9,1), (10,1), (11,1), (12,1), (13,1), (14,1), (15,1), (16,1), (17,1), (18,1), (19,1), (20,1), (21,1), (22,1), (23,1), (8,2), (9,2), (10,2), (11,2), (12,2), (13,2), (14,2), (15,2), (16,2), (17,2), (18,2), (19,2), (20,2), (21,2), (22,2), (23,2), (8,3), (9,3), (10,3), (11,3), (12,3), (13,3), (14,3), (15,3), (16,3), (17,3), (18,3), (19,3), (20,3), (21,3), (22,3), (23,3), (8,4), (9,4), (10,4), (11,4), (12,4), (13,4), (14,4), (15,4), (16,4), (17,4), (18,4), (19,4), (20,4), (21,4), (22,4), (23,4), (8,5), (9,5), (10,5), (11,5), (12,5), (13,5), (14,5), (15,5), (16,5), (17,5), (18,5), (19,5), (20,5), (21,5), (22,5), (23,5), (8,6), (9,6), (10,6), (11,6), (12,6), (13,6), (14,6), (15,6), (16,6), (17,6), (18,6), (19,6), (20,6), (21,6), (22,6), (23,6), (8,7), (9,7), (10,7), (11,7), (12,7), (13,7), (14,7), (15,7), (16,7), (17,7), (18,7), (19,7), (20,7), (21,7), (22,7), (23,7), (0,8), (1,8), (2,8), (3,8), (4,8), (5,8), (6,8), (7,8), (8,8), (9,8), (10,8), (11,8), (12,8), (13,8), (14,8), (15,8), (16,8), (17,8), (18,8), (19,8), (20,8), (21,8), (22,8), (23,8), (24,8), (25,8), (26,8), (27,8), (28,8), (29,8), (30,8), (31,8), (0,9), (1,9), (2,9), (3,9), (4,9), (5,9), (6,9), (7,9), (8,9), (9,9), (10,9), (11,9), (12,9), (13,9), (14,9), (15,9), (16,9), (17,9), (18,9), (19,9), (20,9), (21,9), (22,9), (23,9), (24,9), (25,9), (26,9), (27,9), (28,9), (29,9), (30,9), (31,9), (0,10), (1,10), (2,10), (3,10), (4,10), (5,10), (6,10), (7,10), (8,10), (9,10), (10,10), (11,10), (12,10), (13,10), (14,10), (15,10), (16,10), (17,10), (18,10), (19,10), (20,10), (21,10), (22,10), (23,10), (24,10), (25,10), (26,10), (27,10), (28,10), (29,10), (30,10), (31,10), (0,11), (1,11), (2,11), (3,11), (4,11), (5,11), (6,11), (7,11), (8,11), (9,11), (10,11), (11,11), (12,11), (13,11), (14,11), (15,11), (16,11), (17,11), (18,11), (19,11), (20,11), (21,11), (22,11), (23,11), (24,11), (25,11), (26,11), (27,11), (28,11), (29,11), (30,11), (31,11), (0,12), (1,12), (2,12), (3,12), (4,12), (5,12), (6,12), (7,12), (8,12), (9,12), (10,12), (11,12), (12,12), (13,12), (14,12), (15,12), (16,12), (17,12), (18,12), (19,12), (20,12), (21,12), (22,12), (23,12), (24,12), (25,12), (26,12), (27,12), (28,12), (29,12), (30,12), (31,12), (0,13), (1,13), (2,13), (3,13), (4,13), (5,13), (6,13), (7,13), (8,13), (9,13), (10,13), (11,13), (12,13), (13,13), (14,13), (15,13), (16,13), (17,13), (18,13), (19,13), (20,13), (21,13), (22,13), (23,13), (24,13), (25,13), (26,13), (27,13), (28,13), (29,13), (30,13), (31,13), (0,14), (1,14), (2,14), (3,14), (4,14), (5,14), (6,14), (7,14), (8,14), (9,14), (10,14), (11,14), (12,14), (13,14), (14,14), (15,14), (16,14), (17,14), (18,14), (19,14), (20,14), (21,14), (22,14), (23,14), (24,14), (25,14), (26,14), (27,14), (28,14), (29,14), (30,14), (31,14), (0,15), (1,15), (2,15), (3,15), (4,15), (5,15), (6,15), (7,15), (8,15), (9,15), (10,15), (11,15), (12,15), (13,15), (14,15), (15,15), (16,15), (17,15), (18,15), (19,15), (20,15), (21,15), (22,15), (23,15), (24,15), (25,15), (26,15), (27,15), (28,15), (29,15), (30,15), (31,15), (4,16), (5,16), (6,16), (7,16), (8,16), (9,16), (10,16), (11,16), (20,16), (21,16), (22,16), (23,16), (24,16), (25,16), (26,16), (27,16), (28,16), (29,16), (30,16), (31,16), (32,16), (33,16), (34,16), (35,16), (44,16), (45,16), (46,16), (47,16), (48,16), (49,16), (50,16), (51,16), (4,17), (5,17), (6,17), (7,17), (8,17), (9,17), (10,17), (11,17), (20,17), (21,17), (22,17), (23,17), (24,17), (25,17), (26,17), (27,17), (28,17), (29,17), (30,17), (31,17), (32,17), (33,17), (34,17), (35,17), (44,17), (45,17), (46,17), (47,17), (48,17), (49,17), (50,17), (51,17), (4,18), (5,18), (6,18), (7,18), (8,18), (9,18), (10,18), (11,18), (20,18), (21,18), (22,18), (23,18), (24,18), (25,18), (26,18), (27,18), (28,18), (29,18), (30,18), (31,18), (32,18), (33,18), (34,18), (35,18), (44,18), (45,18), (46,18), (47,18), (48,18), (49,18), (50,18), (51,18), (4,19), (5,19), (6,19), (7,19), (8,19), (9,19), (10,19), (11,19), (20,19), (21,19), (22,19), (23,19), (24,19), (25,19), (26,19), (27,19), (28,19), (29,19), (30,19), (31,19), (32,19), (33,19), (34,19), (35,19), (44,19), (45,19), (46,19), (47,19), (48,19), (49,19), (50,19), (51,19), (0,20), (1,20), (2,20), (3,20), (4,20), (5,20), (6,20), (7,20), (8,20), (9,20), (10,20), (11,20), (12,20), (13,20), (14,20), (15,20), (16,20), (17,20), (18,20), (19,20), (20,20), (21,20), (22,20), (23,20), (24,20), (25,20), (26,20), (27,20), (28,20), (29,20), (30,20), (31,20), (32,20), (33,20), (34,20), (35,20), (36,20), (37,20), (38,20), (39,20), (40,20), (41,20), (42,20), (43,20), (44,20), (45,20), (46,20), (47,20), (48,20), (49,20), (50,20), (51,20), (52,20), (53,20), (54,20), (55,20), (0,21), (1,21), (2,21), (3,21), (4,21), (5,21), (6,21), (7,21), (8,21), (9,21), (10,21), (11,21), (12,21), (13,21), (14,21), (15,21), (16,21), (17,21), (18,21), (19,21), (20,21), (21,21), (22,21), (23,21), (24,21), (25,21), (26,21), (27,21), (28,21), (29,21), (30,21), (31,21), (32,21), (33,21), (34,21), (35,21), (36,21), (37,21), (38,21), (39,21), (40,21), (41,21), (42,21), (43,21), (44,21), (45,21), (46,21), (47,21), (48,21), (49,21), (50,21), (51,21), (52,21), (53,21), (54,21), (55,21), (0,22), (1,22), (2,22), (3,22), (4,22), (5,22), (6,22), (7,22), (8,22), (9,22), (10,22), (11,22), (12,22), (13,22), (14,22), (15,22), (16,22), (17,22), (18,22), (19,22), (20,22), (21,22), (22,22), (23,22), (24,22), (25,22), (26,22), (27,22), (28,22), (29,22), (30,22), (31,22), (32,22), (33,22), (34,22), (35,22), (36,22), (37,22), (38,22), (39,22), (40,22), (41,22), (42,22), (43,22), (44,22), (45,22), (46,22), (47,22), (48,22), (49,22), (50,22), (51,22), (52,22), (53,22), (54,22), (55,22), (0,23), (1,23), (2,23), (3,23), (4,23), (5,23), (6,23), (7,23), (8,23), (9,23), (10,23), (11,23), (12,23), (13,23), (14,23), (15,23), (16,23), (17,23), (18,23), (19,23), (20,23), (21,23), (22,23), (23,23), (24,23), (25,23), (26,23), (27,23), (28,23), (29,23), (30,23), (31,23), (32,23), (33,23), (34,23), (35,23), (36,23), (37,23), (38,23), (39,23), (40,23), (41,23), (42,23), (43,23), (44,23), (45,23), (46,23), (47,23), (48,23), (49,23), (50,23), (51,23), (52,23), (53,23), (54,23), (55,23), (0,24), (1,24), (2,24), (3,24), (4,24), (5,24), (6,24), (7,24), (8,24), (9,24), (10,24), (11,24), (12,24), (13,24), (14,24), (15,24), (16,24), (17,24), (18,24), (19,24), (20,24), (21,24), (22,24), (23,24), (24,24), (25,24), (26,24), (27,24), (28,24), (29,24), (30,24), (31,24), (32,24), (33,24), (34,24), (35,24), (36,24), (37,24), (38,24), (39,24), (40,24), (41,24), (42,24), (43,24), (44,24), (45,24), (46,24), (47,24), (48,24), (49,24), (50,24), (51,24), (52,24), (53,24), (54,24), (55,24), (0,25), (1,25), (2,25), (3,25), (4,25), (5,25), (6,25), (7,25), (8,25), (9,25), (10,25), (11,25), (12,25), (13,25), (14,25), (15,25), (16,25), (17,25), (18,25), (19,25), (20,25), (21,25), (22,25), (23,25), (24,25), (25,25), (26,25), (27,25), (28,25), (29,25), (30,25), (31,25), (32,25), (33,25), (34,25), (35,25), (36,25), (37,25), (38,25), (39,25), (40,25), (41,25), (42,25), (43,25), (44,25), (45,25), (46,25), (47,25), (48,25), (49,25), (50,25), (51,25), (52,25), (53,25), (54,25), (55,25), (0,26), (1,26), (2,26), (3,26), (4,26), (5,26), (6,26), (7,26), (8,26), (9,26), (10,26), (11,26), (12,26), (13,26), (14,26), (15,26), (16,26), (17,26), (18,26), (19,26), (20,26), (21,26), (22,26), (23,26), (24,26), (25,26), (26,26), (27,26), (28,26), (29,26), (30,26), (31,26), (32,26), (33,26), (34,26), (35,26), (36,26), (37,26), (38,26), (39,26), (40,26), (41,26), (42,26), (43,26), (44,26), (45,26), (46,26), (47,26), (48,26), (49,26), (50,26), (51,26), (52,26), (53,26), (54,26), (55,26), (0,27), (1,27), (2,27), (3,27), (4,27), (5,27), (6,27), (7,27), (8,27), (9,27), (10,27), (11,27), (12,27), (13,27), (14,27), (15,27), (16,27), (17,27), (18,27), (19,27), (20,27), (21,27), (22,27), (23,27), (24,27), (25,27), (26,27), (27,27), (28,27), (29,27), (30,27), (31,27), (32,27), (33,27), (34,27), (35,27), (36,27), (37,27), (38,27), (39,27), (40,27), (41,27), (42,27), (43,27), (44,27), (45,27), (46,27), (47,27), (48,27), (49,27), (50,27), (51,27), (52,27), (53,27), (54,27), (55,27), (0,28), (1,28), (2,28), (3,28), (4,28), (5,28), (6,28), (7,28), (8,28), (9,28), (10,28), (11,28), (12,28), (13,28), (14,28), (15,28), (16,28), (17,28), (18,28), (19,28), (20,28), (21,28), (22,28), (23,28), (24,28), (25,28), (26,28), (27,28), (28,28), (29,28), (30,28), (31,28), (32,28), (33,28), (34,28), (35,28), (36,28), (37,28), (38,28), (39,28), (40,28), (41,28), (42,28), (43,28), (44,28), (45,28), (46,28), (47,28), (48,28), (49,28), (50,28), (51,28), (52,28), (53,28), (54,28), (55,28), (0,29), (1,29), (2,29), (3,29), (4,29), (5,29), (6,29), (7,29), (8,29), (9,29), (10,29), (11,29), (12,29), (13,29), (14,29), (15,29), (16,29), (17,29), (18,29), (19,29), (20,29), (21,29), (22,29), (23,29), (24,29), (25,29), (26,29), (27,29), (28,29), (29,29), (30,29), (31,29), (32,29), (33,29), (34,29), (35,29), (36,29), (37,29), (38,29), (39,29), (40,29), (41,29), (42,29), (43,29), (44,29), (45,29), (46,29), (47,29), (48,29), (49,29), (50,29), (51,29), (52,29), (53,29), (54,29), (55,29), (0,30), (1,30), (2,30), (3,30), (4,30), (5,30), (6,30), (7,30), (8,30), (9,30), (10,30), (11,30), (12,30), (13,30), (14,30), (15,30), (16,30), (17,30), (18,30), (19,30), (20,30), (21,30), (22,30), (23,30), (24,30), (25,30), (26,30), (27,30), (28,30), (29,30), (30,30), (31,30), (32,30), (33,30), (34,30), (35,30), (36,30), (37,30), (38,30), (39,30), (40,30), (41,30), (42,30), (43,30), (44,30), (45,30), (46,30), (47,30), (48,30), (49,30), (50,30), (51,30), (52,30), (53,30), (54,30), (55,30), (0,31), (1,31), (2,31), (3,31), (4,31), (5,31), (6,31), (7,31), (8,31), (9,31), (10,31), (11,31), (12,31), (13,31), (14,31), (15,31), (16,31), (17,31), (18,31), (19,31), (20,31), (21,31), (22,31), (23,31), (24,31), (25,31), (26,31), (27,31), (28,31), (29,31), (30,31), (31,31), (32,31), (33,31), (34,31), (35,31), (36,31), (37,31), (38,31), (39,31), (40,31), (41,31), (42,31), (43,31), (44,31), (45,31), (46,31), (47,31), (48,31), (49,31), (50,31), (51,31), (52,31), (53,31), (54,31), (55,31)]

#====== functions =======

def get_file_path(file_name):
    if hasattr(sys, '_MEIPASS'):
        # Running as executable
        base_path = sys._MEIPASS
    else:
        # Running as script
        base_path = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(base_path, file_name)
    return file_path


def point_vs_rect(px,py,rx,ry,rw,rh):
    return px > rx and px <= rx+rw and py > ry and py < ry+rh


def update_skin_from_output(output):
    global output_skin_data, output_skin_surface, upscaled_output_skin_surface, rendered_output_skin, rendered_output_skin_back, render_model, render_model_lock
    i = 0
    for coords in pixel_lookup_table:
        output_skin_data[coords[0]][coords[1]][0] = output[i]*255
        output_skin_data[coords[0]][coords[1]][1] = output[i+1]*255
        output_skin_data[coords[0]][coords[1]][2] = output[i+2]*255
        output_skin_data[coords[0]][coords[1]][3] = 255
        i += 3


    #Copying arm & leg because of bug in MinePI rendering classic 64x32 skins
    #This can be removed once the bug is fixed

    #copy left leg 
    off_x = 0
    off_y = 16
    target_off_x = 16
    target_off_y = 48
    for y in range(16):
        for x in range(4):
            output_skin_data[target_off_x + 4 + x][target_off_y + y] = output_skin_data[off_x + 7 - x][off_y + y]
    for y in range(4):
        for x in range(4):
            output_skin_data[target_off_x + 8 + x][target_off_y + y] = output_skin_data[off_x + 11 - x][off_y + y]
    for y in range(12):
        for x in range(4):
            output_skin_data[target_off_x + 12 + x][target_off_y + 4 + y] = output_skin_data[off_x + 15 - x][off_y + 4 + y]
    for y in range(12):
        for x in range(4):
            output_skin_data[target_off_x + x][target_off_y + 4 + y] = output_skin_data[off_x + 11 - x][off_y + 4 + y]
    for y in range(12):
        for x in range(4):
            output_skin_data[target_off_x + 8 + x][target_off_y + 4 + y] = output_skin_data[off_x + 3 - x][off_y + 4 + y]
            
    #copy left arm 
    off_x = 40
    off_y = 16
    target_off_x = 32
    target_off_y = 48
    for y in range(16):
        for x in range(4):
            output_skin_data[target_off_x + 4 + x][target_off_y + y] = output_skin_data[off_x + 7 - x][off_y + y]
    for y in range(4):
        for x in range(4):
            output_skin_data[target_off_x + 8 + x][target_off_y + y] = output_skin_data[off_x + 11 - x][off_y + y]
    for y in range(12):
        for x in range(4):
            output_skin_data[target_off_x + 12 + x][target_off_y + 4 + y] = output_skin_data[off_x + 15 - x][off_y + 4 + y]
    for y in range(12):
        for x in range(4):
            output_skin_data[target_off_x + x][target_off_y + 4 + y] = output_skin_data[off_x + 11 - x][off_y + 4 + y]
    for y in range(12):
        for x in range(4):
            output_skin_data[target_off_x + 8 + x][target_off_y + 4 + y] = output_skin_data[off_x + 3 - x][off_y + 4 + y]


    pygame.surfarray.blit_array(output_skin_surface, output_skin_data[:,:32,:3])
    pygame.transform.scale(output_skin_surface, (upscaled_skin_width, upscaled_skin_height), upscaled_output_skin_surface)

    with render_model_lock:
        render_model = True

def randomise_sliders():
    global slider_values
    slider_values = np.random.uniform(0,1,latent_space_dim).astype(np.float32)

def slider_range_to_intensity(range):
    c = 2/5
    return range/c if range < c else (range-c)/(1-c)*(intensity_slider_max_value-1)+1

#feed input through model.encode, then update slidervalues with output
def update_sliders_from_model(input):
    global slider_values
    
    output = model_encode.run(None,{model_encode.get_inputs()[0].name:input})[0]
    for i in range(latent_space_dim):
        slider_values[slider_lookup_id[i]] = max(min((output[i] / (2*slider_range+0.000001)) + 0.5, 1), 0)

#feed slidervalues through model.decode, then update skin from output
def update_model_from_sliders():
    output = model_decode.run(None,{model_decode.get_inputs()[0].name:([slider_values[slider_lookup_id[i]]*2*slider_range-slider_range for i in range(latent_space_dim)])})[0]
    if double_feed_through:
        output = model_encode.run(None, {model_encode.get_inputs()[0].name: output})[0]
        output = model_decode.run(None, {model_decode.get_inputs()[0].name: output})[0]
    update_skin_from_output(output)

def get_mouse_slider():
    ind = -1
    if point_vs_rect(mouse_x, mouse_y, slider_offset_x-slider_spacing_x/2, slider_offset_y, 64*slider_spacing_x, slider_spacing_y+slider_height):
        if mouse_y < slider_offset_y + slider_height: #check upper layer
            ind = int(63.0 * (mouse_x - (slider_offset_x-slider_spacing_x/2)) / (63*slider_spacing_x))
        elif mouse_y > slider_offset_y + slider_spacing_y: #check lower layer
            ind = 64 + int(63.0 * (mouse_x - (slider_offset_x-slider_spacing_x/2)) / (63*slider_spacing_x))
    
    return latent_space_dim-1 if ind>=latent_space_dim else ind

def th_render_skin():
    global render_model, render_model_lock, render_data_lock, rendered_output_skin, rendered_output_skin_back

    while running:
        if render_model:
            with render_model_lock:
                render_model = False

            skin = Skin(Image.fromarray(output_skin_data.transpose(1,0,2)))
            asyncio.run(skin.render_skin(hr=20, vr=-10, ratio=8, display_hair=False, display_second_layer=False, aa=False))
            render1 = pygame.surfarray.make_surface(np.array(skin.skin)[:,:,:3].transpose(1,0,2))

            asyncio.run(skin.render_skin(hr=200, vr=-10, ratio=8, display_hair=False, display_second_layer=False, aa=False))
            render2 = pygame.surfarray.make_surface(np.array(skin.skin)[:,:,:3].transpose(1,0,2))

            with render_data_lock:
                rendered_output_skin = render1
                rendered_output_skin_back = render2

        sleep(0.02)


#====== loading resourced ======

model_decode = ort.InferenceSession(get_file_path("traced_model_decode.onnx"), session_options)
model_encode = ort.InferenceSession(get_file_path("traced_model_encode.onnx"), session_options)


# Initialize Pygame
pygame.init()

# Set the width and height of the window
width = 1200
height = 700

# Create the window
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("MSG - Minecraft Skin Generator by BirnB4um")
pygame.display.set_icon(pygame.image.load(get_file_path("icon.png")))
clock = pygame.time.Clock()
running = True

mouse_x = 0
mouse_y = 0

latent_space_dim = 128

font = pygame.font.SysFont("Arial", 20)
font_smaller = pygame.font.SysFont("Arial", 15)

output_skin_data = np.zeros((64,64,4), dtype=np.uint8)
output_skin_surface = pygame.surfarray.make_surface(output_skin_data[:,:32,:3])
rendered_output_skin = output_skin_surface
rendered_output_skin_back = output_skin_surface
upscaled_skin_width = 400
upscaled_skin_height = int(upscaled_skin_width/2)
upscaled_output_skin_surface = pygame.transform.scale(output_skin_surface, (upscaled_skin_width, upscaled_skin_height))


#sort sliders by influence
# influence_values = [[0,i] for i in range(latent_space_dim)]
# for i in range(latent_space_dim):
#     input = np.zeros(latent_space_dim, dtype=np.float32)
#     input[i] = 1
#     output1 = np.array(model_decode.run(None, {model_decode.get_inputs()[0].name:input}))
#     input[i] = -1
#     output2 = np.array(model_decode.run(None, {model_decode.get_inputs()[0].name:input}))
#     influence_values[i][0] = np.sum(np.abs(output1 - output2))
# influence_values.sort(key=lambda x: x[0], reverse=True)
# slider_lookup_id = [influence_values[i][1] for i in range(latent_space_dim)]

slider_lookup_id = [9, 22, 103, 108, 106, 52, 65, 119, 80, 51, 91, 47, 30, 28, 12, 69, 35, 20, 55, 127, 113, 105, 94, 36, 126, 14, 114, 81, 73, 104, 16, 98, 112, 6, 61, 88, 72, 24, 117, 67, 93, 82, 19, 83, 68, 74, 85, 79, 102, 86, 31, 111, 2, 37, 33, 122, 48, 44, 18, 96, 62, 70, 64, 13, 107, 0, 71, 40, 53, 46, 100, 43, 87, 7, 115, 123, 125, 78, 23, 60, 50, 59, 110, 49, 5, 45, 77, 90, 121, 11, 21, 38, 76, 32, 92, 120, 124, 3, 118, 56, 109, 84, 42, 89, 29, 101, 63, 4, 8, 17, 34, 75, 27, 66, 1, 58, 57, 39, 54, 26, 99, 95, 10, 116, 25, 15, 41, 97]

slider_vel = [random.uniform(-0.03, 0.03) for i in range(latent_space_dim)]
slider_values = np.full(latent_space_dim, 0.5, dtype=np.float32)
slider_color = [(random.randint(50,255), random.randint(50,255), random.randint(50,255)) for i in range(latent_space_dim)]
slider_width = 10
slider_height = 150
slider_spacing_x = 18
slider_spacing_y = 20 + slider_height
slider_offset_x = 30
slider_offset_y = 350
pressed_slider_i = -1
mouse_over_slider_i = -1

slider_range = 1.0
intensity_slider_x = 800
intensity_slider_y = 320
intensity_slider_width = 200
intensity_slider_height = 20
intensity_slider_max_value = 10
intensity_slider_value = 2/5
pressed_intensity_slider = False
intensity_slider_text_surface = font.render(f"Intensity: {round(slider_range,2)}", True, (255,255,255))

double_feed_through = False
text_double_feed_trough = font.render(f"Double feed through [D]: {double_feed_through}", True, (255,255,255))


text_randomsize = font.render(f"Randomize sliders [R]", True, (255,255,255))
text_randomsize_normal = font.render(f"Randomize sliders normal dist. [Shift + R]", True, (255,255,255))
text_max_slider = font.render(f"Randomize sliders maximal [M]", True, (255,255,255))
text_reset_slider = font.render(f"Reset sliders [0]", True, (255,255,255))
text_knock_slider = font.render(f"Knock sliders [K]", True, (255,255,255))
text_converge_model = font.render(f"Converge to equilibrium [H]", True, (255,255,255))
text_move_slider = font.render(f"Move sliders [J]", True, (255,255,255))
text_save_skin = font.render(f"Save skin [S]", True, (255,255,255))
text_load_skin = font.render(f"Load skin [L]", True, (255,255,255))

changed_model = True
render_model = True
render_model_lock = threading.Lock()
render_data_lock = threading.Lock()
render_model_thread = threading.Thread(target=th_render_skin)

update_model_from_sliders()
render_model_thread.start()


def generation_screen():
    global pressed_slider_i, slider_values, double_feed_through, mouse_over_slider_i, pressed_intensity_slider
    global intensity_slider_value, slider_range, changed_model, intensity_slider_text_surface, text_double_feed_trough, slider_vel

    for event in event_list:
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_r:
                randomise_sliders()
                slider_vel = [random.uniform(-0.03, 0.03) for i in range(latent_space_dim)]
                changed_model = True
            if event.key == pygame.K_r and pygame.key.get_mods() & pygame.KMOD_SHIFT:
                slider_values = np.clip(np.random.normal(0.5,0.2,latent_space_dim), 0,1).astype(np.float32)
                slider_vel = [random.uniform(-0.03, 0.03) for i in range(latent_space_dim)]
                changed_model = True
            if event.key == pygame.K_0:
                slider_values[:] = 0.5
                changed_model = True
            if event.key == pygame.K_d:
                double_feed_through = not double_feed_through
                text_double_feed_trough = font.render(f"Double feed through [D]: {double_feed_through}", True, (255,255,255))
                changed_model = True
            if event.key == pygame.K_s: #save model
                file_path = asksaveasfilename()
                try:
                    if file_path != "":
                        if not file_path.endswith((".png",".jpg")):
                            file_path = file_path + ".png"
                        Image.fromarray(output_skin_data[:,:32,:].transpose(1,0,2)).save(file_path)
                except:
                    print("Error: couldn't save skin as image")
            if event.key == pygame.K_k: #knock sliders a bit
                for i in range(latent_space_dim):
                    slider_values[i] = max(0, min(1, slider_values[i] + random.uniform(-0.08,0.08)))
                slider_vel = [random.uniform(-0.03, 0.03) for i in range(latent_space_dim)]
                changed_model = True
            if event.key == pygame.K_l: #load model
                file_path = askopenfilename()
                if file_path != "":
                    try:
                        img = np.array(Image.open(file_path).convert("RGB"))
                        if len(img[0]) != 64 or (len(img) != 64 and len(img) != 32):
                            print("Error: Image size isn't supported (use 64x32 or 64x64)")
                            break
                        input = []
                        for coord in pixel_lookup_table:
                            input.append(img[coord[1]][coord[0]][0]/255)
                            input.append(img[coord[1]][coord[0]][1]/255)
                            input.append(img[coord[1]][coord[0]][2]/255)

                        update_sliders_from_model(input)
                        update_model_from_sliders()
                    except:
                        print("Error: couldn't load skin")
                
            if event.key == pygame.K_m: #set slider to max
                for i in range(latent_space_dim):
                    slider_values[i] = 1 if random.uniform(0,1) < 0.5 else 0 if random.uniform(0,1) < 0.2 else 0.5
                slider_vel = [random.uniform(-0.03, 0.03) for i in range(latent_space_dim)]
                changed_model = True

        elif event.type == pygame.MOUSEBUTTONDOWN  and event.button == 1:
            pressed_slider_i = get_mouse_slider() #press slider

            #press slider_range_slider
            if point_vs_rect(mouse_x, mouse_y, intensity_slider_x, intensity_slider_y-intensity_slider_height/2, intensity_slider_width, intensity_slider_height):
                pressed_intensity_slider = True
        
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            pressed_slider_i = -1 #release slider
            pressed_intensity_slider = False
            
        if event.type == pygame.MOUSEMOTION:
            #update mouse over slider
            mouse_over_slider_i = get_mouse_slider() if pressed_slider_i == -1 else pressed_slider_i


    if key_is_pressed("j"):
        for i in range(latent_space_dim):
            new_value = slider_values[i] + slider_vel[i]
            if new_value < 0 or new_value > 1:
                slider_vel[i] *= -1
            slider_values[i] = max(0, min(1, new_value))
            changed_model = True

    if key_is_pressed("h"):
        temp_in = np.zeros(3552, dtype=np.float32)
        i=0
        for coord in pixel_lookup_table:
            temp_in[i] = output_skin_data[coord[0]][coord[1]][0]/255
            temp_in[i+1] = output_skin_data[coord[0]][coord[1]][1]/255
            temp_in[i+2] = output_skin_data[coord[0]][coord[1]][2]/255
            i+=3
        update_sliders_from_model(temp_in)
        changed_model = True

    #update slider
    if pressed_slider_i > -1:
        slider_values[pressed_slider_i] = min(1,max(0, 1-(mouse_y - (slider_offset_y + (slider_spacing_y if pressed_slider_i > 63 else 0)))/slider_height))
        changed_model = True

    #update slider_range_slider
    if pressed_intensity_slider:
        intensity_slider_value = min(max((mouse_x-intensity_slider_x)/intensity_slider_width, 0), 1)
        slider_range = slider_range_to_intensity(intensity_slider_value)
        intensity_slider_text_surface = font.render(f"Intensity: {round(slider_range,2)}", True, (255,255,255))
        changed_model = True

    #update model
    if changed_model:
        changed_model = False
        update_model_from_sliders()


    #draw output skin
    window.blit(upscaled_output_skin_surface, (20,50))
    with render_data_lock:
        window.blit(rendered_output_skin, (420, 30))
        window.blit(rendered_output_skin_back, (600, 30))

    #draw slider_range_slider
    pygame.draw.line(window, (255,255,255), (intensity_slider_x, intensity_slider_y), (intensity_slider_x+intensity_slider_width, intensity_slider_y))
    pygame.draw.circle(window, (255,255,255), (intensity_slider_x + intensity_slider_value * intensity_slider_width, intensity_slider_y), intensity_slider_height/2)
    window.blit(intensity_slider_text_surface, (intensity_slider_x+intensity_slider_width+10, intensity_slider_y-intensity_slider_height/2))

    #draw status text
    window.blit(text_randomsize, (800,40))
    window.blit(text_randomsize_normal,(800,65))
    window.blit(text_max_slider,(800,90))
    window.blit(text_knock_slider,(800,115))
    window.blit(text_reset_slider,(800,140))
    window.blit(text_move_slider,(800,165))
    window.blit(text_converge_model,(800,190))
    window.blit(text_double_feed_trough,(800,215))
    window.blit(text_save_skin,(800,240))
    window.blit(text_load_skin,(800,265))

    #draw slider
    y = slider_offset_y
    slider_i = 0
    for i in range(64):
        if slider_i < latent_space_dim:
            x = int(slider_offset_x + i * slider_spacing_x)
            pygame.draw.line(window, slider_color[slider_i], (x, y), (x, y + slider_height), 1)
            pygame.draw.line(window, slider_color[slider_i], (x-slider_width/2, y+slider_height/2), (x+slider_width/2, y + slider_height/2), 1)
            pygame.draw.circle(window, slider_color[slider_i], (x, y+slider_height*(1-slider_values[slider_i])), slider_width/2)
            slider_i += 1

    for i in range(64):
        if slider_i < latent_space_dim:
            x = int(slider_offset_x + i * slider_spacing_x)
            pygame.draw.line(window, slider_color[slider_i], (x, y + slider_spacing_y), (x, y + slider_spacing_y + slider_height), 1)
            pygame.draw.line(window, slider_color[slider_i], (x-slider_width/2, y+slider_height/2 + slider_spacing_y), (x+slider_width/2, y + slider_spacing_y + slider_height/2), 1)
            pygame.draw.circle(window, slider_color[slider_i], (x, y+slider_spacing_y+slider_height*(1-slider_values[slider_i])), slider_width/2)
            slider_i += 1


#====== Main loop =======

while running:
    mouse_x, mouse_y = pygame.mouse.get_pos()

    # Handle events
    event_list = pygame.event.get()
    for event in event_list:
        if event.type == pygame.QUIT:
            running = False

    if pygame.display.get_active():
        window.fill((0, 0, 0))
        generation_screen()

    pygame.display.flip()
    clock.tick(40)

render_model_thread.join()
pygame.quit()