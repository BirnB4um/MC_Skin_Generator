import os
import onnxruntime as ort
from PIL import Image
import random
from tkinter.filedialog import askopenfilename, asksaveasfilename
import numpy as np
import sys
import pygame
from scipy.signal import convolve2d


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



class App:

    def __init__(self):

        pygame.init()

        self.width = 900
        self.height = 600
        self.background_color = [18,18,20]
        self.running = True
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("MSG - Minecraft Skin Generator by BirnB4um")
        # pygame.display.set_icon(pygame.image.load(get_file_path("icon.png")))
        self.clock = pygame.time.Clock()
        self.keys_pressed = None

        self.font = pygame.font.SysFont("Arial", 18)
        self.font_small = pygame.font.SysFont("Arial", 10)

        self.session_options = ort.SessionOptions()
        self.session_options.intra_op_num_threads = 2
        self.session_options.inter_op_num_threads = 2

        self.model_decode = ort.InferenceSession(get_file_path("model_decode.onnx"), self.session_options)
        self.model_encode = ort.InferenceSession(get_file_path("model_encode.onnx"), self.session_options)

        self.pca_components = np.load(get_file_path("pca_256_components.npy"))
        self.pca_mean = np.load(get_file_path("pca_256_mean.npy"))
        self.pca_std = np.load(get_file_path("pca_256_latentspace_std.npy"))
        self.pca_eigenvalues = np.load(get_file_path("pca_256_explained_variance.npy"))

        self.skin_layout = np.zeros((64, 64))
        self.skin_layout[:8,8:24] = 1
        self.skin_layout[8:16,:32] = 1
        self.skin_layout[20:32,:56] = 1
        self.skin_layout[16:20,4:12] = 1
        self.skin_layout[16:20,20:36] = 1
        self.skin_layout[16:20,44:52] = 1
        self.skin_layout[52:,16:48] = 1
        self.skin_layout[48:,20:28] = 1
        self.skin_layout[48:,36:44] = 1

        self.skin_front_layout = np.load(get_file_path("skin_front_layout.npy"))
        self.skin_front_overlay_layout = np.load(get_file_path("skin_front_overlay_layout.npy"))
        self.skin_back_layout = np.load(get_file_path("skin_back_layout.npy"))
        self.skin_back_overlay_layout = np.load(get_file_path("skin_back_overlay_layout.npy"))
        with open(get_file_path("pca_descriptions.txt"), "r") as file:
            self.slider_descriptions = file.read().splitlines()

        self.slider_move_speed = 0.2
        self.slider_move_target = np.clip((np.random.normal(0, 1, (256))/6) + 0.5, 0, 1)

        self.input_values = np.zeros((1,512), dtype=np.float32)
        self.slider_values = np.full((256), 0.5, dtype=np.float32)
        self.slider_offset = 0
        self.number_of_sliders = 256
        self.slider_spacing = 5
        self.slider_x = 10
        self.slider_y = 390
        self.slider_width = 16
        self.slider_height = 180
        self.slider_knob_height = 8
        self.number_of_sliders_shown = 42
        self.slider_colors = [(random.randint(100,255), random.randint(100,255), random.randint(100,255)) for i in range(self.number_of_sliders)]
        self.mouse_over_slider_i = None
        self.mouse_pressed_slider_i = None
        self.slider_scroll_speed = 3
        self.slider_numbers = [self.font_small.render(str(i+1), True, (255,255,255)) for i in range(256)]

        self.skin_surface = None
        self.skin_array = np.zeros((64, 64, 4), dtype=np.uint8)
        self.skin_render_front_surface = None
        self.skin_render_back_surface = None

        self.overlay = True

        self.sharpen_skin = False
        self.mouse_pressed_sharpen_slider = False
        self.sharpen_slider_x = self.width-200
        self.sharpen_slider_y = 193
        self.sharpen_slider_width = 170
        self.sharpen_slider_height = 18
        self.sharpen_slider_knob_width = 8
        self.sharpen_max = 25
        self.sharpen_min = 4.0001
        self.sharpen_value_default = 12
        self.sharpen_value = self.sharpen_value_default

        self.reduce_colors = False
        self.mouse_pressed_colors_slider = False
        self.number_colors_slider_x = self.width-200
        self.number_colors_slider_y = 264
        self.number_colors_slider_width = 170
        self.number_colors_slider_height = 18
        self.number_colors_slider_knob_width = 8
        self.number_colors_max = 64
        self.number_colors_min = 1
        self.number_of_colors_default = 16
        self.number_of_colors = self.number_of_colors_default

        self.mouse_pressed_range_slider = False
        self.slider_range_x = self.width-200
        self.slider_range_y = 340
        self.slider_range_width = 170
        self.slider_range_height = 18
        self.slider_range_knob_width = 8
        self.slider_range_max = 6
        self.slider_range_min = 0.1
        self.slider_range_default = 3
        self.slider_range_factor = self.slider_range_default


        self.text_randomize_slider = self.font.render("R - Randomize sliders", True, (255,255,255))
        self.text_reset_slider = self.font.render("0 - Reset sliders", True, (255,255,255))
        self.text_save_skin = self.font.render("S - Save skin", True, (255,255,255))
        self.text_load_skin = self.font.render("L - Load skin", True, (255,255,255))
        self.text_move_sliders = self.font.render("M & hold - Move sliders", True, (255,255,255))
        self.text_converge = self.font.render("H - Converge", True, (255,255,255))
        self.text_toggle_overlay = self.font.render("O - Toggle overlay", True, (255,255,255) if self.overlay else (100,100,100))
        self.text_sharpen = self.font.render("X - Sharpen", True, (255,255,255) if self.sharpen_skin else (100,100,100))
        self.text_sharpen_value = self.font.render("Sharpen value: " + str(self.sharpen_value), True, (255,255,255) if self.sharpen_skin else (100,100,100))
        self.text_reduce_colors = self.font.render("C - Reduce colors", True, (255,255,255) if self.reduce_colors else (100,100,100))
        self.text_number_of_colors = self.font.render("Number of colors: " + str(self.number_of_colors), True, (255,255,255) if self.reduce_colors else (100,100,100))
        self.text_slider_range = self.font.render("Slider range: " + str(self.slider_range_factor), True, (255,255,255))
        self.text_slider_description = self.font.render("", True, (255,255,255))

        self.update_inputs_from_sliders()
        self.run_model()


    def reduce_color_palette(self, image, num_colors):
        image = Image.fromarray(np.uint8(image))
        quantized_image = image.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT, kmeans=0, palette=None, dither=Image.FLOYDSTEINBERG)
        quantized_image = quantized_image.convert('RGB')
        return np.array(quantized_image)

    def apply_sharpening_filter(self, image, strength=10, sharpen_alpha=False):
        kernel = np.array([
            [0, -1, 0],
            [-1, strength, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        kernel = kernel / np.sum(kernel)

        sharpened_image = image.copy()

        for i in range(4 if sharpen_alpha else 3):  # Loop through color channels
            sharpened_image[:, :, i] = convolve2d(image[:, :, i], kernel, mode='same')

        return np.clip(sharpened_image, 0, 1)

    def run_model(self):
        model_output = self.model_decode.run(None, {"input": self.input_values})[0][0]
        model_output = model_output.transpose(1, 2, 0)

        # remove overlay
        if not self.overlay:
            model_output[:,:,3] = self.skin_layout

        alpha_mask = model_output[:,:,3:]

        # sharpen skin
        if self.sharpen_skin:
            model_output = self.apply_sharpening_filter(model_output, self.sharpen_value, True)

        # reduce colors
        if self.reduce_colors:
            reduced_model_output = self.reduce_color_palette(model_output[:,:,:3]*255, self.number_of_colors)
            model_output = np.concatenate((reduced_model_output, model_output[:,:,3:]*255), axis=2).astype(np.uint8)
            self.skin_array = model_output
        else:
            self.skin_array = (model_output*255).astype(np.uint8)

        skin_data = (self.skin_array[:,:,:3] * alpha_mask).transpose(1, 0, 2)
        self.skin_surface = pygame.surfarray.make_surface(skin_data)
        self.skin_surface = pygame.transform.scale(self.skin_surface, (300, 300))

        self.skin_array[0,0,:3] = self.background_color # set background color
        self.skin_array[0,0,3] = 0

        rendered_skin = np.full((401,202,4), 255, dtype=np.uint8)
        rendered_skin[:,:,:3] = self.skin_array[self.skin_front_layout[:,:,0],self.skin_front_layout[:,:,1],:3]
        overlay_alpha = self.skin_array[self.skin_front_overlay_layout[:,:,0],self.skin_front_overlay_layout[:,:,1],3:]/255
        rendered_skin[:,:,:3] = rendered_skin[:,:,:3] * (1-overlay_alpha) + overlay_alpha * self.skin_array[self.skin_front_overlay_layout[:,:,0],self.skin_front_overlay_layout[:,:,1],:3]
        self.skin_render_front_surface = pygame.surfarray.make_surface(rendered_skin.transpose(1, 0, 2)[:,:,:3])
        self.skin_render_front_surface = pygame.transform.scale(self.skin_render_front_surface, (150, 300))
        
        rendered_skin[:,:,:3] = self.skin_array[self.skin_back_layout[:,:,0],self.skin_back_layout[:,:,1],:3]
        overlay_alpha = self.skin_array[self.skin_back_overlay_layout[:,:,0],self.skin_back_overlay_layout[:,:,1],3:]/255
        rendered_skin[:,:,:3] = rendered_skin[:,:,:3] * (1-overlay_alpha) + overlay_alpha * self.skin_array[self.skin_back_overlay_layout[:,:,0],self.skin_back_overlay_layout[:,:,1],:3]
        self.skin_render_back_surface = pygame.surfarray.make_surface(rendered_skin.transpose(1, 0, 2)[:,:,:3])
        self.skin_render_back_surface = pygame.transform.scale(self.skin_render_back_surface, (150, 300))

    def load_skin(self):
        file_path = askopenfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All files", "*")])
        if file_path != "":
            try:
                img = np.array(Image.open(file_path).convert("RGBA")).transpose(2,0,1).astype(np.float32)/255

                # if classic skin -> convert to 64x64 skin
                if img.shape[1] == 32 and img.shape[2] == 64:
                    #convert classic skin to normal 64x64 skin
                    new_img = np.zeros((4,64,64), dtype=np.float32)
                    new_img[:,:32,:] = img
                    new_img[:,48:,16:32] = new_img[:,16:32,:16]
                    new_img[:,48:,32:48] = new_img[:,16:32,40:56]
                    img = new_img

                if not (img.shape[1] == 64 and img.shape[2] == 64):
                    print("Error: wrong skin size. image must be 64x64 or 32x64 pixels") 
                    return
                
                new_inputs = self.model_encode.run(None, {"input": img.reshape(1,4,64,64)})[0]
                self.input_values = new_inputs
                self.update_sliders_from_inputs()
                self.run_model()
            except:
                print("Error: couldn't load skin")

    def save_skin(self):
        file_path = asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All files", "*")])
        if file_path != "":
            try:
                Image.fromarray(self.skin_array).save(file_path)
            except:
                print("Error: couldn't save skin as image")

    def update_inputs_from_sliders(self):
        pca_input = (1-(self.slider_values*2)) * self.pca_std * self.slider_range_factor
        self.input_values[0] = np.dot(pca_input, self.pca_components) + self.pca_mean

    def update_sliders_from_inputs(self):
        raw_slider_values = np.dot(self.input_values[0] - self.pca_mean, self.pca_components.T)
        self.slider_values = (1-(raw_slider_values/(self.pca_std * self.slider_range_factor)))/2
        self.slider_values = np.clip(self.slider_values, 0, 1)

    def randomize_slider_values(self):
        self.slider_values = (1-(np.random.normal(0, self.pca_std, 256)/(self.pca_std * self.slider_range_factor)))/2
        self.slider_values = np.clip(self.slider_values, 0, 1)
        self.update_inputs_from_sliders()

    def reset_slider_values(self):
        self.slider_values = np.full((256), 0.5, dtype=np.float32)
        self.update_inputs_from_sliders()

    def update(self):

        # feed output-skin back through model
        if self.keys_pressed[pygame.K_h]:
            self.input_values = self.model_encode.run(None, {"input": [(self.skin_array.transpose(2,0,1)/255).astype(np.float32)]})[0]
            self.update_sliders_from_inputs()
            self.update_inputs_from_sliders()
            self.run_model()
            
        # move to random
        if self.keys_pressed[pygame.K_m]:
            self.slider_values += (self.slider_move_target - self.slider_values) * self.slider_move_speed
            self.update_inputs_from_sliders()
            self.run_model()
            

    def run(self):
        while self.running:
            self.mouse_x, self.mouse_y = pygame.mouse.get_pos()

            # Handle events
            self.keys_pressed = pygame.key.get_pressed()
            event_list = pygame.event.get()
            for event in event_list:
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        self.slider_move_target = np.clip((np.random.normal(0, 1, (256))/6) + 0.5, 0, 1)
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_r:
                        self.randomize_slider_values()
                        self.run_model()
                    elif event.key == pygame.K_0:
                        self.reset_slider_values()
                        self.run_model()
                    elif event.key == pygame.K_s:
                        self.save_skin()
                    elif event.key == pygame.K_l:
                        self.load_skin()
                    elif event.key == pygame.K_o:
                        self.overlay = not self.overlay
                        self.text_toggle_overlay = self.font.render("O - Toggle overlay", True, (255,255,255) if self.overlay else (100,100,100))
                        self.run_model()
                    elif event.key == pygame.K_x:
                        self.sharpen_skin = not self.sharpen_skin
                        self.text_sharpen = self.font.render("X - Sharpen", True, (255,255,255) if self.sharpen_skin else (100,100,100))
                        self.text_sharpen_value = self.font.render("Sharpen value: " + str(round(self.sharpen_value,1)), True, (255,255,255) if self.sharpen_skin else (100,100,100))
                        self.run_model()
                    elif event.key == pygame.K_c:
                        self.reduce_colors = not self.reduce_colors
                        self.text_reduce_colors = self.font.render("C - Reduce colors", True, (255,255,255) if self.reduce_colors else (100,100,100))
                        self.text_number_of_colors = self.font.render("Number of colors: " + str(self.number_of_colors), True, (255,255,255) if self.reduce_colors else (100,100,100))
                        self.run_model()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # left mouse button
                        self.mouse_pressed_slider_i = self.mouse_over_slider_i

                        # mouse over pca sliders
                        if self.mouse_pressed_slider_i != None:
                            self.slider_values[self.mouse_pressed_slider_i] = np.clip((self.mouse_y - self.slider_y - self.slider_knob_height/2) / (self.slider_height-self.slider_knob_height), 0, 1)
                            self.update_inputs_from_sliders()
                            self.run_model()

                        # mouse over sharpen slider
                        if point_vs_rect(self.mouse_x, self.mouse_y, self.sharpen_slider_x, self.sharpen_slider_y, self.sharpen_slider_width, self.sharpen_slider_height):
                            self.mouse_pressed_sharpen_slider = True
                            self.sharpen_value = np.clip(((self.mouse_x - self.sharpen_slider_x - self.sharpen_slider_knob_width/2) / (self.sharpen_slider_width-self.sharpen_slider_knob_width)),0,1)**2 * (self.sharpen_max-self.sharpen_min) + self.sharpen_min
                            self.text_sharpen_value = self.font.render("Sharpen value: " + str(round(self.sharpen_value,1)), True, (255,255,255) if self.sharpen_skin else (100,100,100))
                            self.run_model()
                        
                        # mouse over number of colors slider
                        if point_vs_rect(self.mouse_x, self.mouse_y, self.number_colors_slider_x, self.number_colors_slider_y, self.number_colors_slider_width, self.number_colors_slider_height):
                            self.mouse_pressed_colors_slider = True
                            self.number_of_colors = int(np.clip(((self.mouse_x - self.number_colors_slider_x - self.number_colors_slider_knob_width/2) / (self.number_colors_slider_width-self.number_colors_slider_knob_width)),0,1)**2 * (self.number_colors_max-self.number_colors_min) + self.number_colors_min)
                            self.text_number_of_colors = self.font.render("Number of colors: " + str(self.number_of_colors), True, (255,255,255) if self.reduce_colors else (100,100,100))
                            self.run_model()

                        # mouse over slider range slider
                        if point_vs_rect(self.mouse_x, self.mouse_y, self.slider_range_x, self.slider_range_y, self.slider_range_width, self.slider_range_height):
                            self.mouse_pressed_range_slider = True
                            self.slider_range_factor = np.clip(((self.mouse_x - self.slider_range_x - self.slider_range_knob_width/2) / (self.slider_range_width-self.slider_range_knob_width)),0,1) * (self.slider_range_max-self.slider_range_min) + self.slider_range_min
                            self.text_slider_range = self.font.render("Slider range: " + str(round(self.slider_range_factor,2)), True, (255,255,255))
                            self.update_inputs_from_sliders()
                            self.run_model()
                            

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1: # left mouse button
                        self.mouse_pressed_slider_i = None
                        self.mouse_pressed_colors_slider = False
                        self.mouse_pressed_sharpen_slider = False
                        self.mouse_pressed_range_slider = False

                elif event.type == pygame.MOUSEMOTION:
                    if point_vs_rect(self.mouse_x, self.mouse_y, self.slider_x-self.slider_spacing/2, self.slider_y, (self.slider_width+self.slider_spacing)*self.number_of_sliders_shown, self.slider_height):
                        before_i = self.mouse_over_slider_i
                        self.mouse_over_slider_i = self.slider_offset + int((self.mouse_x - self.slider_x + self.slider_spacing/2) / (self.slider_width + self.slider_spacing))
                        if self.mouse_over_slider_i != before_i:
                            i = self.mouse_over_slider_i if self.mouse_pressed_slider_i == None else self.mouse_pressed_slider_i
                            self.text_slider_description = self.font.render(f"{i+1}: "+self.slider_descriptions[i], True, (255,255,255))
                    else:
                        self.mouse_over_slider_i = None

                    if self.mouse_pressed_slider_i != None:
                        self.slider_values[self.mouse_pressed_slider_i] = np.clip((self.mouse_y - self.slider_y - self.slider_knob_height/2) / (self.slider_height-self.slider_knob_height), 0, 1)
                        self.update_inputs_from_sliders()
                        self.run_model()

                    if self.mouse_pressed_sharpen_slider:
                        self.sharpen_value = np.clip(((self.mouse_x - self.sharpen_slider_x - self.sharpen_slider_knob_width/2) / (self.sharpen_slider_width-self.sharpen_slider_knob_width)),0,1)**2 * (self.sharpen_max-self.sharpen_min) + self.sharpen_min
                        if abs(self.sharpen_value - self.sharpen_value_default) < 0.5:
                            self.sharpen_value = self.sharpen_value_default
                        self.text_sharpen_value = self.font.render("Sharpen value: " + str(round(self.sharpen_value,1)), True, (255,255,255) if self.sharpen_skin else (100,100,100))
                        self.run_model()

                    if self.mouse_pressed_colors_slider:
                        self.number_of_colors = int(np.clip(((self.mouse_x - self.number_colors_slider_x - self.number_colors_slider_knob_width/2) / (self.number_colors_slider_width-self.number_colors_slider_knob_width)),0,1)**2 * (self.number_colors_max-self.number_colors_min) + self.number_colors_min)
                        self.text_number_of_colors = self.font.render("Number of colors: " + str(self.number_of_colors), True, (255,255,255) if self.reduce_colors else (100,100,100))
                        self.run_model()

                    if self.mouse_pressed_range_slider:
                        self.slider_range_factor = np.clip(((self.mouse_x - self.slider_range_x - self.slider_range_knob_width/2) / (self.slider_range_width-self.slider_range_knob_width)),0,1) * (self.slider_range_max-self.slider_range_min) + self.slider_range_min
                        if abs(self.slider_range_factor - self.slider_range_default) < 0.3:
                            self.slider_range_factor = self.slider_range_default
                        self.text_slider_range = self.font.render("Slider range: " + str(round(self.slider_range_factor,2)), True, (255,255,255))
                        self.update_inputs_from_sliders()
                        self.run_model()

                elif event.type == pygame.MOUSEWHEEL:
                    self.slider_offset = np.clip(self.slider_offset - event.y*self.slider_scroll_speed, 0, self.number_of_sliders-self.number_of_sliders_shown)


            self.update()
            

            if pygame.display.get_active():
                self.window.fill(self.background_color)

                # draw skin
                self.window.blit(self.skin_surface, (10, 10))
                self.window.blit(self.skin_render_front_surface, (320, 10))
                self.window.blit(self.skin_render_back_surface, (480, 10))

                # draw pca sliders
                for slider_i in range(self.slider_offset, self.slider_offset+self.number_of_sliders_shown):
                    slider_x = self.slider_x + (self.slider_width + self.slider_spacing) * (slider_i-self.slider_offset)
                    slider_color = self.slider_colors[slider_i]
                    pygame.draw.line(self.window, (slider_color[0]/2, slider_color[1]/2, slider_color[2]/2), 
                                     (slider_x+self.slider_width/2, self.slider_y), 
                                     (slider_x+self.slider_width/2, self.slider_y + self.slider_height))
                    pygame.draw.line(self.window, (slider_color[0]/2, slider_color[1]/2, slider_color[2]/2),
                                     (slider_x, self.slider_y+self.slider_height/2),
                                     (slider_x+self.slider_width, self.slider_y+self.slider_height/2))
                    pygame.draw.rect(self.window, slider_color, pygame.Rect(slider_x, self.slider_y + self.slider_values[slider_i]*(self.slider_height-self.slider_knob_height), self.slider_width, self.slider_knob_height))
                
                    slider_x = self.slider_x + self.slider_width/2 + (self.slider_width + self.slider_spacing) * (slider_i-self.slider_offset) - self.slider_numbers[slider_i].get_width()/2
                    self.window.blit(self.slider_numbers[slider_i], (slider_x, self.slider_y + self.slider_height + 5))

                # draw texts
                self.window.blit(self.text_reset_slider, (self.width-200, 20))
                self.window.blit(self.text_randomize_slider, (self.width-200, 40))
                self.window.blit(self.text_save_skin, (self.width-200, 60))
                self.window.blit(self.text_load_skin, (self.width-200, 80))
                self.window.blit(self.text_move_sliders, (self.width-200, 100))
                self.window.blit(self.text_converge, (self.width-200, 120))
                self.window.blit(self.text_toggle_overlay, (self.width-200, 140))
                self.window.blit(self.text_sharpen, (self.width-200, self.sharpen_slider_y-22))
                self.window.blit(self.text_sharpen_value, (self.width-200, self.sharpen_slider_y+20))
                self.window.blit(self.text_reduce_colors, (self.width-200, self.number_colors_slider_y-22))
                self.window.blit(self.text_number_of_colors, (self.width-200, self.number_colors_slider_y+20))
                self.window.blit(self.text_slider_range, (self.width-200, self.slider_range_y-22))

                if self.mouse_over_slider_i != None or self.mouse_pressed_slider_i != None:
                    self.window.blit(self.text_slider_description, (10, 360))

                # draw number of colors slider
                pygame.draw.line(self.window, (255,255,255) if self.reduce_colors else (100,100,100),
                                    (self.number_colors_slider_x + ((self.number_of_colors_default-self.number_colors_min)/(self.number_colors_max-self.number_colors_min))**0.5 * self.number_colors_slider_width, self.number_colors_slider_y),
                                    (self.number_colors_slider_x + ((self.number_of_colors_default-self.number_colors_min)/(self.number_colors_max-self.number_colors_min))**0.5 * self.number_colors_slider_width, self.number_colors_slider_y + self.number_colors_slider_height))
                pygame.draw.line(self.window, (255,255,255) if self.reduce_colors else (100,100,100),
                                    (self.number_colors_slider_x, self.number_colors_slider_y+self.number_colors_slider_height/2),
                                    (self.number_colors_slider_x+self.number_colors_slider_width, self.number_colors_slider_y+self.number_colors_slider_height/2))
                pygame.draw.rect(self.window, (255,255,255) if self.reduce_colors else (100,100,100), pygame.Rect(self.number_colors_slider_x + ((self.number_of_colors-self.number_colors_min)/(self.number_colors_max-self.number_colors_min))**0.5 *(self.number_colors_slider_width-self.number_colors_slider_knob_width), self.number_colors_slider_y, self.number_colors_slider_knob_width, self.number_colors_slider_height))

                # draw sharpen slider
                pygame.draw.line(self.window, (255,255,255) if self.sharpen_skin else (100,100,100),
                                    (self.sharpen_slider_x + ((self.sharpen_value_default-self.sharpen_min)/(self.sharpen_max-self.sharpen_min))**0.5 * self.sharpen_slider_width, self.sharpen_slider_y),
                                    (self.sharpen_slider_x + ((self.sharpen_value_default-self.sharpen_min)/(self.sharpen_max-self.sharpen_min))**0.5 * self.sharpen_slider_width, self.sharpen_slider_y + self.sharpen_slider_height))
                pygame.draw.line(self.window, (255,255,255) if self.sharpen_skin else (100,100,100),
                                    (self.sharpen_slider_x, self.sharpen_slider_y+self.sharpen_slider_height/2),
                                    (self.sharpen_slider_x+self.sharpen_slider_width, self.sharpen_slider_y+self.sharpen_slider_height/2))
                pygame.draw.rect(self.window, (255,255,255) if self.sharpen_skin else (100,100,100), pygame.Rect(int(self.sharpen_slider_x + ((self.sharpen_value-self.sharpen_min)/(self.sharpen_max-self.sharpen_min))**0.5 * (self.sharpen_slider_width-self.sharpen_slider_knob_width)), self.sharpen_slider_y, self.sharpen_slider_knob_width, self.sharpen_slider_height))

                # draw slider range slider
                pygame.draw.line(self.window, (255,255,255),
                                    (self.slider_range_x + ((self.slider_range_default-self.slider_range_min)/(self.slider_range_max-self.slider_range_min)) * self.slider_range_width, self.slider_range_y),
                                    (self.slider_range_x + ((self.slider_range_default-self.slider_range_min)/(self.slider_range_max-self.slider_range_min)) * self.slider_range_width, self.slider_range_y + self.slider_range_height))
                pygame.draw.line(self.window, (255,255,255),
                                    (self.slider_range_x, self.slider_range_y+self.slider_range_height/2),
                                    (self.slider_range_x+self.slider_range_width, self.slider_range_y+self.slider_range_height/2))
                pygame.draw.rect(self.window, (255,255,255), pygame.Rect(int(self.slider_range_x + ((self.slider_range_factor-self.slider_range_min)/(self.slider_range_max-self.slider_range_min)) * (self.slider_range_width-self.slider_range_knob_width)), self.slider_range_y, self.slider_range_knob_width, self.slider_range_height))

            pygame.display.flip()
            self.clock.tick(40)

        pygame.quit()


if __name__ == "__main__":
    app = App()
    app.run()
