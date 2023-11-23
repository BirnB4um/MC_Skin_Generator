import os
import onnxruntime as ort
from PIL import Image
import random
from time import sleep
import asyncio
from tkinter.filedialog import askopenfilename, asksaveasfilename
import threading
import numpy as np
import sys
import pygame
from keyboard import is_pressed as key_is_pressed
from time import sleep
from sklearn.cluster import KMeans


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

        self.skin_front_layout = np.load(get_file_path("skin_front_layout.npy"))
        self.skin_front_overlay_layout = np.load(get_file_path("skin_front_overlay_layout.npy"))
        self.skin_back_layout = np.load(get_file_path("skin_back_layout.npy"))
        self.skin_back_overlay_layout = np.load(get_file_path("skin_back_overlay_layout.npy"))

        self.input_values = np.zeros((1,512), dtype=np.float32)
        self.slider_values = np.full((256), 0.5, dtype=np.float32)
        self.slider_intensity = 1
        self.slider_range_factor = 3
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
        self.slider_numbers = [self.font_small.render(str(i), True, (255,255,255)) for i in range(256)]


        self.skin_surface = None
        self.skin_array = np.zeros((64, 64, 4), dtype=np.uint8)

        self.skin_render_front_surface = None
        self.skin_render_back_surface = None

        self.overlay = True

        self.number_colors_slider_x = self.width-200
        self.number_colors_slider_y = 165
        self.number_colors_slider_width = 170
        self.number_colors_slider_height = 18
        self.number_colors_slider_knob_width = 8
        self.number_colors_max = 64
        self.number_colors_min = 1
        self.number_of_colors = 16
        self.reduce_colors = False
        self.mouse_pressed_colors_slider = False

        self.text_randomize_slider = self.font.render("R - Randomize sliders", True, (255,255,255))
        self.text_reset_slider = self.font.render("0 - Reset sliders", True, (255,255,255))
        self.text_save_skin = self.font.render("S - Save skin", True, (255,255,255))
        self.text_load_skin = self.font.render("L - Load skin", True, (255,255,255))
        self.text_toggle_overlay = self.font.render("O - Toggle overlay", True, (255,255,255) if self.overlay else (100,100,100))
        self.text_reduce_colors = self.font.render("C - Reduce colors", True, (255,255,255) if self.reduce_colors else (100,100,100))
        self.text_number_of_colors = self.font.render("Number of colors: " + str(self.number_of_colors), True, (255,255,255) if self.reduce_colors else (100,100,100))


        self.update_inputs_from_sliders()
        self.run_model()


    def reduce_color_palette(self, image, num_colors, n_init=5):
        reshaped_image = image.reshape(-1, image.shape[2])
        kmeans = KMeans(n_clusters=num_colors, n_init=n_init)
        kmeans.fit(reshaped_image)
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.predict(reshaped_image)
        reduced_image = cluster_centers[labels].reshape(image.shape[0], image.shape[1], image.shape[2])
        return reduced_image, cluster_centers

    def run_model(self):
        model_output = self.model_decode.run(None, {"input": self.input_values})[0][0]
        model_output = model_output.transpose(1, 2, 0)
        alpha_mask = model_output[:,:,3:]

        if self.reduce_colors:
            reduced_model_output, cluster_centers = self.reduce_color_palette(model_output[:,:,:3], self.number_of_colors, 1)
            model_output = np.concatenate((reduced_model_output, model_output[:,:,3:]), axis=2)

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

                # if classic skin
                if img.shape[1] == 32 and img.shape[2] == 64:
                    #convert classic skin to normal 64x64 skin
                    new_img = np.zeros((4,64,64), dtype=np.float32)
                    new_img[:,:32,:] = img


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
        pca_reduced = np.dot(self.input_values[0] - self.pca_mean, self.pca_components.T)
        self.slider_values = (1-(pca_reduced / (self.pca_std * self.slider_range_factor)))/2
        self.slider_values = np.clip(self.slider_values, 0, 1)
        

    def randomize_slider_values(self):
        self.slider_values = (1-(self.slider_intensity*np.random.normal(0, self.pca_std, 256)/(self.pca_std * self.slider_range_factor)))/2
        self.slider_values = np.clip(self.slider_values, 0, 1)
        self.update_inputs_from_sliders()

    def reset_slider_values(self):
        self.slider_values = np.full((256), 0.5, dtype=np.float32)
        self.update_inputs_from_sliders()

    def update(self):
        pass

    def run(self):
        while self.running:
            self.mouse_x, self.mouse_y = pygame.mouse.get_pos()

            # Handle events
            event_list = pygame.event.get()
            for event in event_list:
                if event.type == pygame.QUIT:
                    self.running = False
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
                    elif event.key == pygame.K_c:
                        self.reduce_colors = not self.reduce_colors
                        self.text_reduce_colors = self.font.render("C - Reduce colors", True, (255,255,255) if self.reduce_colors else (100,100,100))
                        self.text_number_of_colors = self.font.render("Number of colors: " + str(self.number_of_colors), True, (255,255,255) if self.reduce_colors else (100,100,100))
                        self.run_model()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # left mouse button
                        self.mouse_pressed_slider_i = self.mouse_over_slider_i
                        if self.mouse_pressed_slider_i != None:
                            self.slider_values[self.mouse_pressed_slider_i] = np.clip((self.mouse_y - self.slider_y - self.slider_knob_height/2) / (self.slider_height-self.slider_knob_height), 0, 1)
                            self.update_inputs_from_sliders()
                            self.run_model()
                        
                        if point_vs_rect(self.mouse_x, self.mouse_y, self.number_colors_slider_x, self.number_colors_slider_y, self.number_colors_slider_width, self.number_colors_slider_height):
                            self.mouse_pressed_colors_slider = True
                            self.number_of_colors = int(np.clip(((self.mouse_x - self.number_colors_slider_x) / (self.number_colors_slider_width-self.number_colors_slider_knob_width))**2 * (self.number_colors_max-self.number_colors_min) + self.number_colors_min, self.number_colors_min, self.number_colors_max))
                            self.text_number_of_colors = self.font.render("Number of colors: " + str(self.number_of_colors), True, (255,255,255) if self.reduce_colors else (100,100,100))
                            self.run_model()
                            

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1: # left mouse button
                        self.mouse_pressed_slider_i = None
                        self.mouse_pressed_colors_slider = False

                elif event.type == pygame.MOUSEMOTION:
                    if point_vs_rect(self.mouse_x, self.mouse_y, self.slider_x-self.slider_spacing/2, self.slider_y, (self.slider_width+self.slider_spacing)*self.number_of_sliders_shown, self.slider_height):
                        self.mouse_over_slider_i = self.slider_offset + int((self.mouse_x - self.slider_x + self.slider_spacing/2) / (self.slider_width + self.slider_spacing))
                    else:
                        self.mouse_over_slider_i = None

                    if self.mouse_pressed_slider_i != None:
                        self.slider_values[self.mouse_pressed_slider_i] = np.clip((self.mouse_y - self.slider_y - self.slider_knob_height/2) / (self.slider_height-self.slider_knob_height), 0, 1)
                        self.update_inputs_from_sliders()
                        self.run_model()

                    if self.mouse_pressed_colors_slider:
                        self.number_of_colors = int(np.clip(((self.mouse_x - self.number_colors_slider_x) / (self.number_colors_slider_width-self.number_colors_slider_knob_width))**2 * (self.number_colors_max-self.number_colors_min) + self.number_colors_min, self.number_colors_min, self.number_colors_max))
                        self.text_number_of_colors = self.font.render("Number of colors: " + str(self.number_of_colors), True, (255,255,255) if self.reduce_colors else (100,100,100))
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

                # draw sliders
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
                self.window.blit(self.text_randomize_slider, (self.width-200, 20))
                self.window.blit(self.text_reset_slider, (self.width-200, 40))
                self.window.blit(self.text_save_skin, (self.width-200, 60))
                self.window.blit(self.text_load_skin, (self.width-200, 80))
                self.window.blit(self.text_toggle_overlay, (self.width-200, 100))
                self.window.blit(self.text_reduce_colors, (self.width-200, 120))
                self.window.blit(self.text_number_of_colors, (self.width-200, 140))

                # draw number of colors slider
                pygame.draw.line(self.window, (255,255,255) if self.reduce_colors else (100,100,100),
                                    (self.number_colors_slider_x, self.number_colors_slider_y+self.number_colors_slider_height/2),
                                    (self.number_colors_slider_x+self.number_colors_slider_width, self.number_colors_slider_y+self.number_colors_slider_height/2))
                pygame.draw.rect(self.window, (255,255,255) if self.reduce_colors else (100,100,100), pygame.Rect(self.number_colors_slider_x + ((self.number_of_colors-self.number_colors_min)/(self.number_colors_max-self.number_colors_min))**0.5 *(self.number_colors_slider_width-self.number_colors_slider_knob_width), self.number_colors_slider_y, self.number_colors_slider_knob_width, self.number_colors_slider_height))


            pygame.display.flip()
            self.clock.tick(40)

        pygame.quit()


if __name__ == "__main__":
    app = App()
    app.run()
