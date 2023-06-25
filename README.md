# MC_Skin_Generator
Generate new Minecraft Skins with an Autoencoder.  
Run app.py or standalone dist/MSG.exe to start the application.  

- The 128 sliders represent the latent space of the AE
- The *intensity* slider changes the range of the main sliders from 0 to 10 (resulting range: -x to x)

| Keybind | Description |
| ----------- | ----------- |
| R | Randomize sliders |
| Shift + R | Randomize sliders as normal distribution |
| M | Randomize sliders maximal (-1, 0, 1) |
| K | Knock sliders a bit |
| 0 | Reset sliders to 0 |
| J | Move sliders (randomizing/knocking sliders changes velocities) |
| H | Converge sliders to equilibrium by feeding the output back into the input of the AE |
| D | Double feed through: like 'H' but just once |
| S | Save skin as an image |
| L | Load a skin from an image |

---
uses [MinePI](https://github.com/benno1237/MinePI) to render 3D Skin
