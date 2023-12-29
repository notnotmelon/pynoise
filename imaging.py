from PIL import Image
from module import Checkerboard

img = Image.new('L', (256, 256))
checkerboard = Checkerboard()

for x in range(256):
    for y in range(256):
        val = checkerboard.get_value(x/10, y/15, 0)
        val += 1
        val /= 2

        img.putpixel((x,y), val*255)

img.save('test', 'png')
