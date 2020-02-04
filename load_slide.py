# Load in pathology slides using OpenSlide
import openslide as opsl
import parameters as param


print("Opening slide object...")
my_slide = opsl.OpenSlide("{0}/393930.svs".format(param.dir_slides))  # pointer to slide object

print("Reading slide region...")
slide_image = my_slide.read_region(location=(5000,5000), level=1, size=(1000,1000))  # (x,y) of top-left corner, zoom level, (w,h) pixels

print("Saving slide object as PNG...")
slide_image.save("{0}/slide_test.png".format(param.dir_figures))  # save slide object as png

# now we need to find the (x,y) positions of the cores!

print("Closing slide object...")
my_slide.close()
print("Done")

