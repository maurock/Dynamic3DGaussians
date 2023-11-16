import math

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))