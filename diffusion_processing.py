import cv2
import numpy as np
import imageio





in_folder = 'in_pics/'
out_folder = 'out_pics/'

gas_map_folder = 'gas_map_spread/'
gas_folder = 'gas_dencity_spread/'

dencity_file_name = 'dencity'
dencity_map_name = 'map_with_dencity'

# pic_name = 'binary_plann'
pic_name = 'binary_plan'
npy_dir = 'npy_gas_dencity'





def np_array_map_to_cv_img(array):
    height, width = array.shape
    img = np.zeros((height, width, 3))
    img[:,:,0] = 255*(1 - array[:,:])
    img[:,:,1] = 255*(1 - array[:,:])
    img[:,:,2] = 255*(1 - array[:,:])
    return img



arr = np.load('coord_out/plan_binary.npy')
map_img = np_array_map_to_cv_img(arr)
cv2.imwrite('plan.png', map_img)



map_img = cv2.imread(out_folder + '/' + pic_name + '.png')

height, width, depth = map_img.shape
print(width, height, depth)
y, x = int(width*(1 - 0.2)), int(height*(1 - 0.4))



gas_source = [x,y]
source_len = 10
D = 0.12
num_of_times = 300


dencity = np.zeros((width, height), np.uint8)
dencity[:, :] = 0
dencity[x - source_len : x + source_len, y - source_len : y + source_len] = 1


def np_array_to_cv_img(array):
    img = np.zeros((height, width, 3))
    img[:,:,0] = 0
    img[:,:,1] = 255*(array[:,:])
    img[:,:,2] = 255*(array[:,:])
    return img


def gas_spreading(init_dencity, num_of_steps):
    prev_dencity = init_dencity
    for t in range(num_of_steps):
        dencity = gas_spread_step(prev_dencity, t)
        prev_dencity = dencity


def gas_spread_step(prev_dencity, step):

    dencity = calc_cur_dencity(prev_dencity, step)
    cv2.imwrite(gas_folder + dencity_file_name + '_' + str(step)  + '.png', 255*(1 - dencity))
    map_with_dencity = map_img - np_array_to_cv_img(dencity)
    
    cv2.imwrite(gas_map_folder + dencity_map_name + '_' + str(step)  + '.png', map_with_dencity)
    return dencity




def is_on_pic(x,y):
    return 1 < x and x < height - 1 and 1 < y and y < width - 1

def is_source(x,y):
    return (gas_source[0] - x)*(gas_source[0] - x) + (gas_source[1] - y)*(gas_source[1] - y) < source_len*source_len

def is_in_range(x, y, rad):
    return x*x + y*y < rad*rad

def is_wall(x,y):
    return sum(map_img[x, y]) < 255*2.8

def dist_from_src(x, y):
    return (gas_source[0] - x)*(gas_source[0] - x) + (gas_source[1] - y)*(gas_source[1] - y)


def calc_cur_dencity(prev_dencity, step):
    dencity = np.zeros((height, width), np.uint8)
    for x in range(2*step + 1):
        for y in range(2*step + 1):
            if(is_in_range(x - step, y - step, step)):
                xc = gas_source[0] - step + x
                yc = gas_source[1] - step + y
                if(is_on_pic(xc, yc)):
                    if(is_source(xc, yc)):
                        dencity[xc, yc] = 1
                    elif(not is_wall(xc, yc)):
                        dencity[xc, yc] = dencity[xc, yc] + dencity_change_calc(prev_dencity, xc, yc, step)

    return dencity



def dencity_change_calc(dencity, x, y, step):
    sum = dencity[x - 1, y - 1]
    sum = sum + dencity[x + 1, y - 1]
    sum = sum + dencity[x - 1, y + 1]
    sum = sum + dencity[x + 1, y + 1]
    t = step/num_of_times
    coef = 1/(4*3.1415*D*D*t)**(1/2)*2.71**(-dist_from_src(x,y)/(width*height)/(4*D*D*t))
    return sum*coef

def generate_gif(num_of_steps, file_name, folder_name):
    filenames = list()
    for num in range(num_of_steps):
        filenames.append(folder_name + file_name + '_' + str(num) + '.png')
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('gifs/' + file_name + '_movie.gif', images)



def convert_gas_dencity_to_npy(num_of_files):
    for num in range(num_of_files):
        img = cv2.imread(gas_folder + '/' + dencity_file_name + '_' + str(num) + '.png')
        iimg = np.mean(img, axis=2)
        with open(npy_dir + '/' + dencity_file_name + '_' + str(num) + '.npy', 'wb') as f:
            np.save(f, 1 - iimg)



gas_spreading(dencity, num_of_times)
generate_gif(num_of_times, dencity_file_name, gas_folder)
generate_gif(num_of_times, dencity_map_name, gas_map_folder)
convert_gas_dencity_to_npy(num_of_times)



