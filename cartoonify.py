
import math
import _helper
import sys


def separate_channels(image):
    sep_channels_lst = []
    num_colors = len(image[0][0])
    for i in range(num_colors):
        sep_channels_lst.append([])
        for r in range(len(image)):
            sep_channels_lst[i].append([])
            for c in range(len(image[r])):
                sep_channels_lst[i][r].append(image[r][c][i])
    return sep_channels_lst


def combine_channels(channels):
    comb_channels_lst = []
    num_colours = len(channels)
    for r in range(len(channels[0])):
        comb_channels_lst.append([])
        for c in range(len(channels[0][0])):
            comb_channels_lst[r].append([])
            for i in range(num_colours):
                comb_channels_lst[r][c].append(channels[i][r][c])
    return comb_channels_lst

# function that calculates the value of a gray shade
def gray_shade(channel):
    red = channel[0]
    green = channel[1]
    blue = channel[2]
    num_gray = round(red * 0.299 + green * 0.587 + blue * 0.114)
    return num_gray


def RGB2grayscale(colored_image):
    gray_image = []
    for r in range(len(colored_image)):
        gray_image.append([])
        for c in range(len(colored_image[r])):
            gray_image[r].append(gray_shade(colored_image[r][c]))
    return gray_image


def blur_kernel(size):
    kernel_lst = []
    for r in range(size):
        kernel_lst.append([])
        for c in range(size):
            kernel_lst[r].append(1/size**2)
    return kernel_lst


def apply_kernel(image, kernel):
    new_image = []
    for r in range(len(image)):
        if r < len(image):
            new_image.append([])
        for c in range(len(image[r])):
            if r < len(image) and c < len(image[0]):
                new_image[r].append(pixels_value(image, r, c, kernel))

    return new_image


# function that returns an updated value of a specific pixel
def pixels_value(image, p_row, p_col, kernel):
    sum_pix = 0.0
    pixels_matrix = []
    row_count = 0
    k = len(kernel)
    for i in range(p_row - (k // 2), p_row + (k // 2) + 1):
        pixels_matrix.append([])
        for j in range(p_col - (k // 2), p_col + (k // 2) + 1):
            if i < 0 or j < 0 or i >= len(image) or j >= len(image[i]):
                pixels_matrix[row_count].append(image[p_row][p_col])
            else:
                pixels_matrix[row_count].append(image[i][j])
        row_count += 1
    for r in range(k):
        for c in range(len(kernel[r])):
            sum_pix += (pixels_matrix[r][c] * kernel[r][c])
    if sum_pix > 255:
        sum_pix = 255
    elif sum_pix < 0:
        sum_pix = 0
    return round(sum_pix)


# function that receives a specific pixel from an destination image and a list that represents the corners of the pixel [a, b, c, d]
def corners_value(image, y, x):
    # if y is a whole number
    if (y - int(y) == 0) is True:
        a_row = int(y)
        c_row = int(y)
        b_row = int(y)
        d_row = int(y)
    else:
        a_row = int(math.floor(y))
        b_row = int(math.ceil(y))
        c_row = int(math.floor(y))
        d_row = int(math.ceil(y))
    # if x is a whole number
    if (x - int(x) == 0) is True:
        a_col = int(x)
        b_col = int(x)
        c_col = int(x)
        d_col = int(x)
    else:
        a_col = int(math.floor(x))
        b_col = int(math.floor(x))
        c_col = int(math.ceil(x))
        d_col = int(math.ceil(x))

    a = image[a_row][a_col]
    b = image[b_row][b_col]
    c = image[c_row][c_col]
    d = image[d_row][d_col]

    return [a, b, c, d]


def bilinear_interpolation(image, y, x):
    corners = corners_value(image, y, x)
    a = corners[0]
    b = corners[1]
    c = corners[2]
    d = corners[3]

    # in order to calculate the value of the pixel we need y, x to be between 0 - 1
    # new_y / new_x = is calculated to be between these values
    if (y - int(y) == 0) is True:
        new_y = 0
    else:
        new_y = y - math.floor(y)

    if (x - int(x) == 0) is True:
        new_x = 0
    else:
        new_x = x - math.floor(x)

    des_pix_value = a * (1 - new_x)*(1 - new_y) + b * new_y * (1-new_x) + c * new_x * (1 - new_y) + d * new_x * new_y
    return round(des_pix_value)


# function that builds a matrix. the values of each argument in it is -1
def empty_matrix(l, k):
    new_matrix = []
    for i in range(l):
        new_matrix.append([])
        for j in range(k):
            new_matrix[i].append(-1)
    return new_matrix


def resize(image, new_height, new_width):
    l = new_height
    k = new_width
    y = None
    x = None
    new_image = empty_matrix(l, k)
    for i in range(l):
        for j in range(k):
            y = i * (len(image) - 1) / (l - 1)
            x = j * (len(image[0]) - 1) / (k - 1)
            new_image[i][j] = bilinear_interpolation(image, y, x)
    return new_image


def rotate_90(image, direction):
    n = len(image)
    m = len(image[0])
    rotated_image = empty_matrix(m, n)
    if direction == "R":
        for i in range(n):
            for j in range(m):
                rotated_image[j][n-1-i] = image[i][j]
    if direction == "L":
        for i in range(n):
            for j in range(m):
                rotated_image[m-1-j][i] = image[i][j]
    return rotated_image


def get_edges(image, blur_size, block_size, c):
    blur_image = apply_kernel(image, blur_kernel(blur_size))
    avg_image = apply_kernel(blur_image, blur_kernel(block_size))
    for i in range(len(image)):
        for j in range(len(image[i])):
            if blur_image[i][j] < avg_image[i][j] - c:
                avg_image[i][j] = 0
            else:
                avg_image[i][j] = 255
    return avg_image


def quantize(image, N):
    quant_img = empty_matrix(len(image), len(image[0]))
    for i in range(len(quant_img)):
        for j in range(len(quant_img[i])):
            quant_img[i][j] = round(math.floor(image[i][j]*(N/255))*(255/N))
    return quant_img


def quantize_colored_image(image, N):
    sep_colors = separate_channels(image)
    for i in range(len(sep_colors)):
        sep_colors[i] = quantize(sep_colors[i], N)
    return combine_channels(sep_colors)


def add_mask(image1, image2, mask):
    new_image = []
    if type(image1[0][0]) is list:
        channel1 = separate_channels(image1)
        channel2 = separate_channels(image2)
        for i in range(len(image1[0][0])):
            new_image.append(add_mask_for_2D(channel1[i], channel2[i], mask))
        new_image = combine_channels(new_image)
    else:
        new_image = add_mask_for_2D(image1, image2, mask)

    return new_image


def add_mask_for_2D(image1, image2, mask):
    new_image = empty_matrix(len(image1), len(image1[0]))
    for i in range(len(image1)):
        for j in range(len(image1[i])):
            new_image[i][j] = round(image1[i][j] * mask[i][j] + image2[i][j] * (1 - mask[i][j]))

    return new_image


def cartoonify(image, blur_size, th_block_size, th_c, quant_num_shades):
    b_w_image = RGB2grayscale(image)
    edge_img = get_edges(b_w_image, blur_size, th_block_size, th_c)
    quant_img = quantize_colored_image(image, quant_num_shades)
    mask = get_edges(b_w_image, blur_size, th_block_size, th_c)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            mask[i][j] = mask[i][j] / 255
    img2 = separate_channels(image)
    for k in range(len(img2)):
        img2[k] = add_mask(img2[k], edge_img, mask)
    img2 = combine_channels(img2)
    return img2


# the function creats an image in new proportion
def new_proportion(image, max_im_size):
    if len(image) > len(image[0]):
        prop = round((len(image)[0] * max_im_size) / len(image))
        sep_channel = separate_channels(image)
        for i in range(len(sep_channel)):
            sep_channel[i] = resize(sep_channel[i], max_im_size, prop)
    else:
        prop = round((len(image) * max_im_size) / len(image[0]))
        sep_channel = separate_channels(image)
        for i in range(len(sep_channel)):
            sep_channel[i] = (resize(sep_channel[i], prop, max_im_size))
    return combine_channels(sep_channel)


if __name__ == '__main__':
    if len(sys.argv) == 8:
        image = _helper.load_image(sys.argv[1])
        cartoon_dest = sys.argv[2]
        max_im_size = int(sys.argv[3])
        blur_size = int(sys.argv[4])
        th_block_size = int(sys.argv[5])
        th_c = int(sys.argv[6])
        quant_num_shades = int(sys.argv[7])
        if len(image) > max_im_size or len(image[1]) > max_im_size:
            image_resized = new_proportion(image, max_im_size)
        cartoon_img = cartoonify(image_resized, blur_size, th_block_size, th_c, quant_num_shades)
        _helper.save_image(cartoon_img, "cartoon_file.jpg")
        _helper.show_image(cartoon_img)
    else:
        print("unexpected numbers of arguments, please input exactly 7 arguments")













