from PIL import Image

def add_margin(pil_img, top, right, bottom, left, color, flag= False):
    width, height = pil_img.size
    if width == 178 and height == 218:
        return None
    if (width > 178 or height>224) and not flag:
        return add_margin(im,-3,-7,-3,-7,(255,255,255),flag=True)
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def resize(im):
    return im.resize((192, 224))

for img_name in [(str(i).rjust(6, '0')+".jpg") for i in range(2001,202600)]:
    print(img_name)
    im = Image.open(f'data/img_align_celeba/{img_name}')
    im_new = add_margin(im, 3, 7, 3, 7, (255, 255, 255))
    # resize(im) #
    if im_new is not None:
        im_new.save(f'data/img_align_celeba/{img_name}', quality=100)
