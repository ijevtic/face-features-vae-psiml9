from PIL import Image

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

for img_name in [(str(i).rjust(6, '0')+".jpg") for i in range(2,202600)]:
    im = Image.open(f'data/img_align_celeba/{img_name}')
    im_new = add_margin(im, 3, 7, 3, 7, (255, 255, 255))
    im_new.save(f'data/img_align_celeba/{img_name}', quality=100)