def test_inference():

    image = Image.open("data/img_align_celeba/000001.jpg")
    transform=transforms.ToTensor()

    encodings = []
    with torch.no_grad():
        mu, sigma = model.encode(transform(image).unsqueeze(0))
        encodings.append((mu, sigma))

    mu, sigma = encodings[0]

    epsilon = torch.randn_like(sigma)
    z = mu + sigma * epsilon
    out = model.decode(z)
    out = out.view(-1, 3, 224, 192)
    save_image(out, f"generated_ex.png")


test_inference()