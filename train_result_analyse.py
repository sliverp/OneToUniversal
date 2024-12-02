def parse(s):
    try:
        return float(s.split(":")[-1])
    except:
        return 0.0

with open("/home/liyh/OneToUniversal/train.log", 'r', encoding='utf-8') as fp:
    psnrs = []
    mses = []
    ssims = []
    losses = []
    while line := fp.readline():
        if line.startswith('[epoch 1]'):
            items = line.split(":  ")[0].split("|")[1:]
            losses.append(parse(items[0]))
            mses.append(parse(items[1]))
            psnrs.append(parse(items[2]))
            ssims.append(parse(items[3]))
    print(f"loss: {sum(losses)/len(losses)}")
    print(f"psnr: {sum(psnrs)/len(psnrs)}")
    print(f"ssim: {sum(ssims)/len(ssims)}")
    print(f"mse: {sum(mses)/len(mses)}")

    
    