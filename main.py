from train import train, parse_opt
from generate import generate


def main(opt):
    train(**vars(opt))
    # generate(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
