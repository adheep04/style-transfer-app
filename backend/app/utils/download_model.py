from torchvision import models


def main():
    print('loading model...')
    models.vgg19(weights=models.VGG19_Weights)
    print('download complete!')

    
if __name__ == '__main__':
    main()