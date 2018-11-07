# 13 Adversarial Training and GANs

## Artist-Critic Co-Evolution
* Critic is rewarded for distinguishing real images vs generated artist images
* Artist is producing image, and is rewarded for fooling the critic into thinking it is real

## Co-Evolution Paradigms
### Blind Watchmaker
* Human user chooses best images, and after various generations gets better images
    * Artist: Genetic program 
    * Critic: Human
* PicBreeder
    * Artist: Convolutional Pattern Producing NN
    * Critic: Human

### Evolutonary Art (GP Artist, GP or NN Critic)

### Generative Adversarial Networks
* GANS: a **generator (artist)** and a **discriminator (critic)**, both Deep CNNs