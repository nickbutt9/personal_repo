import pygame
import neat
import time
import os
import random
pygame.font.init()

# setting window dimensions
WIN_HEIGHT = 800
WIN_WIDTH = 500
GENERATIONS = 0

# grabbing the images for the game and scaling them by 2x
BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))
SCORE_FONT = pygame.font.SysFont("comicsans", 35)


class Bird:
    # defining some class constants for animation of the bird
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x  # starting x position
        self.y = y  # starting y position
        self.tilt = 0  # how tilted the bird images are
        self.tick_count = 0  # keeps track of when last flapped
        self.vel = 0  # velocity of the bird
        self.height = self.y  # height of the bird in space
        self.img_count = 0  # which bird image is being shown
        self.img = self.IMGS[0]  # the bird images

    def flap(self):
        self.vel = -10.5  # makes the bird jump up
        self.tick_count = 0
        self.height = self.y

    # this is called every frame update to move the bird
    def move(self):
        self.tick_count += 1  # one more frame since last flapped

        # tells how much bird is moving based on when last tick was
        d = self.vel*self.tick_count + 1.5*self.tick_count**2

        # setting a terminal velocity
        if d >= 16:
            d = 16

        # fine-tuning the feeling of flapping upwards
        if d < 0:
            d -= 2

        # updating the y position
        self.y = self.y + d

        # rotating the bird
        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        # choosing which bird image to show
        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        # if tilted ~straight down, make the bird wings level and stop flapping until tilted up again
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2

        # how much to rotate the image
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        # making the image rotate according to the bird location instead of the top left of the screen
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    # getting collisions
    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 180  # how far the pipes are apart
    VEL = 5  # how fast the pipes move

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)  # flipping the pipe image
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False  # if the bird has passed the pipe
        self.set_height()

    def set_height(self):  # setting random heights for the pipes
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP  # making the top and bottom pipes be GAP distance apart

    def move(self):  # moving the pipe with the set VEL
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        # using masks to get pixel perfect collision detection instead of just square hit-boxes
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        # how far the bird is from the top and bottom pipe
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        # returns a point if there is any overlap, otherwise returns None if no collision
        top_point = bird_mask.overlap(top_mask, top_offset)
        bottom_point = bird_mask.overlap(bottom_mask, bottom_offset)

        if top_point or bottom_point:
            return True
        return False


class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        # moves two instances of BASE_IMG
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        # cycles the two images, i.e. when x1 image gets off the screen on the left, move it to far right of the screen
        # doing this ensures that the screen always has seamless base/ground movement
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, birds, pipes, base, score, generation):
    win.blit(BG_IMG, (0,0))
    for pipe in pipes:
        pipe.draw(win)
    base.draw(win)
    for bird in birds:
        bird.draw(win)
    text = SCORE_FONT.render(f'Score: {score}', 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))
    text = SCORE_FONT.render(f'Generation: {generation}', 1, (255, 255, 255))
    win.blit(text, (10, 10))
    text = SCORE_FONT.render(f'Birds alive: {len(birds)}', 1, (255, 255, 255))
    win.blit(text, (10, 60))
    pygame.display.update()


def main(genomes, config):
    global GENERATIONS
    GENERATIONS += 1
    neural_nets = []
    genes = []
    birds = []

    for _, g in genomes:
        neural_net = neat.nn.FeedForwardNetwork.create(g, config)
        neural_nets.append(neural_net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        genes.append(g)

    base = Base(730)
    pipes = [Pipe(600)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    score = 0

    # the main game loop
    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        # determining which pipe we should be using as input to the neural networks
        pipe_index = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_index = 1
        else:  # if no birds left, go to next generation
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            genes[x].fitness += 0.05

            output = neural_nets[x].activate((bird.y, abs(bird.y - pipes[pipe_index].height),
                                              abs(bird.y - pipes[pipe_index].bottom)))
            if output[0] > 0.5:  # checking output of first output since only using 1 output neuron
                bird.flap()

        remove = []
        add_pipe = False
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    genes[x].fitness -= 1  # loosing fitness when hitting pipe to encourage going between them
                    birds.pop(x)
                    neural_nets.pop(x)
                    genes.pop(x)

                # if the pipe gets past, add a new pipe to pipes
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            # set the pipe to be removed from pipes if it's off the screen
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                remove.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            for g in genes:
                g.fitness += 2
            pipes.append(Pipe(600))

        for rem in remove:
            pipes.remove(rem)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                neural_nets.pop(x)
                genes.pop(x)

        base.move()
        draw_window(win, birds, pipes, base, score, GENERATIONS)


# setting up neat for genetic algorithm to pick good birds
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))  # shows some stats for the population
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(main, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
