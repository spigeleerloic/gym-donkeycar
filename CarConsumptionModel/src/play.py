import pygame
import numpy as np
from donkey_environment.ConsumptionWrapper import ConsumptionWrapper
from gym_donkeycar.envs.donkey_env import DonkeyEnv
# Initialize pygame
pygame.init()

# Set up display
SCREEN_WIDTH = 1280  # Increase the screen width
SCREEN_HEIGHT = 720  # Increase the screen height
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Car Control")

# Initialize environment
#env = DonkeyEnv("mountain_track")
env = ConsumptionWrapper("steep-ascent")
# PLAY
obs = env.reset()
done = False

# Define initial steering and throttle values
steering = 0.0
throttle = 0.0

STEERING_SENSITIVITY = 0.35
THROTTLE_SENSITIVITY = 1.0

clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # Adjust steering based on arrow keys
    if keys[pygame.K_LEFT]:
        steering = - STEERING_SENSITIVITY
    elif keys[pygame.K_RIGHT]:
        steering = STEERING_SENSITIVITY
    else:
        steering = 0.0

    # Adjust throttle based on arrow keys
    if keys[pygame.K_UP]:
        throttle = THROTTLE_SENSITIVITY
    elif keys[pygame.K_DOWN]:
        throttle = - THROTTLE_SENSITIVITY
    else:
        throttle = 0.0
    
    if keys[pygame.K_SPACE]:
        print(info, reward)

    action = np.array([steering, throttle])

    # Take a step in the environment
    obs, reward, done, info = env.step(action)

    # Render the observation
    screen.fill((0, 0, 0))  # Clear the screen

    # Convert the observation array into a pygame surface
    obs_surface = pygame.surfarray.make_surface(obs.swapaxes(0, 1))

    # Resize the surface to match the screen dimensions
    obs_resized = pygame.transform.scale(obs_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))

    # Blit the resized observation onto the screen
    screen.blit(obs_resized, (0, 0))

    pygame.display.flip()  # Update the display

    # Limit the frame rate
    clock.tick(60)

    if done:
        obs = env.reset()

pygame.quit()

