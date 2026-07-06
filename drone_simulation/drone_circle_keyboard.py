import pygame

pygame.init()

# Window
width, height = 800, 600
win = pygame.display.set_mode((width, height))
pygame.display.set_caption("Drone Simulator")

# Drone (circle)
x, y = width // 2, height // 2
speed = 5

running = True

while running:
    pygame.time.delay(30)

    # Events (close window)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Keyboard control
    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        y -= speed
    if keys[pygame.K_DOWN]:
        y += speed
    if keys[pygame.K_LEFT]:
        x -= speed
    if keys[pygame.K_RIGHT]:
        x += speed

    # Drawing
    win.fill((0, 0, 0))  # black background
    pygame.draw.circle(win, (0, 255, 0), (x, y), 20)
    pygame.display.update()

pygame.quit()