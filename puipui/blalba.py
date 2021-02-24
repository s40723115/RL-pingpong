import pygame, sys, random
  2. from pygame.locals import *
  3.
  4. # Set up pygame.
  5. pygame.init()
  6. mainClock = pygame.time.Clock()
  7.
  8. # Set up the window.
  9. WINDOWWIDTH = 400
 10. WINDOWHEIGHT = 400
 11. windowSurface = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT),
       0, 32)
 12. pygame.display.set_caption('Collision Detection')
 13.
 14. # Set up the colors.
 15. BLACK = (0, 0, 0)
 16. GREEN = (0, 255, 0)
 17. WHITE = (255, 255, 255)
 18.
 19. # Set up the player and food data structures.
 20. foodCounter = 0
 21. NEWFOOD = 40
 22. FOODSIZE = 20
 23. player = pygame.Rect(300, 100, 50, 50)
 24. foods = []
 25. for i in range(20):
 26.     foods.append(pygame.Rect(random.randint(0, WINDOWWIDTH - FOODSIZE),
           random.randint(0, WINDOWHEIGHT - FOODSIZE), FOODSIZE, FOODSIZE))
 27.
 28. # Set up movement variables.
 29. moveLeft = False
 30. moveRight = False
 31. moveUp = False
 32. moveDown = False
 33.
 34. MOVESPEED = 6
 35.
 36.
 37. # Run the game loop.
 38. while True:
 39.     # Check for events.
 40.     for event in pygame.event.get():
 41.         if event.type == QUIT:
42.             pygame.quit()
 43.             sys.exit()
 44.         if event.type == KEYDOWN:
 45.             # Change the keyboard variables.
 46.             if event.key == K_LEFT or event.key == K_a:
 47.                 moveRight = False
 48.                 moveLeft = True
 49.             if event.key == K_RIGHT or event.key == K_d:
 50.                 moveLeft = False
 51.                 moveRight = True
 52.             if event.key == K_UP or event.key == K_w:
 53.                 moveDown = False
 54.                 moveUp = True
 55.             if event.key == K_DOWN or event.key == K_s:
 56.                 moveUp = False
 57.                 moveDown = True
 58.         if event.type == KEYUP:
 59.             if event.key == K_ESCAPE:
 60.                 pygame.quit()
 61.                 sys.exit()
 62.             if event.key == K_LEFT or event.key == K_a:
 63.                 moveLeft = False
 64.             if event.key == K_RIGHT or event.key == K_d:
 65.                 moveRight = False
 66.             if event.key == K_UP or event.key == K_w:
 67.                 moveUp = False
 68.             if event.key == K_DOWN or event.key == K_s:
 69.                 moveDown = False
 70.             if event.key == K_x:
 71.                 player.top = random.randint(0, WINDOWHEIGHT -
                       player.height)
 72.                 player.left = random.randint(0, WINDOWWIDTH -
                       player.width)
 73.
 74.         if event.type == MOUSEBUTTONUP:
 75.             foods.append(pygame.Rect(event.pos[0], event.pos[1],
                   FOODSIZE, FOODSIZE))
 76.
 77.     foodCounter += 1
 78.     if foodCounter >= NEWFOOD:
 79.         # Add new food.
 80.         foodCounter = 0
 81.         foods.append(pygame.Rect(random.randint(0, WINDOWWIDTH -
               FOODSIZE), random.randint(0, WINDOWHEIGHT - FOODSIZE),
               FOODSIZE, FOODSIZE))
 82.
 83.     # Draw the white background onto the surface.
 84.     windowSurface.fill(WHITE)
 85.
 86.     # Move the player.
 87.     if moveDown and player.bottom < WINDOWHEIGHT:
 88.         player.top += MOVESPEED
 89.     if moveUp and player.top > 0:
 90.         player.top -= MOVESPEED
91.     if moveLeft and player.left > 0:
 92.         player.left -= MOVESPEED
 93.     if moveRight and player.right < WINDOWWIDTH:
 94.         player.right += MOVESPEED
 95.
 96.     # Draw the player onto the surface.
 97.     pygame.draw.rect(windowSurface, BLACK, player)
 98.
 99.     # Check whether the player has intersected with any food squares.
100.     for food in foods[:]:
101.         if player.colliderect(food):
102.             foods.remove(food)
103.
104.     # Draw the food.
105.     for i in range(len(foods)):
106.         pygame.draw.rect(windowSurface, GREEN, foods[i])
107.
108.     # Draw the window onto the screen.
109.     pygame.display.update()
     mainClock.tick(40)