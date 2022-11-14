# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:57:03 2022

@author: David Gomez
"""

import pgzrun
import pygame
import pgzero
from pgzero.builtins import Actor
from random import randint

# Set the size of screen
WIDTH = 800
HEIGHT = 600

# Prep new player-controller actor
balloon = Actor("balloon")
balloon.pos = 400, 300

# Prep new obstacle actors
bird = Actor("bird-up")
bird.pos = randint(800, 1600), randint(10,200)

bird2 = Actor("bird-up")
bird2.pos = randint(800, 1600), randint(10,200)

house = Actor("house")
house.pos = randint(800, 1600), 460

house2 = Actor("house")
house2.pos = randint(800, 1600), 460

tree = Actor("tree")
tree.pos = randint(800, 1600), 450

tree2 = Actor("tree")
tree2.pos = randint(800, 1600), 450

# Global variable declarations
bird_up = True
bird2_up = True
up = False
game_over = False
score = 0
number_of_updates = 0
number_of_updates2 = 0 
lives = 3

scores = []

def update_high_scores():
    global score, scores
    filename = r"/Users/david/Dropbox/SJSU/Fall 2022/EE104/Lab_7/balloon_fight/high-scores.txt"
    scores = []
    # Open high-scores.txt file for READING
    with open(filename, "r") as file:
        line = file.readline() # reads single line in txt file
        high_scores = line.split() # splits the three scores into three diff strings
        for high_score in high_scores: # Loops through list of highscores, checks if player's current highscore greater than the highscores in text file
            if (score > int(high_score)):
                scores.append(str(score) + " ")
                score = int(high_score)
            else:
                scores.append(str(high_score) + " ")
    # open high-scores.txt for WRITING
    with open(filename, "w") as file:
        # Write the top three scores into the file
        for high_score in scores: 
            file.write(high_score)

def display_high_scores():
    screen.draw.text("HIGH SCORES", (350, 150), color="black")
    y = 175
    position = 1
    for high_score in scores:
        screen.draw.text(str(position) + ". " + high_score, (350,y), color="black")
        y+=25
        position+=1

def draw():
    global bird_up, up, game_over, score, number_of_updates, scores, lives
    screen.blit("background", (0,0))
    if not game_over:
        balloon.draw()
        bird.draw()
        bird2.draw()
        house.draw()
        tree.draw()
        house2.draw()
        tree2.draw()
        screen.draw.text("Score: " + str(score), (700,5), color="black")
        screen.draw.text("Lives: " + str(lives), (700,20), color="black")
    else: # if game is over then display the highscore on the screen
        display_high_scores()

# Mouse reactions
def on_mouse_down():
    global up
    up = True
    balloon.y -= 50 # ballon will rise on mouse down by 50 pixels
    
def on_mouse_up():
    global up
    up = False

# make the bird1 flap it's wings
def flap():
    global bird_up 
    if bird_up: # If bird wings are up, switch the image to the bird flapping downwards
        bird.image = "bird-down"
        bird_up = False
    else:
        bird.image = "bird-up"
        bird_up = True

# make the bird2 flap it's wings
def flap2():
    global bird2_up 
    if bird2_up: # If bird wings are up, switch the image to the bird flapping downwards
        bird2.image = "bird-down"
        bird2_up = False
    else:
        bird2.image = "bird-up"
        bird2_up = True

# Runs 60 times a second automatically
def update():
    global game_over, score, number_of_updates, number_of_updates2, lives
    # Gravity will cause balloon to fall as long as game is not over
    if not game_over:
        if not up:
            balloon.y += 1
    # Bird1 flies to the left across screen
    if bird.x > 0:
        bird.x -= 8
        # After a couple of frames, have the bird flap its wings up and down
        if number_of_updates == 9:
            flap()
            number_of_updates = 0
        else:
            number_of_updates += 1
    # If bird disappears to the left, randomly place it off screen on the right and give one point to player
    else:
        bird.x = randint(800, 1600)
        bird.y = randint(10, 200)
        score += 1
        number_of_updates = 0;
    # Bird2 flies to the left across screen
    if bird2.x > 0:
        bird2.x -= 8
        # After a couple of frames, have the bird flap its wings up and down
        if number_of_updates2 == 9:
            flap2()
            number_of_updates2 = 0
        else:
            number_of_updates2 += 1
    # If bird disappears to the left, randomly place it off screen on the right and give one point to player
    else:
        bird2.x = randint(800, 1600)
        bird2.y = randint(10, 200)
        score += 1
        number_of_updates2 = 0;
    # House1 is "moving"
    if house.right > 0:
        house.x -= 6
    # If house1 disappears to the left of screen, place it randomly off screen and give point to player
    else:
        house.x = randint(800,1600)
        score += 1
    # House2 is "moving"
    if house2.right > 0:
        house2.x -= 2
    # If house2 disappears to the left of screen, place it randomly off screen and give point to player
    else:
        house2.x = randint(800,1600)
        score += 1
    # Tree1 is "moving"
    if tree.right > 0:
        tree.x -= 2
    # If tree1 disappears to the left of screen, place it randomly off screen and give point to play
    else:
        tree.x = randint(800,1600)
        score += 1
    # Tree2 is "moving"
    if tree2.right > 0:
        tree2.x -= 3
    # If tree2 disappears to the left of screen, place it randomly off screen and give point to play
    else:
        tree2.x = randint(800,1600)
        score += 1
    
    # Game ends if balloon hits top or bottom of screen
    if balloon.top < 0 or balloon.bottom > 560:
        lives -= 1
        balloon.pos = 400, 300
    
    # Handle collisions to obstacles, game should be over
    if balloon.collidepoint(bird.x, bird.y) or balloon.collidepoint(bird2.x, bird2.y) or balloon.collidepoint(house.x, house.y) or balloon.collidepoint(tree.x, tree.y):
        lives -= 1
        balloon.pos = 400, 300
        
    if lives == 0:
        game_over = True
        update_high_scores()
    
    
pgzrun.go()