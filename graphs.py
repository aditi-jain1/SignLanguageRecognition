# Import necessary libraries (if you are running a copy of this program, you may need to pip install some libraries)
import pygame
import random
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tensorflow as tf
import pyttsx3
import threading
from textblob import TextBlob

# Initialize video settings
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 50 # Padding for resizing hand image
imgSize = 300


# Initialize text-to-speech settings
speechCounter = 0
text_speech = pyttsx3.init("nsss")
speak_label = None
speak_counter = 0
def speak_text(text):
    text_speech.say(text)
    text_speech.runAndWait()
pronunciations = ["ay", "bee", "see", "dee", "ee", "eff", "gee", "aych", "eye", "jay", "kay", "ell", "em", "en", "oh", "pee", "cue", "arr", "ess", "tee", "you", "vee", "double-you", "ex", "why", "zee"]
signed_words = ["", "", ""]

# Initializing Pygame window
pygame.init()
width, height = 900, 830 # Pygame window size
window = pygame.display.set_mode((width, height))
pygame.display.set_caption('Sign Language Prediction')

# Initialize bar graph settings
bar_height = 15
barsList = []
xInitial =  50 # Reference x coordinate for bar graph
yInitial = 410 # Reference y coordinate for bar graph
predictionConfidence = 1.00
# Colors : 1: blue, 2: green, 3: bright yellow, 4: yellow, 5: red, 6: pink
barColors = [(27, 231, 255), (110, 235, 131), (228, 255, 26), (255, 184, 0), (255, 87, 20), (251, 98, 246)]
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# Create Bar class to initialize, store, and display data
class Bar:
    def __init__(self, x, y, length, color, label):
        self.x = x
        self.y = y
        self.height = bar_height
        self.length = length 
        self.color = color
        self.label = label
    def update_length(self, newLength):
        self.length = newLength
    def draw(self):
        pygame.draw.rect(window, self.color, [self.x, self.y, self.length, self.height])
        font = pygame.font.Font(None, 20)
        text = font.render(self.label, True, (187, 218, 255))
        window.blit(text, (self.x- 20, self.y))  # Adjust position as needed

# Initialize bar graph objects representing prediction probabilities for each letter
for i in range(26):
    barsList.append(Bar(xInitial, yInitial + bar_height*i, random.randint(50, 200), random.choice(barColors), labels[i]))
    
# Create divs to define seperate application segments
def create_rounded_rect(width, height, radius, color):
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    rect = pygame.Rect((0, 0, width, height))
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    return surf

# Define divs
graphBackground= create_rounded_rect(350, 440, 15, (61, 109, 161))  # Adjust width, height, radius, and color
vidBackgrounds = create_rounded_rect(870, 275, 10, (0,17,37))
translationBackgrounds = create_rounded_rect(490, 440, 15, (61, 109, 161))
logisticsBackground = create_rounded_rect(470, 100, 8, (0,17,37))
dividerBar = create_rounded_rect(6 ,60, 5, (61, 109, 161))
word1Back = create_rounded_rect(470, 70, 8, (95, 151, 209))
word2Back = create_rounded_rect(470, 70, 8, (80, 131, 186))
word3Back = create_rounded_rect(470, 70, 8, (69, 119, 172))

# Define toggle button to switch between audio mute and unmute settings
# U = Unmute, M = Mute
class toggleButton:
    def __init__(self, x, y, state, color1, color2):
        self.size = 35
        self.x = x
        self.y = y
        self.state = state
        self.color1 = color1
        self.color2 = color2
        self.buffer = 0
    
    def draw(self):
        backdrop = create_rounded_rect(70, self.size, self.size, self.color1)
        window.blit(backdrop, (self.x, self.y))
        circSwitch = pygame.draw.circle(window, self.color2, (self.x + math.ceil(self.size/2.3)+2+self.buffer, self.y + math.ceil(self.size/2.3)+2), math.ceil(self.size/2.6))
        font = pygame.font.Font(None, 27)
        if self.state == 1:
            text = font.render("U", True, (187, 218, 255))
        if self.state == 0:
            text = font.render("M", True, (187, 218, 255))
        window.blit(text, (self.x+self.buffer+10, self.y+10)) 
        text = font.render("Audio", True, (187, 218, 255))
        window.blit(text, (self.x-80, self.y+1)) 
        font = pygame.font.Font(None, 18)
        text = font.render("UNMUTE | MUTE", True, self.color2)
        window.blit(text, (self.x-107, self.y+22)) 

    def setState(self, state):
        self.state = state
        # Button animation
        if state == 1 and self.buffer != 0:
            self.buffer -= 4
        if state == 0 and self.buffer != 36:
            self.buffer += 4
    
# Initialize button settings
buttonState = 1
toggle = toggleButton(805, 15, 0, (9, 40, 83), (61, 109, 161))     


running = True

# Run program
while running:
    # Set up events to recognize key presses and define associated actions
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            # M key = Mute
            if event.key == pygame.K_m:
                buttonState = 0
            # U key = Unmute
            if event.key == pygame.K_u:
                buttonState = 1
            # S key = Speak current word
            if event.key == pygame.K_s:
                blob = TextBlob(signed_words[-1])
                # Perform autocorrection on word
                signed_words[-1] = str(blob.correct())
                speak_text(signed_words[-1])
                signed_words.append("")
            # C key = Clear most recent letter
            if event.key == pygame.K_c:
                signed_words[-1] = ""

    # Set up camera and hand-detector
    success, img = cap.read()
    hands, img = detector.findHands(img)
    frame_h, frame_w, _ = img.shape
    videoDisplay = cv2.resize(img, (math.ceil(frame_w/5), math.ceil(frame_h/5)))
    imgWhite = [0]
    prediction = [0 for i in range(26)] #Initialize confidence scores for each letter to be 0
    lmx = None 
    index = -1
    predictionConfidence = "100%"

    #Image Processing
    if hands:
        hand = hands[0] # Recognition is solely performed on the first identified hand
        x, y, w, h = hand["bbox"]

        imgWhite = np.ones([imgSize, imgSize, 3], np.uint8)
        imgCrop = img[y- offset:y+h + offset, x-offset:x+w+offset]
        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(labels[index])
        
        if aspectRatio <= 1:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hGap+ hCal,:] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(labels[index])

        predictionConfidence = str(round((prediction[index]/sum(prediction))*100, 2)) +"%" # Get prediction confidence
        landmarks = hand["lmList"]  # Get hand landmarks
        lmx = [] # Landmark x coordinates
        lmy = [] # Landmark y coordinates
        for lm in landmarks:
            lmx.append(lm[0])
            lmy.append(lm[1])

    window.fill((9, 40, 83))  # Clear the window
    pygame.draw.rect(window, (0,0,0), [0, 0, width, 65])
    font = pygame.font.Font('AbrilFatface-Regular.ttf', 50)
    text = font.render("gesture", True, (187, 218, 255))
    window.blit(text, (21, -8))
    window.blit(vidBackgrounds, (15, 80))
    window.blit(graphBackground, (20, 375))
    window.blit(translationBackgrounds, (390,375))
    window.blit(logisticsBackground, (400,480))
    window.blit(dividerBar, (580, 500))
    window.blit(word1Back, (400, 600))
    window.blit(word2Back, (400, 670))
    window.blit(word3Back, (400, 740))
    font = pygame.font.Font("PTSerif-Regular.ttf", 18)
    text = font.render("IMAGE PROCESSING", True, (187, 218, 255))
    window.blit(text, (30, 90))
    text = font.render("PREDICTION CONFIDENCE", True, (187, 218, 255))
    window.blit(text, (30, 385))
    font = pygame.font.Font('AbrilFatface-Regular.ttf', 50)
    text = font.render(signed_words[-1], True, (187, 218, 255))
    window.blit(text, (440, 600))
    text = font.render(signed_words[-2], True, (187, 218, 255))
    window.blit(text, (440, 670))
    text = font.render(signed_words[-3], True, (187, 218, 255))
    window.blit(text, (440, 740))
    text = font.render(labels[index], True, (187, 218, 255))
    window.blit(text, (530, 495))
    text = font.render(predictionConfidence, True, (187, 218, 255))
    window.blit(text, (600, 495))

    #Display instructions
    instruct_x = 405
    instruct_y = 385
    instruct_size = 15
    tColor = (0, 17, 37)
    font = pygame.font.Font(None, 22)
    text = font.render("The prediction and confidence are displayed below. If you are", True, tColor)
    window.blit(text, (instruct_x, instruct_y+(0*instruct_size)+5))
    text = font.render("not receiving the desired result, consider slightly adjusting", True, tColor)
    window.blit(text, (instruct_x, instruct_y+(1*instruct_size)+5))
    text = font.render("your gesture or consult the training data. Press the space bar", True, tColor)
    window.blit(text, (instruct_x, instruct_y+(2*instruct_size)+5))
    text = font.render("to signal the end of the word and use the C key to clear the", True, tColor)
    window.blit(text, (instruct_x, instruct_y+(3*instruct_size)+5))
    text = font.render("screen. Press U key to unmute and M to mute audio.", True, tColor)
    window.blit(text, (instruct_x, instruct_y+(4*instruct_size)+5))
    
    '''
    Instruction Template:
    The prediction and confidence are displayed below. If you are
    not receiving the desired result, consider slightly adjusting
    your gesture or consult the training data. Press the space bar
    to signal the end of the word and use the C key to clear the
    screen. Press U key to unmute and M to mute audio.
    '''

    barsIndex = 0
    #draw Landmarks
    if lmx:
        width_desired = 150
        height_desired = 150
        x1 = 690
        y1 = 130
        width_lmx = max(lmx) - min(lmx)
        height_lmx = max(lmy) - min(lmy)
        width_factor = width_desired/width_lmx
        height_factor = height_desired/height_lmx
        max_change = min(width_factor, height_factor)
        for i in range(len(lmx)):
            # Assign colors to landmarks sorted by finger
            if i <= 4:
                colorTemp = barColors[5]
            elif i <= 8:
                colorTemp = barColors[4]
            elif i <= 12:
                colorTemp = barColors[3]
            elif i<= 16:
                colorTemp = barColors[1]
            else:
                colorTemp = barColors[0]
            x_scaled = int((lmx[i]-min(lmx))*max_change+x1)
            y_scaled = int((lmy[i] - min(lmy))*max_change + y1)
            pygame.draw.circle(window, colorTemp, (x_scaled, y_scaled), 5)
    speechCounter += 1
    # Check conditions to speak a label
    if speak_label is None and index != -1 and speak_counter%30 == 0 and buttonState:
        signed_words[-1] += labels[index]
        speak_label = pronunciations[index]
    if speak_label:
        speak_text(speak_label)
        speak_label = None
    speak_counter += 1
    # Update and draw barsList
    for i in barsList:
        i.update_length(prediction[barsIndex]*200)
        i.draw()
        barsIndex += 1
        toggle.setState(buttonState)
    
    toggle.draw()
    frame = cv2.cvtColor(videoDisplay, cv2.COLOR_BGR2RGB)
    frame = pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "RGB")
    window.blit(frame, (30, 120))
    if len(imgWhite) != 1:
        # Resize and display zoomed in hand
        labeledImage = imgWhite
        frame_h, frame_w, _ = labeledImage.shape
        videoDisplay = cv2.resize(labeledImage, (math.ceil(frame_w/1.5), math.ceil(frame_h/1.5)))
        frame = cv2.cvtColor(videoDisplay, cv2.COLOR_BGR2RGB)
        frame = pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "RGB")
        window.blit(frame, (435, 120))

    
    pygame.display.flip()

cap.release()
pygame.quit()

