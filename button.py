from gpiozero import Button
from signal import pause

GPIO = 4
button = Button(GPIO, hold_time=2.5)

def button_pressed():
    print("Button was pressed.")

def button_held():
    print("Button was held.")

def button_released():
    print("Button released.")

button.when_pressed = button_pressed ##runs function when button is pressed
button.when_held = button_held  ##runs function when button is held for 2.5 sec
button.when_released = button_released ##runs function when button is released

pause()  ##keeps program listening for events