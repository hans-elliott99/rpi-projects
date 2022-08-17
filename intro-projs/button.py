from gpiozero import Button
from signal import pause

GPIO = 4
press_counter = 0
press_limit = 3
button = Button(GPIO, hold_time=2.5)

def button_pressed():
    print("PRESSED.")
    #press_counter += 1

def button_held():
	print("HELD.")

def button_released():
	print("RELEASED.")

button.when_pressed = button_pressed ##runs function when button is pressed
button.when_held = button_held  ##runs function when button is held for 2.5 sec
button.when_released = button_released ##runs function when button is released

if press_counter >= press_limit:
    print("that's a lot of pressing")
    press_counter = 0

pause()  ##keeps program listening for events
