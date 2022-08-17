from gpiozero import MotionSensor
from signal import pause

GPIO = 4
motion_sensor = MotionSensor(GPIO, threshold=0.25)

def motion():
    print("Motion detected")

def no_motion():
    print("Motion stopped")

print("Readying sensor")
motion_sensor.wait_for_no_motion() ##pause script until no motion
print("Sensor ready")

motions=0
max_motions=5
#motion_sensor.when_motion = motion
#motion_sensor.when_no_motion = no_motion
while True:
# Allows motion detection to continue until max_motions is surpassed 
    motion_sensor.wait_for_motion() ##pause script until motion
    print("Motion detected!")
    motions += 1
    motion_sensor.wait_for_no_motion() ##pause script until no motion
    print(f"Total Motions:", motions)
    if motions > max_motions:
        print("Shutting down")
        break

#pause() 

