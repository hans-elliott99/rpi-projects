import csv
from pathlib import Path
from datetime import datetime
from gpiozero import LED, Buzzer, MotionSensor
from signal import pause

led = LED(14) 
buzzer = Buzzer(15)
motion_sensor = MotionSensor(4, threshold=0.25)

output_csv_path = Path("motion_detect.csv")
motion = {
    "start_time":None,
    "end_time":None
}

def write_to_csv():
    first_write = not output_csv_path.is_file()
    
    with open(output_csv_path, "a") as file:
        field_names = motion.keys()
        writer = csv.DictWriter(file, field_names)
        if first_write:
            writer.writeheader()
        writer.writerow(motion)

def start_motion():
    led.blink(0.5, 0.5)
    buzzer.beep(0.5, 0.5)
    motion["start_time"] = datetime.now()
    print("motion detected")

def end_motion():
    if motion["start_time"]:
        led.off()
        buzzer.off()
        motion["end_time"] = datetime.now()
        write_to_csv()
        # Reset time dict
        motion["start_time"] = None
        motion["end_time"] = None
    print("motion ended")

# Active
print("Readying sensor...")
motion_sensor.wait_for_no_motion()
print("Sensor ready")

motion_sensor.when_motion = start_motion
motion_sensor.when_no_motion = end_motion

pause()



