import serial
import datetime
import time
import busio
from board import SCL, SDA
import argparse

# Import the PCA9685 module.
from adafruit_pca9685 import PCA9685

def logging(filename):
    # Create the I2C bus interface.
    i2c_bus = busio.I2C(SCL, SDA)

    # Create a simple PCA9685 class instance.
    pca = PCA9685(i2c_bus)

    # Set the PWM frequency to 60Hz.
    pca.frequency = 90

    # Set the PWM duty cycle for channel zero to 50%. duty_cycle is 16 bits to match other PWM objects
    # But the PCA9685 will only actually give 12 bits of resolution.
    # pca.channels[0].duty_cycle = 0x1B30   # servo left, 1108/10280
    # pca.channels[0].duty_cycle = 0x2380   # servo center, 1444/10280
    # pca.channels[0].duty_cycle = 0x2E60   # servo right, 1888/10280

    # pca.channels[1].duty_cycle = 0x2D40   # throttle forward, 1840/10280(10400)
    # pca.channels[1].duty_cycle = 0x2140   # throttle stop, 1352/10280(10400)
    # pca.channels[1].duty_cycle = 0x1A70   # throttle backward, 1076/10280(10400)

    left = 0x1B30
    center = 0x2380
    right = 0x2E60
    forward = 0x2D40
    stop = 0x2140
    backward = 0x1A70

    pca.channels[0].duty_cycle = center
    pca.channels[1].duty_cycle = stop

    f = open(filename, 'w')

    with serial.Serial('/dev/ttyUSB0', 9600, timeout=10) as ser: # open serial port
        start = input('Do you want to start logging? ')[0]
        if start in 'yY':
            ser.write(bytes('YES\n', 'utf-8'))
            while True:
                ser_in = ser.readline().decode('utf-8')
                print(ser_in)
                f.write("{} {}".format(datetime.datetime.now(), ser_in))
                print("{} {}".format(datetime.datetime.now(), ser_in), end='')
                steer = int(ser_in.split(' ')[2].split('/')[0])
                throttle = int(ser_in.split(' ')[5].split('/')[0])

                if int(steer) <= 1000:
                    steer = int(0x2380)
                elif int(steer) <= 1444:
                    steer = int(((0x2380 - 0x1B30) / (1444 - 1108)) * (steer -1444) + 0x2380)
                elif int(steer <= 2000):
                    steer = int(((0x2E60 - 0x2380) / (1888 - 1444)) * (steer - 1444) + 0x2380)
                else:
                    steer = int(ceter)
                
                if int(throttle) <= 1000:
                    throttle = int(0x2140)
                elif int(throttle) >= 1352:
                    throttle = int(((0x2D40 - 0x2140) / (1840 - 1352)) * (throttle - 1352) + 0x2140)
                else:
                    throttle = int(((0x2140 - 0x1A70) / (1352 - 1076)) * (throttle - 1352) + 0x2140)

                pca.channels[0].duty_cycle = steer
                pca.channels[1].duty_cycle = throttle
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('log_file_name', metavar='F', type=str, nargs=1, help="log file's name")
    args = parser.parse_args()
    logging(args.log_file_name[0])

