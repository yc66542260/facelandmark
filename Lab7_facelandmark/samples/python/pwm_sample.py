from machine import PWM
import time
pwm = PWM('pwm',0,1000,240)
time.sleep(1)
pwm.init(0,1000,200)
time.sleep(1)
pwm.init(0,1000,160)
time.sleep(1)
pwm.init(0,1000,120)
time.sleep(1)
pwm.init(0,1000,80)
time.sleep(1)
pwm.init(0,1000,40)
time.sleep(1)
pwm.init(0,1000,0)
time.sleep(1)
pwm.init(0,1000,40)
time.sleep(1)
pwm.init(0,1000,80)
time.sleep(1)
pwm.init(0,1000,120)
time.sleep(1)
pwm.init(0,1000,160)
time.sleep(1)
pwm.init(0,1000,200)
time.sleep(1)
pwm.init(0,1000,240)
pwm.deinit()