import pyvjoy
import time

# Controller Test
testController = pyvjoy.VJoyDevice(1)

testController.set_button(8,1)

time.sleep(10)

testController.set_button(8,0)