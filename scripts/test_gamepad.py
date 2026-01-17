import vgamepad as vg
import keyboard
import time

gamepad = vg.VX360Gamepad()

# Configuration
SMOOTHING = 0.15  # Adjust between 0.01 (very slow) and 1.0 (instant)
is_active = True

# Internal state to track current joystick positions
# [LeftX, LeftY, RightX, RightY]
current_pos = [0.0, 0.0, 0.0, 0.0]

def lerp(current, target, speed):
    return current + (target - current) * speed

print("--- Smooth Drone Controller ---")
print("F1: Toggle | Q: Quit")

keyboard.add_hotkey('f1', lambda: globals().update(is_active=not is_active) or print("Toggled"))

try:
    while True:
        if keyboard.is_pressed('q'):
            break

        if is_active:
            # 1. Determine TARGETS based on keys
            target_ls_x = 32767 if keyboard.is_pressed('d') else (-32768 if keyboard.is_pressed('a') else 0)
            target_ls_y = 32767 if keyboard.is_pressed('w') else (-32768 if keyboard.is_pressed('s') else 0)
            target_rs_x = 32767 if keyboard.is_pressed('right') else (-32768 if keyboard.is_pressed('left') else 0)
            target_rs_y = 32767 if keyboard.is_pressed('up') else (-32768 if keyboard.is_pressed('down') else 0)
            
            targets = [target_ls_x, target_ls_y, target_rs_x, target_rs_y]

            # 2. Smoothly move CURRENT toward TARGET
            for i in range(4):
                current_pos[i] = lerp(current_pos[i], targets[i], SMOOTHING)

            # 3. Apply to Gamepad (converting floats back to integers)
            gamepad.left_joystick(int(current_pos[0]), int(current_pos[1]))
            gamepad.right_joystick(int(current_pos[2]), int(current_pos[3]))
            gamepad.update()
        
        time.sleep(0.01)

finally:
    gamepad.reset()
    gamepad.update()

