import platform

def play_alert():
    system = platform.system()

    try:
        if system == "Windows":
            import winsound
            winsound.Beep(1000, 300)
        else:
            import os
            os.system('printf "\\a"')  # beep sound
    except:
        print("Alert triggered!")