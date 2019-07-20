import win32api, win32con
from random import randint
from run_exe import run
import time
import autopy
error_cnt =0
def smooth_move(point):
    global error_cnt

    while True:
        try:
            x= point[0]
            y = point[1]
            autopy.mouse.smooth_move(x,y)
            error_cnt =0
            break
        except:
            error_cnt+=1
            print("mouse setcursor error :", error_cnt)
            time.sleep(2)
            if error_cnt == 10:
                run('C:\\opcv\\recapcha\\exe\\consoleOn.exe')
                exit()
            continue

def move(point):
    global error_cnt

    while True:
        try:
            x= point[0]
            y = point[1]
            win32api.SetCursorPos((x,y))
            error_cnt =0
            break
        except:
            error_cnt+=1
            print("mouse setcursor error :", error_cnt)
            time.sleep(2)
            if error_cnt == 10:
                run('C:\\opcv\\recapcha\\exe\\consoleOn.exe')
                exit()
            continue
        
def click(point):
    tmp = win32api.GetCursorPos()
    x= point[0]
    y = point[1]
    move(point)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    
    move(tmp)
    
def click_random_point_in_rect(rect,margin=10):
    x,y,w,h = rect
    random_x = randint(x+margin, x+w-margin)
    random_y = randint(y+margin, y+h-margin)
    random_point = [random_x, random_y]
    click(random_point)
