import string
import platform

def fix(a_str):
    if platform.system() == 'Windows':
        a_str = a_str.replace('/','\\')
    return a_str

if __name__ == "__main__":
    print(fix("Hearthstone myndir/Amani enraged full.png"))