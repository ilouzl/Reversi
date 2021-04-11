import sys,tty,termios

class _Getch:
	def __call__(self, l = 1):
		fd = sys.stdin.fileno()
		old_settings = termios.tcgetattr(fd)
		try:
			tty.setraw(sys.stdin.fileno())
			ch = sys.stdin.read(l)
		finally:
			termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
		return ch

def get_key():
	inkey = _Getch()
	while(1):
		k=inkey()
		if k == '\x1b':
			k = inkey(l=2)
			if k[0] == "[" and k[1] in "ABCD":
				break
		if k!='':break
	if k=='[A':
		# print("up")
		return "UP"
	elif k=='[B':
		# print("down")
		return "DOWN"
	elif k=='[C':
		# print("right")
		return "RIGHT"
	elif k=='[D':
		# print("left")
		return "LEFT"
	elif k=='\r':
		# print("enter")
		return "ENTER"
	else:
		# print("not an arrow key!" + k)
		return k

if __name__=='__main__':
	for i in range(20):
		get_key()