import socket

# create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# get local machine name
host = socket.gethostbyname(socket.gethostname())
print(host)
port = 1493

# connection to hostname on the port.
s.connect(('192.168.56.1', port))

# Receive no more than 1024 bytes
msg = s.recv(1024)

s.close()

print (msg.decode('ascii'))