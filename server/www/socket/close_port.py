import socket
import sys

if len(sys.argv) == 2:
    # create a socket object
    serversocket = socket.socket(
        socket.AF_INET, socket.SOCK_STREAM)

    # get local machine name
    host = socket.gethostname()

    port = int(sys.argv[1])

    # bind to the port
    serversocket.close()
else:
    print("please sent 1 argument for port")

