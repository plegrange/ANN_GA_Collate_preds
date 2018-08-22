import socket
import sys

host = '53.35.153.99'
port = 5555

print('# Creating socket')
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except socket.error:
    print('Failed to create socket')
    sys.exit()

print('# Getting remote IP address')
try:
    remote_ip = socket.gethostbyname(host)
except socket.gaierror:
    print('Hostname could not be resolved. Exiting')
    sys.exit()


def connect_to_server():
    print('# Connecting to server ' + host)
    s.connect((remote_ip, port))
    reply = s.recv(1024)
    print(reply)


def send_data_to_server(filename, df):
    print('# Building request message')
    request = "INCOMING NAME\n".encode('utf-8')
    request = request + filename.encode('utf-8')+"\n".encode('utf-8')
    request = request + " ,".encode('utf-8')
    print("Sending file name to server")
    s.send(request)
    request = ""
    for heading in df.columns.values:
        request = request + str(heading) + ","
    request = request + "\n"
    s.send(request.encode('utf-8'))
    for r in range(1, df.shape[0]):
        request = ""
        print("Adding row")
        for c in range(0, df.shape[1]):
            request = request + str(df.iloc[r, c]) + ","
        request = request + "\n"
        s.send(request.encode('utf-8'))
    request = "END\n".encode('utf-8')
    try:
        print("# End of file")
        s.send(request)
        print('# File complete')
    except socket.error:
        print('Send failed')
        sys.exit()

