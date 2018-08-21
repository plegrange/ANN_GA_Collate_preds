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
    request = "".encode('utf-8')
    for heading in df.columns.values:
        request = request + str(heading).encode('utf-8') + ",".encode('utf-8')
    request = request + "\n".encode('utf-8')
    s.send(request)
    for item, value in df.iterrows():
        print("Adding row")
        request = str(item).encode('utf-8')+",".encode('utf-8')
        for entry in value:
            request = request + str(entry).encode('utf-8')+",".encode('utf-8')
        request = request + "\n".encode('utf-8')
        try:
            print("# Sending row to server")
            s.send(request)
            print('# Row sent to server')
        except socket.error:
            print('Send failed')
            sys.exit()
    request = "END\n".encode('utf-8')
    try:
        print("# End of file")
        s.send(request)
        print('# File complete')
    except socket.error:
        print('Send failed')
        sys.exit()

