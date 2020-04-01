import socket
from .my_module import RecommendationAlgoritm

ra = RecommendationAlgoritm()

# Задаем адрес сервера
SERVER_ADDRESS = ('localhost', 8686)

# Настраиваем сокет
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(SERVER_ADDRESS)
server_socket.listen(10)
print('server is running, please, press ctrl+c to stop')

class Progress_bar:
    def __init__(self, id_thread, connection):
        self._progress = 0
        self._connection = connection
    
    def set_progress(self, p: float):
        self._progress = p
        self._connection.send(stuck.pack('f', float(p)))

# Слушаем запросы
while True:
    connection, address = server_socket.accept()
    print("new connection from {address}".format(address=address))

    type_call = connection.recv(1)[0]
    if type_call == 0: # get_recommendation
        user_id = data_call.recv(1024)
        try:
            train_model(progress_bar)
        finally:
            
    elif type_call == 1: # train_model
        

    connection.send(bytes('Hello from server!', encoding='UTF-8'))

    connection.close() 

