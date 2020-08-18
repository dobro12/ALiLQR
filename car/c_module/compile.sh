g++ -fPIC -c ALiLQR.cpp -I/usr/local/include/python3.6m -I/usr/include/python3.6m
g++ -shared -o ALiLQR.so ALiLQR.o
