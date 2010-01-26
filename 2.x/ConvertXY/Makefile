all: example

cxx_extensions.o: /usr/share/python2.5/CXX/cxx_extensions.cxx
	g++ -c /usr/share/python2.5/CXX/cxx_extensions.cxx -I/usr/lib/python2.6/include -I/usr/include/python2.6 -o cxx_extensions.o

cxxextensions.o: /usr/share/python2.5/CXX/cxxextensions.c
	g++ -c /usr/share/python2.5/CXX/cxxextensions.c -I/usr/lib/python2.6/include -I/usr/include/python2.6 -o cxxextensions.o

cxxsupport.o: /usr/share/python2.5/CXX/cxxsupport.cxx
	g++ -c /usr/share/python2.5/CXX/cxxsupport.cxx -I/usr/lib/python2.6/include -I/usr/include/python2.6 -o cxxsupport.o

IndirectPythonInterface.o: /usr/share/python2.5/CXX/IndirectPythonInterface.cxx
	g++ -c /usr/share/python2.5/CXX/IndirectPythonInterface.cxx -I/usr/lib/python2.6/include -I/usr/include/python2.6 -o IndirectPythonInterface.o

example.o: example.cpp
	g++ example.cpp -I/usr/local/include -I/usr/lib/python2.6/include -I/usr/include/python2.6 -c -o example.o

example: example.o cxx_extensions.o cxxextensions.o cxxsupport.o IndirectPythonInterface.o
	g++ example.o cxx_extensions.o cxxextensions.o cxxsupport.o IndirectPythonInterface.o -lpython2.6 -L/usr/lib/python2.6 -o example