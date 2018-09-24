PROG		= test
CXX			= g++
CXXFLAGS	= -std=c++17
LIB			= lib
INCLUDE		= include

$(PROG): rbm-mnist.cpp $(LIB)/libsensei.so $(INCLUDE)
	$(CXX) $(CXXFLAGS) -o $@ $< -I$(INCLUDE) -L$(LIB) -larmadillo -lsensei -Wl,-rpath,$(LIB)

$(LIB)/libsensei.so: ../build/libsensei.so
	mkdir -p $(LIB)
	cp $< $@

$(INCLUDE): ../build/include
	cp -r $< .

.PHONY: clean ../build/libsensei.so ../build/include

clean:
	rm -rf .o $(PROG) $(LIB) $(INCLUDE)

