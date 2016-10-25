FINAL = nnet.test
OBJLIST = gamboge_nnet_test.o main.o
CPPFLAGS = -g -I../include
LDLIBS = -lcppunit

$(FINAL): $(OBJLIST)
	$(CXX) -o $(FINAL) $(OBJLIST) $(LDLIBS)

clean:
	rm -f $(OBJLIST) $(FINAL)

main.o: main.cpp

gamboge_nnet_test.o: gamboge_nnet_test.cpp ../include/gamboge/nnet.h
