OBJLIST = gamboge_nnet_test.o main.o
FINAL = nnet.test
INCDIR = ../include
CTOOL = clang++
CPPFLAGS = -g

%.o: %.cpp
	$(CTOOL) $(CPPFLAGS) -I $(INCDIR) -c -o $@ $<

$(FINAL): $(OBJLIST)
	$(CTOOL) -o $(FINAL) $(OBJLIST)

clean:
	rm -f $(OBJLIST) $(FINAL)

main.o: main.cpp

gamboge_nnet_test.o: gamboge_nnet_test.cpp $(INCDIR)/gamboge/nnet.h
