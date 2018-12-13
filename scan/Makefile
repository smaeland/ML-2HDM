

all:
	g++ ratio_scanner.cpp -std=c++11 \
	-I$(TWOHDMCPATH)/src \
	-I$(ROOTSYS)/include \
	-L$(TWOHDMCPATH)/lib \
	-L$(HIGGSBOUNDSPATH) \
	-L$(HIGGSBOUNDSPATH) \
	-L$(ROOTSYS)/lib \
	-l2HDMC -lgsl -lgslcblas -lm  \
	-DHiggsBounds -lHB -lHS -lgfortran \
	-lCore \
	-lGraf -lGraf3d -lHist -lHtml -lMatrix -lMinuit \
	-lPostscript -lProof -lTree \
	-lGpad -lGui -lGX11 -lRint \
	-lX11 -lm \
	-Wno-gnu-static-float-init \
	-o ratio_scanner

clean:
	rm ratio_scanner
