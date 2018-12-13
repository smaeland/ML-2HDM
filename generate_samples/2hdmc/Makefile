# Compile 2HDMC point calculation binary 

calculate_point:
	g++ calculate_point.cpp \
	-I${TWOHDMCPATH}/src \
	-L${TWOHDMCPATH}/lib \
	-L${HIGGSBOUNDSPATH} \
	-L${HIGGSSIGNALSPATH} \
	-l2HDMC -lgsl -lgslcblas -lm  \
	-DHiggsBounds -lHB -lHS -lgfortran \
	-Wno-gnu-static-float-init \
	-o calculate_point


all:
	calculate_point


clean:
	rm calculate_point
